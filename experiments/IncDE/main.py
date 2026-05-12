import shutil
from datetime import datetime
import logging
import json

from src.utils import *
from src.parse_args import args
from src.data_load.KnowledgeGraph import KnowledgeGraph
from src.model.DLKGE import TransE as DLKGE_TransE
from src.train import *
from src.test import *
import pickle

class Instructor():
    """ The instructor of the model """
    def __init__(self, args) -> None:
        # Ablation experiments
        if args.without_hier_distill:
            args.using_embedding_distill = False
            args.use_multi_layers = False
            args.use_two_stage = True
            args.using_mask_weight = False
            args.using_different_weights = False

        if args.without_two_stage:
            args.using_embedding_distill = True
            args.use_multi_layers = True
            args.use_two_stage = False
            args.using_mask_weight = True
            args.using_different_weights = True

        self.args = args

        """ 1. Prepare for path, logger and device """
        self.prepare()

        """ 2. Load data """
        self.kg = KnowledgeGraph(args)

        """ 3. Create models and optimizer """
        self.model, self.optimizer = self.create_model()

        self.args.logger.info(self.args)

    def create_model(self):
        """ Create KGE model and optimizer """
        model = DLKGE_TransE(self.args, self.kg)
        model.to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.args.learning_rate), weight_decay=self.args.l2)
        return model, optimizer

    def reset_model(self, model=False, optimizer=False):
        """
        Reset model or optimizer
        :param model: If True: reset the model and optimizer
        :param optimizer: If True: reset the optimizer
        """
        if model:
            self.model, self.optimizer = self.create_model()
        if optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args.learning_rate), weight_decay=self.args.l2)

    def prepare(self):
        """ Set data path """
        if not os.path.exists(args.data_path):
            os.mkdir(args.data_path)
        self.args.data_path = args.data_path + args.dataset + "/"

        """ Set save path """
        self.args.save_path = args.save_path + args.dataset
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if self.args.note != '':
            self.args.save_path += self.args.note
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        """ Set log path """
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path #+ datetime.now().strftime("%Y%m%d%H%M%S/")
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset +'-TransE-incDE'
        self.args.log_path = self.args.log_path + '-init' +str(args.init)+'-'+str(args.incremental_epochs)+'e-'+str(args.RN)+'RN'
        if self.args.note != "":
            self.args.log_path += self.args.note

        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = f'{args.log_path}.log'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        """ Set device """
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def next_snapshot_setting(self):
        """ Prepare for next snapshot """
        self.model.switch_snapshot()

    def run(self):
        """ Run the instructor of the model. The training process on all snapshots """
        report_results = PrettyTable()
        report_results.field_names = ['Snapshot', 'Time', 'Whole_MRR', 'Whole_Hits@1', 'Whole_Hits@3', 'Whole_Hits@10']
        json_report_results = {}
        test_results = []
        training_times = []
        BWT = [] # h(n, i) - h(i, i)
        FWT = [] # h(i- 1, i)
        first_learning_res = []
        embeddings_after = []
        embeddings_before = []

        """ training process """
        for ss_id in range(int(self.args.snapshot_num)):
            self.args.snapshot = ss_id
            self.args.snapshot_test = ss_id
            self.args.snapshot_valid = ss_id
            if self.args.use_multi_layers and self.args.using_different_weights:
                if ss_id == 4:
                    self.args.multi_layer_weight *= 10
            if ss_id > 0:
                self.args.epoch_num = self.args.incremental_epochs

            """ preprocess before training on a snapshot """
            self.model.pre_snapshot()
            if self.args.using_mask_weight:
                self.reset_model(optimizer=True)

            if ss_id > 0:
                self.args.test_FWT = True
                res_before = self.test()
                FWT.append(res_before['mrr'])
            self.args.test_FWT = False

            if ss_id == 0 and self.args.reuse_0:
                training_time = 360
                if 'PROVES' in self.args.dataset:
                    self.load_checkpoint('./checkpoint/0model_best_PROVES.tar',ss_id)
                else:
                    self.load_checkpoint('./checkpoint/0model_best.tar',ss_id)
            else:
                training_time = self.train()

            """ prepare result table """
            test_res = PrettyTable()
            test_res.field_names = [
                f'Snapshot:{str(ss_id)}',
                'MRR',
                'Hits@1',
                'Hits@3',
                'Hits@5',
                'Hits@10',
            ]

            if ss_id != 0 and not self.args.reuse_0:
                """ Save and reload the model """
                best_checkpoint = os.path.join(self.args.save_path, f'{str(ss_id)}model_best.tar')
                self.load_checkpoint(best_checkpoint)

            """ After the snapshot, the process of before prediction """
            self.model.snapshot_post_processing()

            """ predict """
            reses = [] # only number
            ss_results = {}
            only_full_results = {}

            for test_ss_id in range(ss_id + 1):
                self.args.snapshot_test = test_ss_id
                res = self.test() # predict results
                if test_ss_id == ss_id:
                    first_learning_res.append(res['mrr'])
                test_res.add_row([
                    test_ss_id, res['mrr'], res['hits1'], res['hits3'], res['hits5'], res['hits10']
                ])
                reses.append(res)

                ss_results[test_ss_id] = {
                'mrr': res['mrr'],
                'hits@1': res['hits1'],
                'hits@3': res['hits3'],
                'hits@10': res['hits10']}

                only_full_results[test_ss_id] = {'only':{
                'mrr': res['mrr'],
                'hits@1': res['hits1'],
                'hits@3': res['hits3'],
                'hits@10': res['hits10']}}


            # LETS CREATE MY COSTUME EVALUATION
            
            for test_ss_id in range(ss_id):
                self.args.snapshot_test = test_ss_id
                self.args.full_entities = self.kg.snapshots[ss_id].num_ent #actual number of entities
                res = self.test()

                only_full_results[test_ss_id]['full']= {
                'mrr': res['mrr'],
                'hits@1': res['hits1'],
                'hits@3': res['hits3'],
                'hits@10': res['hits10']}
            
            self.args.full_entities = None


            if ss_id == int(self.args.snapshot_num) - 1:
                BWT.extend(
                    reses[iid]['mrr'] - first_learning_res[iid]
                    for iid in range(int(self.args.snapshot_num) - 1)
                )
            """ Record all results """
            self.args.logger.info(f"\n{test_res}")
            test_results.append(test_res)

            """ record report results """
            whole_mrr, whole_hits1, whole_hits3, whole_hits10 = self.get_report_results(reses)
            report_results.add_row([ss_id, training_time, whole_mrr, whole_hits1, whole_hits3, whole_hits10])
            training_times.append(training_time)

            json_report_results[ss_id] = {
                'training_time': training_time,
                'mrr': whole_mrr,
                'hits@1': whole_hits1,
                'hits@3': whole_hits3,
                'hits@10': whole_hits10,
                'local': ss_results,
                'cf': only_full_results
            }

            ent,rel,ent2id,rel2id = self.model.return_weights()
            embeddings_after.append({'ent_emb':ent, 'rel_emb':rel,'ent2id':ent2id,'rel2id':rel2id})

            """ After the snapshot, the process after the process """
            if self.args.snapshot < int(self.args.snapshot_num) - 1:
                self.next_snapshot_setting() # Important steps, after prediction
                ent,rel,ent2id,rel2id = self.model.return_weights()
                embeddings_before.append({'ent_emb':ent, 'rel_emb':rel,'ent2id':ent2id,'rel2id':rel2id})
                self.reset_model(optimizer=True)

        self.args.logger.info(f'Final Result:\n{test_results}')
        self.args.logger.info(f'Report Result:\n{report_results}')
        self.args.logger.info(f'Sum_Training_Time:{sum(training_times)}')
        self.args.logger.info(f'Every_Training_Time:{training_times}')
        self.args.logger.info(
            f'Forward transfer: {sum(FWT) / len(FWT)} Backward transfer: {sum(BWT) / len(BWT)}'
        )

        json_report_results['settings'] = {
                'dataset': args.dataset,
                'update_technique': 'incDE',
                'epochs': args.incremental_epochs,
                'NNPP': args.neg_ratio,
                'learning_rate': args.learning_rate,
                'embedding_dimension': args.emb_dim,
                'seed': args.random_seed,
                'lr': args.learning_rate,
                'init': args.init,
                'RN': args.RN,
                'drop_rate':args.drop_rate,
                'noise_rate':args.noise_rate

            }

        # COMPUTE FINAL CORRECTED RESULTS
        mrrs, hits1s, hits3s, hits10s, num_tests = 0, 0, 0, 0, []
        for idx in range(int(self.args.snapshot_num)):
            num_test=len(self.kg.snapshots[idx].test)
            num_tests.append(num_test)
            if idx != int(self.args.snapshot_num)-1:
                mrrs += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['full']['mrr']*num_test
                hits1s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['full']['hits@1']*num_test
                hits3s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['full']['hits@3']*num_test
                hits10s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['full']['hits@10']*num_test

            else:
                mrrs += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['only']['mrr']*num_test
                hits1s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['only']['hits@1']*num_test
                hits3s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['only']['hits@3']*num_test
                hits10s += json_report_results[int(self.args.snapshot_num)-1]['cf'][idx]['only']['hits@10']*num_test



        mrrs = round(mrrs/sum(num_tests), 3)
        hits1s = round(hits1s/sum(num_tests), 3)
        hits3s = round(hits3s/sum(num_tests), 3)
        hits10s = round(hits10s/sum(num_tests), 3)
        
        json_report_results['corrected'] = {
                'mrr': mrrs,
                'hits@1': hits1s,
                'hits@3': hits3s,
                'hits@10': hits10s,
            }
        
        exp_id = args.dataset  + '-incDE-' + str(args.init) + '-' + str(args.learning_rate) + '-' + str(args.RN) + '-' + str(args.epoch_num) + '-' + str(args.random_seed) + str(time.time()).split('.')[0] + '.json'
        json_path = os.path.join('./results/', exp_id)
        with open(json_path, 'w') as f:
            json.dump(json_report_results, f, indent=4)



    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.snapshots[idx].test))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
            ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        """ Training process, return training time """
        start_time = time.time()
        print("Start training =============================")
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)

        """ Trainign iteration """
        for epoch in range(int(self.args.epoch_num)):
            self.args.epoch = epoch
            """ training """
            loss, valid_res = trainer.run_epoch()
            """ early stop """
            if self.args.using_test:
                if epoch > 2:
                    break
            if valid_res[self.args.valid_metrics] > self.best_valid:
                self.best_valid = valid_res[self.args.valid_metrics]
                # self.stop_epoch = max(0, self.stop_epoch - 5)
                self.stop_epoch = 0
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                self.save_model()
                if self.stop_epoch >= self.args.patience and epoch > 30: # Prevent stopping before fitting
                    self.args.logger.info(
                        f'Early Stopping! Snapshot:{self.args.snapshot} Epoch: {epoch} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break
            """ logging """
            if epoch % 1 == 0:
                self.args.logger.info(
                    f"Snapshot:{self.args.snapshot}\tEpoch:{epoch}\tLoss:{round(loss, 3)}\tMRR:{round(valid_res['mrr'] * 100, 3)}\tHits@10:{round(valid_res['hits10'] * 100, 3)}\tBest:{round(self.best_valid * 100, 3)}"
                )
        end_time = time.time()
        return end_time - start_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch # save other information
        out_tar = os.path.join(
            self.args.save_path,
            f'{str(self.args.snapshot)}checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.save_path, f'{str(self.args.snapshot)}model_best.tar'
            )
            shutil.copyfile(out_tar, best_path)

    def load_checkpoint(self, input_file,s_id=-1):
        if os.path.isfile(input_file):
            logging.info(f"=> loading checkpoint \'{input_file}\'")
            checkpoint = torch.load(input_file, map_location=f"cuda:{self.args.gpu}")
            self.model.load_state_dict(checkpoint['state_dict'])
            if s_id == 0:
                self.args.epoch = checkpoint['epoch_id']
            
        else:
            logging.info(f'=> no checking found at \'{input_file}\'')


""" Main function """
if __name__ == "__main__":
    set_seeds(int(args.random_seed))
    ins = Instructor(args)
    ins.run()