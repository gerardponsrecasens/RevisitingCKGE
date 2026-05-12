from ..utils import *
from ..data_load.data_loader import *
from torch.utils.data import DataLoader
import torch.profiler as profiler
import numpy as np

class RetrainBatchProcessor():
    def __init__(self, args, kg):
        self.args = args
        self.kg = kg  # information of snapshot sequence
        '''prepare data'''
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),  # use seed generator
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model.train()
        '''Start training'''
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            '''get loss'''
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.loss(bh.to(self.args.device),
                                       br.to(self.args.device),
                                       bt.to(self.args.device),
                                       by.to(self.args.device) if by is not None else by).float()

            '''update'''
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
            '''post processing'''
            model.epoch_post_processing(bh.size(0))
        return total_loss

class TrainBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.dataset = TrainDatasetMarginLoss(args, kg)
        self.shuffle_mode = True
        # if self.args.use_multi_layers:
        #     self.shuffle_mode = False
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=self.shuffle_mode,
                                      batch_size=int(self.args.batch_size),
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)), # use seed generator
                                      pin_memory=True
                                      ) # real memory is enough, set pin_memory=True, faster!


    def process_epoch(self, model, optimizer):
        model.train()

        """ Start training """
        total_loss = 0.0
        transE_losses = 0.0
        # MAE_losses = 0.0
        token_training_losses = 0.0
        distillation_losses = 0.0
        for b_id, batch in enumerate(self.data_loader):

            """ Get loss """
            bh, br, bt, by = batch
            # print("Before training epoch:")
            # model.check_embedding_trainable_status()
            optimizer.zero_grad()
            if self.args.using_all_data:
                transE_loss,token_training_loss,distillation_loss,batch_loss = model.forward(batch)
            else:
                transE_loss,token_training_loss,distillation_loss,batch_loss =  model.forward(batch)
            """ updata """
            batch_loss.backward()

            optimizer.step()
            # print("After training epoch:")
            # model.check_embedding_trainable_status()
            total_loss += batch_loss.item()
            transE_losses += transE_loss.item()
            token_training_losses +=token_training_loss.item() if isinstance(token_training_loss, torch.Tensor) else token_training_loss
            distillation_losses += distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss
            """ post processing """
            model.epoch_post_processing(bh.size(0))
            if self.args.use_multi_layers and self.args.snapshot > 0:
                """ save previous results """
                old_ent_embeddings = model.old_data_ent_embeddings_weight
                old_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
                value = torch.cat([old_ent_embeddings[:old_len], model.ent_embeddings.weight[old_len:]], dim=0)
                model.register_buffer(f'old_data_ent_embeddings_weight', value.clone().detach())
            if self.args.record:
                with open(loss_save_path, "a", encoding="utf-8") as wf:
                    wf.write(str(batch_loss.item()))
                    wf.write("\t")
        if self.args.record:
            with open(loss_save_path, "a", encoding="utf-8") as wf:
                wf.write("\n")
        return transE_losses,token_training_losses,distillation_losses,total_loss

class DevBatchProcessor():
    def __init__(self, args, kg) -> None:
        self.args = args
        self.kg = kg
        self.batch_size = 100
        """ prepare data """
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.random_seed)),
                                      pin_memory=True)
    def corrupt(self, fact):
        """ generate pos/neg facts from pos facts """
        ss_id = self.args.snapshot
        h, r, t = fact
        prob = 0.5

        """
        random corrupt heads and tails
        1 pos + 10 neg = 11 samples
        """
        neg_h = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        neg_t = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
        pos_h = np.ones_like(neg_h) * h
        pos_t = np.ones_like(neg_t) * t
        rand_prob = np.random.rand(self.args.neg_ratio)
        head = np.where(rand_prob > prob, pos_h, neg_h)
        tail = np.where(rand_prob > prob, neg_t, pos_t)
        facts = [(h, r, t)]
        label = [1]
        for nh, nt in zip(head, tail):
            facts.append((nh, r, nt))
            label.append(-1)
        return facts, label

    def process_epoch(self, model):
        model.eval()

        # batch_size = int(self.args.batch_size * (1 + self.args.neg_ratio))
        batch_size = int(self.args.batch_size)
        # print(f'batch_size:',batch_size)
        num_entities = self.kg.snapshots[self.args.snapshot].num_ent
        num_relations = self.kg.snapshots[self.args.snapshot].num_rel
        dummy_fact = np.column_stack((
            np.random.randint(0, num_entities, batch_size),  # Head
            np.random.randint(0, num_relations, batch_size),  # Relation
            np.random.randint(0, num_entities, batch_size)   # Tail
        ))
        facts = []
        labels = []
        for (h,r,t) in dummy_fact:
            # print(f"Head: {h}, Relation: {r}, Tail: {t}")
            fact, label = self.corrupt((h,r,t))
            facts.append(fact)
            labels.append(label)

        fact_tensor = torch.tensor(facts, dtype=torch.long).cuda()
        label_tensor = torch.tensor(labels, dtype=torch.long).cuda()
        # Flatten fact_tensor and label_tensor
        fact_tensor = fact_tensor.view(-1, 3)  # Shape [3072 * 11, 3]
        label_tensor = label_tensor.view(-1)  # Shape [3072 * 11]

        # print(f'fact_tensor,label_tensor:',fact_tensor.size(),label_tensor.size())
        label_unsqueezed = label_tensor.unsqueeze(1)  # Shape (n, 1)

        # Concatenate fact and label along the last dimension
        combined_tensor = torch.cat((fact_tensor, label_unsqueezed), dim=1)  # Shape (n, 4)
        inputs = torch.split(combined_tensor, 1, dim=1)  # Split into 4 tensors along dim=1

        # Remove the extra dimension (unsqueeze effect)
        split_tensors = [x.squeeze(1) for x in inputs]
        # split_tensors = list(combined_tensor[:, i].unsqueeze(1) for i in range(combined_tensor.size(1))) # Creates a list of tensors
        # print(f"Combined tensor shape: {combined_tensor.shape}")
        # Wrap the inputs into a tuple of tensors
        # wrapped_inputs = (fact_tensor, label_tensor)
        wrapped_inputs = (split_tensors,)
        # Trace the model
        traced_model = torch.jit.trace(model, wrapped_inputs)
        # Save the traced model for later use
        # traced_model.save("traced_model.pt")


        num = 0
        results = {}
        hr2t = self.kg.snapshots[self.args.snapshot].hr2t_all
        """ Start evaluation """
        for batch in self.data_loader:
            # head: (batch_size, 1)
            head, relation, tail, label = batch
            head = head.to(self.args.device)
            relation = relation.to(self.args.device)
            tail = tail.to(self.args.device)
            label = label.to(self.args.device) # (batch_size, ent_num)
            num += len(head)
            stage = "Valid" if self.args.valid else "Test"
            """ Get prediction scores """
            pred = model.predict(head, relation, stage=stage) # (batch_size, num_ent)
            """ filter: filter: If there is more than one tail in the label, we only think that the tail in this triple is right """
            batch_size_range = torch.arange(pred.size()[0], device=self.args.device)
            target_pred = pred[batch_size_range, tail] # take off the score of tail in this triple
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred) # Set all other tail scores to negative infinity
            pred[batch_size_range, tail] = target_pred # restore the score of the tail in this triple
            if self.args.predict_result and stage == "Test":
                logits_sorted, indices_sorted = torch.sort(pred, dim=-1, descending=True)
                predict_result_path = "/data/my_cl_kge/save/predict_result/" + str(self.args.snapshot) + "_" + str(self.args.snapshot_test) + ".txt"
                with open(predict_result_path, "a", encoding="utf-8") as af:
                    batch_num = len(head)
                    for i in range(batch_num):
                        top1 = indices_sorted[i][0]
                        top2 = indices_sorted[i][1]
                        top3 = indices_sorted[i][2]
                        af.write(self.kg.id2entity[head[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2relation[relation[i].detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[tail[i].detach().cpu().item()])
                        af.write("\n")
                        af.write(self.kg.id2entity[top1.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top2.detach().cpu().item()])
                        af.write("\t")
                        af.write(self.kg.id2entity[top3.detach().cpu().item()])
                        af.write("\n")
                        af.write("----------------------------------------------------------")
                        af.write("\n")
            """ rank all candidate entities """
            """ Two sorts can be optimized into one """
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[batch_size_range, tail]
            ranks = ranks.float() # all right tail ranks, (batch_size, 1)
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results[f'hits{k + 1}'] = torch.numel(
                    ranks[ranks <= (k + 1)]
                ) + results.get(f'hits{k + 1}', 0.0)
        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results