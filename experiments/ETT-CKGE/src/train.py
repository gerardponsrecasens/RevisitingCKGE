from .utils import *
from .model.model_process import *

class Trainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        # self.model = model
        self.transE = model
        self.optimizer = optimizer
        self.logger = args.logger
        if self.args.using_all_data:
            self.train_processor = RetrainBatchProcessor(args, kg)
        else:
            self.train_processor = TrainBatchProcessor(args, kg)
        self.valid_processor = DevBatchProcessor(args, kg)

    def run_epoch(self):
        self.args.valid = True
        # transE_loss,token_training_loss,distillation_loss,loss = self.train_processor.process_epoch(self.model, self.optimizer)
        transE_loss,token_training_loss,distillation_loss,loss = self.train_processor.process_epoch(self.transE, self.optimizer)
        # if self.args.using_multi_layer_distance_loss:
        #     # if res[self.args.valid_metrics] > best_valid:
        #     if not hasattr(self, 'first_layer_epoch_num'):
        #         pass
        #     elif self.args.epoch <= self.first_layer_epoch_num:
        #         value = self.model.ent_embeddings.weight
        #         # print(f'value:',value.size())
        #         self.model.register_buffer(f'first_layer_data_ent_embeddings_weight', value.clone().detach())
        #     elif self.args.epoch <= self.second_layer_epoch_num:
        #     # elif hasattr(self, 'first_layer_epoch_num') and hasattr(self, 'second_layer_epoch_num')
        #         value = self.model.ent_embeddings.weight
        #         # print(f'value:',value.size())
        #         self.model.register_buffer(f'second_layer_data_ent_embeddings_weight', value.clone().detach())
        # res = self.valid_processor.process_epoch(self.model)
        res = self.valid_processor.process_epoch(self.transE)
        self.args.valid = False
        return transE_loss,token_training_loss,distillation_loss,loss,res