from .BaseModel import *


class Snapshot(BaseModel):
    def __init__(self, args, kg):
        super(Snapshot, self).__init__(args, kg)

    def switch_snapshot(self):
        '''
        Prepare for training on next snapshot
        '''
        '''expand embeddings for new entities and relations'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        self.ent_embeddings = ent_embeddings
        self.rel_embeddings = rel_embeddings
        '''reinitialize the embeddings'''
        self.reinit_param()
        return 0,0
    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id


class TransE(Snapshot):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: s
        :param rel: r
        :param tail: o
        :param label: label of positive (1) or negative (-1) facts
        :return: training loss
        '''
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss



