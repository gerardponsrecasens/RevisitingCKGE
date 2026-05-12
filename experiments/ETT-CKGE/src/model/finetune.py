from .BaseModel import *
from torch_geometric.nn import GCN

class finetune(BaseModel):
    def __init__(self, args, kg):
        super(finetune, self).__init__(args, kg)
        self.gcn = MAE(args, kg)
        self.args = args
        self.kg = kg
        self.mse_loss_func = nn.MSELoss(size_average=False)
    def switch_snapshot(self):
        '''expand embeddings for new entities and relations '''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()

        '''inherit learned parameters'''
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(
            self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(
            self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

class TransE(finetune):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def MAE_loss(self):
        '''
        Calculate the MAE loss by masking and reconstructing embeddings.
        :return: MAE loss
        '''
        num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot].num_rel
        '''get subgraph(edge indexs and relation types of all facts in the training facts)'''
        edge_index = self.kg.snapshots[self.args.snapshot].edge_index
        edge_type = self.kg.snapshots[self.args.snapshot].edge_type

        '''reconstruct'''
        ent_embeddings, rel_embeddings = self.embedding('Train')
        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        ent_embeddings_reconstruct, rel_embeddings_reconstruct = self.gcn(ent_embeddings, rel_embeddings, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return(self.mse_loss_func(ent_embeddings_reconstruct, ent_embeddings[:num_ent]) / num_ent + self.mse_loss_func(
            rel_embeddings_reconstruct, rel_embeddings[:num_rel]) / num_rel)

    # def forward(self, head, rel, tail=None, label=None):
    def forward(self, inputs):
        head = inputs[0].to(self.args.device)
        rel = inputs[1].to(self.args.device)
        tail = inputs[2].to(self.args.device)
        label = inputs[3].to(self.args.device)
        loss = 0.0
        regular_loss = torch.tensor(0.0).to(self.args.device)
        new_loss = self.new_loss(head, rel, tail, label)
        # print(f'new_loss before loss:',new_loss)
        loss += new_loss
        if self.args.using_reconstruct_loss:
            MAE_loss = self.MAE_loss()
            loss += float(self.args.reconstruct_weight)*MAE_loss
        return loss,new_loss,MAE_loss,regular_loss

class MAE(nn.Module):
    def __init__(self, args, kg):
        super(MAE, self).__init__()
        self.args = args
        self.kg = kg
        '''masked KG auto encoder'''
        self.conv_layers = nn.ModuleList()
        for i in range(args.num_layer):
            self.conv_layers.append(ConvLayer(args, kg))

    def forward(self, ent_embeddings, rel_embeddings, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r):
        '''
        Reconstruct embeddings for all entities and relations
        :param x: input entity embeddings
        :param r: input relation embeddings
        :param edge_index: (s, o)
        :param edge_type: (r)
        :param num_ent: entity number
        :param num_rel: relation number
        :return: reconstructed embeddings
        '''
        x, r = ent_embeddings, rel_embeddings
        for i in range(self.args.num_layer):
            x, r = self.conv_layers[i](x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return x, r

    def get_weight(self, num_ent, num_rel):
        '''get the total number of samples containing each entity or relation'''
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        other_weight = edge_index.size(1)
        ent_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_index[0], dim_size=num_ent)
        rel_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_type, dim_size=num_rel)
        return ent_weight + 1, rel_weight + 1, other_weight

class ConvLayer(nn.Module):
    def __init__(self, args, kg):
        super(ConvLayer, self).__init__()
        self.args = args
        self.kg = kg
        self.GCN_model = GCN(in_channels=self.args.emb_dim, hidden_channels=200, \
                            out_channels=self.args.emb_dim, num_layers=3)
    def forward(self, x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r):
        '''
        Reconstruct embeddings for all entities and relations
        :param x: input entity embeddings
        :param r: input relation embeddings
        :param edge_index: (s, o)
        :param edge_type: (r)
        :param num_ent: entity number
        :param num_rel: relation number
        :return: reconstructed embeddings
        '''
        '''avoid the reliance for learned facts'''
        if old_entity_weight == None:  # for embedding transfer
            edge_index, edge_type = self.add_loop_edge(edge_index, edge_type, num_ent, num_rel)
            r = torch.cat([r, torch.zeros(1, r.size(1)).to(self.args.device)], dim=0)
            neigh_t = torch.index_select(x, 0, edge_index[1])
            neigh_r = torch.index_select(r, 0, edge_type)
            neigh_h = torch.index_select(x, 0, edge_index[0])
            ent_embed = scatter_mean(src=neigh_h + neigh_r, dim=0, index=edge_index[1], dim_size=num_ent)
            rel_embed = scatter_mean(src=neigh_t - neigh_h, dim=0, index=edge_type, dim_size=num_rel + 1)
            ent_embed = self.GCN_model(ent_embed.float(), edge_index,edge_attr = edge_type,batch_size =self.args.batch_size)

            ent_embed = torch.relu(ent_embed.double())
            # print(ent_embed.dtype,rel_embed.dtype)
            return ent_embed, rel_embed[:-1]
        else:
            print(f'MAE is working')
            '''prepare old parameter and the number of |N(x)|'''
            if x.size(0) > old_entity_weight.size(0):
                old_entity_weight = torch.cat((old_entity_weight, torch.zeros(x.size(0)-old_entity_weight.size(0))), dim=0)
                old_x = torch.cat((old_x, torch.zeros(x.size(0)-old_entity_weight.size(0), x.size(1))), dim=0)
            if r.size(0) > old_relation_weight.size(0):
                old_relation_weight = torch.cat((old_relation_weight, torch.zeros(x.size(0) - old_relation_weight.size(0))),dim=0)
                old_r = torch.cat((old_r, torch.zeros(r.size(0) - old_relation_weight.size(0), r.size(1))), dim=0)

            '''add self-loop edges'''
            edge_index, edge_type = self.add_loop_edge(edge_index, edge_type, num_ent, num_rel)
            r = torch.cat([r, torch.zeros(1, r.size(1)).to(self.args.device)], dim=0)

            '''get neighbor embeddings'''
            neigh_t = torch.index_select(x, 0, edge_index[1])
            neigh_r = torch.index_select(r, 0, edge_type)
            neigh_h = torch.index_select(x, 0, edge_index[0])

            '''calculate entity embeddings'''
            ent_embed_new = scatter_add(src=neigh_h + neigh_r, dim=0, index=edge_index[1], dim_size=num_ent)
            ent_embed_old = old_entity_weight.unsqueeze(1) * old_x
            ent_embed = ent_embed_old + ent_embed_new
            ent_involving_num = old_entity_weight + scatter_add(src=torch.ones(edge_index.size(1)), index=edge_index[1], dim_size = num_ent)
            ent_embed = ent_embed/ent_involving_num
            ent_embed = self.GCN_model(ent_embed.float(), edge_index,edge_attr = edge_type,batch_size =self.args.batch_size)
            ent_embed = torch.relu(ent_embed)

            '''calculate relation embeddings'''
            rel_embed_new = scatter_add(src=neigh_t + neigh_h, dim=0, index=edge_index[1], dim_size=num_rel)
            rel_embed_old = old_relation_weight.unsqueeze(1) * old_r
            rel_embed = rel_embed_old + rel_embed_new
            rel_involving_num = old_relation_weight + scatter_add(src=torch.ones(edge_type.size(0)), index=edge_type,
                                                                dim_size=num_rel)
            rel_embed = rel_embed / rel_involving_num

            return ent_embed, rel_embed[:-1]

    def add_loop_edge(self, edge_index, edge_type, num_ent, num_rel):
        '''add self-loop edge for entities'''
        u, v = torch.arange(0, num_ent).unsqueeze(0).to(self.args.device), torch.arange(0, num_ent).unsqueeze(0).to(self.args.device)
        r = torch.zeros(num_ent).to(self.args.device).long()
        loop_edge = torch.cat([u, v], dim=0)
        edge_index = torch.cat([edge_index, loop_edge], dim=-1)
        edge_type = torch.cat([edge_type, r+num_rel], dim=-1)
        return edge_index, edge_type










