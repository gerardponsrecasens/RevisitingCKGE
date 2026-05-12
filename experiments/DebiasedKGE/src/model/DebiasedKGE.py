from .BaseModel import *
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean

class DebiasedKGE(BaseModel):
    def __init__(self, args, kg):
        super(DebiasedKGE, self).__init__(args, kg)
        self.init_old_weight()
        self.huber_loss_func = nn.SmoothL1Loss(reduction='sum')  # Corrected the use of reduction
        self.ent_weight, self.rel_weight, self.other_weight = None, None, None
        self.margin_loss_func = nn.MarginRankingLoss(margin=float(self.args.margin), reduction='sum').to(
            self.args.device)
        self.mse_loss_func = nn.MSELoss(reduction='sum')  # Corrected the use of reduction
        self.num_factors = self.args.num_factors  # Number of disentangled subspaces
        self.ent_factor_embeddings = nn.ModuleList(
            [nn.Embedding(self.kg.num_ent, self.args.emb_dim) for _ in range(self.num_factors)])
        self.rel_factor_embeddings = nn.ModuleList(
            [nn.Embedding(self.kg.num_rel, self.args.emb_dim) for _ in range(self.num_factors)])
        self.factor_labels = {}

    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id
    
    def store_old_parameters(self):
        '''
        Store learned parameters and weights for regularization.
        '''
        self.args.snapshot -= 1
        param_weight = self.get_new_weight()  # Get new weight from the previous snapshot
        self.args.snapshot += 1

        for name, param in self.named_parameters():
            name = name.replace('.', '_')  # Replace dots in the name with underscores
            value = param.data  # Current parameter values
            new_weight = param_weight[name]  # New weight for the current parameter

            if not isinstance(new_weight, torch.Tensor):
                continue  # Skip if new_weight is not a tensor

            old_weight_name = f'old_weight_{name}'
            if not hasattr(self, old_weight_name):
                setattr(self, old_weight_name, torch.zeros_like(new_weight))

            old_weight = getattr(self, old_weight_name)
            self.register_buffer(f'old_data_{name}', value)

            if '_embeddings' in name:
                if self.args.snapshot == 0:
                    old_weight = torch.zeros_like(new_weight)
                else:
                    # Pad old_weight if its size is smaller than new_weight
                    if new_weight.size(0) > old_weight.size(0):
                        padding_size = new_weight.size(0) - old_weight.size(0)
                        padding = torch.zeros(padding_size, old_weight.size(1)).to(self.args.device)
                        old_weight = torch.cat([old_weight, padding], dim=0)

            beta = 0.9
            new_weight = beta * old_weight + (1 - beta) * new_weight
            self.register_buffer(old_weight_name, old_weight + new_weight)

    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_ or 'ent_factor_embeddings' in name_:
                self.register_buffer(f'old_weight_{name_}', torch.tensor([]))
                self.register_buffer(f'old_data_{name_}', torch.tensor([]))
            elif 'rel_embeddings' in name_ or 'rel_factor_embeddings' in name_:
                self.register_buffer(f'old_weight_{name_}', torch.tensor([]))
                self.register_buffer(f'old_data_{name_}', torch.tensor([]))
            else:
                self.register_buffer(f'old_weight_{name_}', torch.tensor(0.0))
                self.register_buffer(f'old_data_{name_}', param.data)

    def switch_snapshot(self):
        '''
        Prepare for training on the next snapshot.
        '''
        # Store parameters from the previous snapshot
        self.store_old_parameters()

        # Expand embedding sizes to accommodate new entities and relations
        ent_embeddings, rel_embeddings = self.expand_embedding_size()

        # Use clone to create new trainable parameters
        # Avoid in-place operations by cloning the weights
        with torch.no_grad():
            new_ent_embeddings = torch.nn.Parameter(ent_embeddings.weight.data.clone())
            new_rel_embeddings = torch.nn.Parameter(rel_embeddings.weight.data.clone())

            # Smooth transition using linear interpolation
            alpha = 0.5  # Adjust alpha to control interpolation smoothness
            new_ent_embeddings.data[:self.kg.snapshots[self.args.snapshot].num_ent] = (
                    alpha * self.ent_embeddings.weight.data.clone() +
                    (1 - alpha) * new_ent_embeddings.data[:self.kg.snapshots[self.args.snapshot].num_ent]
            )
            new_rel_embeddings.data[:self.kg.snapshots[self.args.snapshot].num_rel] = (
                    alpha * self.rel_embeddings.weight.data.clone() +
                    (1 - alpha) * new_rel_embeddings.data[:self.kg.snapshots[self.args.snapshot].num_rel]
            )

        # Update the embedding weights with the new interpolated embeddings
        self.ent_embeddings.weight = new_ent_embeddings
        self.rel_embeddings.weight = new_rel_embeddings

        # Perform embedding transfer if enabled
        if self.args.using_embedding_transfer == 'True':
            reconstruct_ent_embeddings, reconstruct_rel_embeddings = self.reconstruct()
            new_ent_embeddings.data[self.kg.snapshots[self.args.snapshot].num_ent:] = reconstruct_ent_embeddings[
                                                                                      self.kg.snapshots[
                                                                                          self.args.snapshot].num_ent:
                                                                                      ]
            new_rel_embeddings.data[self.kg.snapshots[self.args.snapshot].num_rel:] = reconstruct_rel_embeddings[
                                                                                      self.kg.snapshots[
                                                                                          self.args.snapshot].num_rel:
                                                                                      ]

            self.ent_embeddings.weight = new_ent_embeddings
            self.rel_embeddings.weight = new_rel_embeddings

        # Store the total number of facts associated with each entity or relation
        new_ent_weight, new_rel_weight, new_other_weight = self.get_weight()
        self.register_buffer('new_weight_ent_embeddings_weight', new_ent_weight.clone().detach())
        self.register_buffer('new_weight_rel_embeddings_weight', new_rel_weight.clone().detach())
        # Save regularization weights for other parameters
        self.new_weight_other_weight = new_other_weight

    def reconstruct(self):
        '''
        Reconstruct entity and relation embeddings using a multi-layer GCN with residual connections.
        '''
        num_ent = self.kg.snapshots[self.args.snapshot + 1].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot + 1].num_rel
        edge_index = self.kg.snapshots[self.args.snapshot + 1].edge_index
        edge_type = self.kg.snapshots[self.args.snapshot + 1].edge_type

        old_entity_weight = getattr(self, 'old_weight_entity_embeddings', None)
        old_relation_weight = getattr(self, 'old_weight_relation_embeddings', None)
        old_x = self.old_data_entity_embeddings if hasattr(self, 'old_data_entity_embeddings') else None
        old_r = self.old_data_relation_embeddings if hasattr(self, 'old_data_relation_embeddings') else None

        # Use multi-layer GCN
        ent_embeddings = self.ent_embeddings.weight
        rel_embeddings = self.rel_embeddings.weight

        for _ in range(self.args.num_layers):  # Increase the number of GCN layers
            new_ent_embeddings, new_rel_embeddings = self.gcn(
                ent_embeddings,
                rel_embeddings,
                edge_index,
                edge_type,
                num_ent,
                num_rel,
                old_entity_weight,
                old_relation_weight,
                old_x,
                old_r
            )

            # Residual connections
            ent_embeddings = new_ent_embeddings + ent_embeddings  # Preserve sensitivity to local structure
            rel_embeddings = new_rel_embeddings + rel_embeddings

        return ent_embeddings, rel_embeddings

    def get_new_weight(self):
        '''
        Calculate the regularization weights for entities and relations.
        :return: weights for entities and relations.
        '''
        ent_weight, rel_weight, other_weight = self.get_weight()
        weight = dict()
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_:
                weight[name_] = ent_weight
            elif 'rel_embeddings' in name_:
                weight[name_] = rel_weight
            else:
                weight[name_] = other_weight
        return weight

    def kl_divergence(self, p, q):
        """
        Calculate the KL divergence.
        :param p: First probability distribution, shape [batch_size, embedding_dim]
        :param q: Second probability distribution, shape [batch_size, embedding_dim]
        :return: KL divergence value
        """
        p = F.softmax(p, dim=-1)  # Convert vectors to probability distributions
        q = F.softmax(q, dim=-1)
        return torch.sum(p * torch.log(p / (q + 1e-10)), dim=-1).mean()

    def disentangled_loss(self):
        '''
        Disentangled loss based on KL divergence to minimize correlation between different subspaces.
        '''
        loss = 0.0
        for i in range(self.num_factors):
            for j in range(i + 1, self.num_factors):
                # Get embeddings from different subspaces
                ent_i = self.ent_factor_embeddings[i].weight  # Entity embedding subspace
                ent_j = self.ent_factor_embeddings[j].weight  # Entity embedding subspace
                rel_i = self.rel_factor_embeddings[i].weight  # Relation embedding subspace
                rel_j = self.rel_factor_embeddings[j].weight  # Relation embedding subspace

                # Calculate KL divergence: between entity embedding subspaces
                loss += self.kl_divergence(ent_i, ent_j)

                # Calculate KL divergence: between relation embedding subspaces
                loss += self.kl_divergence(rel_i, rel_j)

        # Return average KL divergence loss
        return loss / (self.num_factors * (self.num_factors - 1) / 2)

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label).mean()

    def alignment_loss(self, new_data, old_data, alignment_weight):
        '''
        Calculate alignment loss to reduce knowledge shift.
        :param new_data: New embeddings (tensor)
        :param old_data: Old embeddings (tensor)
        :param alignment_weight: Weight for alignment regularization
        :return: alignment loss
        '''
        # Use cosine similarity to calculate alignment loss
        sim = torch.nn.functional.cosine_similarity(new_data, old_data)
        alignment_loss = alignment_weight * (1 - sim.mean())  # Mean alignment loss
        return alignment_loss

    def DebiasedKGE_Knowledge_alignment_loss(self):
        """
        Calculate alignment loss based on different similarity metrics (e.g., Huber loss and cosine similarity),
        and adaptively adjust the loss weights.
        """
        if self.args.snapshot == 0:
            return 0.0  # No alignment loss for the first snapshot

        losses = []

        # Iterate over model parameters to align embeddings
        for name, param in self.named_parameters():
            name = name.replace('.', '_')

            # Skip non-embedding parameters
            if 'ent_embeddings' not in name and 'rel_embeddings' not in name:
                continue  # Only align embedding parameters

            # Get current and old (historical) embeddings
            new_data = param
            old_data = getattr(self, f'old_data_{name}')

            # Ensure new and old data have the same length
            if new_data.size(0) != old_data.size(0):
                # Handle length mismatch (truncate or pad)
                min_len = min(new_data.size(0), old_data.size(0))
                new_data = new_data[:min_len]
                old_data = old_data[:min_len]

            # Calculate alignment losses
            huber_loss = self.huber_loss_func(new_data, old_data)
            cosine_loss = self.alignment_loss(new_data, old_data, alignment_weight=0.5)

            # Adaptive weighting
            huber_weight = huber_loss / (huber_loss + cosine_loss)  # Adjust weight according to loss magnitude
            cosine_weight = 1 - huber_weight  # Cosine loss weight is complement of Huber weight

            # Combine losses with adaptive weighting
            loss = huber_weight * huber_loss + cosine_weight * cosine_loss
            losses.append(loss)

        # Return total loss sum
        return torch.sum(torch.stack(losses)) if losses else 0.0


class TransE(DebiasedKGE):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)
        self.gcn = MAE(args, kg)
        self.init_old_weight()

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

    def get_weight(self):
        '''get the total number of samples containing each entity or relation'''
        num_ent = self.kg.snapshots[self.args.snapshot + 1].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot + 1].num_rel
        ent_weight, rel_weight, other_weight = self.gcn.get_weight(num_ent, num_rel)
        return ent_weight, rel_weight, other_weight

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: total loss = new knowledge loss + old knowledge loss + global regularization loss + disentangled loss
        '''
        # New knowledge loss
        new_loss = self.new_loss(head, rel, tail, label) / head.size(0)
        loss_new = float(self.args.new_loss_weight) * new_loss

        # Old knowledge loss (reconstruction loss)
        if self.args.using_reconstruct_loss == 'True':
            MAE_loss = self.MAE_loss()
            loss_old = float(self.args.reconstruct_weight) * MAE_loss
        else:
            loss_old = 0.0

        # Knowledge alignment loss (global regularization loss)
        if self.args.using_Knowledge_alignment == 'True':
            reg_loss = self.DebiasedKGE_Knowledge_alignment_loss()
            loss_reg = float(self.args.regular_weight) * reg_loss
        else:
            loss_reg = 0.0

        # Disentangled loss
        if self.args.using_disentangled_loss == 'True':
            dis_loss = self.disentangled_loss()
            loss_dis = float(self.args.disentangled_weight) * dis_loss
        else:
            loss_dis = 0.0

        total_loss = loss_new + loss_old + loss_reg + loss_dis
        return total_loss


class MAE(nn.Module):
    def __init__(self, args, kg):
        super(MAE, self).__init__()
        self.args = args
        self.kg = kg
        '''masked KG auto encoder'''
        self.conv_layers = nn.ModuleList()
        for i in range(args.num_layers):
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
        for i in range(self.args.num_layers):
            x, r = self.conv_layers[i](x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return x, r

    def get_weight(self, num_ent, num_rel):
        '''get the total number of samples containing each entity or relation'''
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot + 1].edge_index, self.kg.snapshots[
            self.args.snapshot + 1].edge_type
        other_weight = edge_index.size(1)
        ent_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_index[0],
                                 dim_size=num_ent)
        rel_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_type,
                                 dim_size=num_rel)
        return ent_weight + 1, rel_weight + 1, other_weight


class ConvLayer(nn.Module):
    def __init__(self, args, kg):
        super(ConvLayer, self).__init__()
        self.args = args
        self.kg = kg

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
            ent_embed = torch.relu(ent_embed)
            return ent_embed, rel_embed[:-1]
        else:
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
