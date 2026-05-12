from .BaseModel import *
import pickle
from collections import defaultdict



class LKGE(BaseModel):
    def __init__(self, args, kg):
        super(LKGE, self).__init__(args, kg)
        self.init_old_weight()
        self.mse_loss_func = nn.MSELoss(size_average=False)
        self.ent_weight, self.rel_weight, self.other_weight = None, None, None
        self.margin_loss_func = nn.MarginRankingLoss(float(self.args.margin), size_average=False).to(self.args.device)
    
    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id
    def store_old_parameters(self):
        '''
        Store learned paramters and weights for regularization.
        '''
        self.args.snapshot -= 1
        param_weight = self.get_new_weight()
        self.args.snapshot += 1
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            value = param.data
            old_weight = getattr(self, 'old_weight_{}'.format(name))
            new_weight = param_weight[name]
            self.register_buffer('old_data_{}'.format(name), value)
            if '_embeddings' in name:
                if self.args.snapshot == 0:
                    old_weight = torch.zeros_like(new_weight)
                else:
                    old_weight = torch.cat([old_weight, torch.zeros(new_weight.size(0) - old_weight.size(0), 1).to(self.args.device)], dim=0)
            self.register_buffer('old_weight_{}'.format(name), old_weight + new_weight)

    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            elif 'rel_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            else:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor(0.0))
                self.register_buffer('old_data_{}'.format(name_), param.data)

    def switch_snapshot(self):
        '''
        Prepare for the training on next snapshot.
        '''
        '''store old parameters'''
        self.store_old_parameters()
        '''expand embedding size for new entities and relations'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        '''inherit learned paramters'''
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)

        # File path to save the pickle file
        file_path = "entity_embeddings"+str(self.args.snapshot)+"_"+str(self.args.init)+ "_"+self.args.dataset+"_"+self.args.lifelong_name+".pkl"

        # # Save the objects to a pickle file
        # with open(file_path, 'wb') as file:
        #     pickle.dump({'entity2id': self.kg.entity2id, 'new_ent_embeddings': new_ent_embeddings,'relation2id': self.kg.relation2id,'new_rel_embeddings': new_rel_embeddings}, file)
        
        '''embedding transfer'''
        # ## HERE IS WHERE WE CAN PERFORM THE INITIALIZATION
        # print('Initialization can start here')
        # print('This is the information about the snapshot')
        # print(self.args.snapshot)
        # print('The ent2did info')
        # print(len(self.kg.entity2id))
        # print(self.kg.entity2id)
        # print('The info about the embeddings')
        # print(new_ent_embeddings.shape)
        # print(new_ent_embeddings[0].shape)
        # print(new_ent_embeddings[0])

        # Load the new entities added in this snapshot

        random_noise = True
        sd_frac = self.args.RN
        init = self.args.init 

       
        


        ########################## INIT 1 ##################################

        if init == 1:
            with open('./dicts/'+self.args.dataset+'_new_entities.pkl', 'rb') as file:
                new_entities = pickle.load(file)
            new_entities_snapshot = new_entities[self.args.snapshot+1] #We use +1 as this is before going to new snapshot

            # Load the old entities
            old_entities = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_entities += new_entities[previous_snapshot]
            print('Starting CLASS Initialization')
            # Load the class dictionary
            with open('./dicts/dictionary_db.pkl', 'rb') as file:
                class_dict = pickle.load(file) #{'ent_name':[type1,type2],...}
            

            class_to_entities = {} # {'type1':[0,9,245],..} of entities already in the KG

            for entity in old_entities:
                idx = self.kg.entity2id[entity]
                classes = class_dict[entity]
                for c in classes:
                    if c in class_to_entities.keys():
                        class_to_entities[c].append(idx)
                    else:
                        class_to_entities[c] = [idx]
            

            # Assign to each class its average embedding
            class_avg = {} # {'type1': [0.1,0.32,...,0.9],...}

            for c, idx_list in class_to_entities.items():
                initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

                for idx in idx_list:
                    initial += new_ent_embeddings[idx]

                class_avg[c] = initial/len(idx_list)

            class_std = {}
            for c, idx_list in class_to_entities.items():
                total = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

                for idx in idx_list:
                    total += torch.abs(new_ent_embeddings[idx]-class_avg[c])**2

                class_std[c] = (total/len(idx_list))**0.5
            
            for ent in new_entities_snapshot:

                idx = self.kg.entity2id[ent]

                ent_classes = class_dict[ent]

                previous_classes = [i for i in ent_classes if i in class_to_entities.keys()]

                if len(previous_classes) != 0: #some entities do not have any class assigned
                    
                    initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()
                    initial_std = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

                    for ent_class in previous_classes:
                        initial += class_avg[ent_class]
                        initial_std += class_std[ent_class]
                    
                    initial = initial/len(previous_classes)
                    initial_std = initial_std/len(previous_classes)

                    if random_noise:
                        initial += sd_frac*initial_std*torch.randn(1,self.args.emb_dim).to(self.args.device).double()

                    new_ent_embeddings[idx] = initial
        
        ########################### INIT 3 ############################

        if init == 3:
            print('Starting MODEL Initialization')

            with open('./dicts/'+self.args.dataset+'_new_entities.pkl', 'rb') as file:
                new_entities = pickle.load(file)
            new_entities_snapshot = new_entities[self.args.snapshot+1] #We use +1 as this is before going to new snapshot

            # Load the old entities
            old_entities = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_entities += new_entities[previous_snapshot]

                    
            # Relations
            with open('./dicts/'+self.args.dataset+'_new_relations.pkl', 'rb') as file:
                new_relations = pickle.load(file)

            # Load the old entities
            old_relations = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_relations += new_relations[previous_snapshot]

            with open('./dicts/'+self.args.dataset+'_new_triples.pkl', 'rb') as file:
                new_triples = pickle.load(file)
            new_triples_snapshot = new_triples[self.args.snapshot+1]
            
            
            for ent in new_entities_snapshot:
                idx = self.kg.entity2id[ent]

                matching_triples = []
                for head, relation, tail in new_triples_snapshot:
                    if head == ent or tail == ent:
                        matching_triples.append([head, relation, tail])

                ct = 0
                initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

                for triple in matching_triples:

                    head, relation, tail = triple

                    if head == ent:
                        if tail in old_entities and relation in old_relations: #They previosuly exist
                            ct +=1
                            r_idx = self.kg.relation2id[relation]
                            t_idx = self.kg.entity2id[tail]

                            initial += -new_rel_embeddings[r_idx]+new_ent_embeddings[t_idx]
                    else:
                        if head in old_entities and relation in old_entities: #They previosuly exist
                            ct +=1
                            r_idx = self.kg.relation2id[relation]
                            h_idx = self.kg.entity2id[head]
                            initial += new_rel_embeddings[r_idx]+new_ent_embeddings[h_idx]
                
                if ct !=0:
                    initial = initial/ct

                    if random_noise:
                        initial += sd_frac*torch.randn(1,self.args.emb_dim).to(self.args.device).double()

                    new_ent_embeddings[idx] = initial

        
        
        ############################### INITIALIZING RELATIONS ########################################

        rel_init = self.args.rel_init

        if rel_init !=0 and init ==1:

            with open('./dicts/'+self.args.dataset+'_new_relations.pkl', 'rb') as file:
                new_relations = pickle.load(file)
            new_relations_snapshot = new_relations[self.args.snapshot+1] #We use +1 as this is before going to new snapshot

            # Load the old entities
            old_relations = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_relations += new_relations[previous_snapshot]
        

            # Load the dataset that contains for each relation its classes
            with open('./dicts/rel_db.pkl', 'rb') as file:
                class_dict = pickle.load(file)
            # Create a dictionary that for each class has the relations (their idx) containing it
            class_to_relations = defaultdict(list)
            for relation in old_relations:
                idx = self.kg.relation2id[relation]
                for c in class_dict[relation]:
                    class_to_relations[c].append(idx)
            
            emb_dim = self.args.emb_dim
            device = self.args.device

            # Compute class averages using efficient tensor operations
            class_avg = {}

            for c, idx_list in class_to_relations.items():
                idx_tensor = torch.tensor(idx_list, device=device)
                embeddings = new_rel_embeddings[idx_tensor]  # Shape: [N, emb_dim]

                avg = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, emb_dim]

                class_avg[c] = avg

                # Initialize new relation embeddings based on class averages
                for rel in new_relations_snapshot:
                    idx = self.kg.relation2id[rel] #idx of the relation
                    rel_classes = class_dict[rel] #classes of the relation

                    # Only consider classes with prior entity embeddings
                    prev_classes = [c for c in rel_classes if c in class_avg]

                    if prev_classes:
                        avg_stack = torch.cat([class_avg[c] for c in prev_classes], dim=0)  # [K, emb_dim]

                        mean_avg = avg_stack.mean(dim=0, keepdim=True)
                        new_rel_embeddings[idx] = mean_avg
        
        elif rel_init !=0 and init == 3:

            
            with open('./dicts/'+self.args.dataset+'_new_relations.pkl', 'rb') as file:
                new_relations = pickle.load(file)
            new_relations_snapshot = new_relations[self.args.snapshot+1] #We use +1 as this is before going to new snapshot

            # Load the old relations
            old_relations = []
            for previous_snapshot in range(self.args.snapshot+1):
                old_relations += new_relations[previous_snapshot]

            with open('./dicts/'+self.args.dataset+'_new_triples.pkl', 'rb') as file:
                new_triples = pickle.load(file)
            new_triples_snapshot = new_triples[self.args.snapshot+1]
            


            for rel in new_relations_snapshot:
                idx = self.kg.relation2id[rel]

                matching_triples = []
                for head, relation, tail in new_triples_snapshot:
                    if relation == rel:
                        matching_triples.append([head, relation, tail])

                ct = 0
                initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

                for triple in matching_triples:

                    head, relation, tail = triple

                    
                    if tail in old_entities and head in old_entities: #They previosuly exist
                        ct +=1
                        h_idx = self.kg.entity2id[head]
                        t_idx = self.kg.entity2id[tail]

                        initial += new_ent_embeddings[t_idx] -new_ent_embeddings[h_idx]
                    
                
                if ct !=0:
                    initial = initial/ct
                    new_ent_embeddings[idx] = initial



        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings) #Here the update is pushed
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)


        if self.args.using_embedding_transfer == 'True':

            num_ent, num_rel = self.kg.snapshots[self.args.snapshot+1].num_ent, self.kg.snapshots[self.args.snapshot+1].num_rel
            edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type

            reconstruct_ent_embeddings, reconstruct_rel_embeddings = self.reconstruct()
            new_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:] = reconstruct_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:] = reconstruct_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:]
            self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)
        '''store the total number of facts containing each entity or relation'''
        new_ent_weight, new_rel_weight, new_other_weight = self.get_weight()
        self.register_buffer('new_weight_ent_embeddings_weight', new_ent_weight.clone().detach())
        self.register_buffer('new_weight_rel_embeddings_weight', new_rel_weight.clone().detach())
        '''get regularization weights'''
        self.new_weight_other_weight = new_other_weight

        return 10,10

    def reconstruct(self):
        '''
        Reconstruct the entity and relation embeddings.
        '''
        num_ent, num_rel = self.kg.snapshots[self.args.snapshot+1].num_ent, self.kg.snapshots[self.args.snapshot+1].num_rel
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        new_embeddings, rel_embeddings = self.gcn(self.ent_embeddings.weight, self.rel_embeddings.weight, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return new_embeddings, rel_embeddings

    def get_new_weight(self):
        '''
        Calculate the regularization weights for entities and relations.
        :return: weights for entities and relations.
        '''
        ent_weight, rel_weight, other_weight = self.get_weight()
        weight = dict()
        for name, param in self.named_parameters():
            name_ = name.replace('.','_')
            if 'ent_embeddings' in name_:
                weight[name_] = ent_weight
            elif 'rel_embeddings' in name_:
                weight[name_] = rel_weight
            else:
                weight[name_] = other_weight
        return weight

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label).mean()

    def lkge_regular_loss(self):
        '''
        Calculate regularization loss to avoid catastrophic forgetting.
        :return: regularization loss.
        '''
        if self.args.snapshot == 0:
            return 0.0
        losses = []
        '''get samples number of entities and relations'''
        new_ent_weight, new_rel_weight, new_other_weight = self.new_weight_ent_embeddings_weight, self.new_weight_rel_embeddings_weight, self.new_weight_other_weight
        '''calculate regularization loss'''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if 'ent_embeddings' in name:
                new_weight = new_ent_weight
            elif 'rel_embeddings' in name:
                new_weight = new_rel_weight
            else:
                new_weight = new_other_weight
            new_data = param
            old_weight = getattr(self, 'old_weight_{}'.format(name))
            old_data = getattr(self, 'old_data_{}'.format(name))
            if type(new_weight) != int:
                new_weight = new_weight[:old_weight.size(0)]
                new_data = new_data[:old_data.size(0)]
            losses.append((((new_data - old_data) * old_weight / (new_weight+old_weight)) ** 2).sum())
        return sum(losses)


class TransE(LKGE):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)
        self.gcn = MAE(args, kg)

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

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss + MAE loss + regularization loss
        '''
        new_loss = self.new_loss(head, rel, tail, label)/head.size(0)
        loss = new_loss
        if self.args.using_reconstruct_loss == 'True':
            MAE_loss = self.MAE_loss()
            loss += float(self.args.reconstruct_weight)*MAE_loss
        if self.args.using_regular_loss == 'True':
            regular_loss = self.lkge_regular_loss()
            loss += float(self.args.regular_weight)*regular_loss
        return loss

    def get_weight(self):
        '''get the total number of samples containing each entity or relation'''
        num_ent = self.kg.snapshots[self.args.snapshot+1].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot+1].num_rel
        ent_weight, rel_weight, other_weight = self.gcn.get_weight(num_ent, num_rel)
        return ent_weight, rel_weight, other_weight


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





