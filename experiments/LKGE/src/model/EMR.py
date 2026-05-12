from .BaseModel import *
import pickle
from collections import defaultdict




class EMR(BaseModel):
    def __init__(self, args, kg):
        super(EMR, self).__init__(args, kg)
        self.ce = nn.CrossEntropyLoss()
        if 'ENTITY' in self.args.dataset:
            self.args.n_memories = int(self.args.dataset.split('ENTITY')[1])//2
        else:
            self.args.n_memories = 5000

    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id
    
    def pre_snapshot(self):
        '''
        Prepare for training on this snapshot
        '''

        if self.args.snapshot == 0:
            '''sample old facts'''
            self.initialize_memory()
        else:
            '''update old facts'''
            self.update_memory()

    def initialize_memory(self):
        '''sample old facts in first training set'''
        train_data = self.kg.snapshots[0].train_new
        self.memory_data = random.sample(train_data, self.args.n_memories)

    def update_memory(self):
        '''update a half of old facts'''
        random.shuffle(self.memory_data)
        train_data = self.kg.snapshots[self.args.snapshot].train_new
        self.memory_data = self.memory_data[:self.args.n_memories//2] + random.sample(train_data, self.args.n_memories//2)

    def switch_snapshot(self):
        '''prepare for next snapshot'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        
       
                
        # Initialization
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

        if rel_init !=0 and init == 1:

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
                    
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)

        return 10,10

    def replay(self, x, label):
        '''replay old facts'''
        pt_triples, pt_label = self.corrupt(self.memory_data)
        pt_triples, pt_label = torch.LongTensor(pt_triples).to(self.args.device), torch.Tensor(pt_label).to(self.args.device)
        '''merge old and new facts'''
        x = torch.cat([x, pt_triples], dim=0)
        '''get loss'''
        label = torch.cat([label, pt_label], dim=0)
        head, rel, tail = x[:, 0], x[:, 1], x[:, 2]
        loss = self.new_loss(head, rel, tail, label)
        return loss

    def corrupt(self, facts):
        '''
        Create negative samples by randomly corrupt subject or object entity
        :param triples:
        :return: negative samples
        '''
        ss_id = self.args.snapshot
        label = []
        facts_ = []
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
            prob = 0.5
            neg_s = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            neg_o = np.random.randint(0, self.kg.snapshots[ss_id].num_ent - 1, self.args.neg_ratio)
            pos_s = np.ones_like(neg_s) * s
            pos_o = np.ones_like(neg_o) * o
            rand_prob = np.random.rand(self.args.neg_ratio)
            sub = np.where(rand_prob > prob, pos_s, neg_s)
            obj = np.where(rand_prob > prob, neg_o, pos_o)
            facts_.append((s, r, o))
            label.append(1)
            for ns, no in zip(sub, obj):
                facts_.append((ns, r, no))
                label.append(-1)
        return facts_, label


class TransE(EMR):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss
        '''
        x = torch.cat([head.unsqueeze(1), rel.unsqueeze(1), tail.unsqueeze(1)], dim=1)
        if self.args.snapshot > 0:
            new_loss = self.replay(x, label)
        else:
            new_loss = self.new_loss(head, rel, tail, label)
        return new_loss




