from .BaseModel import *
import pickle
import time
from collections import defaultdict




class finetune(BaseModel):
    def __init__(self, args, kg):
        super(finetune, self).__init__(args, kg)
    
    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id

    def switch_snapshot(self):
        '''expand embeddings for new entities and relations '''

        update = 0
        start_time = time.time()
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        end_time = time.time()
        init_time = end_time-start_time

        '''inherit learned parameters'''
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(
            self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(
            self.rel_embeddings.weight.data)
        
        # Save the pickle path
        if self.args.snapshot == 0:
            file_path = "entity_embeddings"+str(self.args.snapshot)+".pkl"
            # Save the objects to a pickle file
            with open(file_path, 'wb') as file:
                pickle.dump({'entity2id': self.kg.entity2id, 'new_ent_embeddings': new_ent_embeddings}, file)
        
        # MY INITIALIZATION

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
                    
            # Load the class dictionary
            with open('./dicts/dictionary_db.pkl', 'rb') as file:
                class_dict = pickle.load(file) #{'ent_name':[type1,type2],...}


            class_to_entities = defaultdict(list)
            for entity in old_entities:
                idx = self.kg.entity2id[entity]
                for c in class_dict[entity]:
                    class_to_entities[c].append(idx)

            # Precompute tensor of new_ent_embeddings
            # Assumes new_ent_embeddings is on the correct device and dtype already
            emb_dim = self.args.emb_dim
            device = self.args.device

            s_t = time.time()
            # Compute class averages and stds using efficient tensor operations
            class_avg = {}
            class_std = {}

            for c, idx_list in class_to_entities.items():
                idx_tensor = torch.tensor(idx_list, device=device)
                embeddings = new_ent_embeddings[idx_tensor]  # Shape: [N, emb_dim]

                avg = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, emb_dim]
                std = embeddings.std(dim=0, unbiased=False, keepdim=True)  # Shape: [1, emb_dim]

                class_avg[c] = avg
                class_std[c] = std
            
            e_t = time.time()
            update = e_t-s_t
            
            start_time = time.time()

            # Initialize new entity embeddings based on class averages
            for ent in new_entities_snapshot:
                idx = self.kg.entity2id[ent]
                ent_classes = class_dict[ent]

                # Only consider classes with prior entity embeddings
                prev_classes = [c for c in ent_classes if c in class_avg]

                if prev_classes:
                    avg_stack = torch.cat([class_avg[c] for c in prev_classes], dim=0)  # [K, emb_dim]
                    std_stack = torch.cat([class_std[c] for c in prev_classes], dim=0)  # [K, emb_dim]

                    mean_avg = avg_stack.mean(dim=0, keepdim=True)
                    mean_std = std_stack.mean(dim=0, keepdim=True)

                    noise = torch.randn_like(mean_avg) * mean_std * sd_frac
                    new_ent_embeddings[idx] = mean_avg + noise
            end_time = time.time()

            init_time = end_time-start_time

        ########################### INIT 3 ############################

        if init == 3:

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
            
            start_time = time.time()


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
                    new_ent_embeddings[idx] = initial
            end_time = time.time()
            init_time = end_time-start_time


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

        return init_time, update


class TransE(finetune):
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
        new_loss = self.new_loss(head, rel, tail, label)
        return new_loss






# class_to_entities = {} # {'type1':[0,9,245],..} of entities already in the KG

            # for entity in old_entities:
            #     idx = self.kg.entity2id[entity]
            #     classes = class_dict[entity]
            #     for c in classes:
            #         if c in class_to_entities.keys():
            #             class_to_entities[c].append(idx)
            #         else:
            #             class_to_entities[c] = [idx]
            

            # # Assign to each class its average embedding
            # class_avg = {} # {'type1': [0.1,0.32,...,0.9],...}

            # for c, idx_list in class_to_entities.items():
            #     initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

            #     for idx in idx_list:
            #         initial += new_ent_embeddings[idx]

            #     class_avg[c] = initial/len(idx_list)

            # class_std = {}
            # for c, idx_list in class_to_entities.items():
            #     total = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

            #     for idx in idx_list:
            #         total += torch.abs(new_ent_embeddings[idx]-class_avg[c])**2

            #     class_std[c] = (total/len(idx_list))**0.5
            
            # start_time = time.time()

            # for ent in new_entities_snapshot:

            #     idx = self.kg.entity2id[ent]
                
            #     ent_classes = class_dict[ent]

            #     previous_classes = [i for i in ent_classes if i in class_to_entities.keys()]

            #     if len(previous_classes) != 0: #some entities do not have any class assigned
            #         initial = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()
            #         initial_std = torch.zeros([1,self.args.emb_dim]).to(self.args.device).double()

            #         for ent_class in previous_classes:
            #             initial += class_avg[ent_class]
            #             initial_std += class_std[ent_class]
                    
            #         initial = initial/len(previous_classes)
            #         initial_std = initial_std/len(previous_classes)

            #         if random_noise:
            #             initial += sd_frac*initial_std*torch.randn(1,self.args.emb_dim).to(self.args.device).double()

            #         new_ent_embeddings[idx] = initial



