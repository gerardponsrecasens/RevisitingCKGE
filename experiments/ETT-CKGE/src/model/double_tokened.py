from .tokened_BaseModel import *

class DLKGE(Tokened_BaseModel):
    def __init__(self, args, kg) -> None:
        super(DLKGE, self).__init__(args, kg)
        self.init_old_weight()
        self.mse_loss_func = nn.MSELoss(size_average=False)

    def return_weights(self):
        ent_embeddings = self.ent_embeddings.weight.data.detach().cpu().numpy()
        rel_embeddings = self.rel_embeddings.weight.data.detach().cpu().numpy()
        ent2id = self.kg.entity2id
        rel2id = self.kg.relation2id

        return ent_embeddings, rel_embeddings,ent2id,rel2id
    
    
    def pre_snapshot(self, optimizer, snapshot_id):
        """
        Prepare the model and optimizer before transitioning to the next snapshot.
        
        Args:
            optimizer (torch.optim.Optimizer): The current optimizer instance.
            snapshot_id (int): The current snapshot index.
        
        Returns:
            torch.optim.Optimizer: Reinitialized optimizer for the next snapshot.
        """
        if snapshot_id > 0:
            print(f"Snapshot {snapshot_id}: Resetting for new snapshot...")

            # Step 1: Freeze token if it exists
            if hasattr(self, 'ent_token'):
                self.ent_token.requires_grad = False
                # print("Token frozen for new snapshot.")

            if hasattr(self, 'rel_token'):
                self.rel_token.requires_grad = False
                print("Token frozen for new snapshot.")

            # Step 2: Unfreeze embeddings
            for param in self.ent_embeddings.parameters():
                param.requires_grad = True
            for param in self.rel_embeddings.parameters():
                param.requires_grad = True
            print("Embeddings are now trainable.")

            # Step 3: Reinitialize token
            print("Reinitializing token...")
            self.ent_token = nn.Parameter(
                torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
            )
            self.ent_token.requires_grad = False

            self.rel_token = nn.Parameter(
                torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
            )
            self.rel_token.requires_grad = False

            # Step 4: Reset optimizer with embeddings and token

            optimizer = torch.optim.Adam(
                [param for name, param in self.named_parameters()
                 if "ent_embeddings" in name or "rel_embeddings" in name],
                lr=self.args.learning_rate
            )

            print("Optimizer reset: Training embeddings and reinitialized token.")

        else:
            optimizer = torch.optim.Adam(
                [param for name, param in self.named_parameters() if "embeddings" in name],
                lr=0.1
            )
            print("Snapshot 0: No changes made to optimizer or model parameters.")

        return optimizer

    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'embeddings' in name_:
                # self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
    def store_old_parameters(self):
        """ store last result """
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            # if param.requires_grad:
            value = param.data
            self.register_buffer(f'old_data_{name}', value.clone().detach())


    def switch_snapshot(self):

        self.store_old_parameters() # save last embedding
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = Parameter(
            self.ent_embeddings.weight.data
        )
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = Parameter(
            self.rel_embeddings.weight.data
        )
        self.ent_embeddings.weight = Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = Parameter(new_rel_embeddings)


class TransE(DLKGE):
    def __init__(self, args, kg) -> None:
        super(TransE, self).__init__(args, kg)
        self.huber_loss = torch.nn.HuberLoss(reduction='sum')
        self.ent_token = None
        self.rel_token = None

    def add_token(self, optimizer):
        """
        Add self.token to the optimizer while ensuring embeddings are frozen
        and excluded from parameter updates.
        """
        self.ent_token = nn.Parameter(
            torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
        )

        self.rel_token = nn.Parameter(
            torch.randn(self.args.token_num, self.args.emb_dim, device=self.args.device).normal_(0, 0.01)
        )
        
        # Ensure embeddings are frozen
        for param in self.ent_embeddings.parameters():
            param.requires_grad = False
        for param in self.rel_embeddings.parameters():
            param.requires_grad = False

        # Remove existing parameter groups for embeddings (if any)
        new_param_groups = []
        for param_group in optimizer.param_groups:
            new_params = [p for p in param_group['params'] if p.requires_grad]
            if new_params:
                param_group['params'] = new_params
                new_param_groups.append(param_group)
        
        # Clear the optimizer and re-add valid parameter groups
        optimizer.param_groups = new_param_groups
        
        # Add token to optimizer explicitly
        optimizer.add_param_group({'params': [self.ent_token,self.rel_token]})
        # optimizer.add_param_group({'params': self.rel_token})
        
        print("Token added to optimizer, embeddings excluded successfully.")

    def token_attention(self,token,frozen_embeddings):
        attention_mask = token @ frozen_embeddings.T  # [T, N]
        attention_mask = attention_mask.sigmoid()  # [T, N]
        attention_mask = attention_mask.T.unsqueeze(-1)  # [N, T, 1]
        
        frozen_embeddings = frozen_embeddings.unsqueeze(1)  # [N, 1, D]
        attention_embeddings = frozen_embeddings * attention_mask  # [N, T, D]
        summed_attention_embeddings = torch.sum(attention_embeddings, dim=1)  # [N, D]
        return summed_attention_embeddings

    def token_base(self,ent_token,frozen_ent_embeddings,rel_token,frozen_rel_embeddings,head,relation,tail,label):

        summed_attention_ent_embeddings = self.token_attention(ent_token,frozen_ent_embeddings)
        summed_attention_rel_embeddings = self.token_attention(rel_token,frozen_rel_embeddings)
        # Index selected entities and relations
        h = torch.index_select(summed_attention_ent_embeddings, 0, head)
        r = torch.index_select(summed_attention_ent_embeddings, 0, relation)
        t = torch.index_select(summed_attention_ent_embeddings, 0, tail)
        
        # Compute score
        score = self.score_fun(h, r, t)
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        return self.margin_loss_func(p_score, n_score, y) / head.size(0)

    def get_TransE_loss(self, head, relation, tail=None, label=None):
        """ Pair wise margin loss: L1-norm (h + r - t) """
        ent_embeddings, rel_embeddings = self.embedding()

        if self.ent_token is not None and self.rel_token is not None and self.args.snapshot == 0:
            # Clone and detach embeddings to ensure frozen weights remain unchanged
            frozen_ent_embeddings = ent_embeddings.clone().detach()
            frozen_rel_embeddings = rel_embeddings.clone().detach()
            loss = self.token_base(self.ent_token,frozen_ent_embeddings,self.rel_token,frozen_rel_embeddings, \
                                    head, relation, tail,label)
            return loss

        if self.args.snapshot > 0:
            if self.ent_token.requires_grad and self.rel_token.requires_grad:
                frozen_ent_embeddings = ent_embeddings.clone().detach()
                frozen_rel_embeddings = rel_embeddings.clone().detach()
                
            # frozen_ent_embeddings = ent_embeddings.clone().detach()
            # frozen_rel_embeddings = rel_embeddings.clone().detach()
                loss = self.token_base(self.ent_token,frozen_ent_embeddings,self.rel_token,frozen_rel_embeddings, \
                                        head, relation, tail,label)
                return loss
        h = torch.index_select(ent_embeddings, 0, head)
        r = torch.index_select(rel_embeddings, 0, relation)
        t = torch.index_select(ent_embeddings, 0, tail)
        score = self.score_fun(h, r, t)

        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.device)
        # print(f'p_score,n_score,y:',p_score.shape,n_score.shape,y.shape)
        return self.margin_loss_func(p_score, n_score, y)/ head.size(0)
        # return self.new_loss(head, relation, tail, label,first_layer_mask,second_layer_mask)

    # def get_old_triples(self):
    #     if isinstance(self.old_triples_weights ,list):
    #         return self.old_triples_weights
    #     return list(self.old_triples_weights.keys())




    def corrupt(self, facts):
        '''
        Create negative samples by randomly corrupt subject or object entity
        :param triples:
        :return: negative samples
        '''
        ss_id = self.args.snapshot
        label = []
        facts_ = []
        prob = 0.5
        for fact in facts:
            s, r, o = fact[0], fact[1], fact[2]
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



    def dice_coeff(self,inputs, eps=1e-12):
        """
        Compute Dice Coefficient Loss for token-entity masks.
        
        Args:
            inputs (torch.Tensor): Input mask with shape [T, N]
            eps (float): Small value to prevent division by zero.
        
        Returns:
            torch.Tensor: Dice Coefficient Loss.
        """
        # Step 1: Pairwise comparison between tokens
        pred = inputs[:, None, :]  # [T, 1, N]
        target = inputs[None, :, :]  # [1, T, N]

        # Step 2: Create a mask to exclude self-comparisons
        mask = torch.ones((inputs.size(0), inputs.size(0)), device=inputs.device)
        mask.fill_diagonal_(0)  # Exclude self-comparisons

        # Step 3: Calculate intersection and union for each token pair
        a = torch.sum(pred * target, dim=-1)  # Intersection: [T, T]
        b = torch.sum(pred * pred, dim=-1) + eps  # Sum of squares of pred: [T, T]
        c = torch.sum(target * target, dim=-1) + eps  # Sum of squares of target: [T, T]

        # Step 4: Compute the Dice Coefficient
        d = (2 * a) / (b + c)  # Dice Coefficient: [T, T]

        # Step 5: Apply the mask to exclude self-comparisons
        d = (d * mask).sum() / mask.sum()
        
        # Step 6: Return Dice Loss (1 - Dice Coefficient)
        return 1-d

    def token_distilliation_loss(self,old_len,old_token,old_embeddings,new_embeddings):
        old_mask = old_token @ old_embeddings.T #[T,D]@[N,D]-->[T,N]
        old_mask = old_mask.sigmoid()
        new_mask = old_token @ new_embeddings.T #[T,D]@[N,D]-->[T,N]
        new_mask = new_mask.sigmoid()
        mask = old_mask * new_mask #[T,N]

        Attentioned_old_embedding = mask.unsqueeze(-1)*old_embeddings.unsqueeze(0) #[T,N,1]@[1,N,D]-->[T,N,D]
        Attentioned_new_embedding = mask.unsqueeze(-1)*new_embeddings.unsqueeze(0) #[T,N,1]@[1,N,D]-->[T,N,D]
        loss = (Attentioned_old_embedding - Attentioned_new_embedding)**2
        # print(f'loss:',loss)
        loss = loss.sum((1))  # [T, D]
        loss = loss / mask.sum((-1)).unsqueeze(-1)
        loss = loss.mean(-1).sum(-1)
        return loss
    def get_token_distilliation_loss(self):
        """ count multy layer loss """
        ent_dice_loss = 0
        rel_dice_loss = 0
        if self.args.snapshot == 0:
            if not self.args.without_div_loss:
                ent_dice_loss = self.dice_coeff(self.ent_token)
                rel_dice_loss = self.dice_coeff(self.rel_token)
            # print(f'ent_dice_loss,rel_dice_loss',ent_dice_loss,rel_dice_loss)
            dice_loss = ent_dice_loss+rel_dice_loss
            return 0.0, dice_loss
        if self.args.snapshot > 0:
            if self.ent_token.requires_grad:
                if not self.args.without_div_loss:
                    ent_dice_loss = self.dice_coeff(self.ent_token)
                    rel_dice_loss = self.dice_coeff(self.rel_token)
                dice_loss = ent_dice_loss+rel_dice_loss
                # print('ejfej')
                return 0.0, dice_loss
        old_ent_len = self.kg.snapshots[self.args.snapshot - 1].num_ent
        old_ent_token=self.old_data_ent_token.clone().detach()
        old_ent_embeddings = self.old_data_ent_embeddings_weight[:old_ent_len]
        new_ent_embeddings = self.ent_embeddings.weight[:old_ent_len]
        ent_loss = self.token_distilliation_loss(old_ent_len,old_ent_token,old_ent_embeddings,new_ent_embeddings) 

        old_rel_len = self.kg.snapshots[self.args.snapshot - 1].num_rel
        old_rel_token=self.old_data_rel_token.clone().detach()
        old_rel_embeddings = self.old_data_rel_embeddings_weight[:old_rel_len]
        new_rel_embeddings = self.rel_embeddings.weight[:old_rel_len]
        rel_loss = self.token_distilliation_loss(old_rel_len,old_rel_token,old_rel_embeddings,new_rel_embeddings)
        # print(f'ent_loss,rel_loss:',ent_loss,rel_loss)
        loss = ent_loss +rel_loss
        return loss,0


    # def forward(self, head, relation, tail=None, label=None):
    def forward(self, inputs):
        head = inputs[0].to(self.args.device)
        relation = inputs[1].to(self.args.device)
        tail = inputs[2].to(self.args.device)
        label = inputs[3].to(self.args.device)

        loss = 0.0;distillation_loss = 0.0;token_training_loss=0.0
        token_distillation_weights = self.args.token_distillation_weight
        if self.args.snapshot == 0:
            token_distillation_weight = 0
        if self.args.snapshot == 1:
            token_distillation_weight = token_distillation_weights[0]
        elif self.args.snapshot == 2:
            token_distillation_weight = token_distillation_weights[1]
        elif self.args.snapshot == 3:
            token_distillation_weight = token_distillation_weights[2]
        elif self.args.snapshot == 4:
            token_distillation_weight = token_distillation_weights[3] 
        """ 0. count initial loss """
        # if self.args.snapshot == 0:
        transE_loss = self.get_TransE_loss(head, relation, tail, label)
        loss += transE_loss
        if self.args.using_token_distillation_loss and self.ent_token is not None and self.rel_token is not None:
            # print(f'yes')
            distillation_loss,token_training_loss = self.get_token_distilliation_loss()
            distillation_loss = distillation_loss * float(token_distillation_weight)
            loss += distillation_loss
        # if self.args.use_multi_layers and (not self.args.without_multi_layers):
            token_training_loss = token_training_loss* float(self.args.div_loss_weight)
            loss += token_training_loss
        # return transE_loss,token_training_loss,distillation_loss,loss
        return (
            transE_loss if isinstance(transE_loss, torch.Tensor) else torch.tensor(transE_loss, device=self.args.device),
            token_training_loss if isinstance(token_training_loss, torch.Tensor) else torch.tensor(token_training_loss, device=self.args.device),
            distillation_loss if isinstance(distillation_loss, torch.Tensor) else torch.tensor(distillation_loss, device=self.args.device),
            loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=self.args.device),
        )
