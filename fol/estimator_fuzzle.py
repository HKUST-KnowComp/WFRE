from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .appfoq import (AppFOQEstimator, IntList, find_optimal_batch,
                     inclusion_sampling)
#ToDo
#1.logic operator
#2.regular for operator
#3.loss function
#4.eval

class Regular_sig():
    def __init__(self):
        self.w = nn.Parameter(torch.tensor([8]))
        self.b = nn.Parameter(torch.tensor([-4]))
        self.sigmoid = nn.Sigmoid()
    def __call__(self, input):
        output = self.sigmoid(self.w * input + self.b)
        return output

class bounded_01:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding, self.min_val, self.max_val)


class projection_net(nn.Module):

    def __init__(self,entity_dim,n_relation,rel_num_base,device):
        super(projection_net, self).__init__()
        self.n_relation = n_relation
        self.n_base = rel_num_base
        self.hidden_dim = entity_dim
        self.device = device

        self.ln0 = nn.LayerNorm(self.hidden_dim)

        self.att = nn.Parameter(
            torch.zeros([n_relation, self.n_base], device=self.device))
        self.r_trans = nn.Parameter(
            torch.zeros(
            [self.n_base,self.hidden_dim,self.hidden_dim],
            device = self.device)
        )
        self.r_bias = nn.Parameter(
            torch.zeros([self.n_base,self.hidden_dim], device=self.device))

        nn.init.orthogonal_(self.r_trans)
        nn.init.xavier_normal_(self.att)
        nn.init.xavier_normal_(self.r_bias)

    def forward(self,emb,proj_ids):
        proj_weights = torch.index_select(self.att, dim=0, index= proj_ids)
        proj_trans = torch.einsum("bi,ijk->bjk",proj_weights,self.r_trans)
        proj_bias = torch.einsum("bi,ij->bj", proj_weights,self.r_bias)

        x = torch.einsum("bij,bj->bi",proj_trans, emb) + proj_bias
        x = self.ln0(x)

        return x # activate is out!


class FuzzleEstiamtor(AppFOQEstimator):

    name = "fuzzle"

    def __init__(self, n_entity, n_relation, gamma, negative_sample_size,
               entity_dim, relation_num_base, t_norm, regular,  device):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.entity_dim = entity_dim 
        self.rel_num_base = relation_num_base
        # equal model's gamma_off
        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False
        )
        self.epsilon = 2.0
        self.negative_size = negative_sample_size
        self.device = device
        t_norm_list = ["Lukasiewicz", "Godel", "product"]
        if t_norm not in t_norm_list:
            assert "does't support this t_norm"
        self.t_norm = t_norm
        if regular =="sigmoid":
            self.p_regular = Regular_sig()
            self.e_regular = Regular_sig()
        elif regular == "bounded":
            self.p_regular = bounded_01(0.0, 1.0)
            self.e_regular = bounded_01(0.0, 1.0)
        else:
            assert "does't support this activate function"

        self.entity_embeddings = nn.Parameter(
            torch.zeros(n_entity, self.entity_dim, device=self.device))
        nn.init.uniform_(tensor=self.entity_embeddings,a=0.0,b=1.0)
        self.projection_net = projection_net(
            entity_dim, n_relation, relation_num_base, self.device)


    def get_entity_embedding(self, entity_ids: torch.Tensor):
        emb = torch.index_select(self.entity_embeddings,
                    dim=0,
                    index=entity_ids.view(-1)
        )
        return self.e_regular(emb)

    def get_projection_embedding(self, proj_ids: torch.Tensor, emb):
        proj_emb = self.p_regular(self.projection_net(emb, proj_ids))
        return proj_emb

    def get_conjunction_embedding(self, conj_emb: List[torch.Tensor]):
        assert len(conj_emb) == 2
        # t_norm only handle two binary
        emb_1 = conj_emb[0]
        emb_2 = conj_emb[1]
        if self.t_norm =="Lukasiewicz":
            x = emb_1 + emb_2 - 1
            x = F.relu(x)
            return x
        elif self.t_norm ==  "Godel":
            x = torch.min(emb_1, emb_2)# need to check!
            return x
        elif self.t_norm == "product":
            x = emb_1 * emb_2
            return x
        else:
            assert "does't support this t_norm"

    def get_disjunction_embedding(self, disj_emb: List[torch.Tensor],
                                  **kwargs):
        union_emb = torch.stack(disj_emb, dim=1)  # batch*disj_num*dim

        return union_emb

    def get_negation_embedding(self, emb: torch.Tensor, **kwargs):
        
        return 1. - emb

    
    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor,
                                 **kwargs):
        assert False, 'Do not use d in Fuzzle'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        assert False, 'Do not use D in Fuzzle'

    def criterion(self,
                  pred_emb: torch.Tensor,
                  answer_set: List[IntList],
                  union: bool = False):
        assert pred_emb.shape[0] == len(answer_set)
        query_emb = pred_emb
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size,
                               entity_num=self.n_entity)  # todo: negative
        answer_embedding = self.get_entity_embedding(
            torch.tensor(chosen_ans, device=self.device)).squeeze()
        if union:
            positive_union_logit = self.compute_logit(
                answer_embedding.unsqueeze(1), query_emb)  # b*disj
            positive_logit = torch.max(positive_union_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, query_emb)
        all_neg_emb = self.get_entity_embedding(torch.tensor(
            chosen_false_ans, device=self.device).view(-1))
        # batch*negative*dim
        all_neg_emb = all_neg_emb.view(-1,
                                       self.negative_size, self.entity_dim)
        if union:  #
            union_negative_logit = self.compute_logit(all_neg_emb.unsqueeze(1), query_emb.unsqueeze(2))
            negative_logit = torch.max(union_negative_logit, dim=1)[0]
        else:
            negative_logit = self.compute_logit(all_neg_emb, query_emb.unsqueeze(1))
        return positive_logit, negative_logit, subsampling_weight.to(
                                                                self.device)

    def compute_logit(self, entity_emb, query_emb):
        cos = nn.CosineSimilarity(dim=-1)
        logit = self.gamma * cos(entity_emb, query_emb)
        return logit

    def compute_all_entity_logit(self,
                                 pred_emb: torch.Tensor,
                                 union: bool = False) -> torch.Tensor:
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_embedding = self.get_entity_embedding(all_entities)  # nentity*dim
        batch_num = find_optimal_batch(all_embedding,
                                       query_dist=pred_emb.unsqueeze(1),
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_embedding, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_part = self.compute_logit(
                    answer_part.unsqueeze(0).unsqueeze(0),
                    pred_emb.unsqueeze(2))  # batch*disj*answer_part*dim
                logit_part = torch.max(union_part, dim=1)[0]
            else:
                logit_part = self.compute_logit(
                    answer_part.unsqueeze(0), pred_emb.unsqueeze(1)
                    )  # batch*answer_part*dim
            logit_list.append(logit_part)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit

