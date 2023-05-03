from typing import List

import torch
from torch import nn, sigmoid
import torch.nn.functional as F
import numpy as np
import math, copy


from .appfoq import (AppFOQEstimator, IntList, inclusion_sampling,
                     find_optimal_batch)


#  WassersteinEmbedding for  dim * num(hyperparameter)

def logcos_plus(x):
    """
    Implement the function in Wasserstein Fisher Rao metric, compute the log(cos_+(x)) for every element
    """
    cos_plus = torch.cos(torch.clamp(x, min=-torch.pi / 2 , max=torch.pi / 2))
    y = - 2 * torch.log(cos_plus + 1e-9)
    return y


def wfr_cost(n, eta):
    """
    Generate Wasserstein Fisher Rao metric's cost matrix.
    Note: When Python deal with the e^(inf), it will overflow and result unstable. Thus we convert these e^(-inf) to 0 by hands.
    Arguement :
    n -- the embedding's grid number.
    eta -- hypermeter to control transport area.
    Return :
    Cost matrix in WFR metric.

    """
    # the cost matrix has inf when distance is too long.
    grid = torch.abs(torch.arange(n).unsqueeze(-1) - torch.arange(n).unsqueeze(-2))
    cost = logcos_plus(grid / (2 * eta))
    real_cost = torch.where(torch.isnan(cost), torch.full_like(cost, torch.inf), cost)
    return real_cost



class distribute:
    """
    Reshape embedding tensor from (batch, d=a*b) to (batch, b, a).
    Note: When Python deal with the e^(inf), it will generate nan. Thus we convert these e^(inf) to 0 by hands.
    Arguement :
    b -- the embedding's grid number.
    """
    def __init__(self, emb_grid):
        self.dim = emb_grid

    def __call__(self, emb):
        num = emb.shape[-1] // self.dim
        chunk_emb = torch.chunk(emb, num, -1)
        stack_emb = torch.stack(chunk_emb, -2)
        return stack_emb


class Conv1d():
    """
    Implement the Convolution layer to compute the matrix multiplication K_{\epsilon} \phi(\psi) in WFR's Sinkhorn iteration.
    Arguement :
    window -- the convolution kernel's element.
    window_size -- the convolution kernel's size.
    batch -- batch tensor to be compute.
    Return :
    computed batch tensor. 

    """
    # it's convenient for batch when applying convolution.
    def __init__(self, window, window_size):
        self.window = window
        self.window_size = window_size

    def __call__(self, batch):
        size = batch.shape
        dim = size[-1]
        x = batch.view(-1,dim).unsqueeze(-2)
        y = F.conv1d(x, self.window, padding=self.window_size // 2)
        return y.view(size)


class WfrSinkHorn():
    """
    Implement the algorithm to compute the dual entropy regularization WFR metric:
    Note: We first get the dual variable by matrix multiplication, and compute the dual probelm. The gradient is just from 
    the dual probelm and the matrix multiplication is detached.
    Augument:
    max_iter: the number of the converge iteration.
    reg: the factor of the entropy regularization.
    reg_m : the factor of the KL penalty.
    eta -- hypermeter to control transport area.
    device -- the device of tensor.
    Input:
    a,b. batch histograms and the shape also could be (batch_1, batch_2, histogram_number).
    Return :
    the dual entropy regularization WFR metrics. 
    """
    def __init__(self, dim, max_iter, reg, reg_m, eta, device):
        self.fi = reg_m / (reg + reg_m)
        self.reg = reg
        self.dim = dim
        self.max_iter = max_iter
        self.device = device
        self.window_size = min(dim-1, math.floor(torch.pi * eta)) * 2 + 1
        self.window = torch.exp(- logcos_plus(torch.tensor([(i - 0.5 * (self.window_size - 1))
                for i in range(self.window_size)], device=self.device) / (2 * eta))).repeat(1, 1, 1)
        self.conv = Conv1d(self.window, self.window_size)
        self.cost = wfr_cost(dim, eta).to(self.device)


    def __call__(self, a, b):
        with torch.no_grad():
            K = torch.exp(-self.cost/self.reg)
            size = (a + b).shape
            u, v = torch.ones(size, device=self.device) / self.dim, torch.ones(size, device=self.device) / self.dim
            for j in range(self.max_iter):
                u = (a / self.conv(v)) ** self.fi
                v = (b / self.conv(u)) ** self.fi
        #        gamma = u.view(size[0], dim, -1) * K * v
            phi, psi = 1 - torch.exp(-torch.log(u) * self.reg), 1 - torch.exp(-torch.log(v) * self.reg)
            u_v = 1-torch.exp(torch.log(u).unsqueeze(-1) + torch.log(v).unsqueeze(-2))
            if torch.isinf(u_v).sum() > 0:
                u_v = torch.where(torch.isinf(u_v), torch.full_like(u_v, 0), u_v)
            remainder = self.reg * torch.mul(u_v, K).sum([-1,-2])
        dual_distance = torch.sum(a * phi, dim=-1) + torch.sum(b * psi, dim=-1) + remainder
        assert torch.isnan(dual_distance).sum() == 0
        # torch.nonzero(torch.isnan(dual_distance)==1)
        return dual_distance


class WassersteinProjection_comp_real(nn.Module):
    """
    Implement relation decompostion Net:
    The idea is every relation has its projection net and reduce the parameter by sharing memory.
    W_r = \sum V_j r_j, b_r = \sum a_j b_j where the V_j, a_j is coefficient and r_j, b_j is basis.
    We conduct the net by nn.Linear and "real" is supporting multi-layer.
    """
    def __init__(self,
                 ent_grid,
                 ent_dim,
                 n_relation,
                 rel_num_base,
                 hidden_dim,
                 num_lay,
                 drop_prob,
                 device):
        super(WassersteinProjection_comp_real, self).__init__()
        self.emb_dim = ent_grid * ent_dim
        self.rel_num_base = rel_num_base
        self.hidden_dim = hidden_dim
        self.n_relation = n_relation
        self.num_lay = num_lay
        self.device = device

        self.weights = nn.Parameter(torch.zeros([self.n_relation, self.rel_num_base, self.num_lay]))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.distribute = distribute(ent_grid)
        self.dropout = nn.Dropout(drop_prob)

        for n1 in range(1, self.num_lay + 1):
            setattr(self, "ln{}".format(n1), nn.LayerNorm(self.emb_dim))
            setattr(self, "layer{}".format(n1), nn.Linear(self.emb_dim, self.rel_num_base*self.emb_dim,bias=True))
        for n1 in range(1, self.num_lay + 1):
            nn.init.xavier_uniform_(getattr(self, "layer{}".format(n1)).weight)


    def forward(self, ent_emb, proj_ids):
        rel_weights = torch.index_select(
            self.weights,
            dim=0,
            index=proj_ids.view(-1))
        x = ent_emb
        for n1 in range(2, self.num_lay + 1):
            y = getattr(self, "layer{}".format(n1))(self.dropout(x)).reshape(-1, self.rel_num_base, self.emb_dim)
            x = torch.einsum("bw,bwj->bj", rel_weights[:, :, n1-1], self.dropout(y))
            x = self.relu(getattr(self, "ln{}".format(n1))(x))

        y = self.layer1(self.dropout(ent_emb)).reshape(-1, self.rel_num_base, self.emb_dim)
        x = torch.einsum("bw,bwj->bj", rel_weights[:, :, 0], self.dropout(y))
        output = self.sigmoid(self.ln1(x))

        return output



class WassersteinProjection_comp(nn.Module):
    # Apply relation decompostion
    
    def __init__(self,
                 ent_grid,
                 ent_dim,
                 n_relation,
                 rel_num_base,
                 hidden_dim,
                 num_lay,
                 drop_prob,
                 device):
        super(WassersteinProjection_comp, self).__init__()
        self.emb_dim = ent_grid * ent_dim
        self.rel_num_base = rel_num_base
        self.hidden_dim = hidden_dim
        self.n_relation = n_relation
        self.num_lay = num_lay
        self.device = device
        self.layer1 = nn.Linear(self.emb_dim, self.rel_num_base*self.emb_dim,bias=True)
        nn.init.xavier_uniform_(self.layer1.weight)
        self.sigmoid = nn.Sigmoid()
        self.distribute = distribute(ent_grid)

        self.relation_base = nn.Parameter(torch.zeros([self.rel_num_base, self.emb_dim, self.emb_dim + 1], device=self.device))
        nn.init.xavier_uniform_(self.relation_base, gain=nn.init.calculate_gain('sigmoid'))
#        self.weight = dgl.nn.pytorch.utils.WeightBasis(shape, self.rel_num_base, self.n_relation)
        self.weights = nn.Parameter(torch.zeros([self.n_relation, self.rel_num_base]))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))
        self.dropout = torch.nn.Dropout(drop_prob)
        self.ln0 = torch.nn.LayerNorm(self.emb_dim)



    def forward(self, ent_emb, proj_ids):
        rel_weights = torch.index_select(
            self.weights,
            dim=0,
            index=proj_ids.view(-1))
        switch = 1
        if switch:
            
            y = self.layer1(self.dropout(ent_emb)).reshape(-1, self.rel_num_base, self.emb_dim)
            x = torch.einsum("bw,bwj->bj", rel_weights, self.dropout(y))
        else:
            rel_emb = torch.matmul(rel_weights,
                self.relation_base.view(self.rel_num_base,-1)
            ).view(rel_weights.shape[0], *self.relation_base.shape[1:3])

            x = torch.matmul(rel_emb[:,:,0:-1], ent_emb.unsqueeze(-1)).squeeze() \
                + rel_emb[:,:,-1]
#        x = torch.flatten(self.ln1(self.distribute(x)),-2)
        output = self.sigmoid(self.ln0(x))

        return output
    
class WassersteinProjection_diag(nn.Module):
    def __init__(self,
                 ent_grid,
                 ent_dim,
                 n_relation,
                 hidden_dim,
                 num_lay,
                 drop_prob,
                 device):
        super(WassersteinProjection_diag, self).__init__()
        self.emb_dim = ent_grid * ent_dim
        self.n_relation = n_relation
        self.hidden_dim = hidden_dim
        self.num_lay = num_lay
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.distribute = distribute(ent_grid)

        self.relation_tran = nn.Parameter(
            torch.zeros([self.n_relation,ent_dim, ent_grid, ent_grid],
            device=self.device))
        self.relation_bias = nn.Parameter(
            torch.zeros([self.n_relation,ent_dim, ent_grid],
            device=self.device))
        nn.init.xavier_uniform_(self.relation_tran)
        nn.init.xavier_uniform_(self.relation_bias)
        self.dropout = torch.nn.Dropout(drop_prob)
#        self.ln0 = LayerNorm_3d(ent_dim, self.device)
        self.norm = nn.LayerNorm(self.emb_dim)
#        self.norm = torch.nn.BatchNorm1d(self.emb_dim, track_running_stats=False) whick norm is better?


    def forward(self, ent_emb, proj_ids):
        rel_tran = torch.index_select(
            self.relation_tran,
            dim=0,
            index=proj_ids.view(-1))
        rel_bias = torch.index_select(
            self.relation_bias,
            dim=0,
            index=proj_ids.view(-1))
        ent_emb = self.distribute(ent_emb)
        x = torch.matmul(rel_tran, ent_emb.unsqueeze(-1)).squeeze() \
            + rel_bias #the last dimensionc is bias 
        output = self.norm(torch.flatten(x,-2,-1))
#        output = self.sigmoid(x)

        return output


class WassersteinEstimator(AppFOQEstimator):
    """
    WFRE for Reasoning over Knowledge Graph, including entity and relation embedding, projection
    , intersection, union(DNF or DM) and negation operations, scoring function based WFR metric.
    Note: Our implement and framewark are based on EFO-1-QA-benchmark https://github.com/rabi-fei/EFO-1-QA-benchmark.git.
    Augument: **model_params
    """

    name = "wasserstein_uot" # does't support MLP now!

    def __init__(self, n_entity, n_relation, gamma, relation_num_base, hidden_dim, num_lay, 
                    negative_sample_size, device, ent_grid, ent_dim, max_iter, scale, reg, eta, 
                    proj_type="decompostion", drop_n=0.1, drop_p=0.05, init_p=0.5, t_norm="Godel", distance="convolution"):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.gamma = nn.Parameter(
            torch.tensor([gamma]),
            requires_grad=False)
        self.emb_dim = ent_dim * ent_grid
        self.rel_num_base = relation_num_base
        self.hidden_dim = hidden_dim
        self.num_lay = num_lay
        self.init_p = init_p #to control e emb initialnation
        self.drop_p = drop_p
        self.drop_n = drop_n
        self.Drop_n = nn.Dropout(drop_n)
        self.negative_size = negative_sample_size
        self.distance = distance
        self.t_norm = t_norm ##one of ["Lukasiewicz", "Godel"-minmax, "product"]
        self.proj_type = proj_type # ["diagonal", "decompostion"]
        self.device = device
        self.ent_grid = ent_grid
        self.ent_dim = ent_dim
        self.max_iter = max_iter
        self.scale = scale
        self.reg = reg
        self.eta = eta
        self.epsilon = 2.0

        #        cost_matrix = CostMatrix(emb_dim, device)
        self.distribute = distribute(self.ent_grid)
        self.Sigimoid = nn.Sigmoid()
        self.WfrSinkHorn = WfrSinkHorn(
            self.ent_grid, max_iter, self.reg, 1, self.eta, self.device)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / self.emb_dim]),
            requires_grad=False)
        self.entity_embedding = nn.Parameter(
                torch.zeros(self.n_entity, self.emb_dim, device=self.device))
        nn.init.uniform_(tensor=self.entity_embedding, a=0.5-init_p, b=0.5+init_p)

        if self.proj_type == "decomposition":
            self.projection_net = WassersteinProjection_comp(
                self.ent_grid, self.ent_dim, self.n_relation, self.rel_num_base, 
                self.hidden_dim, self.num_lay,self.drop_p,self.device)
        elif self.proj_type == "decomposition_real":
            self.projection_net = WassersteinProjection_comp_real(
                self.ent_grid, self.ent_dim, self.n_relation, self.rel_num_base, 
                self.hidden_dim, self.num_lay,self.drop_p,self.device)
        elif self.proj_type == "diagonal":
            self.projection_net = WassersteinProjection_diag(
                self.ent_grid, self.ent_dim, self.n_relation, 
                self.hidden_dim, self.num_lay,self.drop_n,self.device)
        else:
            assert "does't support this projection net"


    def get_entity_embedding(self, entity_ids: torch.Tensor, **kwargs):
        """
        Generate entity embedding for given entity ids.
        Input :
        entity_ids -- batch entities's id.(batch, ids)
        Return :
        Related entities embeddings. (batch, embeddings)
        """
        emb = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=entity_ids.view(-1)
        ).view(list(entity_ids.shape) + [self.emb_dim])

        return torch.clamp(emb, 0.0, 1.0)

    def get_projection_embedding(self, proj_ids, ent_emb):
        """
        Generate projection embedding for given entity embedding and projection ids.
        Input :
        proj_ids -- batch relations's id.(batch, ids)
        ent_emb -- batch entities embedding. (batch, embeddings)
        Return :
        Related query embeddings. (batch, embeddings)
        """
        pro_emb = self.projection_net(ent_emb, proj_ids)
        return pro_emb

    def get_conjunction_embedding(self,
                                  con_emb: List[torch.tensor],
                                  **kwargs) -> any:
        """
        Intersection Operation: generate the new query embedding given sub-query embeddings.
        Input :
        con_emb -- batch query embeddings. [(batch, embeddings)]
        Return :
        query embeddings after intersection. (batch, embeddings)
        """
        all_emb = torch.stack(con_emb)
        if self.t_norm == "Godel":
            conj_emb = torch.min(all_emb, dim=0)[0]
        elif self.t_norm == "product":
            conj_emb = torch.prod(all_emb, dim=0)
        elif self.t_norm == "Lukasiewicz":
            conj_emb = torch.sum(all_emb, dim=0) - all_emb.shape[0] + 1
        else:
            assert "does't support this t_norm"

        return torch.clamp(conj_emb, 0.0, 1.0)

    def get_negation_embedding(self, emb: torch.tensor, **kwargs):
        """
        Complement Operation(negation): generate the new query embedding given query embedding.
        Input :
        proj_ids -- batch relations's id.(batch, ids)
        emb -- batch query embeddings. (batch, embeddings)
        Return :
        query embeddings after negation. (batch, embeddings)
        """
        emb = self.Drop_n(emb - 1./2)*(1-self.drop_n) + 1./2
        neg_emb = 1. - emb

        return neg_emb

    def get_disjunction_embedding(self, disjunction_emb: List[torch.tensor], **kwargs):
        """
        Union Operation: generate the new query embedding given sub-query embeddings.
        Note; DNF compute the sub-distance directly and don't need operation.
        Input :
        disjunction_emb -- batch query embeddings. [(batch, embeddings)]
        Return :
        query embeddings after uniom. (batch, embeddings)
        """
        all_emb = torch.stack(disjunction_emb)

        return all_emb

    def get_difference_embedding(self, lemb: torch.Tensor, remb: torch.Tensor,
                                 **kwargs):
        assert False, 'Do not use d in WFRE'

    def get_multiple_difference_embedding(self, emb: List[torch.Tensor], **kwargs):
        assert False, 'Do not use D in WFRE'

    def criterion(self,
                  query_embedding: torch.tensor,
                  answer_set: List[IntList],
                  union: bool = False):
        """
        Compute the logits given query embeddings and their answer set.
        Note; DNF
        Input :
        query_embedding -- batch query embeddings. [(batch, embeddings)]
        answer_set -- batch nswer set [[ids]]
        Return :
        query embeddings' logit. (batch, logit)
        """
        assert query_embedding.shape[0] == len(answer_set)
        chosen_ans, chosen_false_ans, subsampling_weight = \
            inclusion_sampling(answer_set, negative_size=self.negative_size,
                               entity_num=self.n_entity)  # 负采样，正1，负128
        answer_embedding = self.get_entity_embedding(torch.tensor(chosen_ans, device=self.device)).squeeze()
        #  batch*1*entity_dim
        if union:  # batch*disjunction_num*ent_dim
            union_positive_logit = self.compute_logit(answer_embedding.unsueeze(1),
                                                      query_embedding)
            positive_logit = torch.max(union_positive_logit, dim=1)[0]
        else:
            positive_logit = self.compute_logit(answer_embedding, query_embedding)
        assert torch.isnan(positive_logit).sum() == 0
        all_neg_emb = self.get_entity_embedding(torch.tensor(np.array(chosen_false_ans), device=self.device).view(-1))
        all_neg_emb = all_neg_emb.view(-1, self.negative_size, self.emb_dim)
        #  bacth*negative_size*ent_dim
        if union:  #
            union_negative_logit = self.compute_logit(all_neg_emb.unsqueeze(1), query_embedding.unsqueeze(2))
            negative_logit = torch.max(union_negative_logit, dim=1)[0]
        else:
            negative_logit = self.compute_logit(all_neg_emb, query_embedding.unsqueeze(1))
        assert torch.isnan(negative_logit).sum() == 0
        return positive_logit, negative_logit, subsampling_weight.to(self.device)


    def compute_logit(self, entity_embedding: torch.tensor,
                      query_embedding: torch.tensor):
        """
        Compute the logits given query embeddings and one of their answer entity's embedding.
        Input :
        query_embedding -- batch query embeddings. (batch, embeddings)
        entity_embedding -- batch entity embeddings(answer of the related query.(batch, embeddings)
        Return :
        query embeddings' logit. (batch, logit)
        """
        #  对-1维计算Wasserstein距离，需要能够处理3维，2维的张量
        if self.distance == "convolution":
            entity_embedding = self.distribute(entity_embedding) + 1e-4
            query_embedding = self.distribute(query_embedding) + 1e-4
            pre_distance = self.WfrSinkHorn(entity_embedding, query_embedding)
            distance = torch.mean(pre_distance, dim=-1)
        elif self.distance == "WFR_1":
            pointwise_distance = math.sqrt(2) * self.eta * torch.sqrt(entity_embedding + query_embedding - torch.sqrt(entity_embedding*query_embedding))
            distance = pointwise_distance.mean(-1)
        elif self.distance == "special":
            p_star = torch.exp(( torch.log(entity_embedding+1e-4) + torch.log(query_embedding+1e-4) - 2 - self.reg)/(self.reg + 2))
            pointwise_distance = - p_star * ( torch.log(entity_embedding+1e-4) + torch.log(query_embedding+1e-4)) + (2 + self.reg) * p_star * torch.log(p_star)
            distance = pointwise_distance.mean(-1)
        else:
            kl_loss = nn.KLDivLoss(reduction ="none", log_target=True)
            entity_embedding = torch.log(self.distribute(entity_embedding) + 1e-4)
            query_embedding = torch.log(self.distribute(query_embedding) + 1e-4)
            distance = torch.mean(kl_loss(query_embedding, entity_embedding).sum(-1), dim=-1)

#            distance = (entity_embedding - query_embedding).pow(2).sum(dim=-1)
        logit = self.gamma - distance * self.scale
        assert torch.isnan(logit).sum() == 0
        return logit


    def compute_all_entity_logit(self,
                                 pred_emb: torch.tensor,  # query_dis的数目
                                 union: bool = False) -> torch.tensor:
        """
        Compute the logits given query embeddings and the all the entities in valid and test.
        Note: we would use find_optimal_batch to reduce the compute memory.
        Input :
        query_embedding -- batch query embeddings. (batch, embeddings)
        Return :
        query embeddings' logit. (batch, logit)
        """
        all_entities = torch.LongTensor(range(self.n_entity)).to(self.device)
        all_emb = self.get_entity_embedding(all_entities)
        #  升维度时在find_optimal_batch函数里
        query_dist = pred_emb.unsqueeze(-2)
        batch_num = find_optimal_batch(all_emb,
                                       query_dist=query_dist,
                                       compute_logit=self.compute_logit,
                                       union=union)
        chunk_of_answer = torch.chunk(all_emb, batch_num, dim=0)
        logit_list = []
        for answer_part in chunk_of_answer:
            if union:
                union_logit = self.compute_logit(answer_part.unsqueeze(0).unsqueeze(0), query_dist)
                logit = torch.max(union_logit, dim=0)[0]
            else:
                logit = self.compute_logit(answer_part.unsqueeze(0), query_dist)
            logit_list.append(logit)
        all_logit = torch.cat(logit_list, dim=1)
        return all_logit