# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from params import Params
from dataset import Dataset

class DE_HAKE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_HAKE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs_f = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_i = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_j = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()

        self.time_nl = torch.sinc
        
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)
        nn.init.xavier_uniform_(self.rel_embs_j.weight)

        self.phase_weight = 0.5
        self.modulus_weight = 1

        self.gamma = nn.Parameter(
            torch.Tensor([12]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item()) / (params.s_emb_dim+params.t_emb_dim)]),
            requires_grad=False
        )

        self.pi = 3.14159262358979323846

    # def time_nl(self, x):
    #     return 1- torch.abs(x)
    
    def create_time_embedds(self):

        #khởi tạo nhúng thời gian cho thực thể (ngày - tháng -năm) * 2 chiều xuôi và ngược đây là w trong công thức (1)
        self.m_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        #đây là b trong công thức (1)
        self.m_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        # đây là a trong công thức (1)
        self.m_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.m_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_h = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amps_t = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq_h.weight)
        nn.init.xavier_uniform_(self.d_freq_h.weight)
        nn.init.xavier_uniform_(self.y_freq_h.weight)
        nn.init.xavier_uniform_(self.m_freq_t.weight)
        nn.init.xavier_uniform_(self.d_freq_t.weight)
        nn.init.xavier_uniform_(self.y_freq_t.weight)

        nn.init.xavier_uniform_(self.m_phi_h.weight)
        nn.init.xavier_uniform_(self.d_phi_h.weight)
        nn.init.xavier_uniform_(self.y_phi_h.weight)
        nn.init.xavier_uniform_(self.m_phi_t.weight)
        nn.init.xavier_uniform_(self.d_phi_t.weight)
        nn.init.xavier_uniform_(self.y_phi_t.weight)

        nn.init.xavier_uniform_(self.m_amps_h.weight)
        nn.init.xavier_uniform_(self.d_amps_h.weight)
        nn.init.xavier_uniform_(self.y_amps_h.weight)
        nn.init.xavier_uniform_(self.m_amps_t.weight)
        nn.init.xavier_uniform_(self.d_amps_t.weight)
        nn.init.xavier_uniform_(self.y_amps_t.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        years = years - 2010
        months = months/ 6 - 1
        days = days / 16 - 1
        if h_or_t == "head":
            #áp dụng công thức (1) cho ngày tháng năm 
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
        else:
            #áp dụng công thức (1) cho ngày tháng năm 
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
            
        return emb


    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)
        r_embs3 = self.rel_embs_j(rels)

        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        
        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2, r_embs3
    
    def forward(self, heads, rels, tails, years, months, days):
        phase_head, phase_relation, phase_tail, mod_head, mod_relation, mod_tail, bias_relation = self.getEmbeddings(heads, rels, tails, years, months, days)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=1) * self.phase_weight
        r_score = torch.norm(r_score, dim=1) * self.modulus_weight

        score = (phase_score + r_score)
        score = F.dropout(score, p=self.params.dropout, training=self.training)

        return self.gamma.item() - score
        
        
