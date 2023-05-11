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

class DE_QuatDE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_QuatDE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.ent_transfer      = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_transfer      = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()
        
        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)
        
        self.sigm = torch.nn.Sigmoid()
        self.tanh = nn.Tanh()
    
    def create_time_embedds(self):
            
        self.m_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_freq = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_freq.weight)
        nn.init.xavier_uniform_(self.d_freq.weight)
        nn.init.xavier_uniform_(self.y_freq.weight)

        self.m_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_phi = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_phi.weight)
        nn.init.xavier_uniform_(self.d_phi.weight)
        nn.init.xavier_uniform_(self.y_phi.weight)

        self.m_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.d_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()
        self.y_amp = nn.Embedding(self.dataset.numEnt(), self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.m_amp.weight)
        nn.init.xavier_uniform_(self.d_amp.weight)
        nn.init.xavier_uniform_(self.y_amp.weight)

    def get_time_embedd(self, entities, year, month, day):
        
        y = self.y_amp(entities)*self.time_nl(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.m_amp(entities)*self.time_nl(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.d_amp(entities)*self.time_nl(self.d_freq(entities)*day + self.d_phi(entities))
        
        return y+m+d

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)

        h,r,t = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)

        h_transfer = self.ent_transfer(heads)
        t_transfer = self.ent_transfer(tails)
        r_transfer = self.rel_transfer(rels)
        
        h = torch.cat((h,self.get_time_embedd(heads, years, months, days)), 1)
        t = torch.cat((t,self.get_time_embedd(tails, years, months, days)), 1)
        h_transfer = torch.cat((h_transfer,self.get_time_embedd(heads, years, months, days)), 1)
        t_transfer = torch.cat((t_transfer,self.get_time_embedd(tails, years, months, days)), 1)

        return h,r,t, h_transfer, t_transfer, r_transfer
    
    def _transfer(self, x, x_transfer, r_transfer):
        ent_transfer = self._calc(x, x_transfer)
        ent_rel_transfer = self._calc(ent_transfer, r_transfer)

        return ent_rel_transfer
    
    def _calc(self, h, r):
        # print("pip---------------")
        # print((h.shape, r.shape))

        s_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)

        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        return torch.cat([A, B, C, D], dim=1)
    
    def forward(self, heads, rels, tails, years, months, days):
        h, r, t, h_transfer, t_transfer, r_transfer = self.getEmbeddings(heads, rels, tails, years, months, days)

        # print("pip---------------")
        # print(h.shape, r.shape, t.shape, h_transfer.shape, t_transfer.shape, r_transfer.shape)

        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)
        hr = self._calc(h1, r)
        scores = hr * t1
        scores = F.dropout(scores, p=self.params.dropout, training=self.training)
        score = torch.sum(scores, -1)
        return score
        