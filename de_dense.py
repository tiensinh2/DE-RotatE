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

class DE_DensE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_DensE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        self.ent_embs_x = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_y = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_z = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs_w = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_x = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_y = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        self.rel_embs_z = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()

        self.time_nl = torch.sin
        
        nn.init.xavier_uniform_(self.ent_embs_x.weight)
        nn.init.xavier_uniform_(self.ent_embs_y.weight)
        nn.init.xavier_uniform_(self.ent_embs_z.weight)
        nn.init.xavier_uniform_(self.rel_embs_w.weight)
        nn.init.xavier_uniform_(self.rel_embs_x.weight)
        nn.init.xavier_uniform_(self.rel_embs_y.weight)
        nn.init.xavier_uniform_(self.rel_embs_z.weight)

        self.gamma = nn.Parameter(
            torch.Tensor([12]), 
            requires_grad=False
        )
    
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

        h_x, h_y, h_z = self.ent_embs_x(heads), self.ent_embs_y(heads),self.ent_embs_z(heads)
        t_x, t_y, t_z = self.ent_embs_x(tails), self.ent_embs_y(tails),self.ent_embs_z(tails)
        r_x, r_y, r_z, r_w = self.rel_embs_x(rels), self.rel_embs_y(rels), self.rel_embs_z(rels), self.rel_embs_w(rels)
        
        h_x = torch.cat((h_x,self.get_time_embedd(heads, years, months, days)), 1)
        h_y = torch.cat((h_y,self.get_time_embedd(heads, years, months, days)), 1)
        h_z = torch.cat((h_z,self.get_time_embedd(heads, years, months, days)), 1)
        t_x = torch.cat((t_x,self.get_time_embedd(tails, years, months, days)), 1)
        t_y = torch.cat((t_y,self.get_time_embedd(tails, years, months, days)), 1)
        t_z = torch.cat((t_z,self.get_time_embedd(tails, years, months, days)), 1)
        return h_x, h_y, h_z, r_w, r_x, t_y, r_z, t_x, t_y, t_z
    
    def forward(self, heads, rels, tails, years, months, days):
        head_x, head_y, head_z, rel_w, rel_x, rel_y, rel_z, tail_x, tail_y, tail_z = self.getEmbeddings(heads, rels, tails, years, months, days)
        
        # print(head_x.shape, head_y.shape, head_z.shape, rel_w.shape, rel_x.shape, rel_y.shape, rel_z.shape, tail_x.shape, tail_y.shape, tail_z.shape)

        pi = 3.14159265358979323846
        
        denominator = torch.sqrt(rel_w ** 2 + rel_x ** 2 + rel_y ** 2 + rel_z ** 2)
        w = rel_w / denominator
        x = rel_x / denominator
        y = rel_y / denominator
        z = rel_z / denominator
        
        compute_tail_x = (1 - 2*y*y - 2*z*z) * head_x + (2*x*y - 2*z*w) * head_y + (2*x*z + 2*y*w) * head_z
        compute_tail_y = (2*x*y + 2*z*w) * head_x + (1 - 2*x*x - 2*z*z) * head_y + (2*y*z - 2*x*w) * head_z
        compute_tail_z = (2*x*z - 2*y*w) * head_x + (2*y*z + 2*x*w) * head_y + (1 - 2*x*x - 2*y*y) * head_z
        
        # if self.relation_embedding_has_mod:
        #     compute_tail_x = denominator * compute_tail_x
        #     compute_tail_y = denominator * compute_tail_y
        #     compute_tail_z = denominator * compute_tail_z
        
        delta_x = (compute_tail_x - tail_x)
        delta_y = (compute_tail_y - tail_y)
        delta_z = (compute_tail_z - tail_z)
        
        score1 = torch.stack([delta_x, delta_y, delta_z], dim = 0)
        score1 = score1.norm(dim = 0)
        
        x = -x
        y = -y
        z = -z
        compute_head_x = (1 - 2*y*y - 2*z*z) * tail_x + (2*x*y - 2*z*w) * tail_y + (2*x*z + 2*y*w) * tail_z
        compute_head_y = (2*x*y + 2*z*w) * tail_x + (1 - 2*x*x - 2*z*z) * tail_y + (2*y*z - 2*x*w) * tail_z
        compute_head_z = (2*x*z - 2*y*w) * tail_x + (2*y*z + 2*x*w) * tail_y + (1 - 2*x*x - 2*y*y) * tail_z
        
        # if self.relation_embedding_has_mod:
        #     compute_head_x = compute_head_x / denominator
        #     compute_head_y = compute_head_y / denominator
        #     compute_head_z = compute_head_z / denominator
        
        delta_x2 = (compute_head_x - head_x)
        delta_y2 = (compute_head_y - head_y)
        delta_z2 = (compute_head_z - head_z)
        
        score2 = torch.stack([delta_x2, delta_y2, delta_z2], dim = 0)
        score2 = score2.norm(dim = 0)     
        
        score1 = score1.mean(dim=1)
        score2 = score2.mean(dim=1)

      #         score1 = score1.sum(dim=2)
      #         score2 = score2.sum(dim=2)
        
        score = (score1 + score2) / 2
        
        return self.gamma.item() - score
