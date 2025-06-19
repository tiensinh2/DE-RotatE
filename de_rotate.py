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

class DE_RotatE(torch.nn.Module):
    def __init__(self, dataset, params):
        super(DE_RotatE, self).__init__()
        self.dataset = dataset
        self.params = params
        
        #khai báo các lớp nhúng cho thực thể đầu, quan hệ, thực thể đuôi
        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim).cuda()
        self.rel_embs = nn.Embedding(dataset.numRel(), params.s_emb_dim+params.t_emb_dim).cuda()
        
        self.create_time_embedds()

        #hàm kích hoạt sin (đây là o trong công thức (1) phía dưới)
        self.time_nl = torch.sinc
        
        #khởi tạo tham số ban đầu cho các lớp nhúng
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

        # Đây là thông số bổ xung để chuẩn hóa giá trị loss (của RotatE)
        self.gamma = nn.Parameter(
            torch.Tensor([18]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + 2) / (params.s_emb_dim+params.t_emb_dim)]),
            requires_grad=False
        )
    
    #khởi tạo nhúng thời gian (z = a*o(w*t + b)) (1)
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

        self.time_t = nn.Embedding((32*12*1)//self.params.period + 1, self.params.t_emb_dim).cuda()
        self.time_h = nn.Embedding(((32*12*1))//self.params.period + 1, self.params.t_emb_dim).cuda()
        # self.time_t = nn.Embedding((32*12*11)//self.params.period + 1, self.params.t_emb_dim).cuda()
        # self.time_h = nn.Embedding(((32*12*11))//self.params.period + 1, self.params.t_emb_dim).cuda()

        nn.init.xavier_uniform_(self.time_t.weight)
        nn.init.xavier_uniform_(self.time_h.weight)

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
        times = ((days-1) + (months-1) * 32)//self.params.period
        # times = ((days-1) + (months-1) * 32 + (years - 2005) * 366)//self.params.period
        times = times.long()

        years = years - 2010
        months = months/ 6 - 1
        days = days / 16 - 1
        if h_or_t == "head":
            #áp dụng công thức (1) cho ngày tháng năm 
            emb  = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years  + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days   + self.d_phi_h(entities))
            emb += self.time_h(times).squeeze(1)
        else:
            #áp dụng công thức (1) cho ngày tháng năm 
            emb  = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years  + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days   + self.d_phi_t(entities))
            emb += self.time_t(times).squeeze(1)
            
        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals = None):
        years = years.view(-1,1)
        months = months.view(-1,1)
        days = days.view(-1,1)
        r_embs = self.rel_embs(rels)
        #khởi tạo nhúng ban đầu
        h_embs1 = self.ent_embs_h(heads)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        t_embs2 = self.ent_embs_t(heads)
        
        #bổ xung nhúng thời gian nối vào nhúng ban đầu
        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)
        
        return h_embs1, r_embs, t_embs1, h_embs2, t_embs2
    
    def forward(self, heads, rels, tails, years, months, days):

        #khởi tạo nhúng thời gian
        re_head, relation, re_tail, im_head, im_tail = self.getEmbeddings(heads, rels, tails, years, months, days)
        pi = 3.14159265358979323846

        #phía dưới này là công thức của RotatE
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        #bổ xung dropout
        score = F.dropout(score, p=self.params.dropout, training=self.training)
        score = self.gamma.item() - score.sum(dim = 1)
        return score
