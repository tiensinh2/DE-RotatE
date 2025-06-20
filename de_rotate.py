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

        self.ent_embs_h = nn.Embedding(dataset.numEnt(), params.s_emb_dim)
        self.ent_embs_t = nn.Embedding(dataset.numEnt(), params.s_emb_dim)
        self.rel_embs = nn.Embedding(dataset.numRel(), params.s_emb_dim + params.t_emb_dim)

        self.create_time_embedds()

        self.time_nl = torch.sinc

        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

        self.gamma = nn.Parameter(torch.Tensor([18]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + 2) / (params.s_emb_dim + params.t_emb_dim)]),
            requires_grad=False
        )

    def create_time_embedds(self):
        dim = self.params.t_emb_dim
        ent = self.dataset.numEnt()
        p = self.params.period
        # Frequency embeddings
        self.m_freq_h = nn.Embedding(ent, dim)
        self.m_freq_t = nn.Embedding(ent, dim)
        self.d_freq_h = nn.Embedding(ent, dim)
        self.d_freq_t = nn.Embedding(ent, dim)
        self.y_freq_h = nn.Embedding(ent, dim)
        self.y_freq_t = nn.Embedding(ent, dim)

        # Phase shift (phi)
        self.m_phi_h = nn.Embedding(ent, dim)
        self.m_phi_t = nn.Embedding(ent, dim)
        self.d_phi_h = nn.Embedding(ent, dim)
        self.d_phi_t = nn.Embedding(ent, dim)
        self.y_phi_h = nn.Embedding(ent, dim)
        self.y_phi_t = nn.Embedding(ent, dim)

        # Amplitudes
        self.m_amps_h = nn.Embedding(ent, dim)
        self.m_amps_t = nn.Embedding(ent, dim)
        self.d_amps_h = nn.Embedding(ent, dim)
        self.d_amps_t = nn.Embedding(ent, dim)
        self.y_amps_h = nn.Embedding(ent, dim)
        self.y_amps_t = nn.Embedding(ent, dim)

        # Time position embeddings
        self.time_t = nn.Embedding((32*12)//p + 1, dim)
        self.time_h = nn.Embedding((32*12)//p + 1, dim)

        # Initialize all embeddings
        for emb in [self.time_t, self.time_h,
                    self.m_freq_h, self.d_freq_h, self.y_freq_h,
                    self.m_freq_t, self.d_freq_t, self.y_freq_t,
                    self.m_phi_h, self.d_phi_h, self.y_phi_h,
                    self.m_phi_t, self.d_phi_t, self.y_phi_t,
                    self.m_amps_h, self.d_amps_h, self.y_amps_h,
                    self.m_amps_t, self.d_amps_t, self.y_amps_t]:
            nn.init.xavier_uniform_(emb.weight)

    def get_time_embedd(self, entities, years, months, days, h_or_t):
        times = ((days - 1) + (months - 1) * 32) // self.params.period
        times = times.long()

        years = years - 2010
        months = months / 6 - 1
        days = days / 16 - 1

        if h_or_t == "head":
            emb = self.y_amps_h(entities) * self.time_nl(self.y_freq_h(entities) * years + self.y_phi_h(entities))
            emb += self.m_amps_h(entities) * self.time_nl(self.m_freq_h(entities) * months + self.m_phi_h(entities))
            emb += self.d_amps_h(entities) * self.time_nl(self.d_freq_h(entities) * days + self.d_phi_h(entities))
            emb += self.time_h(times).squeeze(1)
        else:
            emb = self.y_amps_t(entities) * self.time_nl(self.y_freq_t(entities) * years + self.y_phi_t(entities))
            emb += self.m_amps_t(entities) * self.time_nl(self.m_freq_t(entities) * months + self.m_phi_t(entities))
            emb += self.d_amps_t(entities) * self.time_nl(self.d_freq_t(entities) * days + self.d_phi_t(entities))
            emb += self.time_t(times).squeeze(1)

        return emb

    def getEmbeddings(self, heads, rels, tails, years, months, days, intervals=None):
        years = years.view(-1, 1)
        months = months.view(-1, 1)
        days = days.view(-1, 1)

        r_embs = self.rel_embs(rels)

        h_embs1 = self.ent_embs_h(heads)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        t_embs2 = self.ent_embs_t(heads)

        h_embs1 = torch.cat((h_embs1, self.get_time_embedd(heads, years, months, days, "head")), 1)
        t_embs1 = torch.cat((t_embs1, self.get_time_embedd(tails, years, months, days, "tail")), 1)
        h_embs2 = torch.cat((h_embs2, self.get_time_embedd(tails, years, months, days, "head")), 1)
        t_embs2 = torch.cat((t_embs2, self.get_time_embedd(heads, years, months, days, "tail")), 1)

        return h_embs1, r_embs, t_embs1, h_embs2, t_embs2

    def forward(self, heads, rels, tails, years, months, days):
        re_head, relation, re_tail, im_head, im_tail = self.getEmbeddings(
            heads, rels, tails, years, months, days
        )

        pi = 3.14159265358979323846
        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = F.dropout(score, p=self.params.dropout, training=self.training)
        score = self.gamma.item() - score.sum(dim=1)
        return score
