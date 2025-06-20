# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
from params import Params

# Import các mô hình DE
from de_distmult import DE_DistMult
from de_transe import DE_TransE
from de_simple import DE_SimplE
from de_hake1 import DE_HAKE1
from de_hake import DE_HAKE
from de_paire import DE_PaiRE
from de_rotate import DE_RotatE
from de_complex import DE_ComplEx
from de_quatde import DE_QuatDE
from de_convkb import DE_ConvKB

from tester import Tester

class Trainer:
    def __init__(self, dataset, params, model_name):
        instance_gen = globals()[model_name]
        self.model_name = model_name

        # 🔧 Khởi tạo model và chuyển sang GPU đúng cách
        self.model = instance_gen(dataset=dataset, params=params)  # tạo mô hình
        self.model = self.model.to('cuda')                         # chuyển toàn bộ model sang GPU
        self.model = nn.DataParallel(self.model)                   # bọc bằng DataParallel (nếu có nhiều GPU)

        self.dataset = dataset
        self.params = params

    def train(self, early_stop=False):
        self.model.train()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"🔢 Số lượng tham số của mô hình: {num_params:,}")

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.lr, 
            weight_decay=self.params.reg_lambda
        )

        loss_f = nn.CrossEntropyLoss()

        total_start_time = time.time()  # ⏱️ Bắt đầu đo thời gian tổng

        for epoch in range(1, self.params.ne + 1):
            epoch_start_time = time.time()

            last_batch = False
            total_loss = 0.0

            while not last_batch:
                optimizer.zero_grad()

                heads, rels, tails, years, months, days = self.dataset.nextBatch(
                    self.params.bsize, neg_ratio=self.params.neg_ratio
                )
                last_batch = self.dataset.wasLastBatch()

                # forward
                scores = self.model(heads, rels, tails, years, months, days)

                num_examples = int(heads.shape[0] / (1 + self.params.neg_ratio))
                scores = scores.view(num_examples, self.params.neg_ratio + 1)

                # ⚠️ label đúng là vị trí 0 (positive sample)
                labels = torch.zeros(num_examples).long().cuda()

                loss = loss_f(scores, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f"⏱️ Epoch {epoch}/{self.params.ne} mất {epoch_duration:.2f} giây")
            print(f"📉 Loss epoch {epoch}: {total_loss:.4f} ({self.model_name}, {self.dataset.name})")

            if epoch % self.params.save_each == 0:
                self.saveModel(epoch)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"\n✅ Tổng thời gian huấn luyện: {total_duration:.2f} giây")

    def saveModel(self, chkpnt):
        print("💾 Đang lưu mô hình...")
        directory = f"models/{self.model_name}/{self.dataset.name}/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path = directory + self.params.str_() + f"_{chkpnt}.chkpnt"
        torch.save(self.model, model_path)
        print(f"✅ Đã lưu: {model_path}")
