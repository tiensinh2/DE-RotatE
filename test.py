# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from dataset import Dataset
from trainer import Trainer
from tester import Tester
from params import Params

desc = 'Temporal KG Completion methods'
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('-dataset', help='Dataset', type=str, default='icews14', choices = ['icews14', 'icews05-15', 'gdelt'])
parser.add_argument('-model', help='Model', type=str, default='DE_DistMult', choices = ['DE_DistMult', 'DE_TransE', 'DE_SimplE', 'DE_HAKE', 'DE_HAKE1', 'DE_PaiRE', 'DE_RotatE', 'DE_ComplEx', 'DE_QuatDE', 'DE_ConvKB'])
parser.add_argument('-ne', help='Number of epochs', type=int, default=500)
parser.add_argument('-bsize', help='Batch size', type=int, default=512, choices = [512])
parser.add_argument('-lr', help='Learning rate', type=float, default=0.001)
parser.add_argument('-reg_lambda', help='L2 regularization parameter', type=float, default=0.0, choices = [0.0])
parser.add_argument('-emb_dim', help='Embedding dimension', type=int, default=100)
parser.add_argument('-neg_ratio', help='Negative ratio', type=int, default=500, choices = [500])
parser.add_argument('-dropout', help='Dropout probability', type=float, default=0.4)
parser.add_argument('-save_each', help='Save model and validate each K epochs', type=int, default=20, choices = [20])
parser.add_argument('-se_prop', help='Static embedding proportion', type=float, default=0.36)
parser.add_argument('-period', type=int, default=7)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()

dataset = Dataset(args.dataset)

params = Params(
    ne=args.ne, 
    bsize=args.bsize, 
    lr=args.lr, 
    reg_lambda=args.reg_lambda, 
    emb_dim=args.emb_dim, 
    neg_ratio=args.neg_ratio, 
    dropout=args.dropout, 
    save_each=args.save_each, 
    se_prop=args.se_prop,
    period=args.period,
)

print("Testing...")
tester = Tester(dataset, args.save_dir, "test")
tester.test()

