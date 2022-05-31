from logging import root
import os.path as osp
import heapq
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric import seed_everything
from torch_geometric.nn import GATConv, TransformerConv, TAGConv,GCNConv,ResGatedGraphConv,GATv2Conv,ARMAConv,HypergraphConv,EGConv,FAConv,SuperGATConv,FiLMConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import KFold
import random
import datetime
import time

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.conv1 = TransformerConv(in_channels, 128)
        self.conv2 = TransformerConv(128, out_channels)
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index,pos_edge_index,neg_edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        out =x[edge_index[0]] * x[edge_index[1]]
        out = self.linear(out).squeeze(-1)
        return out


def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(data, model, optimizer):
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=int(data.train_pos_edge_index.size(1)))
    optimizer.zero_grad()
    link_logits = model(data.x, data.train_pos_edge_index,data.train_pos_edge_index,neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    aaa = link_logits
    bbb = link_labels
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc as auc3
    precision, recall, thresholds = precision_recall_curve(bbb.cpu(), aaa.cpu().detach().numpy())
    pr = auc3(recall, precision)
    roc = roc_auc_score(bbb.cpu(), aaa.cpu().detach().numpy())
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss,roc,pr


@torch.no_grad()
def test(data, model):
    model.eval()
    result = []
    for prefix in ['val', 'test']:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        link_logits = model(data.x, data.train_pos_edge_index,pos_edge_index,neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import auc as auc3
        precision, recall, thresholds = precision_recall_curve(link_labels.cpu(), link_probs.cpu())
        pr = auc3(recall, precision)
        roc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
        a=[]
        a.append(roc)
        a.append(pr)
        result.append(a)
    return result

def test_max(data, model,bian):
    model.eval()
    result = []
    pos_edge_index = data['test_pos_edge_index']
    neg_edge_index = data['test_neg_edge_index']
    link_logits = model(data.x, data.train_pos_edge_index, bian, neg_edge_index)
    a = link_logits.cpu().detach().numpy().tolist()
    b = pd.Series(a).sort_values()[-6::2]
    c = pd.Series(a).sort_values().index[-6::2]
    druggg = []
    print(b)
    for i in c:
        druggg.append([int(bian[0][i].cpu()),int(bian[1][i].cpu())])

    edge = np.loadtxt('原始点.txt', encoding='utf-8')
    zidian = {}
    for i in range(len(edge)):
        zidian[i] = str(edge[i][0]) + '\t' + str(edge[i][1])


    ggggg= []
    for i in druggg:
        _1 = zidian[i[0]].split('\t')
        _2 = zidian[i[1]].split('\t')
        ggggg.append([int(eval(_1[0])),int(eval(_1[1])),int(eval(_2[0])),int(eval(_2[1]))])
        print(int(eval(_1[0])), end='\t')
        print(int(eval(_1[1])), end='\t\t\t')
        print(int(eval(_2[0])), end='\t')
        print(int(eval(_2[1])))
    _1 = open('drug.txt', 'r', encoding='utf-8')
    drug_1 = _1.readlines()
    _2= open('protein.txt', 'r', encoding='utf-8')
    protein_1 = _2.readlines()
    _3 = open('drug_dict_map.txt', 'r', encoding='utf-8')
    drug_2 = _3.readlines()
    _4 = open('protein_dict_map.txt', 'r', encoding='utf-8')
    protein_2 = _4.readlines()
    drug_num = {}
    pro_num = {}
    drug_name ={}
    pro_name= {}
    for i,j in enumerate(drug_1):
        drug_num[i+1] = j.strip()
    for i, j in enumerate(protein_1):
        pro_num[i+1] = j.strip()
    for i in drug_2:
        aaa = i.split(':')
        drug_name[aaa[0]] = aaa[1].strip()
    for i in protein_2:
        aaa = i.split(':')
        pro_name[aaa[0]] = aaa[1].strip()
    for i in ggggg:
        print(drug_name[drug_num[i[0]]],end='\t')
        print(pro_name[pro_num[i[1]]],end='\t\t\t')
        print(drug_name[drug_num[i[2]]],end='\t')
        print(pro_name[pro_num[i[3]]])

def main():
    a = []
    b = []
    epoch = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge = np.loadtxt('bian.txt', encoding='utf-8')
    x = np.loadtxt('feature50.txt', encoding='utf-8')
    edge = torch.from_numpy(edge).type(torch.LongTensor)
    x = torch.from_numpy(x).type(torch.float32)
    data = Data(x=x, edge_index=edge)
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
    data = data.to(device)
    model = Net(50, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    best_test_roc = 0
    best_test_pr = 0
    loss_list = []
    roc_list = []
    pr_list = []
    for epoch in range(epoch):
        loss,roc_train,pr_train = train(data, model, optimizer)
        if epoch % 100 == 0:
            result = test(data, model)
            val_roc,val_pr = result[0][0],result[0][1]
            test_roc ,test_pr= result[1][0], result[1][1]
            if test_roc > best_test_roc:
                best_test_roc = test_roc
            if test_pr > best_test_pr:
                best_test_pr = test_pr
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}',  f'  train: roc:{roc_train:.4f}, pr:{pr_train:.4f}', f'  Val: roc:{val_roc:.4f}, pr:{val_pr:.4f}',f'  Test: roc:{test_roc:.4f}, pr:{test_pr:.4f},')
            loss_list.append(round(loss.item(),4))
            roc_list.append(round(test_roc,4))
            pr_list.append(round(test_pr,4))
    a.append(best_test_roc)
    b.append(best_test_pr)
    print("loss_list{}".format(loss_list))
    print("roc_list{}".format(roc_list))
    print("pr_list{}".format(pr_list))
    print("auroc = {:.4f}".format(np.mean(a)))
    print("auprc = {:.4f}".format(np.mean(b)))
if __name__ == "__main__":
    start_time =  time.time()
    seed = 22
    seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    main()
    end_time = time.time()
    print(end_time - start_time)
