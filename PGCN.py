#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from rdkit import Chem
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
import torch.optim as optim
from torch.utils.data import Dataset
import torch_geometric
from torch_geometric.nn import GCNConv, ARMAConv
from torch_geometric.nn import global_add_pool
from torch_geometric.data import DataLoader, Data
import copy
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score, f1_score
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Polypeptide import three_to_one
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
from biotite.structure import filter_amino_acids
import seaborn as sn
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--device', type=str)
args = parser.parse_args()


args.batch_size = 8 # 16
args.epoch = 500
args.lr = 0.0000001
args.optim = 'Adam'
args.step_size = 10
args.gamma = 0.9
args.dropout = 0.2
args.n_features = 194
args.conv_dim1 = 128
args.conv_dim2 = 128
args.conv_dim3 = 128
args.concat_dim = 128
args.pred_dim1 = 128
args.pred_dim2 = 128
args.pred_dim3 = 128
args.out_dim = 2
# args.seed = 10
args.output_name = f'./EC500_{args.seed}'


np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device(args.device)
print('seed:', args.seed)
print('device:', device)


H1_dict = {'A' : 0.62, 
           'C' : 0.29, 
           'D' : -0.9, 
           'E' : -0.74, 
           'F' : 1.19, 
           'G' : 0.48, 
           'H' : -0.4, 
           'I' : 1.38, 
           'K' : -1.5, 
           'L' : 1.06, 
           'M' : 0.64, 
           'N' : -0.78, 
           'P' : 0.12, 
           'Q' : -0.85, 
           'R' : -2.53, 
           'S' : -0.18, 
           'T' : -0.05, 
           'V' : 1.08, 
           'W' : 0.81, 
           'Y' : 0.26}

H2_dict = {'A' : -0.5, 
           'C' : -1, 
           'D' : 3, 
           'E' : 3, 
           'F' : -2.5, 
           'G' : 0, 
           'H' : -0.5, 
           'I' : -1.8, 
           'K' : 3, 
           'L' : -1.8, 
           'M' : -1.3, 
           'N' : 2, 
           'P' : 0, 
           'Q' : 0.2, 
           'R' : 3, 
           'S' : 0.3, 
           'T' : -0.4, 
           'V' : -1.5, 
           'W' : -3.4, 
           'Y' : -2.3}

PL_dict = {'A' : 8.1, 
           'C' : 5.5, 
           'D' : 13, 
           'E' : 12.3, 
           'F' : 5.2, 
           'G' : 9, 
           'H' : 10.4, 
           'I' : 5.2, 
           'K' : 11.3, 
           'L' : 4.9, 
           'M' : 5.7, 
           'N' : 11.6, 
           'P' : 8, 
           'Q' : 10.5, 
           'R' : 10.5, 
           'S' : 9.2, 
           'T' : 8.6, 
           'V' : 5.9, 
           'W' : 5.4, 
           'Y' : 6.2}

SASA_dict = {'A' : 1.181, 
             'C' : 1.461, 
             'D' : 1.587, 
             'E' : 1.862, 
             'F' : 2.228, 
             'G' : 0.881, 
             'H' : 2.025, 
             'I' : 1.81, 
             'K' : 2.258, 
             'L' : 1.931, 
             'M' : 2.034, 
             'N' : 1.655, 
             'P' : 1.468, 
             'Q' : 1.932, 
             'R' : 2.56, 
             'S' : 1.298, 
             'T' : 1.525, 
             'V' : 1.645, 
             'W' : 2.663, 
             'Y' : 2.368}

pKa_dict = {'A' : 2.34,
            'R' : 2.17,
            'N' : 2.02,
            'D' : 1.88,
            'C' : 1.96,
            'E' : 2.19,
            'Q' : 2.17,
            'G' : 2.34,
            'H' : 1.82,
            'O' : 1.82,
            'I' : 2.36,
            'L' : 2.36,
            'K' : 2.18,
            'M' : 2.28,
            'F' : 1.83,
            'P' : 1.99,
            'U' : 0,
            'S' : 2.21,
            'T' : 2.09,
            'W' : 2.83,
            'Y' : 2.20,
            'V' : 2.32}

pKb_dict = {'A' : 9.69,
            'R' : 9.04,
            'N' : 8.80,
            'D' : 9.60,
            'C' : 10.28,
            'E' : 9.67,
            'Q' : 9.13,
            'G' : 9.60,
            'H' : 9.17,
            'O' : 9.65,
            'I' : 9.60,
            'L' : 9.60,
            'K' : 8.95,
            'M' : 9.21,
            'F' : 9.13,
            'P' : 10.60,
            'U' : 0,
            'S' : 9.15,
            'T' : 9.10,
            'W' : 9.39,
            'Y' : 9.11,
            'V' : 9.62}

pI_dict = {'A' : 6.00,
           'R' : 10.76,
           'N' : 5.41,
           'D' : 2.77,
           'C' : 5.07,
           'E' : 3.22,
           'Q' : 5.65,
           'G' : 5.97,
           'H' : 7.59,
           'O' : 0,
           'I' : 6.02,
           'L' : 5.98,
           'K' : 9.74,
           'M' : 5.74,
           'F' : 5.48,
           'P' : 6.30,
           'U' : 5.68,
           'S' : 5.68,
           'T' : 5.60,
           'W' : 5.89,
           'Y' : 5.66,
           'V' : 5.96}

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

AA = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def aa_features(x):
    return np.array(one_of_k_encoding(x, AA) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).aromaticity()), [0, 1]) +  
                    one_of_k_encoding(int(ProteinAnalysis(x).isoelectric_point()), [4, 5, 6, 8, 9]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).gravy()), [0, 1, 2, 3, 4, -4, -3, -1]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).secondary_structure_fraction()[0]), [0, 1]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).secondary_structure_fraction()[1]), [0, 1]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).secondary_structure_fraction()[2]), [0, 1]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).molar_extinction_coefficient()[0]), [0, 1490, 5500]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).molar_extinction_coefficient()[1]), [0, 1490, 5500]) + 
                    one_of_k_encoding(int(ProteinAnalysis(x).molecular_weight()), [121, 131, 132, 165, 133, 105, 75, 204, 174, 
                                                                                   146, 115, 147, 149, 117, 119, 181, 89, 155]) + 
                    one_of_k_encoding(H1_dict[x], list(set(list(H1_dict.values())))) + 
                    one_of_k_encoding(H2_dict[x], list(set(list(H2_dict.values())))) + 
                    one_of_k_encoding(PL_dict[x], list(set(list(PL_dict.values())))) + 
                    one_of_k_encoding(SASA_dict[x], list(set(list(SASA_dict.values())))) + 
                    one_of_k_encoding(pKa_dict[x], list(set(list(pKa_dict.values())))) + 
                    one_of_k_encoding(pKb_dict[x], list(set(list(pKb_dict.values())))) + 
                    one_of_k_encoding(pI_dict[x], list(set(list(pI_dict.values())))))

def adjacency2edgeindex(adjacency):
    start = []
    end = []
    adjacency = adjacency - np.eye(adjacency.shape[0], dtype=int)
    for x in range(adjacency.shape[1]):
        for y in range(adjacency.shape[0]):
            if adjacency[x, y] == 1:
                start.append(x)
                end.append(y)

    edge_index = np.asarray([start, end])
    return edge_index

AMINOS =  ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 
           'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET']

def filter_20_amino_acids(array):
    return ( np.in1d(array.res_name, AMINOS) & (array.res_id != -1) )

SEQ = ['A', 'T', 'G', 'C', 'U']
def filter_seq(array):
    return ( np.in1d(array.res_name, SEQ) & (array.res_id != -1) )

def protein_analysis(pdb_id):
    file_name = rcsb.fetch(pdb_id, "mmtf", './data/pdb')
    array = strucio.load_structure(file_name)
#     protein_mask = filter_amino_acids(array)
    protein_mask = filter_20_amino_acids(array)
    try:
        array = array[protein_mask]
    except:
        array = array[0]
        array = array[protein_mask]
    try:
        ca = array[array.atom_name == "CA"]
    except:
        array = array[0]
        ca = array[array.atom_name == "CA"]
    
    seq = ''.join([three_to_one(str(i).split(' CA')[0][-3:]) for i in ca])
    # 7 Angstrom adjacency threshold
    threshold = 7
    cell_list = struc.CellList(ca, cell_size=threshold)
    A = cell_list.create_adjacency_matrix(threshold)
    A = np.where(A == True, 1, A)

    return [aa_features(aa) for aa in seq], adjacency2edgeindex(A)

def pro2vec(pdb_id):
    node_f, edge_index = protein_analysis(pdb_id)
    data = Data(x=torch.tensor(node_f, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long))
    # print(data)
    return data

def make_pro(df):
    pro_key = []
    pro_value = []
    for i in range(df.shape[0]):
        pro_key.append(df['PDB ID'].iloc[i])
        pro_value.append(df['Class'].iloc[i])
    return pro_key, pro_value

def save_graph(graph_path, pdb_id):
    vec = pro2vec(pdb_id)
    np.save(graph_path+pdb_id+'_e.npy', vec.edge_index)
    np.save(graph_path+pdb_id+'_n.npy', vec.x)
    
def load_graph(graph_path, pdb_id):
    n = np.load(graph_path+pdb_id+'_n.npy')
    e = np.load(graph_path+pdb_id+'_e.npy')
    N = torch.tensor(n, dtype=torch.float)
    E = torch.tensor(e, dtype=torch.long)
    data = Data(x=N, edge_index=E)
    return data

def make_vec(pro, value, class_size):
    X = []
    Y = []
    for i in tqdm(range(len(pro))):
        m = pro[i]
        y = value[i]
        try:
            v = load_graph('./data/graph/', m)
            if v.x.shape[0] < 100000:
                X.append(v)
                Y.append(y)
        except:
            continue
    for i, data in enumerate(X):
        y = Y[i]      
        y = np.array([int(i) for i in y.split('/')])
        data.y = torch.tensor(y)
    return X

def df_check(df):
    df['pro2vec'] = 'Yes'
    for i in range(df.shape[0]):
        try:
            save_graph('./data/graph/', df['PDB ID'].iloc[i])
        except:
            df['pro2vec'].iloc[i] = 'No'
            continue
    df = df[df['pro2vec'] != 'No'].reset_index(drop=True)
    del df['pro2vec']
    return df


def metrics(pro_total, pred_pro_total):
    # TN FP
    # FN TP
    multi_conf = multilabel_confusion_matrix(pro_total.T, pred_pro_total.T)
    a = [((multi_conf[i][0, 0] + multi_conf[i][1, 1])/multi_conf[i].sum()) for i in range(class_size)] # (TP+TN)/(TP+FP+FN+TN)
    p = [((multi_conf[i][1, 1])/(multi_conf[i][0, 1] + multi_conf[i][1, 1])) for i in range(class_size)] # TP/(TP+FP)
    r = [((multi_conf[i][1, 1])/(multi_conf[i][1, 0] + multi_conf[i][1, 1])) for i in range(class_size)] # TP/(TP+FN)
    f = [2*((i*j)/(i+j)) for i, j in zip(p, r)] # 2*(Recall*Precision)/(Recall+Precision)
    acc = np.mean(a)
    mac_p = precision_score(pro_total.T, pred_pro_total.T, average='macro')
    mac_r = recall_score(pro_total.T, pred_pro_total.T, average='macro')
    mac_f = f1_score(pro_total.T, pred_pro_total.T, average='macro')
#     print(pd.DataFrame({'EC1': [round(a[0], 3), round(p[0], 3), round(r[0], 3), round(f[0], 3)], 
#                         'EC2': [round(a[1], 3), round(p[1], 3), round(r[1], 3), round(f[1], 3)], 
#                         'EC3': [round(a[2], 3), round(p[2], 3), round(r[2], 3), round(f[2], 3)], 
#                         'EC4': [round(a[3], 3), round(p[3], 3), round(r[3], 3), round(f[3], 3)], 
#                         'EC5': [round(a[4], 3), round(p[4], 3), round(r[4], 3), round(f[4], 3)], 
#                         'EC6': [round(a[5], 3), round(p[5], 3), round(r[5], 3), round(f[5], 3)]}, 
#                        index=['Acc', 'Pre', 'Rec', 'F1s']))
#     print('- accuracy : %.4f' % acc)
#     print('- macro_precision : %.4f' % mac_p)
#     print('- macro_recall : %.4f' % mac_r)
#     print('- macro_f1 : %.4f' % mac_f)
#     plt.figure(figsize = (8, 1))
#     for i in range(len(multi_conf)):
#         plt.subplot(1, class_size, i+1)   
#         sn.heatmap(multi_conf[i], annot=True, cmap='Blues', cbar=False, fmt="d", annot_kws={"size": 8})
#         plt.tick_params(left=False, bottom=False)
#         plt.xticks(fontsize=0)
#         plt.yticks(fontsize=0)
# #         plt.xlabel('Pred')
# #         plt.ylabel('True')
#     plt.tick_params(left=False, bottom=False)
#     plt.tight_layout()
#     plt.show()
    return a, p, r, f, acc, mac_p, mac_r, mac_f



class GCNlayer(nn.Module):
    def __init__(self, n_features, conv_dim1, conv_dim2, conv_dim3, concat_dim, dropout):
        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dim1 = conv_dim1
        self.conv_dim2 = conv_dim2
        self.conv_dim3 = conv_dim3
        self.concat_dim =  concat_dim
        self.dropout = dropout
        
        self.conv1 = ARMAConv(self.n_features, self.conv_dim1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = BatchNorm1d(self.conv_dim1)
        self.conv2 = ARMAConv(self.conv_dim1, self.conv_dim2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = BatchNorm1d(self.conv_dim2)
        self.conv3 = ARMAConv(self.conv_dim2, self.conv_dim3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.bn3 = BatchNorm1d(self.conv_dim3)
        self.conv4 = ARMAConv(self.conv_dim3, self.conv_dim3)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.bn4 = BatchNorm1d(self.conv_dim3)
        self.conv5 = ARMAConv(self.conv_dim3, self.conv_dim3)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.bn5 = BatchNorm1d(self.conv_dim3)
        self.conv6 = ARMAConv(self.conv_dim3, self.concat_dim)
        nn.init.xavier_uniform_(self.conv6.weight)
        self.bn6 = BatchNorm1d(self.concat_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class FClayer(nn.Module):
    def __init__(self, concat_dim, pred_dim1, pred_dim2, pred_dim3, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dim1 = pred_dim1
        self.pred_dim2 = pred_dim2
        self.pred_dim3 = pred_dim3
        self.out_dim = out_dim
        self.dropout = dropout

        self.fc1 = Linear(self.concat_dim, self.pred_dim1)
        self.bn1 = BatchNorm1d(self.pred_dim1)
        self.fc2 = Linear(self.pred_dim1, self.pred_dim2)
        self.bn2 = BatchNorm1d(self.pred_dim2)
        self.fc3 = Linear(self.pred_dim2, self.pred_dim3)
        self.bn3 = BatchNorm1d(self.pred_dim3)
        self.fc4 = Linear(self.pred_dim3, self.pred_dim3)
        self.bn4 = BatchNorm1d(self.pred_dim3)
        self.fc5 = Linear(self.pred_dim3, self.pred_dim3)
        self.bn5 = BatchNorm1d(self.pred_dim3)
        self.fc6 = Linear(self.pred_dim3, self.pred_dim3)
        self.bn6 = BatchNorm1d(self.pred_dim3)
        self.fc7 = Linear(self.pred_dim3, self.pred_dim3)
        self.bn7 = BatchNorm1d(self.pred_dim3)
        self.fc8 = Linear(self.pred_dim3, self.out_dim)
    
    def forward(self, data):
        x = F.relu(self.fc1(data))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc5(x))
        x = self.bn5(x)
        x = F.relu(self.fc6(x))
        x = self.bn6(x)
        x = F.relu(self.fc7(x))
        x = self.bn7(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc8(x)
        return x
    
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = GCNlayer(args.n_features, 
                              args.conv_dim1, 
                              args.conv_dim2, 
                              args.conv_dim3, 
                              args.concat_dim, 
                              args.dropout)
        self.fc1 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
        self.fc2 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
        self.fc3 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
        self.fc4 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
        self.fc5 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
        self.fc6 = FClayer(args.concat_dim, 
                           args.pred_dim1, 
                           args.pred_dim2, 
                           args.pred_dim3, 
                           args.out_dim, 
                           args.dropout)
      
    def forward(self, pro):
        x = self.conv1(pro)
        x1 = F.log_softmax(self.fc1(x), dim=1)
        x2 = F.log_softmax(self.fc2(x), dim=1)
        x3 = F.log_softmax(self.fc3(x), dim=1)
        x4 = F.log_softmax(self.fc4(x), dim=1)
        x5 = F.log_softmax(self.fc5(x), dim=1)
        x6 = F.log_softmax(self.fc6(x), dim=1)
        return torch.cat([x1, x2, x3, x4, x5, x6])



def make_pred(outputs):
    return torch.tensor([torch.max(outputs[i], 1)[1].tolist() for i in range(outputs.shape[0])])

def make_outputs(outputs):
    return outputs.reshape(class_size, args.batch_size, -1)

def make_labels(y):
    return y.reshape(args.batch_size, class_size).T

def get_loss(outputs, labels):
    loss_func = nn.CrossEntropyLoss()
    loss_1 = loss_func(outputs[0], labels[0])
    loss_2 = loss_func(outputs[1], labels[1])
    loss_3 = loss_func(outputs[2], labels[2])
    loss_4 = loss_func(outputs[3], labels[3])
    loss_5 = loss_func(outputs[4], labels[4])
    loss_6 = loss_func(outputs[5], labels[5])
    loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
    return loss

def save_checkpoint(epoch, model, optimizer, filename):
    state = {'Epoch': epoch,
             'State_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)
    
def train(model, device, optimizer, train_loader, criterion, args):
    pro_total = torch.zeros((6, 1)).int()
    pred_pro_total = torch.zeros((6, 1)).int()
    train_correct = 0
    train_total = 0
    train_loss = 0
    for i, pro in enumerate(train_loader):
        pro = pro.to(device)
        labels = make_labels(pro.y).to(device)
        outputs = model(pro)
        outputs = make_outputs(outputs)
        loss = criterion(outputs, labels)
        train_total += pro.y.size(0)
        predicted = make_pred(outputs)
        pro_total = torch.cat([pro_total, labels.cpu()], dim=1)
        pred_pro_total = torch.cat([pred_pro_total, predicted], dim=1)
        train_correct += (predicted == labels.cpu()).sum().item()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    pro_total = pro_total[:,1:]
    pred_pro_total = pred_pro_total[:,1:]
    train_loss /= len(train_loader)
    a, p, r, f, acc, mac_p, mac_r, mac_f = metrics(pro_total, pred_pro_total)
    print('- loss : %.4f' % train_loss)
    print('- accuracy : %.4f' % acc)
    return model, train_loss, a, p, r, f, acc, mac_p, mac_r, mac_f

def test(model, device, test_loader, criterion, args):
    model.eval()
    pro_total = torch.zeros((6, 1)).int()
    pred_pro_total = torch.zeros((6, 1)).int()
    test_correct = 0
    test_total = 0
    test_loss = 0
    with torch.no_grad():
        for i, pro in enumerate(test_loader):
            pro = pro.to(device)
            labels = make_labels(pro.y).to(device)
            outputs = model(pro)
            outputs = make_outputs(outputs)
            loss = criterion(outputs, labels)
            test_total += pro.y.size(0)
            predicted = make_pred(outputs)
            test_loss += loss.item()
            pro_total = torch.cat([pro_total, labels.cpu()], dim=1)
            pred_pro_total = torch.cat([pred_pro_total, predicted], dim=1)
            test_correct += (predicted == labels.cpu()).sum().item()
    pro_total = pro_total[:,1:]
    pred_pro_total = pred_pro_total[:,1:]
    test_loss /= len(test_loader)
    a, p, r, f, acc, mac_p, mac_r, mac_f = metrics(pro_total, pred_pro_total)
    print('- loss : %.4f' % test_loss)
    print('- accuracy : %.4f' % acc)
    return pro_total, pred_pro_total, test_loss, a, p, r, f, acc, mac_p, mac_r, mac_f

def experiment(model, train_loader, test_loader, device, args):
    time_start = time.time()
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = get_loss
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.step_size,
                                          gamma=args.gamma)
    
    list_train_acc = []
    list_train_loss = []
    list_test_acc = []
    list_test_loss = []
    list_test_macp = []
    list_test_macr = []
    list_test_macf = []
    for epoch in tqdm(range(args.epoch)):
        scheduler.step()
        print('- Epoch :', epoch+1)
        print('[Train]')
        model, train_l, _, _, _, _, train_a, _, _, _  = train(model, device, optimizer, train_loader, criterion, args)
#         print('[Valid]')
#         pro_total, pred_pro_total, test_l, a, p, r, f, test_a, mac_p, mac_r, mac_f = test(model, device, train_loader, criterion, args)
        print('[Test]')
        pro_total, pred_pro_total, test_l, a, p, r, f, test_a, mac_p, mac_r, mac_f = test(model, device, test_loader, criterion, args)
        list_train_acc.append(train_a)
        list_train_loss.append(train_l)
        list_test_acc.append(test_a)
        list_test_loss.append(test_l)
        list_test_macp.append(mac_p)
        list_test_macr.append(mac_r)
        list_test_macf.append(mac_f)
        print()
    print('[Final Test]')
    pro_total, pred_pro_total, test_loss, accuracy, precision, recall, f1score, test_acc, macro_precision, macro_recall, macro_f1score     = test(model, device, test_loader, criterion, args)
    multi_conf = multilabel_confusion_matrix(pro_total.T, pred_pro_total.T)
    
    time_end = time.time()
    time_required = time_end - time_start
    
    args.list_train_acc = list_train_acc
    args.list_train_loss = list_train_loss
    args.list_test_acc = list_test_acc
    args.list_test_loss = list_test_loss
    args.multi_conf = multi_conf
    args.test_acc = test_acc
    args.test_loss = test_loss
    args.list_test_macp = list_test_macp
    args.list_test_macr = list_test_macr
    args.list_test_macf = list_test_macf
    # args.pro_total = pro_total
    # args.pred_pro_total = pred_pro_total
    args.test_acc = test_acc
    args.test_loss = test_loss
    args.accuracy = accuracy
    args.precision = precision
    args.recall = recall
    args.f1score = f1score
    args.macro_precision = macro_precision
    args.macro_recall = macro_recall
    args.macro_f1score = macro_f1score
    args.multi_conf = multi_conf
    args.time_required = time_required
    
    save_checkpoint(epoch, model, optimizer, f'{str(args.output_name)}_mymodel.pt')
    return args, pro_total, pred_pro_total


def make_plots(df, pro_true, pro_pred):

    train_loss = df['list_train_loss'].iloc[0]
    train_acc = df['list_train_acc'].iloc[0]
    test_loss = df['list_test_loss'].iloc[0]
    test_acc = df['list_test_acc'].iloc[0]
    list_test_macp = df['list_test_macp'].iloc[0]
    list_test_macr = df['list_test_macr'].iloc[0]
    list_test_macf = df['list_test_macf'].iloc[0]
    a = df['accuracy'].iloc[0]
    p = df['precision'].iloc[0]
    r = df['recall'].iloc[0]
    f = df['f1score'].iloc[0]
    accuracy = df['test_acc'].iloc[0]
    macro_precision = df['macro_precision'].iloc[0]
    macro_recall = df['macro_recall'].iloc[0]
    macro_f1score = df['macro_f1score'].iloc[0]
    multi_conf = df['multi_conf'].iloc[0]

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.plot([e for e in range(len(train_acc))], [float(t)*0.01 for t in train_acc], label="train_acc", c='blue')
    plt.plot([e for e in range(len(test_acc))], [float(t)*0.01 for t in test_acc], label="test_acc", c='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()
    plt.savefig(f'{str(args.output_name)}_acc.png')

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.plot([e for e in range(len(list_test_macp))], [float(t) for t in list_test_macp], label="macro_precision", c='blue')
    plt.plot([e for e in range(len(list_test_macr))], [float(t) for t in list_test_macr], label="macro_recal", c='orange')
    plt.plot([e for e in range(len(list_test_macf))], [float(t) for t in list_test_macf], label="macro_f1score", c='red')
    plt.xlabel("Epoch")
    plt.ylabel("Scores")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.show()
    plt.savefig(f'{str(args.output_name)}_micro.png')
    
    print()
    multi_conf = np.array(multi_conf)
    print(pd.DataFrame({'EC1': [round(a[0]/100, 3), round(p[0], 3), round(r[0], 3), round(f[0], 3)], 
                        'EC2': [round(a[1]/100, 3), round(p[1], 3), round(r[1], 3), round(f[1], 3)], 
                        'EC3': [round(a[2]/100, 3), round(p[2], 3), round(r[2], 3), round(f[2], 3)], 
                        'EC4': [round(a[3]/100, 3), round(p[3], 3), round(r[3], 3), round(f[3], 3)], 
                        'EC5': [round(a[4]/100, 3), round(p[4], 3), round(r[4], 3), round(f[4], 3)], 
                        'EC6': [round(a[5]/100, 3), round(p[5], 3), round(r[5], 3), round(f[5], 3)]}, 
                       index=['Acc', 'Pre', 'Rec', 'F1s']))
    for i in range(len(multi_conf)):
        plt.figure(figsize = (6, 5))
        sn.heatmap(multi_conf[i], annot=True, cmap='Blues', fmt="d", annot_kws={"size": 20})
        plt.tick_params(left=False, bottom=False)
        plt.tight_layout()
        plt.xlabel('Predicted Class', fontsize=16)
        plt.ylabel('True Class', fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)
#         plt.show()
        plt.savefig(f'{str(args.output_name)}_conf_{i+1}.png')
    
    print()
    print('[Total]')
    print('- total_accuracy : %.4f' % accuracy)    
    print('- macro_precision : %.4f' % macro_precision)
    print('- macro_recall : %.4f' % macro_recall)
    print('- macro_f1score : %.4f' % macro_f1score)
    print()



df = pd.read_csv('PDB_EnzyNet_Dataset.csv')


class_list = []
for ec in df['EC']:
    for i in ec[1:-1].split(', '):
        class_list += i.split('/')

class_size = len(list(set(class_list)))

df['Class'] = 0
for i in range(df.shape[0]):
    ec = [int(i) for i in df['EC'].iloc[i][1:-1].split(', ')]
    cl = [0 for i in range(class_size)]
    for e in ec:
        cl[e-1] = 1
    df['Class'].iloc[i] = '/'.join([str(i) for i in cl])

class_dict = {}
for i in range(class_size):
    class_dict[i+1] = class_list.count(str(i+1))
print(class_dict)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=args.seed)#, stratify=df['Class'])
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

class_list = []
for ec in X_train['EC']:
    for i in ec[1:-1].split(', '):
        class_list += i.split('/')

class_dict = {}
for i in range(class_size):
    class_dict[i+1] = class_list.count(str(i+1))
print(class_dict)

class_list = []
for ec in X_test['EC']:
    for i in ec[1:-1].split(', '):
        class_list += i.split('/')

class_dict = {}
for i in range(class_size):
    class_dict[i+1] = class_list.count(str(i+1))
print(class_dict)

train_pro_key, train_pro_value = make_pro(X_train)
test_pro_key, test_pro_value = make_pro(X_test)

train_X = make_vec(train_pro_key, train_pro_value, class_size)
test_X = make_vec(test_pro_key, test_pro_value, class_size)



model = Net(args)
model = model.to(device)

train_loader = DataLoader(train_X, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_X, batch_size=args.batch_size, shuffle=False, drop_last=True)

dict_result = dict()
args.exp_name = 'Test'
results, pro_total, pred_pro_total = experiment(model, train_loader, test_loader, device, args)
result = vars(results)
dict_result[args.exp_name] = copy.deepcopy(result)
torch.cuda.empty_cache()

result_df = pd.DataFrame(dict_result).transpose()
result_df.to_json(f'{str(args.output_name)}_results.json', orient='table')

np.save(f'{str(args.output_name)}_true.npy', pro_total)
np.save(f'{str(args.output_name)}_pred.npy', pred_pro_total)

make_plots(result_df, pro_total, pred_pro_total)





