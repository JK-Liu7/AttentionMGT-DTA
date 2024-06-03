import dgl
import pandas as pd
import torch
import numpy as np

from dgl import load_graphs
from torch.utils.data import DataLoader, Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')


class DTIDataset(Dataset):
    def __init__(self, dataset='Davis', compound_graph=None, compound_id=None, protein_graph=None, protein_embedding=None, protein_id=None, label=None):

        self.dataset = dataset
        self.compound_graph, _ = load_graphs(compound_graph)
        self.compound_graph = list(self.compound_graph)

        self.protein_graph, _ = load_graphs(protein_graph)
        self.protein_graph = list(self.protein_graph)
        self.protein_embedding = np.load(protein_embedding, allow_pickle=True)

        self.compound_id = np.load(compound_id, allow_pickle=True)
        self.protein_id = np.load(protein_id, allow_pickle=True)
        self.label = np.load(label, allow_pickle=True)


    def __len__(self):
        return len(self.label)


    def __getitem__(self, idx):

        compound_len = self.compound_graph[idx].num_nodes()
        protein_len = self.protein_graph[idx].num_nodes()
        return self.compound_graph[idx], self.protein_graph[idx], self.protein_embedding[idx], compound_len, protein_len, self.label[idx]


    def collate(self, sample):
        batch_size = len(sample)

        compound_graph, protein_graph, protein_embedding, compound_len, protein_len, label = map(list, zip(*sample))
        max_protein_len = max(protein_len)

        for i in range(batch_size):
            if protein_embedding[i].shape[0] < max_protein_len:
                protein_embedding[i] = np.pad(protein_embedding[i], ((0, max_protein_len-protein_embedding[i].shape[0]), (0, 0)), mode='constant', constant_values = (0,0))

        compound_graph = dgl.batch(compound_graph).to(device)

        protein_graph = dgl.batch(protein_graph).to(device)
        protein_embedding = torch.FloatTensor(protein_embedding).to(device)
        label = torch.FloatTensor(label).to(device)
        return compound_graph, protein_graph, protein_embedding, label

