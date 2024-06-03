import timeit

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

from metrics import *
from DTIDataset import DTIDataset
from models.net import MDGTDTInet

device = torch.device('cuda')

def train(model, device, train_loader, optimizer):

    model.train()
    for batch_idx, data in enumerate(train_loader):
        label = data[-1].to(device)
        compound_graph, protein_graph, protein_embedding = data[:-1]
        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        output = model(compound_graph, protein_graph,  protein_embedding)
        loss = criterion(output, label.view(-1, 1).float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in test_loader:
            label = data[-1].to(device)
            compound_graph, protein_graph, protein_embedding = data[:-1]
            compound_graph = compound_graph.to(device)
            protein_graph = protein_graph.to(device)
            protein_embedding = protein_embedding.to(device)
            output = model(compound_graph, protein_graph, protein_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, label.view(-1, 1).cpu()), 0)

    total_labels = total_labels.numpy().flatten()
    total_preds = total_preds.numpy().flatten()

    MSE = mse(total_labels, total_preds)
    RMSE = rmse(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)
    return MSE, RMSE, CI, RM2


if __name__ == '__main__':

    dataset = 'Davis'
    file_path = 'data/' + dataset + '/processed'

    fold = 1
    epochs = 1000
    batch = 4
    lr = 1e-4

    train_set = DTIDataset(dataset=dataset, compound_graph=file_path + '/train/fold/' + str(fold) +'/compound_graph.bin',
                           compound_id=file_path + '/train/fold/' + str(fold) +'/compound_id.npy',
                           protein_graph=file_path + '/train/fold/' + str(fold) +'/protein_graph.bin',
                           protein_embedding=file_path + '/train/fold/' + str(fold) +'/protein_embedding.npy',
                           protein_id=file_path + '/train/fold/' + str(fold) +'/protein_id.npy',
                           label=file_path + '/train/fold/' + str(fold) +'/label.npy')
    test_set = DTIDataset(dataset=dataset, compound_graph=file_path + '/test/fold/' + str(fold) +'/compound_graph.bin',
                          compound_id=file_path + '/test/fold/' + str(fold) +'/compound_id.npy',
                          protein_graph=file_path + '/test/fold/' + str(fold) +'/protein_graph.bin',
                          protein_embedding=file_path + '/test/fold/' + str(fold) +'/protein_embedding.npy',
                          protein_id=file_path + '/test/fold/' + str(fold) +'/protein_id.npy',
                          label=file_path + '/test/fold/' + str(fold) +'/label.npy')

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=train_set.collate, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch, shuffle=False, collate_fn=test_set.collate, drop_last=True)

    model = MDGTDTInet(compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1)
    model.to(device)

    start = timeit.default_timer()
    best_ci = 0
    best_mse = 100
    best_r2 = 0
    best_epoch = -1
    file_model = 'model_save/' + dataset + '/fold/' + str(fold) + '/'

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=80, verbose=True, min_lr=1e-5)
    criterion = nn.MSELoss()

    Indexes = ('Epoch\t\tTime\t\tMSE\t\tRMSE\t\tCI\t\tr2')

    """Start training."""
    print('Training on ' + dataset + ', fold:' + str(fold))
    print(Indexes)

    for epoch in range(epochs):
        train(model, device, train_loader, optimizer)
        mse_test, rmse_test, ci_test, rm2_test = test(model, device, test_loader)
        scheduler.step(mse_test)
        end = timeit.default_timer()
        time = end - start
        ret = [epoch + 1, round(time, 2), round(mse_test, 5), round(rmse_test, 5), round(ci_test, 5), round(rm2_test, 5)]
        print('\t\t'.join(map(str, ret)))
        if mse_test < best_mse:
            if mse_test < 0.600:
                torch.save(model.state_dict(), file_model + 'Epoch:' + str(epoch + 1) + '.pt')
                print("model has been saved")
            best_epoch = epoch + 1
            best_mse = mse_test
            print('MSE improved at epoch ', best_epoch, ';\tbest_mse:', best_mse)

