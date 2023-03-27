import os
import pickle
import timeit

import deepchem
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
from scipy import sparse as sp
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis.analysis import distances
from itertools import product, groupby, permutations
from scipy.spatial import distance_matrix
from dgl import load_graphs
import warnings

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda')

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
					obtain_self_dist(res) +  # 5
					obtain_dihediral_angles(res)  # 4
					)


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):

    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def load_protein(protpath, explicit_H=False, use_chirality=True):

    mol = Chem.MolFromPDBFile(protpath, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def prot_to_graph(id, prot_pdb, cutoff=10.0):
    """obtain the residue graphs"""
    pk = deepchem.dock.ConvexHullPocketFinder()
    prot = Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False)
    Chem.AssignStereochemistryFrom3D(prot)
    u = mda.Universe(prot)
    g = dgl.DGLGraph()
    # Add nodes
    num_residues = len(u.residues)
    g.add_nodes(num_residues)
    res_feats = np.array([calc_res_features(res) for res in u.residues])

    esm_feats = np.load('data/Davis/processed/ESM_embedding/' + id + '.npy', allow_pickle=True)
    len_esm = np.size(esm_feats, 0)
    if len_esm < num_residues:
        esm_feats = np.pad(esm_feats, ((0, num_residues-len_esm), (0,0)), 'constant')
    elif len_esm > num_residues:
        esm_feats = esm_feats[:num_residues, :]

    prot_feats = np.concatenate((res_feats, esm_feats), axis=1)

    g.ndata["feats"] = torch.tensor(prot_feats)
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)
    g.add_edges(src_list, dst_list)
    g.ndata["ca_pos"] = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    g.ndata["center_pos"] = torch.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    cadist = torch.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    cedist = torch.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = torch.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))
    g.edata["feats"] = torch.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm)], dim=1)
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")

    ca_pos = np.array(np.array([obtain_ca_pos(res) for res in u.residues]))

    pockets = pk.find_pockets(prot_pdb)
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        idxs = []
        for idx in range(ca_pos.shape[0]):
            if x_min < ca_pos[idx][0] < x_max and y_min < ca_pos[idx][1] < y_max and z_min < ca_pos[idx][2] < z_max:
                idxs.append(idx)

    g_pocket = dgl.node_subgraph(g, idxs)
    g_pocket = laplacian_positional_encoding(g_pocket, pos_enc_dim=8)
    return g_pocket


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


if __name__ == '__main__':
    dataset = 'Davis'
    fold = 1
    file_path = 'data/' + dataset + '/DTA/fold/'
    file_path_protein = 'data/' + dataset + '/' + dataset + '_protein_mapping.csv'
    dir_output = ('data/' + dataset + '/processed/')
    os.makedirs(dir_output, exist_ok=True)

    train_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_train.csv')
    test_data = pd.read_csv(file_path + str(fold) + '/' + dataset + '_test.csv')

    raw_data_protein = pd.read_csv(file_path_protein)
    # raw_data_protein = pd.read_csv(file_path_protein)
    protein_id_unique = raw_data_protein['PROTEIN ID'].values
    # protein_seq_unique = raw_data_protein['PROTEIN_SEQUENCE'].values

    protein_id_train = train_data['PROTEIN_ID'].values
    protein_id_test = test_data['PROTEIN_ID'].values

    N = len(protein_id_unique)
    protein_max_len = 1000
    proteins_id_train, proteins_id_test, proteins_id_val = list(), list(), list()

    start = timeit.default_timer()

    # protein_graph process
    # for no, data in enumerate(protein_id_unique):
    #     proteins_g = list()
    #     print('/'.join(map(str, [no + 1, N])))
    #     # seq = protein_seq_unique[no]
    #     pdb_pdb = 'data/' + dataset + '/PDB/' + data + '.pdb'
    #     # protein_pdb = load_protein(pdb_path, explicit_H=False, use_chirality=True)
    #     protein_graph = prot_to_graph(data, pdb_pdb, cutoff=10.0)
    #     # protein_embedding = label_sequence(data, CHARPROTSET, protein_max_len)
    #     proteins_g.append(protein_graph)
    #     dgl.save_graphs(dir_output + '/protein_graph_PDB/' + data + '.bin', list(proteins_g))
    #     # np.save(dir_output + '/protein_embedding/' + protein_id + '.npy', list(protein_embedding))
    #     end = timeit.default_timer()
    #     time = end - start
    #     print(round(time, 2))

    ## ESM_embedding extraction
    for no, data in enumerate(protein_id_unique):
        # proteins_g = list()
        print('/'.join(map(str, [no + 1, N])))
        # seq = protein_seq_unique[no]
        # protein_id = protein_to_id[seq]
        protein_graph, _ = load_graphs(
            'data/' + dataset + '/processed' + '/pocket_graph_ESM_PDB/' + str(data) + '.bin')
        feats = protein_graph[0].ndata['feats'][:, 41:]
        np.save(dir_output + '/ESM_embedding_pocket_PDB/' + str(data) + '.npy', feats)
        end = timeit.default_timer()
        time = end - start
        print(round(time, 2))

    # Protein_embedding process
    # proteins_embedding_train, proteins_embedding_test = [], []
    # N = len(protein_id_train)
    # for no, id in enumerate(protein_id_train):
    #     print('/'.join(map(str, [no + 1, N])))
    #     protein_embedding_train = np.load('data/' + dataset + '/processed' + '/ESM_embedding_max/' + str(id) + '.npy', allow_pickle=True)
    #     proteins_embedding_train.append(protein_embedding_train)
    # print(len(proteins_embedding_train))
    # np.save(dir_output + '/train/fold/' + str(fold) + '/protein_embedding_max.npy', proteins_embedding_train)
    #
    # N = len(protein_id_test)
    # for no, id in enumerate(protein_id_test):
    #     print('/'.join(map(str, [no + 1, N])))
    #     protein_embedding_test = np.load('data/' + dataset + '/processed' + '/ESM_embedding_max/' + str(id) + '.npy', allow_pickle=True)
    #     proteins_embedding_test.append(protein_embedding_test)
    # print(len(proteins_embedding_test))
    # np.save(dir_output + '/test/fold/' + str(fold) + '/protein_embedding_max.npy', proteins_embedding_test)
    #
    # ## Protein_graph process
    proteins_graph_train, proteins_graph_test = [], []
    # N = len(protein_id_train)
    # for no, id in enumerate(protein_id_train):
    #     print('/'.join(map(str, [no + 1, N])))
    #     protein_graph_train, _ = load_graphs('data/' + dataset + '/processed' + '/pocket_graph/' + str(id) + '.bin')
    #     proteins_graph_train.append(protein_graph_train[0])
    # print(len(proteins_graph_train))
    # dgl.save_graphs(dir_output + '/train/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_train)
    #
    # N = len(protein_id_test)
    # for no, id in enumerate(protein_id_test):
    #     print('/'.join(map(str, [no + 1, N])))
    #     protein_graph_test, _ = load_graphs('data/' + dataset + '/processed' + '/pocket_graph/' + str(id) + '.bin')
    #     proteins_graph_test.append(protein_graph_test[0])
    # print(len(proteins_graph_test))
    # dgl.save_graphs(dir_output + '/test/fold/' + str(fold) + '/protein_graph.bin', proteins_graph_test)
    #
    ## Protein_id process
    # N = len(protein_id_train)
    # for no, id in enumerate(protein_id_train):
    #     print('/'.join(map(str, [no + 1, N])))
    #     proteins_id_train.append(id)
    # np.save(dir_output + '/train/fold/' + str(fold) + '/protein_id.npy', proteins_id_train)
    #
    # N = len(protein_id_test)
    # for no, id in enumerate(protein_id_test):
    #     print('/'.join(map(str, [no + 1, N])))
    #     proteins_id_test.append(id)
    # np.save(dir_output + '/test/fold/' + str(fold) + '/protein_id.npy', proteins_id_test)

    print('The preprocess of ' + dataset + ' dataset has finished!')
