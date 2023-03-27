import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from models import gt_net_compound, gt_net_protein


if torch.cuda.is_available():
    device = torch.device('cuda')


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]
        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x



class MDGTDTInet(nn.Module):
    def __init__(self, compound_dim=128, protein_dim=128, gt_layers=10, gt_heads=8, out_dim=1):
        super(MDGTDTInet, self).__init__()
        self.compound_dim = compound_dim
        self.protein_dim = protein_dim
        self.n_layers = gt_layers
        self.n_heads = gt_heads

        self.crossAttention = SelfAttention(hid_dim=self.compound_dim, n_heads=1, dropout=0.2)

        self.compound_gt = gt_net_compound.GraphTransformer(device, n_layers=gt_layers, node_dim=44, edge_dim=10, hidden_dim=compound_dim,
                                                        out_dim=compound_dim, n_heads=gt_heads, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)
        self.protein_gt = gt_net_protein.GraphTransformer(device, n_layers=gt_layers, node_dim=41, edge_dim=5, hidden_dim=protein_dim,
                                                       out_dim=protein_dim, n_heads=gt_heads, in_feat_dropout=0.0, dropout=0.2, pos_enc_dim=8)


        self.protein_embedding_fc = nn.Linear(320, self.protein_dim)
        self.protein_fc = nn.Linear(self.protein_dim * 2, self.protein_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.joint_attn_prot, self.joint_attn_comp = nn.Linear(compound_dim, compound_dim), nn.Linear(compound_dim, compound_dim)
        self.modal_fc = nn.Linear(protein_dim*2, protein_dim)

        self.fc_out = nn.Linear(compound_dim, out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(compound_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )


    def dgl_split(self, bg, feats):
        max_num_nodes = int(bg.batch_num_nodes().max())
        batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                       dim=1).reshape(-1).type(torch.long).to(bg.device)
        cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
        idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
        idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
        size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
        out = feats.new_full(size, fill_value=0)
        out[idx] = feats
        out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
        return out


    def forward(self, compound_graph, protein_graph,  protein_embedding):
        compound_feat = self.compound_gt(compound_graph)
        compound_feat_x = self.dgl_split(compound_graph, compound_feat)
        compound_feats = compound_feat_x

        protein_feat = self.protein_gt(protein_graph)
        protein_feat_x = self.dgl_split(protein_graph, protein_feat)
        protein_embedding = self.protein_embedding_fc(protein_embedding)
        protein_feats = self.crossAttention(protein_embedding, protein_feat_x, protein_feat_x)

        # compound-protein interaction
        inter_comp_prot = self.sigmoid(torch.einsum('bij,bkj->bik', self.joint_attn_prot(self.relu(protein_feats)), self.joint_attn_comp(self.relu(compound_feats))))
        inter_comp_prot_sum = torch.einsum('bij->b', inter_comp_prot)
        inter_comp_prot = torch.einsum('bij,b->bij', inter_comp_prot, 1/inter_comp_prot_sum)

        # compound-protein joint embedding
        cp_embedding = self.tanh(torch.einsum('bij,bkj->bikj', protein_feats, compound_feats))
        cp_embedding = torch.einsum('bijk,bij->bk', cp_embedding, inter_comp_prot)

        x = self.classifier(cp_embedding)

        return x

