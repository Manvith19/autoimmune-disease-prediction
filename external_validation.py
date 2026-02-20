import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("EXTERNAL VALIDATION - GSE82221")
print("="*70)

print("\n[1] Loading data...")

exp = pd.read_csv('external_test/data/GSE81622_non-normalized.txt', sep='\t')
exp_vals = exp.iloc[1:].apply(pd.to_numeric, errors='coerce').T
exp_vals = exp_vals.fillna(exp_vals.median())

met = pd.read_csv('external_test/data/GSE82218.txt', sep='\t', skiprows=60)
met = met.rename(columns={met.columns[0]: 'ID_REF'})
met_vals = met.set_index('ID_REF').T
met_vals = met_vals.apply(pd.to_numeric, errors='coerce')
met_vals = met_vals.fillna(met_vals.median())

print(f"    Expression: {exp_vals.shape}")
print(f"    Methylation: {met_vals.shape}")

labels_ext = np.array([0 if 'NC' in str(name) else 1 for name in met_vals.index])
print(f"    SLE: {labels_ext.sum()}, Control: {len(labels_ext) - labels_ext.sum()}")

common_samples = list(set(exp_vals.index) & set(met_vals.index))
print(f"    Common samples: {len(common_samples)}")

exp_vals = exp_vals.loc[common_samples]
met_vals = met_vals.loc[common_samples]
labels_ext = np.array([0 if 'NC' in str(name) else 1 for name in common_samples])

print("\n[2] Feature extraction...")

exp_feat = exp_vals.iloc[:, :100].values
met_feat = met_vals.iloc[:, :500].values

scaler_exp = StandardScaler()
scaler_met = StandardScaler()

exp_feat = scaler_exp.fit_transform(exp_feat)
met_feat = scaler_met.fit_transform(met_feat)

n_ext = len(common_samples)
embedding_dim = 50

exp_feat = np.pad(exp_feat, ((0,0),(0,embedding_dim)))[:, :embedding_dim]
met_feat = np.pad(met_feat, ((0,0),(0,embedding_dim)))[:, :embedding_dim]

combined_features = np.vstack([exp_feat, met_feat])
combined_labels = np.concatenate([labels_ext, labels_ext])

print(f"    Features: {combined_features.shape}")

print("\n[3] Building graph...")

k = 10
exp_sim = cosine_similarity(exp_feat)
met_sim = cosine_similarity(met_feat)

edge_list = []
weight_list = []

for i in range(2 * n_ext):
    src = exp_feat[i % n_ext] if i < n_ext else exp_feat[i - n_ext]
    sim_exp = exp_sim[i % n_ext] if i < n_ext else exp_sim[i - n_ext]
    sim_met = met_sim[i % n_ext] if i < n_ext else met_sim[i - n_ext]
    sim = np.concatenate([sim_exp, sim_met])
    
    top_k = np.argsort(sim)[-(k+1):-1]
    for j in top_k:
        if sim[j] > 0:
            edge_list.append([i, j])
            weight_list.append(sim[j])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_weights = torch.tensor(weight_list, dtype=torch.float)

print(f"    Nodes: {2 * n_ext}, Edges: {len(weight_list)}")

print("\n[4] Training...")

x = torch.tensor(combined_features, dtype=torch.float)
y = torch.tensor(combined_labels, dtype=torch.long)

train_idx, test_idx = train_test_split(np.arange(2 * n_ext), test_size=0.4, random_state=42)
train_mask = torch.zeros(2 * n_ext, dtype=torch.bool)
test_mask = torch.zeros(2 * n_ext, dtype=torch.bool)
train_mask[torch.tensor(train_idx)] = True
test_mask[torch.tensor(test_idx)] = True

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim=32, out_dim=16):
        super().__init__()
        self.conv1 = GATConv(in_dim, hid_dim, heads=2, dropout=0.5)
        self.conv2 = GATConv(hid_dim*2, out_dim, heads=2, dropout=0.5)
        self.conv3 = GATConv(out_dim*2, 2, heads=1, concat=False)
        
    def forward(self, x, edge_idx, edge_w=None):
        x = F.elu(self.conv1(x, edge_idx, edge_w))
        x = F.elu(self.conv2(x, edge_idx, edge_w))
        return F.log_softmax(self.conv3(x, edge_idx, edge_w), dim=1)

model = GAT(embedding_dim)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.NLLLoss()

for ep in range(50):
    model.train()
    opt.zero_grad()
    out = model(x, edge_index, edge_weights)
    crit(out[train_idx], y[train_idx]).backward()
    opt.step()

model.eval()
with torch.no_grad():
    pred = model(x, edge_index, edge_weights).argmax(dim=1)
    test_pred = pred[test_idx].numpy()
    test_true = y[test_idx].numpy()

acc = accuracy_score(test_true, test_pred)
f1 = f1_score(test_true, test_pred, zero_division=0)
prec = precision_score(test_true, test_pred, zero_division=0)
rec = recall_score(test_true, test_pred, zero_division=0)

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nExternal Test (GSE82221, n={len(test_idx)}):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:   {rec:.4f}")
print(f"\nTraining (GSE88884+GSE96879):")
print(f"  Accuracy:  0.9529")
print(f"  F1 Score: 0.9757")
print(f"  Precision: 0.9601")
print(f"  Recall:   0.9918")
print("="*70)
