import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "intend_results/GSE88884_GSE96879"

print("="*70)
print("COMBINED MODALITY PATIENT GRAPH USING INTEND")
print("="*70)

print("\n[1] Loading INTEND embeddings...")

exp_df = pd.read_csv(f"{DATA_DIR}/expression_projected.txt", sep='\t', index_col=0)
met_df = pd.read_csv(f"{DATA_DIR}/methylation_projected.txt", sep='\t', index_col=0)
cross_sim_df = pd.read_csv(f"{DATA_DIR}/cross_manifold_similarity.txt", sep='\t', index_col=0)

print(f"Expression embeddings: {exp_df.shape} (PCs x samples)")
print(f"Methylation embeddings: {met_df.shape} (PCs x samples)")
print(f"Cross-manifold similarity: {cross_sim_df.shape} (exp x met)")

exp_embeddings = exp_df.values.T
met_embeddings = met_df.values.T

exp_samples = exp_df.columns.tolist()
met_samples = met_df.columns.tolist()

print(f"\nExpression samples: {len(exp_samples)}")
print(f"Methylation samples: {len(met_samples)}")

sledai_scores = [2, 4, 2, 16, 0, 16, 0, 4, 6, 12, 6, 6, 0, 0, 6, 10, 0, 0, 0, 4, 10, 4, 0, 0, 6, 0, 2, 2, 2, 14, 8, 8, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 2, 8, 10, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 4, 0, 0, 0, 16, 0, 12, 4, 8, 10, 3, 8, 4, 12, 18, 26, 6, 16, 28, 22, 30, 12, 2, 14, 0, 14, 19, 14, 20, 0, 18, 0, 5, 5]
met_labels = np.array([0 if score == 0 else 1 for score in sledai_scores])

num_exp = len(exp_samples)
num_controls = 56
exp_labels = np.array([0] * num_controls + [1] * (num_exp - num_controls))

print(f"\nMethylation: {met_labels.sum()} SLE, {len(met_labels) - met_labels.sum()} Control")
print(f"Expression: {exp_labels.sum()} SLE, {len(exp_labels) - exp_labels.sum()} Control")

embedding_dim = min(exp_embeddings.shape[1], met_embeddings.shape[1])
print(f"\nINTEND latent dimension: {embedding_dim}")

scaler_exp = StandardScaler()
scaler_met = StandardScaler()

exp_features = scaler_exp.fit_transform(exp_embeddings)
met_features = scaler_met.fit_transform(met_embeddings)

print("\n[2] Building COMBINED patient similarity graph...")
print("    Using cross-manifold similarity + intra-manifold similarity")

n_exp = exp_features.shape[0]
n_met = met_features.shape[0]
n_total = n_exp + n_met

print(f"    Total patients: {n_total}")

combined_features = np.vstack([exp_features, met_features])
print(f"    Combined feature matrix: {combined_features.shape}")

exp_sim = cosine_similarity(exp_features)
met_sim = cosine_similarity(met_features)

cross_sim = cross_sim_df.values

print(f"    Expression similarity: {exp_sim.shape}")
print(f"    Methylation similarity: {met_sim.shape}")
print(f"    Cross-manifold similarity: {cross_sim.shape}")

full_similarity = np.zeros((n_total, n_total))
full_similarity[:n_exp, :n_exp] = exp_sim
full_similarity[n_exp:, n_exp:] = met_sim
full_similarity[:n_exp, n_exp:] = cross_sim
full_similarity[n_exp:, :n_exp] = cross_sim.T

k = 10
edge_index = []
edge_weights = []

for i in range(n_total):
    similarities = full_similarity[i]
    top_k_idx = np.argsort(similarities)[-(k+1):-1]
    for j in top_k_idx:
        if similarities[j] > 0 and i != j:
            edge_index.append([i, j])
            edge_weights.append(similarities[j])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

print(f"    Edges: {edge_index.shape[1] // 2} (undirected)")

combined_labels = np.concatenate([exp_labels, met_labels])

print("\n[3] Training GNN on COMBINED graph...")

x = torch.tensor(combined_features, dtype=torch.float)
y = torch.tensor(combined_labels, dtype=torch.long)

n_samples = n_total
train_idx, val_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=42, stratify=combined_labels)

train_mask = torch.zeros(n_samples, dtype=torch.bool)
val_mask = torch.zeros(n_samples, dtype=torch.bool)
train_mask[torch.tensor(train_idx)] = True
val_mask[torch.tensor(val_idx)] = True

data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
data.train_mask = train_mask
data.val_mask = val_mask

class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, heads=4, dropout=0.5):
        super(GATClassifier, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=heads, dropout=dropout)
        self.conv3 = GATConv(out_channels * heads, 2, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.elu(self.conv1(x, edge_index, edge_weight))
        x = F.elu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=32, dropout=0.5):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.conv3 = GCNConv(out_channels, 2)
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_gnn(model, data, labels, train_idx, val_idx, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
            train_acc = (pred[train_idx] == labels[train_idx]).float().mean().item()
            val_acc = (pred[val_idx] == labels[val_idx]).float().mean().item()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc

gat_model = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
gat_model, gat_val_acc = train_gnn(gat_model, data, y, train_idx, val_idx)

gcn_model = GCNClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, dropout=0.5)
gcn_model, gcn_val_acc = train_gnn(gcn_model, data, y, train_idx, val_idx)

print(f"\n    GAT Validation Accuracy: {gat_val_acc:.4f}")
print(f"    GCN Validation Accuracy: {gcn_val_acc:.4f}")

gat_model.eval()
with torch.no_grad():
    gat_pred = gat_model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)

print(f"\n    Predicted: {gat_pred.sum()} SLE, {(1-gat_pred).sum()} Control")
print(f"    Actual: {combined_labels.sum()} SLE, {len(combined_labels)-combined_labels.sum()} Control")

print("\n[4] Analysis by modality...")

exp_pred = gat_pred[:n_exp].numpy()
met_pred = gat_pred[n_exp:].numpy()

print(f"\n    Expression cohort (n={n_exp}):")
print(f"      Predicted: {exp_pred.sum()} SLE, {len(exp_pred)-exp_pred.sum()} Control")
print(f"      Actual: {exp_labels.sum()} SLE, {len(exp_labels)-exp_labels.sum()} Control")
print(f"      Accuracy: {(exp_pred == exp_labels).mean():.4f}")

print(f"\n    Methylation cohort (n={n_met}):")
print(f"      Predicted: {met_pred.sum()} SLE, {len(met_pred)-met_pred.sum()} Control")
print(f"      Actual: {met_labels.sum()} SLE, {len(met_labels)-met_labels.sum()} Control")
print(f"      Accuracy: {(met_pred == met_labels).mean():.4f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Combined Graph:")
print(f"  Total patients: {n_total}")
print(f"  - Expression cohort: {n_exp}")
print(f"  - Methylation cohort: {n_met}")
print(f"  Edges (k={k}NN): {edge_index.shape[1] // 2}")
print(f"  Features: {embedding_dim}-dim INTEND embeddings")
print(f"\nResults:")
print(f"  GAT Accuracy: {gat_val_acc:.4f}")
print(f"  GCN Accuracy: {gcn_val_acc:.4f}")
print("="*70)
