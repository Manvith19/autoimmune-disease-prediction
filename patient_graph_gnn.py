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

print("Loading INTEND embeddings...")

exp_df = pd.read_csv(f"{DATA_DIR}/expression_projected.txt", sep='\t', index_col=0)
met_df = pd.read_csv(f"{DATA_DIR}/methylation_projected.txt", sep='\t', index_col=0)

print(f"Expression embeddings: {exp_df.shape}")
print(f"Methylation embeddings: {met_df.shape}")

exp_embeddings = exp_df.values.T
met_embeddings = met_df.values.T

exp_samples = exp_df.columns.tolist()
met_samples = met_df.columns.tolist()

print(f"Expression samples: {len(exp_samples)}, Methylation samples: {len(met_samples)}")

sledai_scores = [2, 4, 2, 16, 0, 16, 0, 4, 6, 12, 6, 6, 0, 0, 6, 10, 0, 0, 0, 4, 10, 4, 0, 0, 6, 0, 2, 2, 2, 14, 8, 8, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 2, 8, 10, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 4, 0, 0, 0, 16, 0, 12, 4, 8, 10, 3, 8, 4, 12, 18, 26, 6, 16, 28, 22, 30, 12, 2, 14, 0, 14, 19, 14, 20, 0, 18, 0, 5, 5]
met_labels = np.array([0 if score == 0 else 1 for score in sledai_scores])
print(f"Methylation labels: {met_labels.sum()} SLE, {len(met_labels) - met_labels.sum()} Control")

num_exp_samples = len(exp_samples)
num_controls = 56
exp_labels = np.array([0] * num_controls + [1] * (num_exp_samples - num_controls))
print(f"Expression labels: {exp_labels.sum()} SLE, {num_exp_samples - exp_labels.sum()} Control")

embedding_dim = min(exp_embeddings.shape[1], met_embeddings.shape[1])
exp_features = exp_embeddings[:, :embedding_dim]
met_features = met_embeddings[:, :embedding_dim]

print(f"Using {embedding_dim}-dimensional embeddings")

scaler = StandardScaler()
exp_features = scaler.fit_transform(exp_features)
met_features = scaler.fit_transform(met_features)

def create_knn_graph(features, k=10):
    n_samples = features.shape[0]
    similarity_matrix = cosine_similarity(features)
    
    edge_index = []
    edge_weights = []
    
    for i in range(n_samples):
        similarities = similarity_matrix[i]
        top_k_idx = np.argsort(similarities)[-(k+1):-1]
        for j in top_k_idx:
            if similarities[j] > 0:
                edge_index.append([i, j])
                edge_weights.append(similarities[j])
                edge_index.append([j, i])
                edge_weights.append(similarities[j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_weights

print("\nCreating patient similarity graphs...")

k = 10

exp_edge_index, exp_edge_weights = create_knn_graph(exp_features, k=k)
met_edge_index, met_edge_weights = create_knn_graph(met_features, k=k)

print(f"Expression graph: {exp_features.shape[0]} nodes, {exp_edge_index.shape[1]} edges")
print(f"Methylation graph: {met_features.shape[0]} nodes, {met_edge_index.shape[1]} edges")

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

print("\n" + "="*60)
print("Training GAT Classifier on Methylation Data (SLE vs Control)")
print("="*60)

met_x = torch.tensor(met_features, dtype=torch.float)
met_y = torch.tensor(met_labels, dtype=torch.long)

n_samples = met_features.shape[0]
train_idx, val_idx = train_test_split(np.arange(n_samples), test_size=0.2, random_state=42, stratify=met_labels)

train_mask = torch.zeros(n_samples, dtype=torch.bool)
val_mask = torch.zeros(n_samples, dtype=torch.bool)
train_mask[torch.tensor(train_idx)] = True
val_mask[torch.tensor(val_idx)] = True

data_met = Data(x=met_x, edge_index=met_edge_index, edge_weight=met_edge_weights, y=met_y)
data_met.train_mask = train_mask
data_met.val_mask = val_mask

gat_model = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
gat_model, gat_val_acc = train_gnn(gat_model, data_met, met_y, train_idx, val_idx)

gcn_model = GCNClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, dropout=0.5)
gcn_model, gcn_val_acc = train_gnn(gcn_model, data_met, met_y, train_idx, val_idx)

print(f"\nGAT Validation Accuracy: {gat_val_acc:.4f}")
print(f"GCN Validation Accuracy: {gcn_val_acc:.4f}")

gat_model.eval()
gcn_model.eval()

with torch.no_grad():
    gat_pred = gat_model(data_met.x, data_met.edge_index, data_met.edge_weight).argmax(dim=1)
    gcn_pred = gcn_model(data_met.x, data_met.edge_index, data_met.edge_weight).argmax(dim=1)
    
print("\n" + "="*60)
print("GAT Predictions Distribution:")
print(f"  Predicted SLE: {(gat_pred == 1).sum().item()}")
print(f"  Predicted Control: {(gat_pred == 0).sum().item()}")
print(f"  Actual SLE: {(met_labels == 1).sum()}")
print(f"  Actual Control: {(met_labels == 0).sum()}")

print("\n" + "="*60)
print("Semi-supervised Training (Using All Labels)")
print("="*60)

full_train_mask = torch.ones(n_samples, dtype=torch.bool)
data_met_full = Data(x=met_x, edge_index=met_edge_index, edge_weight=met_edge_weights, y=met_y)
data_met_full.train_mask = full_train_mask

gat_model_full = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
optimizer = torch.optim.Adam(gat_model_full.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

for epoch in range(300):
    gat_model_full.train()
    optimizer.zero_grad()
    out = gat_model_full(data_met_full.x, data_met_full.edge_index, data_met_full.edge_weight)
    loss = criterion(out, met_y)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        gat_model_full.eval()
        with torch.no_grad():
            pred = gat_model_full(data_met_full.x, data_met_full.edge_index, data_met_full.edge_weight).argmax(dim=1)
            acc = (pred == met_y).float().mean().item()
            print(f"Epoch {epoch}: Train Accuracy = {acc:.4f}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Methylation Data:")
print(f"  Total patients: {met_labels.shape[0]}")
print(f"  SLE patients: {(met_labels == 1).sum()}")
print(f"  Control patients: {(met_labels == 0).sum()}")
print(f"  k-NN edges: {met_edge_index.shape[1] // 2}")
print(f"  Best GAT Val Accuracy: {gat_val_acc:.4f}")
print(f"  Best GCN Val Accuracy: {gcn_val_acc:.4f}")

print("\n" + "="*60)
print("Training GAT Classifier on Expression Data (SLE vs Control)")
print("="*60)

exp_x = torch.tensor(exp_features, dtype=torch.float)
exp_y = torch.tensor(exp_labels, dtype=torch.long)

n_exp_samples = exp_features.shape[0]
exp_train_idx, exp_val_idx = train_test_split(np.arange(n_exp_samples), test_size=0.2, random_state=42, stratify=exp_labels)

exp_train_mask = torch.zeros(n_exp_samples, dtype=torch.bool)
exp_val_mask = torch.zeros(n_exp_samples, dtype=torch.bool)
exp_train_mask[torch.tensor(exp_train_idx)] = True
exp_val_mask[torch.tensor(exp_val_idx)] = True

data_exp = Data(x=exp_x, edge_index=exp_edge_index, edge_weight=exp_edge_weights, y=exp_y)
data_exp.train_mask = exp_train_mask
data_exp.val_mask = exp_val_mask

gat_model_exp = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
gat_model_exp, gat_exp_val_acc = train_gnn(gat_model_exp, data_exp, exp_y, exp_train_idx, exp_val_idx)

gcn_model_exp = GCNClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, dropout=0.5)
gcn_model_exp, gcn_exp_val_acc = train_gnn(gcn_model_exp, data_exp, exp_y, exp_train_idx, exp_val_idx)

print(f"\nExpression Data - GAT Validation Accuracy: {gat_exp_val_acc:.4f}")
print(f"Expression Data - GCN Validation Accuracy: {gcn_exp_val_acc:.4f}")

print("\n" + "="*60)
print("Final Summary")
print("="*60)
print(f"Methylation: {met_labels.shape[0]} patients, {met_labels.sum()} SLE, {len(met_labels)-met_labels.sum()} Control")
print(f"  GAT: {gat_val_acc:.4f}, GCN: {gcn_val_acc:.4f}")
print(f"Expression: {exp_labels.shape[0]} patients, {exp_labels.sum()} SLE, {len(exp_labels)-exp_labels.sum()} Control")
print(f"  GAT: {gat_exp_val_acc:.4f}, GCN: {gcn_exp_val_acc:.4f}")

print("\nGraph construction complete!")
print("Nodes = Patients, Edges = kNN similarity based on INTEND embeddings")
