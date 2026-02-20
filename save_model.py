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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "intend_results/GSE88884_GSE96879"
OUTPUT_DIR = "patient_graph_model"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }

print("="*80)
print("INTEND + GNN PATIENT CLASSIFICATION MODEL")
print("SLE vs Control Classification using Combined Methylation & Expression Data")
print("="*80)

print("\n[1] Loading data...")

exp_df = pd.read_csv(f"{DATA_DIR}/expression_projected.txt", sep='\t', index_col=0)
met_df = pd.read_csv(f"{DATA_DIR}/methylation_projected.txt", sep='\t', index_col=0)
cross_sim_df = pd.read_csv(f"{DATA_DIR}/cross_manifold_similarity.txt", sep='\t', index_col=0)

exp_embeddings = exp_df.values.T
met_embeddings = met_df.values.T

exp_samples = exp_df.columns.tolist()
met_samples = met_df.columns.tolist()

sledai_scores = [2, 4, 2, 16, 0, 16, 0, 4, 6, 12, 6, 6, 0, 0, 6, 10, 0, 0, 0, 4, 10, 4, 0, 0, 6, 0, 2, 2, 2, 14, 8, 8, 0, 0, 8, 6, 8, 0, 0, 0, 0, 0, 2, 8, 10, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 4, 0, 0, 0, 16, 0, 12, 4, 8, 10, 3, 8, 4, 12, 18, 26, 6, 16, 28, 22, 30, 12, 2, 14, 0, 14, 19, 14, 20, 0, 18, 0, 5, 5]
met_labels = np.array([0 if score == 0 else 1 for score in sledai_scores])

num_exp = len(exp_samples)
num_controls = 56
exp_labels = np.array([0] * num_controls + [1] * (num_exp - num_controls))

embedding_dim = min(exp_embeddings.shape[1], met_embeddings.shape[1])

print(f"    Expression samples: {len(exp_samples)}")
print(f"    Methylation samples: {len(met_samples)}")
print(f"    Total patients: {len(exp_samples) + len(met_samples)}")
print(f"    INTEND embedding dimension: {embedding_dim}")

print("\n[2] Preprocessing...")

scaler_exp = StandardScaler()
scaler_met = StandardScaler()

exp_features = scaler_exp.fit_transform(exp_embeddings)
met_features = scaler_met.fit_transform(met_embeddings)

n_exp = exp_features.shape[0]
n_met = met_features.shape[0]
n_total = n_exp + n_met

combined_features = np.vstack([exp_features, met_features])
exp_sim = cosine_similarity(exp_features)
met_sim = cosine_similarity(met_features)
cross_sim = cross_sim_df.values

combined_labels = np.concatenate([exp_labels, met_labels])

print(f"    SLE patients: {combined_labels.sum()}")
print(f"    Control patients: {len(combined_labels) - combined_labels.sum()}")

print("\n[3] Building combined similarity graph...")

BEST_K = 15
BEST_CROSS_WEIGHT = 0.5
BEST_CROSS_THRESHOLD = 0.35

k = BEST_K
cross_weight = BEST_CROSS_WEIGHT
cross_threshold = BEST_CROSS_THRESHOLD

full_similarity = np.zeros((n_total, n_total))
full_similarity[:n_exp, :n_exp] = exp_sim
full_similarity[n_exp:, n_exp:] = met_sim

if cross_threshold > 0:
    cross_mask = cross_sim > cross_threshold
    cross_filtered = cross_sim * cross_mask
    full_similarity[:n_exp, n_exp:] = cross_filtered * cross_weight
    full_similarity[n_exp:, :n_exp] = (cross_filtered.T * cross_weight)
else:
    full_similarity[:n_exp, n_exp:] = cross_sim * cross_weight
    full_similarity[n_exp:, :n_exp] = (cross_sim.T * cross_weight)

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

print(f"    k-NN: {k}")
print(f"    Cross-manifold weight: {cross_weight}")
print(f"    Cross-manifold threshold: {cross_threshold}")
print(f"    Edges: {edge_index.shape[1] // 2}")

print("\n[4] Creating train/val/test split (60/20/20)...")

train_idx, temp_idx = train_test_split(np.arange(n_total), test_size=0.4, random_state=42, stratify=combined_labels)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=combined_labels[temp_idx])

train_mask = torch.zeros(n_total, dtype=torch.bool)
val_mask = torch.zeros(n_total, dtype=torch.bool)
test_mask = torch.zeros(n_total, dtype=torch.bool)

train_mask[torch.tensor(train_idx)] = True
val_mask[torch.tensor(val_idx)] = True
test_mask[torch.tensor(test_idx)] = True

print(f"    Training samples: {len(train_idx)}")
print(f"    Validation samples: {len(val_idx)}")
print(f"    Test samples: {len(test_idx)}")

x = torch.tensor(combined_features, dtype=torch.float)
y = torch.tensor(combined_labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

print("\n[5] Defining GAT model...")

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

print("\n[6] Training model...")

model = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

best_val_acc = 0
best_model_state = None
epochs = 150

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
        train_acc = (pred[train_idx] == y[train_idx]).float().mean().item()
        val_acc = (pred[val_idx] == y[val_idx]).float().mean().item()
        
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()
    
    if epoch % 30 == 0:
        print(f"    Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

model.load_state_dict(best_model_state)

print("\n[7] Evaluating on test set...")

model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
    
    train_pred = pred[train_idx].numpy()
    train_true = y[train_idx].numpy()
    train_metrics = compute_metrics(train_true, train_pred)
    
    val_pred = pred[val_idx].numpy()
    val_true = y[val_idx].numpy()
    val_metrics = compute_metrics(val_true, val_pred)
    
    test_pred = pred[test_idx].numpy()
    test_true = y[test_idx].numpy()
    test_metrics = compute_metrics(test_true, test_pred)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nTraining Set ({len(train_idx)} samples):")
print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"  F1 Score: {train_metrics['f1']:.4f}")
print(f"  Precision: {train_metrics['precision']:.4f}")
print(f"  Recall:   {train_metrics['recall']:.4f}")

print(f"\nValidation Set ({len(val_idx)} samples):")
print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
print(f"  F1 Score: {val_metrics['f1']:.4f}")
print(f"  Precision: {val_metrics['precision']:.4f}")
print(f"  Recall:   {val_metrics['recall']:.4f}")

print(f"\n*** TEST SET - COMPLETELY UNSEEN ({len(test_idx)} samples) ***")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  F1 Score: {test_metrics['f1']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:   {test_metrics['recall']:.4f}")

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(test_true, test_pred)
print(f"                Predicted")
print(f"              Control  SLE")
print(f"Actual Control   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"Actual SLE       {cm[1,0]:4d}  {cm[1,1]:4d}")

print("\nClassification Report:")
print(classification_report(test_true, test_pred, target_names=['Control', 'SLE']))

print("\n[8] Saving model...")

model_path = f"{OUTPUT_DIR}/gat_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_dim': embedding_dim,
    'hidden_channels': 64,
    'out_channels': 32,
    'heads': 4,
    'dropout': 0.5,
    'k': k,
    'cross_weight': cross_weight,
    'cross_threshold': cross_threshold,
    'n_exp': n_exp,
    'n_met': n_met,
    'n_total': n_total,
    'test_metrics': test_metrics,
}, model_path)

scaler_exp_path = f"{OUTPUT_DIR}/scaler_exp.npy"
scaler_met_path = f"{OUTPUT_DIR}/scaler_met.npy"
np.save(scaler_exp_path, scaler_exp)
np.save(scaler_met_path, scaler_met)

np.save(f"{OUTPUT_DIR}/exp_sim.npy", exp_sim)
np.save(f"{OUTPUT_DIR}/met_sim.npy", met_sim)
np.save(f"{OUTPUT_DIR}/cross_sim.npy", cross_sim)
np.save(f"{OUTPUT_DIR}/combined_labels.npy", combined_labels)
np.save(f"{OUTPUT_DIR}/train_idx.npy", train_idx)
np.save(f"{OUTPUT_DIR}/val_idx.npy", val_idx)
np.save(f"{OUTPUT_DIR}/test_idx.npy", test_idx)
torch.save({'edge_index': edge_index, 'edge_weights': edge_weights}, f"{OUTPUT_DIR}/graph_edges.pt")

print(f"    Model saved to: {model_path}")
print(f"    Scalers saved to: {OUTPUT_DIR}/")
print(f"    Graph structure saved to: {OUTPUT_DIR}/graph_edges.pt")

print("\n" + "="*80)
print("MODEL SUMMARY")
print("="*80)
print(f"Architecture: GAT (Graph Attention Network)")
print(f"  Input: {embedding_dim}-dim INTEND embeddings")
print(f"  Hidden: 64 channels, 4 attention heads")
print(f"  Output: 32 channels -> 2 classes (SLE/Control)")
print(f"\nGraph Construction:")
print(f"  k-NN: {k}")
print(f"  Cross-manifold weight: {cross_weight}")
print(f"  Cross-manifold threshold: {cross_threshold}")
print(f"  Total edges: {edge_index.shape[1] // 2}")
print(f"\nData Split:")
print(f"  Train: {len(train_idx)} ({100*len(train_idx)/n_total:.0f}%)")
print(f"  Val: {len(val_idx)} ({100*len(val_idx)/n_total:.0f}%)")
print(f"  Test: {len(test_idx)} ({100*len(test_idx)/n_total:.0f}%)")
print(f"\nTest Set Performance:")
print(f"  Accuracy:  {test_metrics['accuracy']*100:.2f}%")
print(f"  F1 Score:  {test_metrics['f1']*100:.2f}%")
print(f"  Precision: {test_metrics['precision']*100:.2f}%")
print(f"  Recall:    {test_metrics['recall']*100:.2f}%")
print("="*80)
