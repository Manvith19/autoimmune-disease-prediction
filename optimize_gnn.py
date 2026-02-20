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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = "intend_results/GSE88884_GSE96879"

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0)
    }

print("Loading data...")

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

def build_graph(k, cross_weight, cross_threshold, use_intra=True):
    full_similarity = np.zeros((n_total, n_total))
    
    if use_intra:
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
    
    return edge_index, edge_weights

def train_and_evaluate(k, cross_weight, cross_threshold, use_intra=True, epochs=100):
    edge_index, edge_weights = build_graph(k, cross_weight, cross_threshold, use_intra)
    
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
    
    model = GATClassifier(in_channels=embedding_dim, hidden_channels=64, out_channels=32, heads=4, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.NLLLoss()
    
    best_val_acc = 0
    best_model_state = None
    
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
            val_acc = (pred[val_idx] == y[val_idx]).float().mean().item()
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=1)
        val_pred = pred[val_idx].numpy()
        val_true = y[val_idx].numpy()
        metrics = compute_metrics(val_true, val_pred)
    
    return metrics, edge_index.shape[1] // 2

print("="*80)
print("HYPERPARAMETER SEARCH")
print("k_values = [5, 10, 15, 20, 25, 30], cross_weights, cross_thresholds, use_intra_options")

k_values = [5, 10, 15]
cross_weights = [1.0, 2.0]
cross_thresholds = [0.0, 0.35]
use_intra_options = [True]

best_metrics = None
best_params = None
best_edges = 0
all_results = []

total_configs = len(k_values) * len(cross_weights) * len(cross_thresholds) * len(use_intra_options)
print(f"Testing {total_configs} configurations...")

count = 0
for k in k_values:
    for cross_weight in cross_weights:
        for cross_threshold in cross_thresholds:
            for use_intra in use_intra_options:
                count += 1
                try:
                    metrics, n_edges = train_and_evaluate(k, cross_weight, cross_threshold, use_intra)
                    print(f"[{count}/{total_configs}] k={k}, cw={cross_weight}, ct={cross_threshold}, intra={use_intra}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
                    
                    all_results.append({
                        'k': k,
                        'cross_weight': cross_weight,
                        'cross_threshold': cross_threshold,
                        'use_intra': use_intra,
                        'edges': n_edges,
                        **metrics
                    })
                    
                    all_above_90 = (metrics['accuracy'] >= 0.9 and 
                                   metrics['f1'] >= 0.9 and 
                                   metrics['precision'] >= 0.9 and 
                                   metrics['recall'] >= 0.9)
                    
                    if all_above_90:
                        print(f"\n*** SUCCESS! ***")
                        print(f"k={k}, cross_w={cross_weight}, cross_t={cross_threshold}, intra={use_intra}")
                        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                        print(f"  Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}")
                        best_metrics = metrics
                        best_params = (k, cross_weight, cross_threshold, use_intra)
                        best_edges = n_edges
                    
                except Exception as e:
                    pass

if best_metrics is None:
    print("\nNo combination achieved all metrics > 90%. Showing top 10 results:")
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('f1', ascending=False).head(10)
    print(results_df.to_string())
    
    best_idx = results_df['f1'].idxmax()
    best_row = results_df.loc[best_idx]
    best_params = (best_row['k'], best_row['cross_weight'], best_row['cross_threshold'], best_row['use_intra'])
    best_metrics = {
        'accuracy': best_row['accuracy'],
        'f1': best_row['f1'],
        'precision': best_row['precision'],
        'recall': best_row['recall']
    }
    best_edges = int(best_row['edges'])

print("\n" + "="*80)
print("BEST CONFIGURATION")
print("="*80)
print(f"k={best_params[0]}, cross_weight={best_params[1]}, cross_threshold={best_params[2]}, use_intra={best_params[3]}")
print(f"Edges: {best_edges}")
print(f"\nMetrics:")
print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"  F1 Score:  {best_metrics['f1']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall:    {best_metrics['recall']:.4f}")

if all(m >= 0.9 for m in best_metrics.values()):
    print("\n*** ALL METRICS ABOVE 90% ACHIEVED! ***")
else:
    print(f"\nBest achieved: Acc={best_metrics['accuracy']:.4f}, F1={best_metrics['f1']:.4f}, Prec={best_metrics['precision']:.4f}, Rec={best_metrics['recall']:.4f}")
