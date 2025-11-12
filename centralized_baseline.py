# centralized_baseline.py
import pandas as pd, numpy as np
from federated_learning_model import DiabetesModelTrainer
from run_phase1 import prepare_dataloaders

dfs = [pd.read_parquet(f"data/processed/institution_{i}_ml_data.parquet") for i in [1,2,3]]
df = pd.concat(dfs, ignore_index=True)
X = np.vstack(df['features'].values).astype('float32')
y = df['hypo_risk'].values
train_loader, val_loader = prepare_dataloaders(X, y, batch_size=64, train_split=0.8)
t = DiabetesModelTrainer(input_dim=X.shape[1], hidden_dims=[128,64,32])
t.train(train_loader, val_loader, epochs=20)
print("Centralized eval on val set:", t.evaluate(val_loader))
