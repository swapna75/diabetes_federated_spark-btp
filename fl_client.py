"""
Flower Federated Learning Client
Represents one institution in the federated setup
"""

import flwr as fl
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from federated_learning_model import DiabetesModelTrainer, DiabetesDataset


class DiabetesFlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning"""
    
    def __init__(
        self, 
        institution_name: str,
        trainer: DiabetesModelTrainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        local_epochs: int = 5
    ):
        self.institution_name = institution_name
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        
        print(f"[{institution_name}] Flower client initialized")
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return current local model parameters"""
        print(f"[{self.institution_name}] Sending parameters to server")
        return self.trainer.get_parameters()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model on local data"""
        print(f"\n[{self.institution_name}] Starting local training (Round {config.get('round', '?')})")
        
        # ⭐⭐⭐ VALIDATE INCOMING PARAMETERS ⭐⭐⭐
        has_nan = False
        for i, param in enumerate(parameters):
            if np.isnan(param).any() or np.isinf(param).any():
                print(f"WARNING: NaN/Inf detected in parameter {i} from server!")
                has_nan = True
        
        if has_nan:
            print("CRITICAL: Received corrupted parameters from server. Using current parameters instead.")
            # Don't set the corrupted parameters, keep current ones
        else:
            # Set global parameters only if valid
            self.trainer.set_parameters(parameters)
        
        # Train locally
        history = self.trainer.train(
            self.train_loader, 
            self.val_loader,
            epochs=self.local_epochs,
            verbose=False
        )
        
        # Get updated parameters
        updated_parameters = self.trainer.get_parameters()
        
        # ⭐⭐⭐ VALIDATE OUTGOING PARAMETERS ⭐⭐⭐
        for i, param in enumerate(updated_parameters):
            if np.isnan(param).any() or np.isinf(param).any():
                print(f"ERROR: NaN/Inf in parameter {i} after training!")
                # Return original parameters instead
                return parameters, len(self.train_loader.dataset), {
                    "train_loss": 999.0,
                    "train_accuracy": 0.0,
                    "val_loss": 999.0,
                    "val_accuracy": 0.0,
                    "val_f1": 0.0,
                    "val_auc": 0.0
                }
        
        # Calculate number of training examples
        num_examples = len(self.train_loader.dataset)
        
        # Prepare metrics
        metrics = {
            "train_loss": float(history['train_loss'][-1]) if history['train_loss'] else 999.0,
            "train_accuracy": float(history['train_acc'][-1]) if history['train_acc'] else 0.0,
        }
        
        if history['val_metrics']:
            val_metrics = history['val_metrics'][-1]
            metrics.update({
                "val_loss": float(val_metrics['loss']) if not np.isnan(val_metrics['loss']) else 999.0,
                "val_accuracy": float(val_metrics['accuracy']),
                "val_f1": float(val_metrics['f1']),
                "val_auc": float(val_metrics['auc'])
            })
        
        print(f"[{self.institution_name}] Training complete")
        print(f"  Train Loss: {metrics.get('train_loss', 999):.4f}, Train Acc: {metrics.get('train_accuracy', 0):.4f}")
        if 'val_accuracy' in metrics:
            print(f"  Val Acc: {metrics['val_accuracy']:.4f}, Val F1: {metrics['val_f1']:.4f}")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local data
        
        Args:
            parameters: Global model parameters
            config: Configuration dictionary
        
        Returns:
            Loss, number of examples, metrics
        """
        print(f"[{self.institution_name}] Evaluating global model...")
        
        # Set parameters
        self.trainer.set_parameters(parameters)
        
        # Evaluate
        metrics = self.trainer.evaluate(self.val_loader)
        
        # Number of validation examples
        num_examples = len(self.val_loader.dataset)
        
        print(f"[{self.institution_name}] Evaluation complete")
        print(f"  Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Return loss and metrics
        return float(metrics['loss']), num_examples, metrics


def load_institution_data(institution_name: str, data_path: str = "data/processed"):
    """
    Load preprocessed data for an institution with DATA VALIDATION
    """
    print(f"Loading data for {institution_name}...")
    
    file_path = f"{data_path}/{institution_name}_ml_data.parquet"
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} records from {file_path}")
        
        # ⭐⭐⭐ VALIDATE DATA ⭐⭐⭐
        features = np.array([np.array(f) for f in df['features']])
        labels = df['hypo_risk'].values
        
        # Check for NaN/Inf in features
        if np.isnan(features).any() or np.isinf(features).any():
            print(f"WARNING: NaN/Inf detected in features for {institution_name}!")
            print(f"  NaN count: {np.isnan(features).sum()}")
            print(f"  Inf count: {np.isinf(features).sum()}")
            
            # Replace NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"  Replaced NaN/Inf with 0")
        
        # Check for NaN/Inf in labels
        if np.isnan(labels).any() or np.isinf(labels).any():
            print(f"WARNING: NaN/Inf in labels for {institution_name}!")
            labels = np.nan_to_num(labels, nan=0)
        
        # Clip extreme values
        features = np.clip(features, -1e6, 1e6)
        
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        print(f"Feature range: [{features.min():.2f}, {features.max():.2f}]")
        
        return features, labels  # ⭐ THIS RETURN IS INSIDE THE TRY BLOCK
        
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using IMPROVED synthetic data.")
        
        # ⭐ IMPROVED SYNTHETIC DATA GENERATION ⭐
        n_samples = 2000
        input_dim = 10
        
        # Create more realistic features
        np.random.seed(hash(institution_name) % 2**32)
        
        # Generate realistic glucose patterns
        n_hypo = int(n_samples * 0.35)
        n_normal = n_samples - n_hypo
        
        # Hypoglycemia cases: glucose < 70
        glucose_hypo = np.random.uniform(40, 69, n_hypo)
        glucose_ma_hypo = np.random.uniform(45, 75, n_hypo)
        glucose_std_hypo = np.random.uniform(5, 20, n_hypo)
        glucose_roc_hypo = np.random.uniform(-15, -2, n_hypo)
        steps_hypo = np.random.randint(0, 50, n_hypo)
        hr_hypo = np.random.randint(70, 110, n_hypo)
        hour_hypo = np.random.choice([0, 1, 2, 3, 4, 5, 22, 23], n_hypo)
        
        # Normal cases: glucose >= 70
        glucose_normal = np.random.normal(130, 35, n_normal).clip(70, 350)
        glucose_ma_normal = np.random.normal(135, 30, n_normal).clip(75, 300)
        glucose_std_normal = np.random.uniform(2, 15, n_normal)
        glucose_roc_normal = np.random.normal(0, 5, n_normal)
        steps_normal = np.random.randint(20, 150, n_normal)
        hr_normal = np.random.randint(60, 100, n_normal)
        hour_normal = np.random.randint(6, 22, n_normal)
        
        # Combine
        features = np.column_stack([
            np.concatenate([glucose_hypo, glucose_normal]),
            np.concatenate([glucose_ma_hypo, glucose_ma_normal]),
            np.concatenate([glucose_std_hypo, glucose_std_normal]),
            np.concatenate([glucose_roc_hypo, glucose_roc_normal]),
            np.concatenate([steps_hypo, steps_normal]),
            np.concatenate([hr_hypo, hr_normal]),
            np.concatenate([hour_hypo, hour_normal]),
            np.random.randint(0, 7, n_samples),
            np.concatenate([np.ones(n_hypo), np.zeros(n_normal)]),
            np.random.randint(0, 2, n_samples)
        ]).astype(np.float32)
        
        labels = np.concatenate([np.ones(n_hypo), np.zeros(n_normal)]).astype(np.int64)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        features = features[indices]
        labels = labels[indices]
        
        print(f"Created {n_samples} synthetic samples")
        print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
        print(f"Class distribution: {np.bincount(labels)}")
        
        return features, labels  # ⭐ THIS RETURN IS INSIDE THE EXCEPT BLOCK


def create_client_fn(institution_name: str, input_dim: int, local_epochs: int = 5):
    """
    Factory function to create Flower client
    """
    def client_fn(cid: str) -> fl.client.Client:
        # Load data
        features, labels = load_institution_data(institution_name)
        
        # Create dataloaders
        dataset = DiabetesDataset(features, labels)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Create trainer
        trainer = DiabetesModelTrainer(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            learning_rate=0.0005,
            weight_decay=1e-4
        )
        
        # Create and return client
        return DiabetesFlowerClient(
            institution_name=institution_name,
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            local_epochs=local_epochs
        ).to_client()
    
    return client_fn


if __name__ == "__main__":
    import sys
    
    # Get institution name from command line
    institution_name = sys.argv[1] if len(sys.argv) > 1 else "institution_1"
    
    print(f"Starting Flower client for {institution_name}")
    
    # Load data to determine input dimension
    features, labels = load_institution_data(institution_name)
    input_dim = features.shape[1]
    
    # Create dataloaders with better settings
    dataset = DiabetesDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # ⭐ LARGER BATCH SIZE
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create trainer with improved settings
    trainer = DiabetesModelTrainer(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],  # ⭐ Larger network
        learning_rate=0.0005,  # ⭐ Lower learning rate
        weight_decay=1e-4  # ⭐ Regularization
    )
    
    # Create client
    client = DiabetesFlowerClient(
        institution_name=institution_name,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        local_epochs=5  # ⭐ REDUCE from 10 to 5
    )
    
    # Start client
    print(f"\nConnecting to Flower server...")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client
    )