import logging
import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

from src.lens2.classifier import Lens2Predictor
from src.lens2.dataset import generate_golden_dataset

logger = logging.getLogger(__name__)

def train_lens2_classifier(settings, project_root):
    """
    Trains the Lens 2 Supervised Predictor using the Golden Dataset.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: TRAINING LENS 2 PREDICtor (Classifier)")
    logger.info("=" * 60)
    
    dataset_path = Path(project_root) / settings.lens2.dataset_save_path
    
    # 1. Ensure dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}. Please generate it first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Filter undefined scenarios if necessary, but we might want to learn them
    # For now, map S_UNDEFINED (-1) to class 8
    df["scenario_id"] = df["scenario_id"].replace(-1, 8)
    
    # 2. Split Data Based on Settings
    train_end = pd.to_datetime(settings.data.train_end).date()
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    train_mask = df['date'] <= train_end
    val_mask = df['date'] > train_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    
    # Feature columns (everything except date and scenario_id)
    feature_cols = [c for c in df.columns if c not in ["date", "scenario_id"]]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df["scenario_id"].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df["scenario_id"].values
    
    # Handle NaNs
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    
    # 3. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save Scaler
    model_dir = Path(project_root) / settings.lens2.model_save_path
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "lens2_scaler.pkl")
    
    # 4. PyTorch Preparation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=settings.lens2.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.lens2.batch_size, shuffle=False)
    
    # 5. Model Initialization
    model = Lens2Predictor(
        input_dim=len(feature_cols),
        hidden_dim=settings.lens2.net_arch,
        num_classes=9
    ).to(device)
    
    criterion = nn.CrossEntropyLoss() # Weights can be added if imbalanced
    optimizer = optim.Adam(model.parameters(), lr=settings.lens2.learning_rate)
    
    # 6. Training Loop
    best_val_acc = 0.0
    epochs = settings.lens2.n_epochs
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_dir / "lens2_best.pth")
            
    logger.info(f"Training Complete. Best Val Accuracy: {best_val_acc:.4f}")
    
    # Final Evaluation Report
    logger.info("Generating Classification Report on Validation Set...")
    model.load_state_dict(torch.load(model_dir / "lens2_best.pth"))
    model.eval()
    
    final_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1)
            final_preds.extend(preds.cpu().numpy())
            
    report = classification_report(val_true, final_preds, zero_division=0)
    logger.info(f"\\n{report}")
    
    logger.info(f"Lens 2 Classifier saved to {model_dir}")
