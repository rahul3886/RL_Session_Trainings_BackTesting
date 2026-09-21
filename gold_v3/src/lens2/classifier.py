import torch
import torch.nn as nn
import torch.nn.functional as F

class Lens2Predictor(nn.Module):
    """
    Supervised Classifier predicting NY Scenarios (S1-S8) based on London Geometry.
    """
    def __init__(self, input_dim: int, hidden_dim: list, num_classes: int = 9):
        # 9 classes: S_UNDEFINED (-1) mapped to 8, plus S1-S8 (0-7)
        super(Lens2Predictor, self).__init__()
        
        layers = []
        last_dim = input_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            layers.append(nn.BatchNorm1d(h_dim))
            last_dim = h_dim
            
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(last_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
