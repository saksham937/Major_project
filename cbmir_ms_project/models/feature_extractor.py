import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='inception_v3', use_gpu=True):
        """
        Research Paper Architecture:
        InceptionV3 Backbone
        -> Global Average Pooling
        -> Dense(512) -> Dropout(0.2) + GELU
        -> Dense(256) -> Dropout(0.2) + GELU
        -> Dense(32) (Final 32-dimensional feature embedding)
        """
        super(FeatureExtractor, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.model_name = model_name
        
        if model_name != 'inception_v3':
            print("Warning: Paper mandates InceptionV3. Falling back to InceptionV3.")
            
        # 1. Base Model (InceptionV3)
        weights = models.Inception_V3_Weights.DEFAULT
        base_model = models.inception_v3(weights=weights)
        base_model.fc = nn.Identity() # Remove default 1000-class head
        self.features = base_model
        
        # 2. Custom Layers matching the paper
        # InceptionV3 outputs 2048 features after its internal GAP
        self.custom_layers = nn.Sequential(
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 32)  # Final 32-dim feature vector
        )
        
        self.embedding_size = 32
        
        self.to(self.device)
        self.eval() # Set to evaluation mode for offline extraction
        
    def forward(self, x):
        """
        Extracts the 32-dimensional vector.
        """
        x = x.to(self.device)
        with torch.no_grad():
            # Extract 2048-dim features from InceptionV3
            base_features = self.features(x)
            
            # Pass through custom head
            final_embedding = self.custom_layers(base_features)
            
        return final_embedding

if __name__ == "__main__":
    extractor = FeatureExtractor()
    dummy_input = torch.randn(1, 3, 299, 299)
    out = extractor(dummy_input)
    print(f"Extraction Pipeline Output shape: {out.shape}") # Should be [1, 32]
