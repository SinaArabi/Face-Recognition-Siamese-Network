import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(SiameseNetwork, self).__init__()

        # ✅ Load Pretrained ResNet18 (Feature Extractor)
        self.backbone = models.resnet18(pretrained=pretrained)

        # ✅ Remove classification head (FC layer)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Replace FC with Identity layer

        # ✅ Embedding Head (Projects to embedding_dim)
        self.embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x):
        """Passes one image through the network to get its embedding."""
        x = self.backbone(x)  # Extract features
        x = self.embedding_head(x)  # Convert to embedding space
        return F.normalize(x, p=2, dim=1)  # ✅ Normalize embeddings (L2 norm)

    def forward_two_input(self, x1, x2):
        """Processes two images and returns their distance (similarity)."""
        emb1 = self.forward(x1)  # First image embedding
        emb2 = self.forward(x2)  # Second image embedding

        # Compute similarity (Euclidean Distance or Cosine Similarity)
        similarity = F.cosine_similarity(emb1, emb2, dim=1)  # Ranges from -1 to 1

        return similarity    # Smaller distance = More similar

