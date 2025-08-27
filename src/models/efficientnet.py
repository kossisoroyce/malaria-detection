import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import timm


class MalariaEfficientNet(nn.Module):
    """EfficientNet-based malaria classification model with clinical focus."""
    
    def __init__(self, model_name='efficientnet-b0', num_classes=1, pretrained=True, dropout=0.2):
        super(MalariaEfficientNet, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # Get feature dimension
        feature_dim = self.backbone._fc.in_features
        
        # Replace classifier for binary classification
        self.backbone._fc = nn.Identity()
        
        # Custom classifier head for clinical use
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # For Grad-CAM visualization
        self.feature_maps = None
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # Extract features
        features = self.backbone.extract_features(x)
        
        # Register hook for Grad-CAM
        if features.requires_grad:
            h = features.register_hook(self.activations_hook)
        
        self.feature_maps = features
        
        # Global average pooling
        pooled = F.adaptive_avg_pool2d(features, 1)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self):
        return self.feature_maps


class MalariaEfficientNetTimm(nn.Module):
    """Alternative implementation using timm library for more model options."""
    
    def __init__(self, model_name='efficientnet_b0', num_classes=1, pretrained=True, dropout=0.2):
        super(MalariaEfficientNetTimm, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def create_model(model_name='efficientnet-b0', num_classes=1, pretrained=True, dropout=0.2, use_timm=False):
    """Factory function to create malaria detection model."""
    
    if use_timm:
        # Convert model name for timm
        timm_name = model_name.replace('-', '_')
        return MalariaEfficientNetTimm(timm_name, num_classes, pretrained, dropout)
    else:
        return MalariaEfficientNet(model_name, num_classes, pretrained, dropout)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in malaria detection."""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClinicalLoss(nn.Module):
    """Custom loss function prioritizing sensitivity for clinical use."""
    
    def __init__(self, sensitivity_weight=2.0, specificity_weight=1.0):
        super(ClinicalLoss, self).__init__()
        self.sensitivity_weight = sensitivity_weight
        self.specificity_weight = specificity_weight
    
    def forward(self, inputs, targets):
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply weights based on true labels
        # Higher weight for missing positive cases (false negatives)
        weights = torch.where(
            targets == 1, 
            self.sensitivity_weight,  # Weight for positive cases
            self.specificity_weight   # Weight for negative cases
        )
        
        weighted_loss = bce_loss * weights
        return weighted_loss.mean()


def model_summary(model, input_size=(3, 224, 224)):
    """Print model summary and parameter count."""
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing EfficientNet-B0 model:")
    model = create_model('efficientnet-b0', num_classes=1, pretrained=False)
    model_summary(model)
    
    print("\nTesting Timm EfficientNet-B0 model:")
    model_timm = create_model('efficientnet-b0', num_classes=1, pretrained=False, use_timm=True)
    model_summary(model_timm)
    
    # Test loss functions
    print("\nTesting loss functions:")
    dummy_logits = torch.randn(4, 1)
    dummy_targets = torch.tensor([1., 0., 1., 0.])
    
    bce_loss = F.binary_cross_entropy_with_logits(dummy_logits.squeeze(), dummy_targets)
    focal_loss = FocalLoss()(dummy_logits.squeeze(), dummy_targets)
    clinical_loss = ClinicalLoss()(dummy_logits.squeeze(), dummy_targets)
    
    print(f"BCE Loss: {bce_loss:.4f}")
    print(f"Focal Loss: {focal_loss:.4f}")
    print(f"Clinical Loss: {clinical_loss:.4f}")
