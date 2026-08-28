"""
ResNet Model Architecture for Food Spoilage Detection

This module provides flexible ResNet model building functions supporting
multiple architectures, pretrained weights, and custom configurations.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, Any


# Supported ResNet architectures
SUPPORTED_ARCHITECTURES = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
}

# Pretrained weights mapping
WEIGHTS_MAPPING = {
    'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
    'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
    'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
    'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
    'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,
}


class FoodSpoilageResNet(nn.Module):
    """
    ResNet model wrapper for food spoilage classification.
    
    This class provides a clean interface for creating ResNet models
    with customizable architecture, pretrained weights, and dropout.
    
    Args:
        architecture (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 
                           'resnet101', 'resnet152')
        num_classes (int): Number of output classes (default: 2 for fresh/spoiled)
        pretrained (bool): Whether to use pretrained ImageNet weights (default: True)
        dropout (float): Dropout rate before final layer (default: 0.5)
    """
    
    def __init__(
        self,
        architecture: str = 'resnet50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super(FoodSpoilageResNet, self).__init__()
        
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from {list(SUPPORTED_ARCHITECTURES.keys())}"
            )
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout
        
        # Load base ResNet model
        weights = WEIGHTS_MAPPING[architecture] if pretrained else None
        self.backbone = SUPPORTED_ARCHITECTURES[architecture](weights=weights)
        
        # Get number of features from the final layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer with custom classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def freeze_backbone(self):
        """
        Freeze all layers except the final classifier.
        Useful for transfer learning with limited data.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final classifier
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_layers(self, num_layers: int):
        """
        Freeze the first num_layers of the backbone.
        
        Args:
            num_layers (int): Number of layer groups to freeze
        """
        layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]
        
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get the number of parameters in the model.
        
        Args:
            trainable_only (bool): Count only trainable parameters
        
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def build_resnet(
    architecture: str = 'resnet50',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = False
) -> FoodSpoilageResNet:
    """
    Build a ResNet model for food spoilage classification.
    
    Args:
        architecture (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 
                           'resnet101', 'resnet152')
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Use pretrained ImageNet weights (default: True)
        dropout (float): Dropout rate before final layer (default: 0.5)
        freeze_backbone (bool): Freeze backbone layers for transfer learning (default: False)
    
    Returns:
        FoodSpoilageResNet: Configured ResNet model
    
    Example:
        >>> model = build_resnet(architecture='resnet50', num_classes=2, pretrained=True)
        >>> model = build_resnet('resnet18', pretrained=False, dropout=0.3)
    """
    model = FoodSpoilageResNet(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


def build_resnet_from_config(config: Dict[str, Any]) -> FoodSpoilageResNet:
    """
    Build ResNet model from configuration dictionary.
    
    Args:
        config (dict): Configuration dictionary with model parameters.
                      Expected keys: 'architecture', 'pretrained', 'num_classes', 'dropout'
    
    Returns:
        FoodSpoilageResNet: Configured ResNet model
    
    Example:
        >>> config = {
        ...     'architecture': 'resnet50',
        ...     'pretrained': True,
        ...     'num_classes': 2,
        ...     'dropout': 0.5
        ... }
        >>> model = build_resnet_from_config(config)
    """
    model_config = config.get('model', config)  # Support nested or flat config
    
    architecture = model_config.get('architecture', 'resnet50')
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)
    dropout = model_config.get('dropout', 0.5)
    
    return build_resnet(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


def load_model(
    checkpoint_path: str,
    architecture: str = 'resnet50',
    num_classes: int = 2,
    dropout: float = 0.5,
    device: str = 'cpu'
) -> FoodSpoilageResNet:
    """
    Load a saved model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to saved model checkpoint (.pt or .pth file)
        architecture (str): ResNet architecture (must match saved model)
        num_classes (int): Number of classes (must match saved model)
        dropout (float): Dropout rate (must match saved model)
        device (str): Device to load model on ('cpu' or 'cuda')
    
    Returns:
        FoodSpoilageResNet: Loaded model
    
    Example:
        >>> model = load_model('models/resnet_spoilage.pt', device='cuda')
    """
    # Create model with same architecture
    model = build_resnet(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=False,  # Don't load ImageNet weights
        dropout=dropout
    )
    
    # Load saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_model_info(model: FoodSpoilageResNet) -> Dict[str, Any]:
    """
    Get information about the model.
    
    Args:
        model (FoodSpoilageResNet): Model instance
    
    Returns:
        dict: Model information including architecture, parameters, etc.
    """
    return {
        'architecture': model.architecture,
        'num_classes': model.num_classes,
        'pretrained': model.pretrained,
        'dropout': model.dropout_rate,
        'total_parameters': model.get_num_parameters(trainable_only=False),
        'trainable_parameters': model.get_num_parameters(trainable_only=True),
    }


# Convenience function for backward compatibility
def create_model(num_classes: int = 2, pretrained: bool = True) -> FoodSpoilageResNet:
    """
    Simple wrapper for creating default ResNet50 model.
    Kept for backward compatibility.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
    
    Returns:
        FoodSpoilageResNet: ResNet50 model
    """
    return build_resnet(
        architecture='resnet50',
        num_classes=num_classes,
        pretrained=pretrained
    )
