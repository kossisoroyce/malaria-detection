import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GradCAM:
    """Grad-CAM implementation for model explainability in clinical settings."""
    
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        if self.target_layer_name:
            target_layer = dict(self.model.named_modules())[self.target_layer_name]
        else:
            # Use the last convolutional layer by default
            target_layer = self._find_last_conv_layer()
        
        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        conv_layers = []
        
        def find_conv_recursive(module):
            for child in module.children():
                if isinstance(child, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    conv_layers.append(child)
                else:
                    find_conv_recursive(child)
        
        find_conv_recursive(self.model)
        
        if not conv_layers:
            raise ValueError("No convolutional layers found in the model")
        
        return conv_layers[-1]
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap."""
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            # For binary classification, use the single output
            class_idx = 0 if output.dim() == 1 else 1
        
        # Backward pass
        self.model.zero_grad()
        if output.dim() == 1:
            # Single output (binary classification)
            output.backward()
        else:
            # Multiple outputs
            output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.cpu().numpy()
    
    def __del__(self):
        """Remove hooks when object is destroyed."""
        for hook in self.hooks:
            hook.remove()


def visualize_gradcam(image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Visualize Grad-CAM overlay on original image."""
    
    # Convert image to numpy if it's a tensor
    if torch.is_tensor(image):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image (assuming ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Resize CAM to match image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Apply colormap to CAM
    cam_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    cam_colored = cam_colored.astype(np.float32) / 255
    
    # Overlay CAM on image
    overlayed = alpha * cam_colored + (1 - alpha) * image
    
    return overlayed, cam_resized


def create_gradcam_visualization(model, image_tensor, save_path=None, 
                               title="Grad-CAM Visualization"):
    """Create complete Grad-CAM visualization with subplots."""
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(image_tensor.unsqueeze(0))
    
    # Create visualization
    overlayed, cam_resized = visualize_gradcam(image_tensor, cam)
    
    # Original image for display
    if torch.is_tensor(image_tensor):
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        original = image_tensor.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original = std * original + mean
    original = np.clip(original, 0, 1)
    
    # Create subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlayed)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, cam


def batch_gradcam_analysis(model, dataloader, num_samples=10, save_dir=None):
    """Generate Grad-CAM visualizations for a batch of samples."""
    
    model.eval()
    gradcam = GradCAM(model)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if len(results) >= num_samples:
                break
            
            for i in range(min(images.size(0), num_samples - len(results))):
                image = images[i]
                label = labels[i].item()
                
                # Get prediction
                pred_logit = model(image.unsqueeze(0))
                pred_prob = torch.sigmoid(pred_logit).item()
                pred_class = int(pred_prob > 0.5)
                
                # Generate Grad-CAM
                cam = gradcam.generate_cam(image.unsqueeze(0))
                
                # Create visualization
                overlayed, cam_resized = visualize_gradcam(image, cam)
                
                result = {
                    'image_idx': len(results),
                    'true_label': label,
                    'pred_prob': pred_prob,
                    'pred_class': pred_class,
                    'correct': (pred_class == label),
                    'original_image': image,
                    'cam': cam_resized,
                    'overlayed': overlayed
                }
                
                results.append(result)
                
                # Save individual visualization if directory provided
                if save_dir:
                    fig, _ = create_gradcam_visualization(
                        model, image,
                        save_path=f"{save_dir}/gradcam_sample_{len(results)}.png",
                        title=f"Sample {len(results)}: True={label}, Pred={pred_prob:.3f}"
                    )
                    plt.close(fig)
    
    return results


def create_gradcam_summary_plot(results, save_path=None):
    """Create summary plot showing multiple Grad-CAM results."""
    
    n_samples = len(results)
    cols = min(5, n_samples)
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
    
    if rows == 1:
        axes = axes.reshape(2, -1)
    
    for i, result in enumerate(results):
        row = (i // cols) * 2
        col = i % cols
        
        # Original image
        axes[row, col].imshow(result['overlayed'])
        axes[row, col].set_title(
            f"Sample {i+1}\nTrue: {result['true_label']}, "
            f"Pred: {result['pred_prob']:.3f}"
        )
        axes[row, col].axis('off')
        
        # Heatmap
        im = axes[row + 1, col].imshow(result['cam'], cmap='jet')
        axes[row + 1, col].set_title('Heatmap')
        axes[row + 1, col].axis('off')
    
    # Hide unused subplots
    for i in range(n_samples, rows * cols):
        row = (i // cols) * 2
        col = i % cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.suptitle('Grad-CAM Analysis Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def clinical_gradcam_report(results):
    """Generate clinical interpretation of Grad-CAM results."""
    
    print("=" * 60)
    print("GRAD-CAM CLINICAL ANALYSIS")
    print("=" * 60)
    
    total_samples = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    accuracy = correct_predictions / total_samples
    
    print(f"Total samples analyzed: {total_samples}")
    print(f"Correct predictions: {correct_predictions} ({accuracy:.1%})")
    
    # Analyze by prediction correctness
    correct_results = [r for r in results if r['correct']]
    incorrect_results = [r for r in results if not r['correct']]
    
    print(f"\nCorrect predictions: {len(correct_results)}")
    print(f"Incorrect predictions: {len(incorrect_results)}")
    
    # Analyze confidence distribution
    high_conf_correct = sum(1 for r in correct_results if abs(r['pred_prob'] - 0.5) > 0.3)
    low_conf_incorrect = sum(1 for r in incorrect_results if abs(r['pred_prob'] - 0.5) < 0.3)
    
    print(f"\nHigh confidence correct predictions: {high_conf_correct}")
    print(f"Low confidence incorrect predictions: {low_conf_incorrect}")
    
    print("\nClinical Recommendations:")
    print("- Review cases with low confidence scores")
    print("- Validate Grad-CAM highlights with clinical experts")
    print("- Use visualizations to build clinician trust")
    print("- Flag unusual activation patterns for manual review")
    
    return {
        'total_samples': total_samples,
        'accuracy': accuracy,
        'correct_predictions': correct_predictions,
        'high_conf_correct': high_conf_correct,
        'low_conf_incorrect': low_conf_incorrect
    }


if __name__ == "__main__":
    # Test Grad-CAM with dummy model and data
    print("Testing Grad-CAM implementation...")
    
    # Create dummy model (simplified EfficientNet-like)
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = torch.nn.Linear(32, 1)
        
        def forward(self, x):
            features = self.features(x)
            features = features.view(features.size(0), -1)
            return self.classifier(features)
    
    # Test
    model = DummyModel()
    dummy_image = torch.randn(3, 224, 224)
    
    try:
        gradcam = GradCAM(model)
        cam = gradcam.generate_cam(dummy_image.unsqueeze(0))
        print(f"Generated CAM shape: {cam.shape}")
        print("Grad-CAM test successful!")
        
    except Exception as e:
        print(f"Grad-CAM test failed: {e}")
