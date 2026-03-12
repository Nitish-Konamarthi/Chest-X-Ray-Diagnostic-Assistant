import os
import numpy as np
import sys
from PIL import Image

import cv2
import torch
import torchvision.transforms as transforms

# Handle both relative and absolute imports
try:
    from .DensenetModels import DenseNet121, DenseNet169, DenseNet201
except ImportError:
    from DensenetModels import DenseNet121, DenseNet169, DenseNet201

#------------------------------------------------------------------------------------
# Grad-CAM++ Utilities for Model Interpretability
#------------------------------------------------------------------------------------

def generate_gradcam(model, inp, device, image_rgb=None, blend_alpha=0.5):
    """
    Generate Grad-CAM++ (Gradient-weighted Class Activation Map) visualization.
    
    Grad-CAM++ improves upon Grad-CAM by using pixel-wise weighting of gradients,
    providing better localization especially for multiple instances of objects.
    
    This is a reusable function for both web apps and batch processing.
    Uses gradient information to show which regions contributed most to predictions.
    
    Args:
        model: DenseNet121 model (loaded and in eval mode)
        inp: Input tensor (1, 3, 224, 224) already preprocessed and on device
        device: torch device (cuda or cpu)
        image_rgb: Original PIL Image (RGB) for blending (optional)
        blend_alpha: Blend intensity (0.0=all X-ray, 1.0=all heatmap, default=0.5)
    
    Returns:
        numpy array: Grad-CAM++ heatmap (224, 224, 3) in RGB format, or None if failed
        
    Example:
        >>> from HeatmapGenerator import generate_gradcam
        >>> heatmap = generate_gradcam(model, inp, device, image_rgb, blend_alpha=0.6)
        >>> st.image(heatmap, caption="Grad-CAM++ Attention Map")
    """
    try:
        # Step 1: Forward pass to identify top prediction
        with torch.no_grad():
            output = model(inp)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            top_class = np.argmax(probs)
        
        # Step 2: Register hooks to capture feature maps and gradients
        activations = None
        gradients = None
        
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()
        
        # Find the last convolutional layer (features for DenseNet)
        target_layer = None
        if hasattr(model, 'features'):
            target_layer = model.features
        elif hasattr(model, 'densenet121') and hasattr(model.densenet121, 'features'):
            target_layer = model.densenet121.features
        
        if target_layer is None:
            raise RuntimeError("Could not find features layer in model")
        
        # Register hooks
        handle_forward = target_layer.register_forward_hook(forward_hook)
        handle_backward = target_layer.register_backward_hook(backward_hook)
        
        # Step 3: Forward pass with gradient computation
        inp_grad = inp.clone().detach().to(device)
        inp_grad.requires_grad_(True)
        
        output_grad = model(inp_grad)
        score = output_grad[0, top_class]
        
        # Step 4: Backward pass to compute gradients
        model.zero_grad()
        score.backward()
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # Step 5: Compute Grad-CAM++ weights
        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients or activations")
        
        # Get dimensions
        b, k, u, v = gradients.size()
        
        # Calculate alpha weights using 2nd and 3rd order gradients
        # alpha = grad^2 / (2 * grad^2 + sum(activation * grad^3))
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        
        # Avoid division by zero
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        
        # Compute alpha
        alpha = alpha_num.div(alpha_denom + 1e-7)
        
        # Apply ReLU to gradients (only positive influences)
        positive_gradients = torch.nn.functional.relu(gradients)
        
        # Calculate importance weights
        weights = (alpha * positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        
        # Generate weighted activation map
        gradcam = (weights * activations).sum(1, keepdim=True)
        gradcam = torch.nn.functional.relu(gradcam)
        
        # Normalize to [0, 1]
        gradcam_min = gradcam.min()
        gradcam_max = gradcam.max()
        if (gradcam_max - gradcam_min).item() > 0:
            gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min)
        
        # Convert to numpy and resize
        gradcam_np = gradcam.squeeze().cpu().numpy()
        
        # Resize to 224x224
        gradcam_resized = cv2.resize(gradcam_np, (224, 224))
        
        # Normalize again after resize
        if gradcam_resized.max() > 0:
            gradcam_resized = gradcam_resized / gradcam_resized.max()
        
        # Convert to uint8 for colormap
        gradcam_uint8 = np.uint8(255 * gradcam_resized)
        
        # Apply colormap (JET: red=high importance, blue=low)
        heatmap_colored = cv2.applyColorMap(gradcam_uint8, cv2.COLORMAP_JET)
        
        # If original image provided, blend them
        if image_rgb is not None:
            img_np = np.array(image_rgb.resize((224, 224))).astype(np.uint8)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 3:  # RGB to BGR for OpenCV
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Blend images using blend_alpha parameter
            # blend_alpha: 0.0 = 100% X-ray, 0.5 = 50/50, 1.0 = 100% heatmap
            original_weight = 1.0 - blend_alpha
            blended = cv2.addWeighted(img_np, original_weight, heatmap_colored, blend_alpha, 0)
            blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            return blended_rgb
        else:
            # Return heatmap only (convert BGR to RGB)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            return heatmap_rgb
            
    except Exception as e:
        print(f"Grad-CAM++ generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


#------------------------------------------------------------------------------------
# Legacy HeatmapGenerator Class for Batch Processing
#------------------------------------------------------------------------------------

class HeatmapGenerator:
    """
    Class for batch heatmap generation and file saving using Grad-CAM++.
    
    For real-time web applications, use generate_gradcam() function directly.
    This class provides backward compatibility and convenience methods for batch processing.
    
    Example:
        >>> h = HeatmapGenerator("model.pth", "DENSE-NET-121", 14, 224)
        >>> h.generate("input.png", "output_heatmap.png")
    """
    
    def __init__(self, pathModel, nnArchitecture, nnClassCount, transCrop):
        """
        Initialize HeatmapGenerator.
        
        Args:
            pathModel: Path to model checkpoint
            nnArchitecture: Model architecture ('DENSE-NET-121', 'DENSE-NET-169', 'DENSE-NET-201')
            nnClassCount: Number of output classes (14 for CheXNet)
            transCrop: Image size after preprocessing (224)
        """
       
        # Initialize the network
        if nnArchitecture == 'DENSE-NET-121': 
            model = DenseNet121(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-169': 
            model = DenseNet169(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-201': 
            model = DenseNet201(nnClassCount, True)
        else:
            raise ValueError(f"Unknown architecture: {nnArchitecture}")
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)

        # Load checkpoint
        if not os.path.exists(pathModel):
            raise FileNotFoundError(f"Model checkpoint not found at {pathModel}")
        
        modelCheckpoint = torch.load(pathModel, map_location=self.device)
        state_dict = modelCheckpoint.get('state_dict', modelCheckpoint)
        
        # Remove DataParallel wrapper if present
        new_state = {}
        for k, v in state_dict.items():
            nk = k.replace("module.", "") if k.startswith("module.") else k
            new_state[nk] = v
        
        model.load_state_dict(new_state, strict=False)
        self.model = model
        self.model.eval()
        
        # Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transformSequence = transforms.Compose([
            transforms.Resize(transCrop),
            transforms.ToTensor(),
            normalize
        ])
        self.transCrop = transCrop
    
    def generate(self, pathImageFile, pathOutputFile, transCrop=None):
        """
        Generate Grad-CAM++ heatmap and save to file.
        
        Args:
            pathImageFile: Path to input image
            pathOutputFile: Path to save output heatmap image
            transCrop: Image crop size (uses self.transCrop if not specified)
            
        Returns:
            numpy array: Generated heatmap image
        """
        
        if transCrop is None:
            transCrop = self.transCrop
        
        # Load and preprocess image
        imageData = Image.open(pathImageFile).convert('RGB')
        image_tensor = self.transformSequence(imageData)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Generate Grad-CAM++ using shared function
        heatmap_img = generate_gradcam(self.model, image_tensor, self.device, imageData)
        
        if heatmap_img is None:
            raise RuntimeError("Failed to generate Grad-CAM++")
        
        # Save output
        output_dir = os.path.dirname(pathOutputFile)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Convert RGB to BGR for cv2.imwrite
        heatmap_bgr = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(pathOutputFile, heatmap_bgr)
        print(f"✓ Grad-CAM++ heatmap saved to {pathOutputFile}")
        
        return heatmap_img


#------------------------------------------------------------------------------------
# Command-line Interface for Batch Processing
#------------------------------------------------------------------------------------

if __name__ == "__main__":
    pathInputImage = 'test/00009285_000.png'
    pathOutputImage = 'test/heatmap.png'
    pathModel = 'chexnet/models/m-25012018-123527.pth.tar'

    nnArchitecture = 'DENSE-NET-121'
    nnClassCount = 14
    transCrop = 224

    try:
        h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
        h.generate(pathInputImage, pathOutputImage, transCrop)
        print("✓ Batch Grad-CAM++ heatmap generation completed successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)