"""
Grad-CAM++ heatmap generation for model interpretability.

Produces visual overlays showing which image regions contributed most
to the model's predictions.  Uses pixel-wise gradient weighting for
better localization than standard Grad-CAM.

Usage::

    from chexray_diagnosis.services.heatmap import generate_gradcam

    heatmap_np = generate_gradcam(model, input_tensor, device, original_image)
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Optional


def generate_gradcam(
    model: torch.nn.Module,
    inp: torch.Tensor,
    device: torch.device,
    image_rgb: Optional[Image.Image] = None,
    blend_alpha: float = 0.5,
) -> Optional[np.ndarray]:
    """Generate a Grad-CAM++ heatmap.

    Args:
        model:       Loaded DenseNet model (eval mode).
        inp:         Pre-processed input tensor ``(1, 3, 224, 224)``.
        device:      ``torch.device`` for computation.
        image_rgb:   Original PIL image (RGB) for blending. If ``None``,
                     returns the raw coloured heatmap.
        blend_alpha: Blend intensity (0.0 = all X-ray, 1.0 = all heatmap).

    Returns:
        ``numpy.ndarray`` (224, 224, 3) in RGB, or ``None`` on failure.
    """
    try:
        # 1. Forward pass — identify top predicted class
        with torch.no_grad():
            output = model(inp)
            probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            top_class = int(np.argmax(probs))

        # 2. Hook into the last convolutional block
        activations = None
        gradients = None

        def fwd_hook(_mod, _inp, out):
            nonlocal activations
            activations = out.detach()

        def bwd_hook(_mod, _grad_in, grad_out):
            nonlocal gradients
            gradients = grad_out[0].detach()

        # Find features layer
        target_layer = None
        if hasattr(model, "features"):
            target_layer = model.features
        elif hasattr(model, "densenet121") and hasattr(model.densenet121, "features"):
            target_layer = model.densenet121.features

        if target_layer is None:
            raise RuntimeError("Cannot locate features layer in model")

        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_backward_hook(bwd_hook)

        # 3. Forward + backward with gradients
        inp_grad = inp.clone().detach().to(device).requires_grad_(True)
        out_grad = model(inp_grad)
        score = out_grad[0, top_class]

        model.zero_grad()
        score.backward()

        h_fwd.remove()
        h_bwd.remove()

        if gradients is None or activations is None:
            raise RuntimeError("Failed to capture gradients / activations")

        # 4. Grad-CAM++ weighting
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = (
            gradients.pow(2).mul(2)
            + activations.mul(gradients.pow(3))
            .view(b, k, u * v)
            .sum(-1, keepdim=True)
            .view(b, k, 1, 1)
        )
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom),
        )
        alpha = alpha_num.div(alpha_denom + 1e-7)

        positive_gradients = torch.nn.functional.relu(gradients)
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        gradcam = torch.nn.functional.relu((weights * activations).sum(1, keepdim=True))

        # Normalise
        gc_min, gc_max = gradcam.min(), gradcam.max()
        if (gc_max - gc_min).item() > 0:
            gradcam = (gradcam - gc_min) / (gc_max - gc_min)

        cam = gradcam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam /= cam.max()
        cam_uint8 = np.uint8(255 * cam)

        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # 5. Optionally blend with the original image
        if image_rgb is not None:
            img_np = np.array(image_rgb.resize((224, 224))).astype(np.uint8)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            blended = cv2.addWeighted(img_np, 1.0 - blend_alpha, heatmap_bgr, blend_alpha, 0)
            return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        else:
            return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Grad-CAM++ generation failed: {e}")
        return None
