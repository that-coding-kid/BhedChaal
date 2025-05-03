import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

class CSRNet(torch.nn.Module):
    """
    CSRNet model for crowd density estimation
    """
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = torch.nn.Conv2d(64, 1, kernel_size=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        if load_weights:
            self._load_pretrained_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        layers = []
        d_rate = 2 if dilation else 1
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v
        return torch.nn.Sequential(*layers)
    
    def _load_pretrained_weights(self):
        # Check if weights file exists, otherwise download
        weights_path = 'weights.pth'
        if not os.path.exists(weights_path):
            print(f"CSRNet weights not found at {weights_path}. Please download them.")
            print("You can download from here: https://github.com/leeyeehoo/CSRNet-pytorch/blob/master/checkpoints/")
            return False
        
        try:
            # Load weights
            pretrained_dict = torch.load(weights_path, map_location=self.device)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print("CSRNet weights loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading CSRNet weights: {e}")
            return False

class CrowdDensityEstimator:
    """
    Class for estimating crowd density in images using CSRNet
    """
    def __init__(self, model_path=None):
        """Initialize crowd density estimation model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Crowd Density Estimator on {self.device}...")
        
        # Create model
        self.model = CSRNet(load_weights=True if model_path else False)
        
        # Try to load custom weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded crowd density model from {model_path}")
            except Exception as e:
                print(f"Error loading crowd density model: {e}")
                print("Continuing with default model...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # For image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Heat map parameters
        self.heat_map = None
        self.heat_map_alpha = 0.7
        self.heat_map_decay = 0.95
        self.heat_map_threshold = 0.1
        self.colormap = cv2.COLORMAP_JET
    
    def estimate_density(self, frame):
        """
        Estimate crowd density in the given frame
        
        Parameters:
        - frame: Input frame (BGR format)
        
        Returns:
        - density_map: Estimated density map
        - estimated_count: Estimated people count
        - colorized_density: Colorized density map for visualization
        """
        # Make a copy of the frame
        frame_copy = frame.copy()
        
        try:
            # Convert to RGB and preprocess
            frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Resize to a reasonable size if too large
            # CSRNet works better with smaller images
            orig_size = frame_pil.size
            if max(orig_size) > 1024:
                scale_factor = 1024 / max(orig_size)
                new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
                frame_pil = frame_pil.resize(new_size, Image.LANCZOS)
            
            # Transform image for the model
            X = self.transform(frame_pil).unsqueeze(0).to(self.device)
            
            # Forward pass through the model
            with torch.no_grad():
                density_map = self.model(X).squeeze().detach().cpu().numpy()
            
            # Resize density map to original frame size
            density_map = cv2.resize(density_map, (frame.shape[1], frame.shape[0]))
            
            # Update heat map with temporal smoothing
            if self.heat_map is None:
                self.heat_map = density_map
            else:
                # Apply decay to existing heat map and add new density map
                self.heat_map = self.heat_map * self.heat_map_decay + density_map * (1 - self.heat_map_decay)
            
            # Create a normalized and colorized density map for visualization
            norm_density = self.heat_map / (np.max(self.heat_map) + 1e-10)
            colorized_density = cv2.applyColorMap((norm_density * 255).astype(np.uint8), self.colormap)
            
            # Apply threshold mask to hide low-density areas
            mask = norm_density > self.heat_map_threshold
            mask = mask.astype(np.uint8) * 255
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, 3, axis=-1)
            
            # Blend density map with mask to hide low-density areas
            colorized_density = cv2.bitwise_and(colorized_density, mask)
            
            # Calculate approximate people count
            estimated_count = int(np.sum(self.heat_map))
            
            return self.heat_map, estimated_count, colorized_density
            
        except Exception as e:
            print(f"Error in density estimation: {e}")
            # Return empty results on error
            return np.zeros_like(frame[:, :, 0], dtype=np.float32), 0, frame_copy
    
    def reset_heat_map(self):
        """Reset the heat map"""
        self.heat_map = None
        print("Density heat map reset")