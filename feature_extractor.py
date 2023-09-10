import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.net = model
        for p in self.net.parameters():
            p.requires_grad = False
        self.features = nn.Sequential(*list(self.net.children())[:-1])
        
        # Register hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        self.net.layer4[-1].register_forward_hook(self.save_activations)
        self.net.layer4[-1].register_backward_hook(self.save_gradients)
        
    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def get_activations(self):
        return self.activations

    def get_activations_gradient(self):
        return self.gradients
    
    def forward(self, x):
        return self.features(x)
