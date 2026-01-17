import pytest
import torch
import numpy as np
from src.models.wifi_transformer import WiFiTransformer

class TestWiFiTransformer:
    def setup_method(self):
        self.model = WiFiTransformer()
    
    def test_model_initialization(self):
        assert self.model is not None
        assert isinstance(self.model, WiFiTransformer)
    
    def test_forward_pass(self):
        # Test with sample input
        batch_size = 2
        seq_len = 100
        input_dim = 90
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = self.model(x)
        
        assert output is not None
        assert isinstance(output, torch.Tensor)
        # Add more specific assertions based on your model implementation
