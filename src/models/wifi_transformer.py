import torch
import torch.nn as nn
import numpy as np

class WiFiTransformer(nn.Module):
    def __init__(self, input_dim=90, d_model=256, num_heads=8, num_layers=6, num_joints=17, num_classes=100):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_joints = num_joints
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output heads
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints * 2)  # x, y coordinates for each joint
        )
        
        self.id_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print('WiFiTransformer initialized with:')
        print(f'  - Input dim: {input_dim}')
        print(f'  - Model dim: {d_model}')
        print(f'  - Num heads: {num_heads}')
        print(f'  - Num layers: {num_layers}')
        print(f'  - Num joints: {num_joints}')
        print(f'  - Num classes: {num_classes}')

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch_size, d_model)
        
        # Output heads
        pose_output = self.pose_head(x)  # (batch_size, num_joints * 2)
        id_output = self.id_head(x)      # (batch_size, num_classes)
        
        return pose_output, id_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

