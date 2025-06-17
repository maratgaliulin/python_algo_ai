import torch
import torch.nn as nn

class OHLCTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Replace LSTM with Transformer components
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        
        # Transformer processes entire sequence
        out = self.transformer(x)  # (batch_size, seq_len, hidden_size)
        
        # Take last timestep (like LSTM) and apply same FC structure
        out = self.dropout(out[:, -1, :])  # (batch_size, hidden_size)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# class OHLCTransformer(nn.Module):
#     def __init__(self, input_size=617, hidden_size=256, num_layers=4, output_size=1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # 1. Input projection layer to handle high dimensionality
#         self.input_proj = nn.Sequential(
#             nn.Linear(input_size, 617),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(617, hidden_size),
#             nn.LayerNorm(hidden_size)
#         )
        
#         # 2. Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=8,  # Increased attention heads
#             dim_feedforward=hidden_size*4,  # Larger feedforward network
#             dropout=0.3,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # 3. Output head
#         self.output_head = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, output_size),
#             nn.Tanh()
#         )
        
#     def forward(self, x):
#         # Input shape: (batch_size, seq_len, input_size)
        
#         # Step 1: Project high-dim input to manageable size
#         x = self.input_proj(x)  # (batch_size, seq_len, hidden_size)
        
#         # Step 2: Apply transformer
#         # Generate padding mask if needed (for variable length sequences)
#         # padding_mask = (x.sum(dim=-1) == 0)  # Example mask creation
#         out = self.transformer(x)  # (batch_size, seq_len, hidden_size)
        
#         # Step 3: Take last timestep and predict
#         out = out[:, -1, :]  # (batch_size, hidden_size)
#         return self.output_head(out)