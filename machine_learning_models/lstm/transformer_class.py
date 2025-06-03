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
            nhead=4,  # Number of attention heads (adjust based on hidden_size)
            dropout=0.3,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Keep the same fully connected structure
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.tanh = nn.Tanh()
        # self.swish = Swish()
        # self.swish_alt = SwishAlt()
        
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