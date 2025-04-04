import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=16, num_layers=1, dropout=0.1, **kwargs):
        super(LSTM, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True
                          , bidirectional=True, dropout=dropout) #lstm
        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2,num_classes)
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)#hidden state
        c_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) #internal state
        # Propagate input through LSTM
        out, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state   
        # out = self.layer_norm(out[:, -1, :])
        out = self.fc(out) 
        
        return out

class DenseModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=16, num_layers=1, dropout=0.1, **kwargs):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First dense layer
        self.relu = nn.ReLU()                         # Activation function
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]) # Hidden layers
        self.dropout = nn.Dropout(dropout)               # Dropout for regularization
        self.fc3 = nn.Linear(hidden_size, num_classes) # Output layer
    
    def forward(self, x):
        x = self.fc1(x)       # Pass through first dense layer
        x = self.dropout(self.relu(x))      # Apply ReLU activation
        for hidden_layer in self.hidden_layers:
            x = self.dropout(self.relu(hidden_layer(x)))
        x = self.fc3(x)       # Output layer
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_features, num_classes, num_heads=4, num_layers=1, hidden_dim=64, dropout=0.1, **kwargs):
        super(TimeSeriesTransformer, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Linear transformation to project input features to the model's hidden dimension
        self.input_projection = nn.Linear(num_features, hidden_dim)
        
        # Positional encoding to add sequence information
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Fully connected layer for binary classification at each time step
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, sequence_length, hidden_dim)
        
        # Apply positional encoding
        x = self.positional_encoding(x)
        
        # Pass through the Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, hidden_dim)
        
        # Apply the classification layer at each time step
        x = self.fc(x)  # (batch_size, sequence_length, 1)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)