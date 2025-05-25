import torch
import torch.nn as nn

class LongShortTermMemory(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, num_layers=2, dropout=0.5):
        super(LongShortTermMemory, self).__init__()
        
        # Embedding layer to convert word indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_length]
    
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        output = self.dropout(output)
        logits = self.fc(output)  # [batch_size, seq_length, vocab_size]

        return logits, hidden
    


