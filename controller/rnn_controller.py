import torch
import torch.nn as nn

class RNNController(nn.Module):
    def __init__(self, num_layers, hidden_size, output_size):
        super(RNNController, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.fc(x[-1])
        return out
