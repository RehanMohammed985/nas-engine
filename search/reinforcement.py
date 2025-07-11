from controller.rnn_controller import RNNController
import torch
import torch.optim as optim

def train_controller():
    model = RNNController(num_layers=2, hidden_size=64, output_size=10)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for step in range(100):
        input_seq = torch.randn(5, 1, 1)
        logits = model(input_seq)
        reward = (logits.sum() / 10).item()
        loss = -reward  # REINFORCE-style surrogate
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f\"Step {step} | Reward: {reward:.4f}\")
