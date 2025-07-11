# nas-engine
Neural Architecture Search Engine using Reinforcement Learning and Evolutionary Algorithms

# Neural Architecture Search Engine

An automated system for discovering optimal neural network architectures using Reinforcement Learning and Evolutionary Algorithms. Built to drastically reduce manual model design and hyperparameter tuning.

## 🔍 Overview

Neural Architecture Search (NAS) is a subfield of AutoML that focuses on automating the design of deep neural network architectures. This project implements:

- Reinforcement Learning (RL)-based controller to propose model architectures.
- Evaluation of sampled architectures via proxy training.
- Evolutionary mutation strategies for exploration.
- Multi-objective optimization (accuracy, latency, parameter count).

## 🚀 Features

- RNN-based controller trained with REINFORCE algorithm
- Search space for CNNs (Conv layers, filters, pooling, etc.)
- Fast architecture validation using weight-sharing and proxy tasks
- Evolutionary mutation and crossover strategies
- PyTorch-based modular design
- Extensible search and evaluation pipeline

## 📦 Installation

bash
git clone https://github.com/RehanMohammed985/nas-engine.git
cd nas-engine
pip install -r requirements.txt

🧠 Usage

python train_controller.py
python evaluate_architecture.py
🔧 Project Structure

nas-engine/
├── controller/             # RL controller
│   └── rnn_controller.py
├── search/                 # Search algorithms
│   ├── evolutionary.py
│   └── reinforcement.py
├── models/                 # Candidate architectures
│   └── cnn_space.py
├── train_controller.py     # Main training script
├── evaluate_architecture.py# Evaluator script
├── utils.py
├── requirements.txt
└── README.md
📊 Objectives

Accuracy maximization
Latency minimization (mobile-first)
Parameter efficiency
🤖 Sample Output

Best Architecture:
[Conv3x3-32, Conv3x3-64, MaxPool, FC-128]
Accuracy: 91.2% | Params: 1.2M | Latency: 28ms
📎 License

MIT


</details>

---

### 📄 `requirements.txt`

> ✅ Path: root

txt
torch>=1.12
numpy
tqdm
matplotlib

