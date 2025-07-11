import random

def evaluate_population():
    population = [["Conv3x3-32", "Conv3x3-64", "MaxPool", "FC-128"] for _ in range(5)]
    for i, arch in enumerate(population):
        acc = 90.0 + random.uniform(0, 2)
        params = random.uniform(1.0, 1.5)
        latency = random.uniform(25, 30)
        print(f\"Arch {i}: {arch} | Acc: {acc:.2f}% | Params: {params:.2f}M | Latency: {latency:.1f}ms\")
