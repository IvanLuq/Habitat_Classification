import numpy as np
from utils import load_training_data
from model import predict

patches, labels = load_training_data()

idx = np.random.choice(len(labels), size=50, replace=False)
hits = 0
for i in idx:
    hits += (predict(patches[i]) == int(labels[i]))
print("Top-1 accuracy on 50 random training samples:", hits/50)
