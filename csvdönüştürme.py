import torch
import pandas as pd
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# Veri dönüşümlerini tanımlama
transform = transforms.Compose([
    transforms.ToTensor(),  # Görüntüyü tensöre dönüştürme
])

# MNIST veri kümesini yükleme
mnist_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

# Veri kümesini pandas DataFrame'e dönüştürme
data = []
for i in range(len(mnist_dataset)):
    image, label = mnist_dataset[i]
    image = image.numpy().reshape(-1)  # Görüntüyü tek boyutlu diziyi dönüştürme
    data.append([label] + image.tolist())

# DataFrame oluşturma
columns = ['label'] + [f'pixel_{i}' for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)

# CSV olarak kaydetme
df.to_csv('mnist_data.csv', index=False)