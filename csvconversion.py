import torch
import pandas as pd
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),  
])


mnist_dataset = MNIST(root='./data', train=True, transform=transform, download=True)


data = []
for i in range(len(mnist_dataset)):
    image, label = mnist_dataset[i]
    image = image.numpy().reshape(-1)  
    data.append([label] + image.tolist())


columns = ['label'] + [f'pixel_{i}' for i in range(len(data[0]) - 1)]
df = pd.DataFrame(data, columns=columns)


df.to_csv('mnist_data.csv', index=False)
