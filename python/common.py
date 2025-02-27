import numpy
import torchvision
import torch

batch_size = 64
learning_rate = 0.01
epochs = 10

# Neural network architecture
input_size = 28 * 28  # 28x28 pixels
hidden_size1 = 512
hidden_size2 = 256
output_size = 10  # Digits 0-9

train_dataset = torchvision.datasets.MNIST(root='../data', train = True, transform = torchvision.transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='../data', train = False, transform = torchvision.transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = len(test_dataset), shuffle = False)

