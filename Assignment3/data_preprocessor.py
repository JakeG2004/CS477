# Calculate the mean and std. dev of our dataset
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the variables
transform = transforms.ToTensor()
dataset = datasets.ImageFolder(root="./train", transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16)

# Create dummy variables
mean = 0.
std = 0.
num_batches = 0

# Process each image in the data loader
for images, _ in loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    num_batches += batch_samples

mean /= num_batches
std /= num_batches

print(mean, std)