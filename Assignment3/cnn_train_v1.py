import torch
from PIL import Image
import os
import csv
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

# Create a dataloader for the unlabeled data
class TestImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith((".jpg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)

# Data transform (convert to tensor, then normalize to the values from data_preprocessing.py)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.684, 0.603, 0.556], std=[0.265, 0.322, 0.356])
])

# Load the training data
train_set = datasets.ImageFolder(root="./train", transform=transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

# Load the testing data
test_set = TestImageDataset(root="./test", transform=transform)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

# Create the CNN Model
class CNN(nn.Module):
    def __init__(self):
        # Init the super class
        super(CNN, self).__init__()
        
        # First layer: convolution. RGB in, 6 features out. Kernel size 5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        
        # Second layer: maxpool. 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third layer: convolution: 6 channel in, 16 features out, kernel size 5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # Fourth layer, Linear. 16x5x5 in, 120 out
        self.fc1 = nn.Linear(in_features=16*22*22, out_features=120)
        
        # Fifth layer, Linear. 120 in, 84 out
        self.fc2 = nn.Linear(120, 84)
        
        # Sixth layer, output layer, Linear. 84 in, 7 out
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        # Convolution and pooling, twice. Using ReLU
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Reshape the tensor to flatten in
        x = x.view(-1, 16 * 22 * 22)
        
        # Basic ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Final linear transform
        x = self.fc3(x)
        return x

# Training loop
# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
net = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Train
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print('[Epoch %d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))

print('Finished Training!')

# Save the model
torch.save(net.state_dict(), "cnn_model_state.pth")
print("Saved Model!")
