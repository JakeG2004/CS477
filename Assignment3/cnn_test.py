import torch
from PIL import Image
import os
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn

# Test dataset
class TestImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if f.lower().endswith(".jpg")
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

# Transform
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.626, 0.523, 0.469], std=[0.278, 0.340, 0.370])
])

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

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5)
        
        # Fourth layer, Linear. 16x5x5 in, 120 out
        self.fc1 = nn.Linear(in_features=256*2*2, out_features=240)
        
        # Fifth layer, Linear. 120 in, 84 out
        self.fc2 = nn.Linear(240, 120)

        self.fc3 = nn.Linear(120, 84)
        
        # Sixth layer, output layer, Linear. 84 in, 7 out
        self.fc4 = nn.Linear(84, 7)

    def forward(self, x):
        # Convolution and pooling, twice. Using ReLU
        x = self.pool(torch.relu(self.conv1(x)))  # 100 -> 48
        x = self.pool(torch.relu(self.conv2(x)))  # 48 -> 22
        x = self.pool(torch.relu(self.conv3(x)))  # 22 -> 9
        x = self.pool(torch.relu(self.conv4(x)))  # 9 -> 2
        
        # Reshape the tensor to flatten in
        x = x.view(x.size(0), -1)
        
        # Basic ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Final linear transform
        x = self.fc4(x)
        return x


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CNN().to(device)
net.load_state_dict(torch.load("cnn_model_state.pth", map_location=device))
net.eval()

# Make predictions
predictions = []
with torch.no_grad():
    for images, filenames in test_loader:
        images = images.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        for fname, pred in zip(filenames, predicted):
            predictions.append((fname.split('.')[0], pred.item()))

# Save CSV
csv_path = "test_predictions.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "PredictedClass"])
    for idx, pred_class in predictions:
        writer.writerow([idx, pred_class])

print(f"Predictions saved to {csv_path}")
