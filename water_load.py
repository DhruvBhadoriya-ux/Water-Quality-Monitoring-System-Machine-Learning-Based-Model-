import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load train dataset
train_dataset = datasets.ImageFolder("water_dataset_images/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load test dataset
test_dataset = datasets.ImageFolder("water_dataset_images/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(train_dataset.classes)  # ['clean', 'polluted']


#define the model

class WaterCNN(nn.Module):
    def __init__(self):
        super(WaterCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = WaterCNN()

#train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
#save the model
torch.save(model.state_dict(), "water_model.pth")


