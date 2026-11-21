import torch
import torch.nn as nn #neural network module of pytorch
import torch.optim as optim
import random

# pytorch for building and training neural networks , torchvision for resnet18 and image datasets, and torch.utils.data for data loading and manipulation , pillow for rading images , and random for selecting small datasets

from torchvision import datasets, transforms,models
from torch.utils.data import DataLoader,Subset


 #--1---load and preprocess images

transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

# imagefolder library reads the structure defined and assign labels based on folder name
dataset=datasets.ImageFolder("dataset/train",transform=transform) 
# print(dataset.classes,dataset.class_to_idx,len(dataset))   ->['fake', 'real'] {'fake': 0, 'real': 1} 42

class_indices = {i: [] for i in range(len(dataset.classes))}

# dataset.samples contains information about every image, including its file path and label.enumerate() gives us the index along with each item.
for i, (_, label) in enumerate(dataset.samples):
    class_indices[label].append(i) #Stores each image's index in the list corresponding to its class.

train_indices = []
val_indices = []

for label in class_indices:
    indices = class_indices[label]
    random.shuffle(indices)

    train_indices.extend(indices[:80])
    val_indices.extend(indices[80:])

train_data = Subset(dataset, train_indices)
val_data = Subset(dataset, val_indices)

#--2---create data loaders for training and validation, batches
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

model=models.resnet18(weights=None) # cause we are not using pretrained weights
num_features=model.fc.in_features
model.fc=nn.Sequential(nn.Dropout(0.2),nn.Linear(num_features,2)) # 2 classes

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
#print(model)

#training of the model using cross-entropy loss and Adam optimizer with a learning rate of 0.0001 for 5 epochs. The training loop iterates over the training data, computes the loss, performs backpropagation, and updates the model parameters. It also calculates and prints the accuracy after each epoch.
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 5

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
#pytorch train cycle consists of 4 steps: forward pass, loss calculation, backward pass, and parameter update. The optimizer.zero_grad() 
        optimizer.zero_grad()#clear old gradients

        outputs = model(images)#image passed through the model to get predictions

        loss = criterion(outputs, labels) # compare the predicted outputs with the true labels to compute the loss

        loss.backward()#calculates gradients and tell us how to update weights to reduce loss

        optimizer.step()#use the gradients to update the model's parameters

        total_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()

    accuracy = correct / len(train_data) * 100

    avg_loss = total_loss / len(train_loader)

    print(
    f"Epoch {epoch+1}/{epochs}, "
    f"Loss: {avg_loss:.4f}, "
    f"Accuracy: {accuracy:.2f}%"
    )

# ---3---Evaluation phase

model.eval()

correct = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()

accuracy = correct / len(val_data) * 100

print(f"Validation Accuracy: {accuracy:.2f}%")

torch.save(
    {
        "model_state": model.state_dict(), #contains learned weights to a file
        "class_to_idx": dataset.class_to_idx #saves the mapping between class names and numerical labels
    },
    "veriface_model.pth" # Saved checkpoint file
)
print("Model saved successfully!")