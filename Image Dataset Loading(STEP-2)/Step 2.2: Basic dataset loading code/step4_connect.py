import torch
from load_dataset import train_loader, num_classes
from step3_model import SimpleCNN

# model create
model = SimpleCNN(num_classes=num_classes)

# train_loader থেকে 1 batch নাও
for images, labels in train_loader:
#    print("Real images shape:", images.shape)
#    print("Real labels:", labels)

    # images-ই এখন x
    outputs = model(images)

#    print("Model output shape:", outputs.shape)
    break