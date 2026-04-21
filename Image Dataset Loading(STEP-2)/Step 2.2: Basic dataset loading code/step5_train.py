import torch
import torch.nn as nn
import torch.optim as optim

# Step 2 থেকে data আনছি
from load_dataset import train_loader, val_loader, num_classes

# Step 3 থেকে model আনছি
from step3_model import SimpleCNN


# -------------------------------
# 1. Model create
# -------------------------------
model = SimpleCNN(num_classes=num_classes)

# -------------------------------
# 2. Loss Function
# -------------------------------
criterion = nn.CrossEntropyLoss()

# -------------------------------
# 3. Optimizer
# -------------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# 4. Training loop
# -------------------------------
num_epochs = 10

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print("Average Loss:", avg_loss)


    # -------------------------------
# Validation (Evaluation)
# -------------------------------

model.eval()   # 👉 model evaluation mode

correct = 0
total = 0

with torch.no_grad():   # 👉 gradient বন্ধ (memory save + faster)
    for images, labels in val_loader:

        outputs = model(images)

        # prediction বের করা
        _, predicted = torch.max(outputs, 1)

        # total samples count
        total += labels.size(0)

        # correct prediction count
        correct += (predicted == labels).sum().item()

# accuracy calculate
accuracy = 100 * correct / total

print(f"\nValidation Accuracy: {accuracy:.2f}%")