# ================================
# Step 2: Dataset Loading + Split
# ================================

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# -------------------------------
# 1. Dataset path (IMPORTANT)
# -------------------------------
# 👉 এখানে তোমার dataset-এর exact path দাও
data_dir = "/home/ashraful/programming/PyTorch/Image Dataset Loading(STEP-2)/Step 2.1: ImageFolder/data/PlantVillage"

# -------------------------------
# 2. Transform (image preprocessing)
# -------------------------------
# Resize → fixed size
# ToTensor → image কে tensor এ convert করে
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# 3. Load dataset using ImageFolder
# -------------------------------
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# -------------------------------
# 4. Basic info print
# -------------------------------
class_names = dataset.classes
num_classes = len(dataset.classes)
print("Dataset path exists:", os.path.exists(data_dir))
print("Classes:", dataset.classes)
print("Class to index:", dataset.class_to_idx)
print("Total images:", len(dataset))

# -------------------------------
# 5. Train / Validation split
# -------------------------------
train_size = int(0.7 * len(dataset))   # 70% training
val_size = len(dataset) - train_size   # remaining validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("\nTrain size:", len(train_dataset))
print("Validation size:", len(val_dataset))

# -------------------------------
# 6. DataLoader (batch তৈরি)
# -------------------------------
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# -------------------------------
# 7. Train loader থেকে 1টা batch test
# -------------------------------
for images, labels in train_loader:
    print("\n--- Train Batch ---")
    print("Image shape:", images.shape)
    print("Labels:", labels)
    break

# -------------------------------
# 8. Validation loader থেকে 1টা batch test
# -------------------------------
for images, labels in val_loader:
    print("\n--- Validation Batch ---")
    print("Image shape:", images.shape)
    print("Labels:", labels)
    break