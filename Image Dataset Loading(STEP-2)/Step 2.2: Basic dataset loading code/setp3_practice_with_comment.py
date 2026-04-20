# ================================
# Step 3: Simple CNN Model
# ================================

import torch
import torch.nn as nn


# -------------------------------
# CNN Model Definition
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()  
        # 👉 nn.Module-এর constructor call (mandatory)

        # -------------------------------
        # First Convolution Block
        # -------------------------------
        # Input: [B, 3, 224, 224]
        # Output: [B, 16, 224, 224]
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB image → 3 channels
            out_channels=16,    # 16 filters (feature maps)
            kernel_size=3,      # 3x3 filter
            padding=1           # output size same রাখে
        )

        # Activation Function
        self.relu = nn.ReLU()  
        # 👉 negative value → 0, positive → same

        # Pooling Layer
        self.pool = nn.MaxPool2d(2, 2)  
        # 👉 size half করে (224 → 112)

        # -------------------------------
        # Second Convolution Block
        # -------------------------------
        # Input: [B, 16, 112, 112]
        # Output: [B, 32, 112, 112]
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # -------------------------------
        # Flatten Layer
        # -------------------------------
        self.flatten = nn.Flatten()
        # 👉 [B, 32, 56, 56] → [B, 100352]

        # -------------------------------
        # Fully Connected Layer
        # -------------------------------
        self.fc = nn.Linear(
            in_features=32 * 56 * 56,  # Flatten output size
            out_features=num_classes   # Final class সংখ্যা
        )


    # -------------------------------
    # Forward Pass (Data Flow)
    # -------------------------------
    def forward(self, x):

        print("Input Shape:", x.shape)
        # 👉 [B, 3, 224, 224]

        # ---- Conv Block 1 ----
        x = self.conv1(x)
        print("After Conv1:", x.shape)
        # 👉 [B, 16, 224, 224]

        x = self.relu(x)
        x = self.pool(x)
        print("After Pool1:", x.shape)
        # 👉 [B, 16, 112, 112]

        # ---- Conv Block 2 ----
        x = self.conv2(x)
        print("After Conv2:", x.shape)
        # 👉 [B, 32, 112, 112]

        x = self.relu(x)
        x = self.pool(x)
        print("After Pool2:", x.shape)
        # 👉 [B, 32, 56, 56]

        # ---- Flatten ----
        x = self.flatten(x)
        print("After Flatten:", x.shape)
        # 👉 [B, 100352]

        # ---- Fully Connected ----
        x = self.fc(x)
        print("After FC:", x.shape)
        # 👉 [B, num_classes]

        return x


# -------------------------------
# Test Run (Dummy Input)
# -------------------------------
if __name__ == "__main__":

    # Model create (temporary 10 classes)
    model = SimpleCNN(num_classes=10)

    # Dummy input (batch=4, RGB image, 224x224)
    x = torch.randn(4, 3, 224, 224)

    # Forward pass
    output = model(x)

    # Final output shape
    print("Final output shape:", output.shape)