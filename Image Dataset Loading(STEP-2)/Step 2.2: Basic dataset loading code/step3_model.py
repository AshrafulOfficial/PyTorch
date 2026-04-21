import torch
import torch.nn as nn
#from load_dataset import num_classes

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*56*56, num_classes)

#unnecessary learn houar somoy jaate print na hoy, tai print ghulo comment kore dicchi
    def forward(self, x):
#        print("Input Shape:", x.shape)
        x = self.conv1(x)

#        print("After Conv1:", x.shape)
        x = self.relu(x)
        x = self.pool(x)

#        print("After Pool1:", x.shape)
        x = self.conv2(x)

#        print("After Conv2:", x.shape)
        x = self.relu(x)
        x = self.pool(x)

#        print("After pool2:", x.shape)
        x = self.flatten(x)

#        print("After flatten:", x.shape)
        x = self.fc(x)

#        print("After fc:", x.shape)

        return x

#if __name__ == "__main__":
 #   model = SimpleCNN(num_classes)
  #  x = torch.randn(4, 3, 224, 224)
  #  output = model(x)
  #  print("Final output shape:", output.shape)

