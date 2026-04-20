# SimpleCNN — সম্পূর্ণ ব্যাখ্যা (Bangla + English)

---

## পুরো code

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)

    x = torch.randn(4, 3, 224, 224)
    output = model(x)

    print("Output shape:", output.shape)
```

---

## ১) First Big Picture: এই code-এর কাজ কী?

এই code একটা simple CNN image classification model বানাচ্ছে।

মানে:

- **input:** image tensor
- model image-এর features detect করবে
- **output:** class scores দিবে

তোমার crop disease detection-এ later:

- input = leaf image
- output = healthy / diseased class

---

## ২) এই code কোন কোন topic থেকে এসেছে?

এই একটা code-এর ভিতরে অনেকগুলো PyTorch topic আছে:

| Topic | কী কাজ করে |
|---|---|
| Tensor | Data structure |
| nn.Module | Model base class |
| Conv2d | Feature extraction |
| ReLU | Activation function |
| MaxPool2d | Size reduction |
| Flatten | 4D → 2D conversion |
| Linear | Final classification |
| Forward pass | Data flow path |

---

## ৩) `import torch`

```python
import torch
```

Main PyTorch library import করছি।

`torch` দিয়ে আমরা করি:
- tensor বানাই
- random input বানাই
- shape দেখি
- operations করি

এই code-এ use হয়েছে:

```python
x = torch.randn(4, 3, 224, 224)
```

---

## ৪) `import torch.nn as nn`

```python
import torch.nn as nn
```

`torch.nn` module import করছি। `nn` মানে neural network components।

এর ভিতরে আছে:
- `nn.Module`
- `nn.Conv2d`
- `nn.ReLU`
- `nn.MaxPool2d`
- `nn.Flatten`
- `nn.Linear`

---

## ৫) `class SimpleCNN(nn.Module):`

```python
class SimpleCNN(nn.Module):
```

এর মানে:
- `SimpleCNN` নামে একটা model class বানাচ্ছি
- এটা `nn.Module` থেকে **inherit** করছে

### `nn.Module` কী?

এটা PyTorch-এর **base class for all models**।

কেন দরকার? কারণ এতে PyTorch automatically handle করে:
- model parameters
- layer registration
- training/eval mode
- save/load
- optimizer-এর জন্য weights access

> সহজভাবে: **nn.Module = model বানানোর standard framework**

---

## ৬) `def __init__(self, num_classes):`

```python
def __init__(self, num_classes):
```

এটা **constructor**। Model create হওয়ার সাথে সাথে এই অংশ run হয়।

এখানে model-এর **layers define** করি, অর্থাৎ model-এর body parts declare করি।

---

## ৭) `num_classes` কেন parameter হিসেবে নিলাম?

কারণ output class সংখ্যা fixed নাও হতে পারে।

- যদি dataset-এ 2 class: `Tomato_healthy`, `Tomato_diseased` → `num_classes = 2`
- যদি 10টা disease class → `num_classes = 10`

Model flexible রাখার জন্য বাইরে থেকে নিচ্ছি:

```python
num_classes = len(dataset.classes)
```

---

## ৮) `super().__init__()`

```python
super().__init__()
```

Parent class `nn.Module`-এর constructor call করে।

> **Beginner rule:** `nn.Module` inherit করলে `super().__init__()` অবশ্যই লিখবে।

---

## ৯) First Conv Block

```python
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
```

### `Conv2d` কী?

Convolutional layer। এটা image-এর উপর ছোট ছোট filter চালায় এবং feature বের করে।

### `nn.Conv2d(3, 16, kernel_size=3, padding=1)` breakdown

| Parameter | মান | মানে কী |
|---|---|---|
| `3` | in_channels | RGB image = 3 channels |
| `16` | out_channels | 16টা filter শিখবে |
| `kernel_size=3` | filter size | 3×3 window দিয়ে scan |
| `padding=1` | border padding | output size same রাখে |

**Shape effect:**

```
[B, 3, 224, 224]  →  [B, 16, 224, 224]
```

---

## ১০) `self.relu = nn.ReLU()`

```python
self.relu = nn.ReLU()
```

**Activation function।**

### ReLU rule:
- positive value → same রাখে
- negative value → 0 করে

```
[-2, 4, -1, 6]  →  [0, 4, 0, 6]
```

### কেন দরকার?

Activation ছাড়া network শুধু linear transformation করবে, যা খুব limited। ReLU model-এ **non-linearity** আনে, এতে complex pattern শিখতে পারে।

---

## ১১) `self.pool = nn.MaxPool2d(2, 2)`

```python
self.pool = nn.MaxPool2d(2, 2)
```

**Pooling layer।** Feature map-এর height-width কমিয়ে দেয়।

- `kernel size = 2`, `stride = 2`
- প্রতিটি 2×2 area থেকে **maximum value** নেয়

```
[[1, 5],   →  5
 [2, 3]]
```

### কেন pooling use করি?
- size কমে → computation কমে → memory কমে
- important strong features retain হয়
- edge device-এর জন্য ভালো

**Shape effect:**

```
[B, 16, 224, 224]  →  [B, 16, 112, 112]
```

---

## ১২) `self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)`

```python
self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
```

**Second convolution layer।**

- input channel = `16` কারণ `conv1` output দিয়েছিল 16 channels
- output channel = `32` কারণ deeper layer-এ more complex features দরকার

```
Before conv2:  [B, 16, 112, 112]
After conv2:   [B, 32, 112, 112]
After pool:    [B, 32, 56, 56]
```

---

## ১৩) `self.flatten = nn.Flatten()`

```python
self.flatten = nn.Flatten()
```

4D tensor → 2D tensor।

### কেন দরকার?

Convolution output 4D হয়, কিন্তু Linear layer 2D input চায়:

```
[B, 32, 56, 56]  →  [B, 32*56*56]  =  [B, 100352]
```

---

## ১৪) `self.fc = nn.Linear(32 * 56 * 56, num_classes)`

```python
self.fc = nn.Linear(32 * 56 * 56, num_classes)
```

**Final fully connected layer।**

### `32 * 56 * 56` কোথা থেকে এলো?

| Step | Shape |
|---|---|
| Input image | `[B, 3, 224, 224]` |
| After conv1 | `[B, 16, 224, 224]` |
| After pool1 | `[B, 16, 112, 112]` |
| After conv2 | `[B, 32, 112, 112]` |
| After pool2 | `[B, 32, 56, 56]` |
| After flatten | `[B, 100352]` |

So: `32 × 56 × 56 = 100352`

### `num_classes` কেন?

Final output-এ প্রতিটি class-এর জন্য একটা score দরকার:

```
num_classes=10  →  output shape = [B, 10]
num_classes=2   →  output shape = [B, 2]
```

---

## ১৫-২৩) `def forward(self, x):` — পুরো data flow

```python
def forward(self, x):
    x = self.conv1(x)    # [B, 3, 224, 224] → [B, 16, 224, 224]
    x = self.relu(x)     # shape same
    x = self.pool(x)     # [B, 16, 224, 224] → [B, 16, 112, 112]

    x = self.conv2(x)    # [B, 16, 112, 112] → [B, 32, 112, 112]
    x = self.relu(x)     # shape same
    x = self.pool(x)     # [B, 32, 112, 112] → [B, 32, 56, 56]

    x = self.flatten(x)  # [B, 32, 56, 56]  → [B, 100352]
    x = self.fc(x)       # [B, 100352]       → [B, num_classes]

    return x
```

> **Note:** `return x` এখনো final label না — এটা raw scores / **logits**। Training-এ `CrossEntropyLoss` এই logits use করবে।

---

## ২৪) পুরো Data Flow একনজরে

```
Input image:   [4, 3,  224, 224]
Conv1:         [4, 16, 224, 224]
ReLU:          [4, 16, 224, 224]
Pool:          [4, 16, 112, 112]
Conv2:         [4, 32, 112, 112]
ReLU:          [4, 32, 112, 112]
Pool:          [4, 32,  56,  56]
Flatten:       [4, 100352]
Linear:        [4, 10]
```

---

## ২৫) Test অংশ

```python
if __name__ == "__main__":
```

এর মানে: এই file direct run করলে নিচের code execute হবে, কিন্তু অন্য file থেকে import করলে হবে না।

```python
model = SimpleCNN(num_classes=10)
x = torch.randn(4, 3, 224, 224)  # dummy input — real image না, শুধু shape test
output = model(x)
print("Output shape:", output.shape)
# Expected: torch.Size([4, 10])
```

---

## ২৬) Simple Intuition দিয়ে বোঝা

ধরো model একটা **disease inspector:**

| Layer | কাজ |
|---|---|
| **Conv1** | Basic feature দেখে — edge, color, small pattern |
| **Conv2** | Meaningful pattern দেখে — spot, damaged texture, infection |
| **Pool** | Important parts রেখে size ছোট করে |
| **Flatten** | সব information এক line-এ সাজায় |
| **FC** | Final decision নেয়: কোন class? |

---

## ২৭) Common Confusion Clear

| প্রশ্ন | উত্তর |
|---|---|
| conv1 আর conv2 কি manually feature detect করে? | না। Training-এ filters নিজেরাই শেখে। |
| Linear final class দেয়? | না, raw scores দেয়। |
| ReLU কেন একটাই variable? | Same activation বারবার use করা যায়। |
| Same pool বারবার use করা যায়? | হ্যাঁ, same pooling layer আবার use করা যায়। |

---

## ২৮) এই Architecture-এর Limitations

এই model শেখার জন্য ভালো, কিন্তু thesis-এর final model হিসেবে best না:

- `32 * 56 * 56` অনেক বড় feature vector
- Linear layer heavy হয়ে যায়
- CPU / low RAM-এ training slow
- Edge deployment-এর জন্য **MobileNetV2 better**

> **Learning bridge:** SimpleCNN → CNN concept বুঝবে → তারপর MobileNetV2

---

## ২৯) Thesis-এর সাথে Connection

তোমার thesis focus: energy efficiency, edge deployment, low compute, pruning, quantization।

এই code তোমাকে foundation দিচ্ছে:
- CNN কী
- image shape কীভাবে যায়
- conv/pool কী
- model structure কী

> এগুলো না বুঝলে MobileNetV2 বোঝা কঠিন হবে।

---

## ৩০) Exercise — নিজে করো

### Exercise 1: Shape Print করো

```python
def forward(self, x):
    print("Input:", x.shape)
    x = self.conv1(x)
    print("After conv1:", x.shape)
    x = self.relu(x)
    x = self.pool(x)
    print("After pool1:", x.shape)

    x = self.conv2(x)
    print("After conv2:", x.shape)
    x = self.relu(x)
    x = self.pool(x)
    print("After pool2:", x.shape)

    x = self.flatten(x)
    print("After flatten:", x.shape)

    x = self.fc(x)
    print("After fc:", x.shape)

    return x
```

### Exercise 2: নিজে Answer করো

1. কেন input channels = 3?
2. কেন conv2-এর input = 16?
3. কেন pooling size half করে?
4. flatten না দিলে কী problem?
5. num_classes যদি 2 হয়, output shape কী হবে?

---

## ৩১) Final Version (Shape Print সহ)

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        print("Input shape:", x.shape)

        x = self.conv1(x)
        print("After conv1:", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool1:", x.shape)

        x = self.conv2(x)
        print("After conv2:", x.shape)

        x = self.relu(x)
        x = self.pool(x)
        print("After pool2:", x.shape)

        x = self.flatten(x)
        print("After flatten:", x.shape)

        x = self.fc(x)
        print("After fc:", x.shape)

        return x


if __name__ == "__main__":
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print("Final output shape:", output.shape)
```

---

## ৩২) Must Know Topics Summary

| Category | Topics |
|---|---|
| **Tensor** | shape, batch dimension, channels, image size |
| **Model** | nn.Module, model class, __init__, forward |
| **Layers** | Conv2d, ReLU, MaxPool2d, Flatten, Linear |
| **Concept** | feature extraction, non-linearity, pooling, classification |

---

*পরের ধাপে: Conv2d exactly কীভাবে image scan করে — visual intuition সহ।*
