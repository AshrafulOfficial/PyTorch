# 🧩 PyTorch Dataset Loading & Split (Step 2) — Deep Explanation

## 🎯 Goal
এই অংশে আমরা শিখলাম কিভাবে:
- Image dataset load করতে হয়
- Transform apply করতে হয়
- Train / Validation split করতে হয়
- DataLoader দিয়ে batch তৈরি করতে হয়

👉 খুব সহজভাবে: **model-কে data খাওয়ানোর জন্য data ready করা**

---

# 🧠 Full Pipeline (Mind Picture)
```
Image Folder → Load → Resize → Tensor → Split → Batch → Ready for Model
```

---

# 🔷 1. Imports (Tools আনা)
```python
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
```

### 🧠 Deep Explanation
- `os` → file/folder আছে কিনা check করতে
- `datasets` → ready dataset loader (ImageFolder)
- `transforms` → image change করার rules
- `DataLoader` → batch machine (image group করে)
- `random_split` → dataset ভাগ করার tool

👉 analogy:
- dataset = বই
- DataLoader = বইয়ের bundle

---

# 🔷 2. Dataset Path (data কোথায়?)
```python
data_dir = "/home/ashraful/.../PlantVillage"
```

### 🧠 Explanation
👉 এখানে বলছি: "আমার data এই folder-এ আছে"

### ✔ Correct Structure
```
PlantVillage/
 ├── Tomato_diseased/
 │     ├── img1.jpg
 │
 ├── Tomato_healthy/
 │     ├── img2.jpg
```

👉 Folder name = class label

---

# 🔷 3. Transform (Image → Model-ready)
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

### 🧠 Step-by-step

### 1. Resize
সব image same size বানায়

👉 কেন?
Model fixed input চায়

### 2. ToTensor
image → number বানায়

👉 আগে:
- JPG image

👉 পরে:
- tensor (numbers)

👉 value range:
```
0–255 → 0–1
```

---

# 🔷 4. Dataset Load (Core step)
```python
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
```

### 🧠 Inside what happens?

👉 PyTorch internally:
1. folder scan করে
2. class detect করে
3. image load করে
4. label assign করে

👉 Example:
```
Tomato_diseased → 0
Tomato_healthy → 1
```

---

# 🔷 5. Dataset Info
```python
print(dataset.classes)
print(dataset.class_to_idx)
print(len(dataset))
```

### 🧠 Output meaning
```
['Tomato_diseased', 'Tomato_healthy']
```
👉 class list

```
{'Tomato_diseased': 0, 'Tomato_healthy': 1}
```
👉 mapping

```
4
```
👉 total images

---

# 🔷 6. Train / Validation Split
```python
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
```

### 🧠 Why split?
👉 model cheating বন্ধ করার জন্য

- Train → শেখা
- Validation → পরীক্ষা

### Example
Total = 4

→ Train = 2
→ Validation = 2

---

# 🔷 7. DataLoader (Batch Machine)
```python
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
```

### 🧠 Explanation

### batch_size
একবারে কয়টা image যাবে

### shuffle=True
👉 random order
👉 better learning

### shuffle=False
👉 validation-এ দরকার নেই

---

# 🔷 8. Batch Output
```python
for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break
```

### 🧠 Example Output
```
torch.Size([2, 3, 224, 224])
tensor([1, 0])
```

---

# 📊 Shape Breakdown
```
[batch, channel, height, width]
```

| Part | Meaning |
|------|--------|
| batch | কয়টা image |
| channel | RGB (3) |
| height | image height |
| width | image width |

---

# 🔥 Key Understanding

👉 এখন তোমার data এমন:
```
images → [batch, 3, 224, 224]
labels → [batch]
```

👉 এইটাই model-এর input হবে

---

# ❗ Very Important

👉 এটা model না
👉 এটা data preparation

---

# 🧠 Real Analogy

- Dataset = বই
- DataLoader = বইয়ের bundle
- Model = ছাত্র

👉 এখনো ছাত্র পড়া শুরু করে নাই 😄

---

# 🚀 Next Step

👉 Step 3: Model + Forward Pass

---

# 🎯 Final Summary

- ImageFolder → dataset
- Transform → preprocess
- Split → train/val
- DataLoader → batch
- Shape → structure
- Label → number

---

🔥 **You are now ready for Model Training**

