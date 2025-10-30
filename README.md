# Tiny-VGG Convolutional Neural Network on Fashion-MNIST

A compact convolutional neural network inspired by **VGG-like** architectures, trained on the **Fashion-MNIST** dataset for image classification tasks.

![Model Output](https://imgur.com/7rtsP5K.png)

---

## 📚 Overview

The **Tiny-VGG** is a simplified version of the classical VGG-16 network, designed to be lightweight and easy to train on small datasets such as **Fashion-MNIST**.

It aims to:
- Demonstrate the power of convolutional feature extraction.
- Provide an educational implementation for deep learning beginners.
- Achieve high accuracy with minimal parameters.

---

## ⚙️ Architecture

| Layer Type | Configuration | Output Shape |
|-------------|----------------|----------------|
| **Input** | 1 × 28 × 28 grayscale image | 1 × 28 × 28 |
| **Conv2D + ReLU** | 32 filters, 3×3 kernel, padding=1 | 32 × 28 × 28 |
| **Conv2D + ReLU** | 32 filters, 3×3 kernel, padding=1 | 32 × 28 × 28 |
| **MaxPool2D** | 2×2 | 32 × 14 × 14 |
| **Conv2D + ReLU** | 64 filters, 3×3 kernel, padding=1 | 64 × 14 × 14 |
| **Conv2D + ReLU** | 64 filters, 3×3 kernel, padding=1 | 64 × 14 × 14 |
| **MaxPool2D** | 2×2 | 64 × 7 × 7 |
| **Flatten** | — | 3136 |
| **Linear + ReLU** | 3136 → 128 | 128 |
| **Dropout(0.3)** | — | 128 |
| **Output (Linear)** | 128 → 10 | 10 classes |

---

## 🧩 Implementation (PyTorch)

```python
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self, in_channels: int, hidden_units: int, out_channels: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units * 2, hidden_units * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
