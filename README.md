# Brain MRI Skull-Stripping using 3D U-Net

<p align="left">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

This repository contains a pre-trained **3D U-Net** model for **Brain MRI Skull-Stripping** along with the training and architecture implementation notebook.

The project focuses on medical image segmentation for removing non-brain tissues from MRI scans using deep learning.

---

# Model Information

| Model File                                  | Application               | Architecture | Input Shape         | Loss Function  |
| :------------------------------------------ | :------------------------ | :----------- | :------------------ | :------------- |
| `skull_stripping_unet_3d_soft_dice_loss.h5` | Brain MRI Skull-Stripping | 3D U-Net     | `(128, 128, 96, 1)` | Soft Dice Loss |

---

# Architecture Overview

The model is based on a **3D U-Net** architecture designed for volumetric medical image segmentation.

## Key Components

* **Encoder Path**

  * 3D convolutional layers
  * Feature extraction using hierarchical representations
  * Max pooling for spatial downsampling

* **Bottleneck Layer**

  * Deep compressed feature representation
  * Captures contextual information from MRI volumes

* **Decoder Path**

  * 3D transposed convolutions
  * Skip connections for preserving spatial details
  * Reconstruction of segmentation masks

* **Optimization**

  * Soft Dice Loss for segmentation accuracy
  * Designed for handling class imbalance in medical imaging

---

# Training & Architecture Notebook

The complete model architecture and training pipeline are available in:

* [brain-mri-segmentation.ipynb](https://github.com/FirdowsRahaman/Saved-Model/blob/master/brain-mri-segmentation.ipynb)

The notebook includes:

* MRI preprocessing
* Data loading
* 3D U-Net architecture definition
* Model training
* Loss function implementation
* Evaluation pipeline
* Model saving

---

# Getting Started

## Prerequisites

Install the required dependencies:

```bash
pip install tensorflow numpy nibabel pillow
```

---

# Repository Structure

```text
Saved-Model/
│
├── brain-mri-segmentation.ipynb
├── skull_stripping_unet_3d_soft_dice_loss.h5
└── README.md
```

---

# Load the Pre-trained Model

```python
from tensorflow.keras.models import load_model

model = load_model(
    "skull_stripping_unet_3d_soft_dice_loss.h5",
    compile=False
)

model.summary()
```

---

# Input Specifications

| Parameter          | Value               |
| :----------------- | :------------------ |
| Input Volume Shape | `(128, 128, 96, 1)` |
| Data Type          | MRI Volumes         |
| Output             | Binary Brain Mask   |
| Framework          | TensorFlow / Keras  |

---

# Use Cases

* Brain extraction from MRI scans
* Neuroimaging preprocessing
* Medical image segmentation research
* Deep learning experimentation in radiology

---

# References

* U-Net: Convolutional Networks for Biomedical Image Segmentation
* TensorFlow
* Keras
* Medical Imaging Deep Learning Research

---

# ‍ Author

**Firdows Rahaman**

GitHub Repository:

* [https://github.com/FirdowsRahaman/Saved-Model](https://github.com/FirdowsRahaman/Saved-Model)
