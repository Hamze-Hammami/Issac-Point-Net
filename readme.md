# Isaac Point Cloud 🚀

## Overview  
Isaac Point Cloud is an open-source project that provides a structured approach to:
- **📡 Capturing point cloud data** in Isaac Sim using Physics LiDAR.
- **🏷️ Auto-labeling objects** lable instances by adding themin groups.
- **🤖 Training a PointNet-based model** for object detection and segmentation.
- **🔍 Visualizing** the resulting point clouds and bounding boxes.

This repository serves as a **concept proposal** rather than an actively developed project. Contributions and research from the community are welcome.

- **Scene:**![image](https://github.com/user-attachments/assets/807031e2-7b27-4cd2-911d-891227e507f6)
- **PCD DATA (from ISSAC-sim):**![image](https://github.com/user-attachments/assets/d7def4fe-d3bb-4376-be91-b529dda42833)
- **Labeled Data:** ![image](https://github.com/user-attachments/assets/98d2182c-5fe8-4e66-a5a0-45f4d5e87b6d)
- **Prediction:**![image](https://github.com/user-attachments/assets/0c2fcec1-636a-429b-a46b-1c0904eb326b)
- **Processed Prediction:**![image](https://github.com/user-attachments/assets/5e27b23d-85ea-49fb-9ae8-60dafdac42e7)
- **predict (Green) Vs Lable (Blue):**![image](https://github.com/user-attachments/assets/ad8badab-2e8c-43ba-a865-d7113fc512b4)




---

## Features 🏆

### 1 Physics LiDAR-Based Point Cloud Capture  

### 2 Auto-Labeling of Simulation Data  

### 3 Training a PointNet-Based Model  

## Prposed Goals 🔍

### 1 live data prediction  

### 2 class handeling (curruntly only instances / 1 class)




## Data Formats

### PCD v0.7 Format  
Point cloud files are saved using the PCD v0.7 format. An example header is as follows:

```
.PCD v0.7 - Point Cloud Data file format
FIELDS x y z instance
SIZE 4 4 4 4
TYPE F F F I
COUNT 1 1 1 1
WIDTH N
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS N
DATA ascii
```

- **x, y, z:** The 3D coordinates for each point.
- **instance:** An integer representing the object instance label (e.g., background is labeled as `0`).

---

## Pretrained Example weights from Hugging Face 🤗

### How to Pull the Example Model  
A pretrained PointNet-based model is available on Hugging Face to help jump-start experimentation. You can download it using either of the following methods:

#### Option 1: Clone the Repository
```bash
git clone https://huggingface.co/Hamze-Hammami/ISSAC-PointNet
```

#### Option 2: Download from Hugging Face Model Hub
[Download the Model Here](https://huggingface.co/Hamze-Hammami/ISSAC-PointNet/tree/main/models)

#### Option 3: Install via Hugging Face Hub
```bash
pip install huggingface_hub
```

Then, download and load the model:
```python
from huggingface_hub import hf_hub_download
import torch
from train import PointNetDetector  # Ensure your PointNetDetector is defined as in this repo

# Download the pretrained model from Hugging Face Hub
model_path = hf_hub_download(repo_id="Hamze-Hammami/ISSAC-PointNet", filename="pointnet_detector.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the model
model = PointNetDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Pretrained PointNetDetector model loaded successfully!")
```

Source: [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)

---

## Training, Testing, and Visualization Scripts

### 🏋️ Training the PointNet Model
The training script can be used to fine-tune a PointNet-based model on labeled point cloud data. The current setup has been tested on a limited dataset of five PCD files.

Run the training script with:
```bash
python train.py
```

### 🧪 Testing the Model on a Labeled PCD File
To evaluate the trained model and compare the predicted cuboids with the actual labeled cuboids, run:
```bash
python test.py
```

The script will load a labeled PCD file, pass it through the model, and visualize the predicted bounding boxes.

### 🎨 Visualizing Point Cloud Data
An experimental script for visualizing labeled point cloud data is provided. This script supports toggling between predicted, merged, and corrected bounding boxes:
```bash
python visualise_pcd.py
```

---

## Author ✍️
Developed by **Hamze Hammami**. Contributions and improvements are welcome!
