# Isaac Point Net 🚀

## Overview  
Isaac Point Net is an open-source project that provides a structured approach to 3D pixel and cuboid training machine learning model with pytorch and issac sim:


### **Scene & PCD Data (through ISSAC)**

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/user-attachments/assets/807031e2-7b27-4cd2-911d-891227e507f6" alt="Scene" width="50%">
    <img src="https://github.com/user-attachments/assets/d7def4fe-d3bb-4376-be91-b529dda42833" alt="PCD Data" width="50%">
</div>

- **📡 Capturing point cloud data** in Isaac Sim using Physics LiDAR.

  
 ### **Labeled Data:**  
![Labeled Data](https://github.com/user-attachments/assets/98d2182c-5fe8-4e66-a5a0-45f4d5e87b6d)

- **🏷️ Auto-labeling objects** lable instances by adding themin groups.
  
### **Model Architecture** 
<img width="1073" alt="Drawing5" src="https://github.com/user-attachments/assets/9bc6ae73-745c-4bcd-b6bb-d9c36fc41872" alt="PCD Data" width="80%">

- **🏗️ Training a Model based ona PointNet Architecture** for object detection and segmentation.

### **Prediction vs Processed Prediction**

<div style="display: flex; justify-content: center; gap: 20px;">
    <img src="https://github.com/user-attachments/assets/0c2fcec1-636a-429b-a46b-1c0904eb326b" alt="Scene" width="45%">
    <img src="https://github.com/user-attachments/assets/5e27b23d-85ea-49fb-9ae8-60dafdac42e7" alt="PCD Data" width="45%">
</div>

- **☁️ processing point cloud** processing predction to enahnce object accuracy.

This repository serves as a **concept proposal** rather than an actively developed project. Contributions and research from the community are welcome.


---

## Features 🏆

1- Physics LiDAR-Based Point Cloud Capture ( through ISSAC ) 

2- Auto-Labeling of Simulation Data  

3- Training a PointNet-Based Model  

## Prposed Goals 🔍

1- live data prediction  

2- class handeling (curruntly only instances / 1 class)




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

```

Source: [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)

---

## auto-Labeling, Training, Testing, and Visualization Scripts


### 📝 auto-Labeling PointCloud data through issac 
This script is an experimental implementation for capturing and labeling cones in a simulated Isaac Sim environment. The script utilizes a simulated 3D LiDAR sensor to extract and label point cloud data.

Running via Script Editor (Isaac Sim)
To run the script directly within Isaac Sim, follow these steps:
1. Open **Isaac Sim** and load your scene.
2. Navigate to **Window → Script Editor**.
3. open the script scripts/ISSAC_LABEL_PCD.py, Copy and paste the script into the editor and run it 
4. Click **Run** to execute the script.

currently script captures static scenes, finding a way to capture live scenes would speed up operations


### 🏋️ Training the PointNet Model
The training script can be used to Train a PointNet-based model on labeled point cloud data. The current setup has been tested on a limited dataset of five PCD files.

Run the training script with:
```bash
python scripts/train.py
```

### 🧪 Testing the Model on a Labeled PCD File
To evaluate the trained model and compare the predicted cuboids with the actual labeled cuboids, run:
```bash
python scripts/test.py
```


### 🎨 Visualizing Point Cloud Data
An experimental script for visualizing labeled point cloud data is provided. This script supports toggling between predicted, merged, and corrected bounding boxes:
```bash
python scripts/visualise_pcd.py
```

---

## Author ✍️
Developed by **@Hamze-Hammami**. Contributions and improvements are welcome!


