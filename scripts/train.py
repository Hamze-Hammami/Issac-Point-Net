#|--------------------------------------------------------|
#|                                                        |
#|                                                        |
#|                   __                                   |                                                                                                                                                                 
#|            ------|__|------                            |
#|               _.-|  |_          __                     |
#|.----.     _.-'   |/\| |        |__|0<                  |
#||    |__.-'____________|________|_|____ _____           |-------------------------------------------------------------------------------------------------------------|
#||  .-""-."" |                       |   .-""-.""""--._  |- Script: train pointnet                                                                                     | 
#|'-' ,--. `  |                       |  ' ,--. `      _`.|- this script is an expirmnetal script meant for training point cloud data, curruntly tested on 5 pcd points |
#| ( (    ) ) |                       | ( (    ) )--._".-.|-  max, does not garantuee a working model as many background and full scene scenes were not tested          |
#|  . `--' ;\__________________..--------. `--' ;--------'|                                                                                                             |
#|   `-..-'                               `-..-'          |- By: Hamze Hammami                                                                                          |
#|                                                        |                                                                                                             |
#|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions to load PCD and process data
def load_labeled_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    labels = []
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            if line.startswith('DATA'):
                is_data_section = True
                continue
            if is_data_section:
                parts = line.split()
                if len(parts) == 4:
                    labels.append(int(parts[3]))
    labels = np.array(labels)
    return points, labels

def extract_cuboids(points, labels):
    cuboids = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == 0:
            continue  # Skip background
        label_points = points[labels == label]
        min_bound = np.min(label_points, axis=0)
        max_bound = np.max(label_points, axis=0)
        cuboid = np.hstack((min_bound, max_bound))
        cuboids.append(cuboid)
    return np.array(cuboids)

class PointCloudDetectionDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = []
        for file_path in self.file_paths:
            points, labels = load_labeled_pcd(file_path)
            cuboids = extract_cuboids(points, labels)
            self.data.append((points, labels, cuboids))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        points, labels, cuboids = self.data[idx]
        return points, labels, cuboids


def detection_collate_fn(batch):
    """
    Collate function for object detection dataset.
    """
    max_num_points = max([len(points) for points, _, _ in batch])
    
    point_clouds = []
    labels_list = []
    bbox_targets = []
    
    for points, labels, cuboids in batch:
        num_points = len(points)
        # Pad point clouds
        padded_points = np.pad(points, ((0, max_num_points - num_points), (0, 0)), mode='constant')
        point_clouds.append(padded_points)
        
        # Objectness labels: 1 if point belongs to an object, 0 otherwise
        obj_labels = np.zeros(num_points)
        bbox_label = np.zeros((num_points, 6))
        
        for cuboid in cuboids:
            min_bound = cuboid[:3]
            max_bound = cuboid[3:]
            center = (min_bound + max_bound) / 2
            size = max_bound - min_bound
            # Find points within this cuboid
            in_bbox = np.all((points >= min_bound) & (points <= max_bound), axis=1)
            obj_labels[in_bbox] = 1
            bbox_label[in_bbox, :3] = center - points[in_bbox]  # Center offset
            bbox_label[in_bbox, 3:] = size  # Size
        
        # Pad labels
        obj_labels = np.pad(obj_labels, (0, max_num_points - num_points), mode='constant')
        bbox_label = np.pad(bbox_label, ((0, max_num_points - num_points), (0, 0)), mode='constant')
        
        labels_list.append(obj_labels)
        bbox_targets.append(bbox_label)
    
    # Convert to tensors
    point_clouds = torch.tensor(point_clouds, dtype=torch.float32).permute(0, 2, 1)  # (batch_size, 3, num_points)
    labels_list = torch.tensor(labels_list, dtype=torch.float32)  # (batch_size, num_points)
    bbox_targets = torch.tensor(bbox_targets, dtype=torch.float32)  # (batch_size, num_points, 6)
    
    return point_clouds, labels_list, bbox_targets



### PointNet Model Definition ###

class PointNetDetector(nn.Module):
    """
    PointNet model for 3D object detection with per-point predictions.
    """
    def __init__(self):
        super(PointNetDetector, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        
        self.conv_obj = nn.Conv1d(1024, 1, 1)  
        self.conv_bbox = nn.Conv1d(1024, 6, 1) 
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, _, num_points = x.size()
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))  
        
    
        obj_scores = self.sigmoid(self.conv_obj(x))  
        

        bbox_preds = self.conv_bbox(x)  
        
        return obj_scores.squeeze(1), bbox_preds.permute(0, 2, 1)  



### Training Loop ###

def train_pointnet_detector(pcd_file_paths, epochs=10, batch_size=4, lr=0.001):
    # Load dataset and create DataLoader
    dataset = PointCloudDetectionDataset(pcd_file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn)
    
    # Initialize the model, loss functions, and optimizer
    model = PointNetDetector().to(device)
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (points, obj_labels, bbox_targets) in enumerate(dataloader):
            points = points.to(device)  # (batch_size, 3, num_points)
            obj_labels = obj_labels.to(device)  # (batch_size, num_points)
            bbox_targets = bbox_targets.to(device)  # (batch_size, num_points, 6)
            
            optimizer.zero_grad()
            
            # Forward pass
            obj_scores, bbox_preds = model(points)
            
            # Objectness loss
            loss_cls = criterion_cls(obj_scores, obj_labels)
            
            # Bounding box regression loss (only for positive samples)
            positive_mask = obj_labels > 0
            if positive_mask.sum() > 0:
                loss_reg = criterion_reg(bbox_preds[positive_mask], bbox_targets[positive_mask])
            else:
                loss_reg = torch.tensor(0.0).to(device)
            
            loss = loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item()}')
        
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {running_loss / len(dataloader)}')
    
    # Save the trained model
    torch.save(model.state_dict(), 'pointnet_detector.pth')
    print("Model saved as 'pointnet_detector.pth'.")

def test_pointnet_detector(pcd_file_path, model_path, conf_threshold=0.5, nms_threshold=0.3):
    # Load the trained model
    model = PointNetDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load the test data
    points, labels = load_labeled_pcd(pcd_file_path)
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 3, N)
    
    with torch.no_grad():
        obj_scores, bbox_preds = model(points_tensor)
        obj_scores = obj_scores.cpu().numpy().squeeze()  # (N,)
        bbox_preds = bbox_preds.cpu().numpy().squeeze()  # (N, 6)
    
    # Apply confidence threshold
    mask = obj_scores >= conf_threshold
    obj_scores = obj_scores[mask]
    bbox_preds = bbox_preds[mask]
    points = points[mask]
    
    # Convert bbox predictions to absolute coordinates
    centers = points + bbox_preds[:, :3]
    sizes = bbox_preds[:, 3:]
    min_bounds = centers - sizes / 2
    max_bounds = centers + sizes / 2
    boxes = np.hstack((min_bounds, max_bounds))
    
    # Apply Non-Maximum Suppression (NMS)
    keep_indices = nms_3d(boxes, obj_scores, nms_threshold)
    boxes = boxes[keep_indices]
    obj_scores = obj_scores[keep_indices]
    
    # Visualize results
    visualize_detection_results(points, boxes)


def nms_3d(boxes, scores, iou_threshold):
    """
    Custom implementation of 3D Non-Maximum Suppression (NMS).
    """
    # If there are no boxes, return an empty array
    if len(boxes) == 0:
        return []

    # Convert boxes and scores to PyTorch tensors if they are not already
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)

    # Compute the volume of each box
    volumes = (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])

    # Sort scores in descending order and get the sorted indices
    _, order = scores.sort(descending=True)

    # List to keep track of boxes to keep
    keep = []

    while order.numel() > 0:
        # Pick the box with the highest score
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # Get the coordinates of the boxes with the highest score
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        zz1 = torch.max(boxes[i, 2], boxes[order[1:], 2])
        xx2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        yy2 = torch.min(boxes[i, 4], boxes[order[1:], 4])
        zz2 = torch.min(boxes[i, 5], boxes[order[1:], 5])

        # Compute the dimensions of the overlap region
        w = torch.clamp(xx2 - xx1, min=0)
        h = torch.clamp(yy2 - yy1, min=0)
        d = torch.clamp(zz2 - zz1, min=0)

        # Compute the volume of the overlap
        overlap = w * h * d

        # Compute the Intersection over Union (IoU)
        iou = overlap / (volumes[i] + volumes[order[1:]] - overlap)

        # Keep boxes with IoU less than the threshold
        order = order[1:][iou < iou_threshold]

    return keep
def visualize_detection_results(points, boxes):
    """
    Visualize point cloud and detected bounding boxes.
    """
    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Create bounding boxes
    bboxes = []
    for box in boxes:
        min_bound = box[:3]
        max_bound = box[3:]
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = (0, 1, 0)  # Green for predictions
        bboxes.append(bbox)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd] + bboxes)



### Main Function ###

if __name__ == "__main__": # for a more robust implmentaion make a fucntion to read all files from a specfied dir 
    pcd_file_paths = [
        "PCD_Train_Data\labeled_initial_scene1.pcd",
        "PCD_Train_Data\labeled_initial_scene2.pcd",
        "PCD_Train_Data\labeled_initial_scene3.pcd",
        "PCD_Train_Data\labeled_initial_scene4.pcd",
        "PCD_Train_Data\labeled_initial_scene5.pcd",
    ]
    # Train the model
    train_pointnet_detector(pcd_file_paths, epochs=500, batch_size=1, lr=0.0001)
    

