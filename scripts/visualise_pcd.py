#|--------------------------------------------------------|
#|                                                        |
#|                                                        |
#|                   __                                   |                                                                                                                                                                 
#|            ------|__|------                            |
#|               _.-|  |_          __                     |
#|.----.     _.-'   |/\| |        |__|0<                  |
#||    |__.-'____________|________|_|____ _____           |-------------------------------------------------------------------------------------------------------------|
#||  .-""-."" |                       |   .-""-.""""--._  |- Script: visualise PCD                                                                                      | 
#|'-' ,--. `  |                       |  ' ,--. `      _`.|- this script is an expirmnetal script meant for vislaising pre labled point cloud data.                     |
#| ( (    ) ) |                       | ( (    ) )--._".-.|- folowing a PCD v0.7 labled data with following Field (x y z instance)                                      |
#|  . `--' ;\__________________..--------. `--' ;--------'|                                                                                                             |
#|   `-..-'                               `-..-'          |- By: Hamze Hammami                                                                                          |
#|                                                        |                                                                                                             |
#|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|


# IMPORTANT: make sure to change the file names in  line 26

import open3d as o3d
import numpy as np
import os

# Path to the labeled point cloud file
# change file name  
file_path = '<file>.pcd' 

# Function to load a PCD file with instance labels
def load_labeled_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)  # Load the point cloud
    points = np.asarray(pcd.points)  # Extract the 3D points

    # Read the PCD file manually to extract the instance labels
    labels = []
    with open(file_path, 'r') as f:
        is_data_section = False
        for line in f:
            if line.startswith('DATA'):
                is_data_section = True
                continue
            
            if is_data_section:
                parts = line.split()
                if len(parts) == 4:  # x, y, z, instance_id
                    labels.append(int(parts[3]))  # Append instance label

    labels = np.array(labels)  # Convert to a numpy array
    return points, labels

# Function to compute the minimum distance between two bounding boxes
def compute_bbox_distance(bbox1, bbox2):
    # Get the center of each bounding box
    center1 = bbox1.get_center()
    center2 = bbox2.get_center()
    
    # Calculate the Euclidean distance between the centers
    distance = np.linalg.norm(center1 - center2)
    
    return distance

# Function to merge two bounding boxes
def merge_bounding_boxes(bbox1, bbox2):
    # Compute the new min and max bounds by combining both bounding boxes
    min_bound = np.minimum(bbox1.min_bound, bbox2.min_bound)
    max_bound = np.maximum(bbox1.max_bound, bbox2.max_bound)
    
    # Create a new bounding box that encompasses both
    merged_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return merged_bbox

# Function to create and merge close bounding boxes and 3D markers for each instance label
def create_and_merge_bounding_boxes(points, labels, distance_threshold=0.5):
    unique_labels = np.unique(labels)
    bounding_boxes = []
    label_markers = []

    for label in unique_labels:
        if label == 0:
            # Skip the initial scene (label 0)
            continue

        # Get points associated with the current label
        label_points = points[labels == label]

        # Compute the min and max bounds for the points
        min_bound = np.min(label_points, axis=0)
        max_bound = np.max(label_points, axis=0)

        # Create an axis-aligned bounding box (AABB)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        # Assign a random color to each bounding box
        bbox.color = np.random.rand(3)
        bounding_boxes.append(bbox)

    # Merge close bounding boxes
    merged_boxes = []
    used_boxes = set()

    for i, bbox1 in enumerate(bounding_boxes):
        if i in used_boxes:
            continue
        merged_bbox = bbox1
        for j, bbox2 in enumerate(bounding_boxes):
            if j <= i or j in used_boxes:
                continue
            distance = compute_bbox_distance(merged_bbox, bbox2)
            if distance < distance_threshold:
                merged_bbox = merge_bounding_boxes(merged_bbox, bbox2)
                used_boxes.add(j)  # Mark the second box as used
        merged_boxes.append(merged_bbox)
        used_boxes.add(i)

    # Create label markers (3D spheres) for the merged boxes
    for idx, bbox in enumerate(merged_boxes):
        marker_position = bbox.get_center() + np.array([0, 0, bbox.get_extent()[2] / 2 + 0.2])
        sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere_marker.translate(marker_position)
        sphere_marker.paint_uniform_color(np.random.rand(3))  # Assign a random color to each marker
        label_markers.append(sphere_marker)

    return merged_boxes, label_markers

def main():
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    # Load points and labels
    points, labels = load_labeled_pcd(file_path)

    # Create a colored point cloud based on labels
    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(points)

    # Create and merge close bounding boxes and label markers (3D spheres)
    distance_threshold = 0.5  # Define the threshold for stacking/merging bounding boxes
    merged_boxes, label_markers = create_and_merge_bounding_boxes(points, labels, distance_threshold)

    # Create the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set a dark background color (black)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0, 0, 0])  # Set background to black

    # Add point cloud, bounding boxes, and 3D markers
    vis.add_geometry(labeled_pcd)
    for bbox in merged_boxes:
        vis.add_geometry(bbox)
    for marker in label_markers:
        vis.add_geometry(marker)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
