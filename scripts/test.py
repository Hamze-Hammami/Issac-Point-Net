#|--------------------------------------------------------|
#|                                                        |
#|                                                        |
#|                   __                                   |                                                                                                                                                                 
#|            ------|__|------                            |
#|               _.-|  |_          __                     |
#|.----.     _.-'   |/\| |        |__|0<                  |-------------------------------------------------------------------------------------------------------------|
#||    |__.-'____________|________|_|____ _____           |- Script: Test pointnet                                                                                      |
#||  .-""-."" |                       |   .-""-.""""--._  |- this script is an expirmnetal script built to tese a pointcloud model on a labled pcd file                 | 
#|'-' ,--. `  |                       |  ' ,--. `      _`.|- then compare the predicted cuboids with the actual labled cuboids                                          |
#| ( (    ) ) |                       | ( (    ) )--._".-.|- and might also not work due to fruquent API changes                                                        |
#|  . `--' ;\__________________..--------. `--' ;--------'|                                                                                                             |
#|   `-..-'                               `-..-'          |- By: Hamze Hammami                                                                                          |
#|                                                        |                                                                                                             |
#|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from train import PointNetDetector, load_labeled_pcd, extract_cuboids

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

global_bbox_index = 0

def test_pointnet_detector(pcd_file_path, model_path, output_road_mask_file, conf_threshold=0.5, nms_threshold=0.3, floor_z_threshold=-1.0):
    global global_bbox_index
    
    model = PointNetDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    points, labels = load_labeled_pcd(pcd_file_path)
    ground_truth_cuboids = extract_cuboids(points, labels)  

    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  

    with torch.no_grad():
        obj_scores, bbox_preds = model(points_tensor)
        obj_scores = obj_scores.cpu().numpy().squeeze()  
        bbox_preds = bbox_preds.cpu().numpy().squeeze()  

    mask = obj_scores >= conf_threshold
    obj_scores = obj_scores[mask]
    bbox_preds = bbox_preds[mask]
    predicted_points = points[mask]

 
    centers = predicted_points + bbox_preds[:, :3]
    sizes = bbox_preds[:, 3:]
    min_bounds = centers - sizes / 2
    max_bounds = centers + sizes / 2
    predicted_boxes = np.hstack((min_bounds, max_bounds))

    stacked_pred_boxes = stack_overlapping_boxes(predicted_boxes, nms_threshold)


    further_corrected_boxes = correct_bounding_boxes(points, stacked_pred_boxes, floor_z_threshold)


    old_corrected_boxes = correct_bounding_boxes(points, stacked_pred_boxes, floor_z_threshold=None)


    visualize_detection_results(points, predicted_boxes, stacked_pred_boxes, old_corrected_boxes, further_corrected_boxes, ground_truth_cuboids)


def stack_overlapping_boxes(boxes, nms_threshold):
    """
    Stack overlapping cuboids by merging them into a single cuboid.
    """
    merged_boxes = []
    
    while len(boxes) > 0:
        current_box = boxes[0]
        boxes = boxes[1:]


        overlap_mask = []
        for i, box in enumerate(boxes):
            if is_overlapping(current_box, box):
                current_box = merge_boxes(current_box, box)
                overlap_mask.append(i)

        boxes = np.delete(boxes, overlap_mask, axis=0)
        merged_boxes.append(current_box)

    return np.array(merged_boxes)

def is_overlapping(box1, box2):
    """
    Check if two cuboids (axis-aligned bounding boxes) overlap.
    """
    return not (box1[3] < box2[0] or box1[0] > box2[3] or
                box1[4] < box2[1] or box1[1] > box2[4] or
                box1[5] < box2[2] or box1[2] > box2[5])

def merge_boxes(box1, box2):
    """
    Merge two overlapping cuboids by creating a new cuboid that encompasses both.
    """
    min_bound = np.minimum(box1[:3], box2[:3])
    max_bound = np.maximum(box1[3:], box2[3:])
    return np.hstack((min_bound, max_bound))

def correct_bounding_boxes(points, stacked_boxes, floor_z_threshold):
    """
    Correct the bounding boxes by finding the min and max points inside each stacked box,
    optionally ignoring points that are below the floor_z_threshold on the Z-axis.
    This creates a tighter bounding box around the points within the stacked region.
    """
    corrected_boxes = []
    for box in stacked_boxes:

        min_bound = box[:3]
        max_bound = box[3:]
        if floor_z_threshold is not None:
            mask = (points[:, 2] > floor_z_threshold) & np.all((points >= min_bound) & (points <= max_bound), axis=1)
        else:
            mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)


        points_in_box = points[mask]
        if len(points_in_box) > 0:
            new_min_bound = np.min(points_in_box, axis=0)
            new_max_bound = np.max(points_in_box, axis=0)
            corrected_box = np.hstack((new_min_bound, new_max_bound))
            corrected_boxes.append(corrected_box)
        else:
            corrected_boxes.append(box) 

    return np.array(corrected_boxes)

def visualize_detection_results(points, pred_boxes, stacked_boxes, old_corrected_boxes, further_corrected_boxes, gt_boxes):
    """
    Visualize the full point cloud, and allow toggling between predicted, stacked, corrected (old and further), actual (ground truth), and combined (further + actual) bounding boxes using arrow keys.
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = np.array([0, 0, 0])
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(pcd)


    pred_bboxes = create_bboxes(pred_boxes, color=(1, 0, 0)) 
    stacked_bboxes = create_bboxes(stacked_boxes, color=(1, 1, 0)) 
    old_corrected_bboxes = create_bboxes(old_corrected_boxes, color=(0, 0.5, 0))  
    further_corrected_bboxes = create_bboxes(further_corrected_boxes, color=(0, 1, 0)) 
    gt_bboxes = create_bboxes(gt_boxes, color=(0, 0, 1))  

    combined_bboxes = further_corrected_bboxes + gt_bboxes

    all_bboxes = [pred_bboxes, stacked_bboxes, old_corrected_bboxes, further_corrected_bboxes, gt_bboxes, combined_bboxes]

    def update_bboxes(vis):
        global global_bbox_index

        vis.clear_geometries()
        vis.add_geometry(pcd)

   
        bboxes = all_bboxes[global_bbox_index]
        for bbox in bboxes:
            vis.add_geometry(bbox)
        return False

    def change_bbox(vis, direction):
        global global_bbox_index
        if direction == "next":
            global_bbox_index = (global_bbox_index + 1) % len(all_bboxes)
        elif direction == "prev":
            global_bbox_index = (global_bbox_index - 1) % len(all_bboxes)
        update_bboxes(vis)
        return False


    vis.register_key_callback(262, lambda vis: change_bbox(vis, "next")) 
    vis.register_key_callback(263, lambda vis: change_bbox(vis, "prev"))


    update_bboxes(vis)


    vis.run()
    vis.destroy_window()

def create_bboxes(boxes, color):
    """
    Create bounding box geometries from the given boxes and color them.
    """
    bboxes = []
    for box in boxes:
        min_bound = box[:3]
        max_bound = box[3:]
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = color
        bboxes.append(bbox)
    return bboxes


if __name__ == "__main__":
    test_pcd_file = "labeled_initial_scene.pcd"
    model_path = "pointnet_detector_200.pth"
    output_road_mask_file = "road_mask.pcd"
    floor_z_threshold = -1.0  
    test_pointnet_detector(test_pcd_file, model_path, output_road_mask_file, floor_z_threshold=floor_z_threshold)
