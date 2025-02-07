#|--------------------------------------------------------|
#|                                                        |
#|                                                        |
#|                   __                                   |                                                                                                                                                                 
#|            ------|__|------                            |
#|               _.-|  |_          __                     |
#|.----.     _.-'   |/\| |        |__|0<                  |-------------------------------------------------------------------------------------------------------------|
#||    |__.-'____________|________|_|____ _____           |- Script: Label Issac                                                                                        |
#||  .-""-."" |                       |   .-""-.""""--._  |- this script is an expirmnetal script built for an ISSAC sim scene which has access to a Simulated 3D Lidar | 
#|'-' ,--. `  |                       |  ' ,--. `      _`.|- for capturing Cones on a simulation, the code will need some fine tuning to work for you scenes on issac   |
#| ( (    ) ) |                       | ( (    ) )--._".-.|- and might also not work due to fruquent API changes                                                        |
#|  . `--' ;\__________________..--------. `--' ;--------'|- comments are unavilabel at the momnets                                                                     |
#|   `-..-'                               `-..-'          |                                                                                                             |
#|                                                        |- By: Hamze Hammami                                                                                          |
#|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

import omni
import asyncio
import os
from omni.isaac.range_sensor import _range_sensor
import numpy as np
from pxr import UsdGeom, UsdPhysics, Gf, Usd

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()
lidarInterface = _range_sensor.acquire_lidar_sensor_interface()

max_range_threshold = 50.0  
lidarPath = "/Group/CAR/Lidar" 
ads_cad_path = "/Group/CAR/FS_AI_ADS_DV_CAD"
cube_path = "/Cube"

base_file_path = 'C:/Users/hamze/OneDrive/Desktop/PointClouds'
if not os.path.exists(base_file_path):
    os.makedirs(base_file_path)

def toggle_colliders(paths, state):
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)
            enabled_attr = collisionAPI.GetCollisionEnabledAttr()
            if not enabled_attr:
                enabled_attr = collisionAPI.CreateCollisionEnabledAttr()
            enabled_attr.Set(state)

            rigidBodyAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
            rigidBodyAPI.CreateKinematicEnabledAttr().Set(True)
            print(f"{path}: Collider {'enabled' if state else 'disabled'} and set as kinematic")

def modify_translate_op(prim_path, move_out=True):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        xform = UsdGeom.Xformable(prim)
        translate_ops = xform.GetOrderedXformOps()
        
        for op in translate_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                if move_out:
                    op.Set(Gf.Vec3f(1000.0, 1000.0, 1000.0))  
                    print(f"{prim_path} moved out of scene.")
                else:
                    op.Set(Gf.Vec3f(0.0, 0.0, 0.0))  
                    print(f"{prim_path} restored to original position.")
                return
        xform.AddTranslateOp().Set(Gf.Vec3f(1000.0, 1000.0, 1000.0))  
        print(f"Added translate operation for {prim_path} and moved out.")

async def capture_lidar_data(lidar_path):
    await asyncio.sleep(1.0)  
    timeline.pause()  
    pointcloud = lidarInterface.get_point_cloud_data(lidar_path)
    
    if pointcloud is not None and len(pointcloud) > 0:
        pc_data = np.array(pointcloud).reshape(-1, 3)
        distances = np.linalg.norm(pc_data, axis=1)
        valid_points = pc_data[distances <= max_range_threshold]
    else:
        valid_points = np.array([])  
        print("Point cloud data is None or empty.")
    
    timeline.play()
    return valid_points  

def label_cone_points_in_initial_pc(initial_pc, cone_pc, labels, instance_id):
    tolerance = 1e-5  
    for cone_point in cone_pc:
        distances = np.linalg.norm(initial_pc - cone_point, axis=1)
        indices = np.where(distances < tolerance)[0]
        labels[indices] = instance_id
    return labels

def save_labeled_point_cloud(pc_data, labels, file_name):
    file_path = f'{base_file_path}/{file_name}'
    with open(file_path, 'w') as file:
        file.write("# .PCD v0.7 - Point Cloud Data file format\n")
        file.write("VERSION 0.7\n")
        file.write("FIELDS x y z instance\n")
        file.write("SIZE 4 4 4 4\n")
        file.write("TYPE F F F I\n")
        file.write("COUNT 1 1 1 1\n")
        file.write("WIDTH {}\n".format(len(pc_data)))
        file.write("HEIGHT 1\n")
        file.write("POINTS {}\n".format(len(pc_data)))
        file.write("DATA ascii\n")
        for point, label in zip(pc_data, labels):
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {label}\n")
    print(f"Labeled Point Cloud saved to {file_path}")

async def process_cones_with_instances(lidar_path, group_path):
    cone_paths = [prim.GetPath().pathString for prim in stage.Traverse()
                  if prim.GetPath().HasPrefix(group_path)]
    car_and_cube_paths = [cube_path, ads_cad_path]
    
    toggle_colliders(car_and_cube_paths + cone_paths, True)
    
    initial_pc = await capture_lidar_data(lidar_path)
    if initial_pc.size == 0:
        print("No points captured in the initial scene.")
        return
    
    labels = np.zeros(len(initial_pc), dtype=int)  

    toggle_colliders([cube_path], False)
    modify_translate_op(ads_cad_path, move_out=True)
    toggle_colliders(cone_paths, False)

    instance_id = 1

    for cone_prim_path in cone_paths:
        toggle_colliders([cone_prim_path], True)
        await asyncio.sleep(0.5)
        cone_pc = await capture_lidar_data(lidar_path)
        if cone_pc.size != 0:
            labels = label_cone_points_in_initial_pc(initial_pc, cone_pc, labels, instance_id)
        else:
            print(f"No points captured for cone {cone_prim_path}.")
        instance_id += 1
        toggle_colliders([cone_prim_path], False)
        await asyncio.sleep(0.5)

    toggle_colliders(car_and_cube_paths + cone_paths, True)
    modify_translate_op(ads_cad_path, move_out=False)

    save_labeled_point_cloud(initial_pc, labels, "labeled_initial_scene.pcd")

async def main():
    group_path = "/Group/cone"
    await process_cones_with_instances(lidarPath, group_path)

asyncio.ensure_future(main())
