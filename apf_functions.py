import numpy as np
from stl import mesh
import open3d as o3d

# x : current state [x, y, z, roll, pitch, yaw]

def make_point_cloud(mesh):
    pcd = o3d.geometry.PointCloud()
    points = mesh.vectors.reshape(-1, 3)  # (N, 3, 3) -> (3N, 3)
    unique_points = np.unique(points, axis=0)  
    pcd.points = o3d.utility.Vector3dVector(unique_points)
    return np.asarray(pcd.points)

def get_closest_point(pcd, x):
    distances = np.linalg.norm(pcd - x, axis=1)
    idx = np.argmin(distances)
    point_min = pcd[idx]
    dis_min = distances[idx]
    return point_min, dis_min

def attractive_potential_field(x, goal, d_goal):
    k_a1 = 0.5
    k_a2 = 0.5
    if np.linalg.norm(x - goal) < d_goal:
        return k_a1 * 0.5 * d_goal ** 2
    else:
        return k_a2 * np.linalg.norm(x - goal)

def repulsive_potential_field(x, pcd, r_min):
    point_min, dis_min = get_closest_point(pcd, x)
    k_r = 0.5
    if dis_min < r_min:
        return k_r * 0.5 * (1 / dis_min) ** 2
    else:
        return 0
    
def attractive_force(x, goal, d_goal):
    k_a1 = 0.5
    k_a2 = 0.5
    if np.linalg.norm(x - goal) < d_goal:
        return k_a1 * (x - goal)
    else:
        return k_a2 * (x - goal) / np.linalg.norm(x - goal)
    
def repulsive_force(x, pcd, r_min):
    point_min, dis_min = get_closest_point(pcd, x)
    k_r = 0.5
    if dis_min < r_min:
        return -k_r * (x - point_min) / dis_min ** 3
    else:
        return np.array([0.0, 0.0, 0.0])
    
def total_force(mesh, x, goal, d_goal, r_min):
    pcd = make_point_cloud(mesh)
    F_a = attractive_force(x, goal, d_goal)
    F_r = repulsive_force(x, pcd, r_min)
    return F_a + F_r

