import numpy as np
import apf_functions as apf
from stl import mesh
# initial states of the systems
xyz_rpy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
v_xyz_0 = np.array([0.0, 0.0, 0.0])
v_rpy_0 = np.array([0.0, 0.0, 0.0])
goal = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
d_goal = 0.1
r_min = 0.1

mesh = mesh.Mesh.from_file('models/model.stl')
force = apf.total_force(mesh, xyz_rpy, goal, d_goal, r_min)
