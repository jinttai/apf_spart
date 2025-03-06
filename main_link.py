import numpy as np
import apf_functions as apf 
import spart_functions as spart
import urdf2robot as u2r
from stl import mesh

# initial states of the systems
xyz_rpy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
v_xyz_0 = np.array([0.0, 0.0, 0.0])
v_rpy_0 = np.array([0.0, 0.0, 0.0])
goal = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
d_goal = 0.1
r_min = 0.1

# spart_init
robot, robot_key = u2r.urdf_to_robot('models/SC_ur10e.urdf')
n = robot['n_q']
t0 = np.random.rand(6,1)
tL = np.random.rand(6,n)
P0 = np.random.rand(6,6)
pm = np.random.rand(6,n)
Bi0 = np.random.rand(6,6,n)
Bij = np.random.rand(6,6,n,n)
u0 = np.random.rand(6,1)
um = np.random.rand(n,1)
u0dot = np.random.rand(6,1)
umdot = np.random.rand(n,1)
R0 = np.random.rand(3,3)
r0 = np.random.rand(3,1)
qm = np.random.rand(n,1)

t0dot, tLdot = spart.accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
RJ, RL, rJ, rL, e, g = spart.kinematics(R0, r0, qm, robot)
r_cm = spart.center_of_mass(r0, rL, robot)
Bij, Bi0, P0, pm = spart.diff_kinematics(R0, r0, rL, e, g, robot)
t0_dot, tL_dot = spart.accelerations(t0,tL,P0,pm,Bi0,Bij,u0,um,u0dot,umdot,robot)
I0, Im = spart.inertia_projection(R0, RL, robot)
M0_tilde, Mm_tilde = spart.mass_composite_body(I0, Im, Bij, Bi0, robot)
H0, H0m, Hm = spart.generalized_inertia_matrix(M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)
C0, C0m, Cm0, Cm = spart.convective_inertia_matrix(t0, tL, I0, Im, M0_tilde, Mm_tilde, Bij, Bi0, P0, pm, robot)

# apf_init
mesh = mesh.Mesh.from_file('models/model.stl')




force = apf.total_force(mesh, xyz_rpy, goal, d_goal, r_min)
