from firedrake import *
from firedrake import *
import firedrake as fd
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
import numpy as np
print = PETSc.Sys.Print
# fem tools


Constant = fd.Constant
dot = fd.dot
inner = fd.inner
grad = fd.grad
transpose = fd.transpose
div = fd.div
jump = fd.jump
avg = fd.avg
outer = fd.outer
sign = fd.sign
split = fd.split
dx = fd.dx
dS = fd.dS
# --------------------
# 1. Mesh (Periodic)
# --------------------
n = 64
Lx = Ly = 1.0
mesh = fd.PeriodicRectangleMesh(n, n, Lx, Ly)
x, y = fd.SpatialCoordinate(mesh)
# Time stepping
dt = 0.001
Dt = Constant(dt)


# --------------------------
# 4. Physical parameters
# --------------------------
g = Constant(5)
f = Constant(5)
H = Constant(2.5)




# --------------------------
# 2. Function spaces
# --------------------------
V1 = fd.FunctionSpace(mesh, "BDM", 1)   # Velocity (H(div))
V2 = fd.FunctionSpace(mesh, "DG", 0)    # Depth (L2)
V = V1 *V1* V2

uGD0 = fd.Function(V)
u0, G0,  D0 = uGD0.subfunctions


# Function to check Courant number
def check_courant(u, dt, mesh):
    h = fd.CellDiameter(mesh)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    cfl_expr = fd.sqrt(fd.dot(u, u)) * dt / h
    cfl = fd.project(cfl_expr, Q)
    print(f"Courant: max = {cfl.dat.data.max():.4f}, min = {cfl.dat.data.min():.4f}")





# intial condtions
# Vortex parameters
x0, y0 = 0.5, 0.5     # Center of vortex
R = 0.1               # Vortex radius
A = 0.1               # Amplitude of height perturbation

# Spatial coordinates
x, y = fd.SpatialCoordinate(mesh)

# Depth field: Gaussian bump
r2 = (x - x0)**2 + (y - y0)**2
D_expr = H + A * fd.exp(-r2 / R**2)

# Compute gradient of depth
gradD = fd.grad(D_expr)

# Geostrophically balanced velocity: u = (g/f) * perp(grad D)
u_expr = (g/f) * fd.as_vector([gradD[1], -gradD[0]])

# Optional: Add a weak zonal jet perturbation for richer structure
jet_strength = 0.5
u_expr += fd.as_vector([jet_strength * fd.sin(2 * np.pi * y), 0])

# Interpolate into initial condition fields
D0.interpolate(D_expr)
u0.interpolate(u_expr)


# u_expr = fd.as_vector([fd.sin(4*fd.pi*y), fd.sin(4*fd.pi*x)])
# u0.interpolate(u_expr)
# D_expr = 1 + (1/4*fd.pi)*fd.sin(4*fd.pi*x) + (1/4*fd.pi)*fd.sin(4*fd.pi*y)
# D0.interpolate(D_expr)




# set up for bilinear form
uGD1 = fd.Function(V)
uGD1.assign(uGD0)
u1, G1, D1 = split(uGD1)
u0, G0, D0 = split(uGD0)

def both(u):
    return 2*fd.avg(u)


def perp(u):
    return fd.as_vector([-u[1], u[0]])

n = fd.FacetNormal(mesh)


# mid point rule
uh = 0.5 * (u1 + u0)
Dh = 0.5 * (D1 + D0)

du, dG, dD = fd.TestFunctions(V)
Upwind = 0.5 * (sign(fd.dot(uh, n)) + 1)


F = (1/Dt)*(G1-G0)
ubar = F/Dh


eqn =  (1/Dt)*inner(du, u1-u0)*dx
eqn -= inner(perp(grad(inner(du, perp(ubar)))), uh)*dx
eqn += inner(both(perp(n)*inner(du, perp(ubar))), both(Upwind*uh))*dS
eqn += inner(du, f*perp(ubar))*dx
eqn -= div(du)*(inner(uh,uh)/2 + g*(Dh))*dx
eqn += (1/Dt)*dD*(D1-D0)*dx
eqn += dD*div(F)*dx
# G definition
eqn += inner(F - Dh*uh, dG)*dx



# probelm
uD_problem = fd.NonlinearVariationalProblem(eqn, uGD1)


# soolver parameters
lu_parameters = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    #'ksp_monitor': None,
    'snes_rtol': 1e-5,
    'snes_atol': 1.0e-5,
    'snes_stol': 0,
    'ksp_type': 'fgmres',
    'ksp_max_it': 50,
    #'pc_type': 'fieldsplit',
    'ksp_error_if_not_converged': None,
    'pc_fieldsplit_type': 'multiplicative',
    #'fieldsplit_0_fields': '1',
    #'fieldsplit_1_fields': '0',
    'fieldsplit_ksp_type': 'preonly',
    'fieldsplit_pc_type': 'lu',
    'fieldsplit_pc_factor_mat_solver_type': 'mumps',
}
uD_solver = fd.NonlinearVariationalSolver(uD_problem, solver_parameters=lu_parameters)



# 9. Output
# --------------------------
t = 0.0
# Create a VTK file writer for velocity
V_CG = fd.VectorFunctionSpace(mesh, "CG", 1)
u_proj = fd.Function(V_CG, name="ProjectedVelocity")
ufile = VTKFile("CH_output0/NLRSW/velocity_output.pvd", project_output=True)
u1_func, G1_func, D1_func = uGD1.subfunctions
u_proj.assign(fd.project(u1_func, V_CG))
ufile.write(u_proj, time=t)

# --------------------------
# 10. Time-stepping loop
# --------------------------
T = 5.0
t = 0.0
ndump = 5
dumpn = 0
energy_all = []



print(f"Starting time loop from t=0 to t={T}")
while (t < T - 0.5 * dt):
    t += dt
    uD_solver.solve()
    uGD0.assign(uGD1)
    # energy_all.append((fd.assemble(energy) - energy_0)/energy_0)
    energy = 0.5 * fd.assemble(dot(u0, u0)*H*dx +g*D0*D0*dx )
    #energy_all.append(energy)
    print(f"t = {t:.6f}, Energy = {energy:.6f}", dumpn, ndump)

    dumpn += 1
    if dumpn == ndump:
        energy_all.append(energy)
        check_courant(u0, dt, mesh)
        dumpn = 0
        u_proj.assign(fd.project(u1_func, V_CG))
        ufile.write(u_proj, time=t)
        
    
# np save energy and plot energy
np.savetxt("CH_output0/NLRSW/energy_rsw.txt", energy_all)
np.save("CH_output0/NLRSW/energy.npy", energy_all)
# plt the energy
plt.plot(np.arange(len(energy_all)) * dt*ndump, energy_all, label="Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy over Time")
plt.legend()
plt.savefig("CH_output0/NLRSW/energy_plot.png")