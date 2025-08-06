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



# --- Physical constants ---
g = fd.Constant(10)      # gravity
f = fd.Constant(10)      # Coriolis parameter
H = fd.Constant(1.0)     # mean fluid depth
A = 0.1                  # amplitude for perturbations

# --- Domain size ---
Lx = Ly = 1.0

# --- Mesh and spatial coordinates ---
n = 128
mesh = fd.PeriodicRectangleMesh(n, n, Lx, Ly)
x, y = fd.SpatialCoordinate(mesh)


# Time stepping
dt = 0.00025
Dt = Constant(dt)




degree = 1
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)   # Velocity (H(div))
V2 = fd.FunctionSpace(mesh, "DG", degree)    # Depth (L2)
V0 = fd.FunctionSpace(mesh, "CG", degree+2) # pv
V = V1 * V2

uD0 = fd.Function(V)
u0, D0 = uD0.subfunctions


# Function to check Courant number
def check_courant(u, dt, mesh):
    h = fd.CellDiameter(mesh)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    cfl_expr = fd.sqrt(fd.dot(u, u)) * dt / h
    cfl = fd.project(cfl_expr, Q)
    print(f"Courant: max = {cfl.dat.data.max():.4f}, min = {cfl.dat.data.min():.4f}")



# ------------------------------
# Initial Condition Options
# ------------------------------
def init_vortex():
    """Gaussian vortex + optional zonal jet"""
    x0, y0 = 0.5, 0.5
    R = 0.1
    r2 = (x - x0)**2 + (y - y0)**2
    D_expr = H + A * fd.exp(-r2 / R**2)
    gradD = fd.grad(D_expr)
    u_expr = (g / f) * fd.as_vector([gradD[1], -gradD[0]])

    # Optional: zonal jet
    jet_strength = 0.5
    u_expr += fd.as_vector([jet_strength * fd.sin(2 * fd.pi * y), 0])

    D0.interpolate(D_expr)
    u0.interpolate(u_expr)

def init_cosine():
    """Cosine perturbation with geostrophic balance"""
    kx = 2 * fd.pi / Lx
    ky = 2 * fd.pi / Ly
    eta = A * fd.cos(kx * x) * fd.cos(ky * y)
    h_expr = H + eta

    u_x = -g/f * (-ky * A * fd.sin(kx * x) * fd.sin(ky * y))  # -∂η/∂y
    u_y =  g/f * (-kx * A * fd.sin(kx * x) * fd.sin(ky * y))  # ∂η/∂x
    u_expr = fd.as_vector((u_x, u_y))

    D0.project(h_expr)
    u0.project(u_expr)

def init_rest_plus_perturbation():
    """Small cosine perturbation, fluid initially at rest"""
    kx = 2 * fd.pi / Lx
    ky = 2 * fd.pi / Ly
    h_expr = H + A * fd.cos(kx * x) * fd.cos(ky * y)
    u_expr = fd.as_vector((0.0, 0.0))

    D0.project(h_expr)
    u0.project(u_expr)

# Dictionary of ICs
initial_conditions = {
    "vortex": init_vortex,
    "cosine": init_cosine,
    "rest_plus_perturbation": init_rest_plus_perturbation,
}

# Choose initial condition here
chosen_ic = "cosine"  # Options: "vortex", "cosine", "rest_plus_perturbation"
initial_conditions[chosen_ic]()  # Apply it
print(f"Applied initial condition: {chosen_ic}")



def both(u):
    return 2*fd.avg(u)


def perp(u):
    return fd.as_vector([-u[1], u[0]])


def u_op(v, u, h):
    n = fd.FacetNormal(mesh)
    Upwind = 0.5 * (sign(fd.dot(u, n)) + 1)
    K = 0.5*fd.inner(u, u)
    return (fd.inner(v, f*perp(u))*dx
                - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*dx
                + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                           both(Upwind*u))*dS
                - fd.div(v)*g*(h + K)*dx)


def h_op(phi, u, h):
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi), u)*h*dx
            + fd.jump(phi)*(uup('+')*h('+')
                     - uup('-')*h('-'))*dS
            )



# set up for bilinear form
uD1 = fd.Function(V)
uD1.assign(uD0)
u1, D1 = split(uD1)
u0, D0 = split(uD0)

# implicit rule
# uh = u1 
# Dh = D1



# implicit midpoint rule
uh = 0.5 * (u1 + u0)
Dh = 0.5 * (D1 + D0)



du, dD = fd.TestFunctions(V)

testeqn = (
        inner(du, u1 - u0)*dx
        + Dt*u_op(du, uh, Dh)
        + dD*(D1 - D0)*dx
        + Dt*h_op(dD, uh, Dh))


# F_uD = ( 
#     inner(du, u1-u0)*dx 
#     + Dt*inner(du, f*perp(uh))*dx
#     - Dt*g*div(du)*Dh*dx
#     + dD*(D1-D0)*dx
#     +Dt*H*dD*div(uh)*dx
# )

nuD_problem = fd.NonlinearVariationalProblem(testeqn, uD1)
# probelm
# uD_problem = fd.NonlinearVariationalProblem(F_uD, uD1)


# solver parameters
lu_parameters = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'snes_rtol': 1e-5,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}




uD_solver = fd.NonlinearVariationalSolver(nuD_problem, solver_parameters=lu_parameters)


# solver for vorticity
q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), u0)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)









# Create a VTK file writer for velocity
outfile = VTKFile("Output0/NLRSW/Veldepthvort.pvd", project_output=True)
Vv_CG = fd.VectorFunctionSpace(mesh, "CG", 2)
Vs_CG = fd.FunctionSpace(mesh, "CG", 1)
u_proj = fd.Function(Vv_CG, name="ProjectedVelocity")
D_proj = fd.Function(Vs_CG, name="ProjectedDepth")
u1_func,  D1_func = uD1.subfunctions
u_proj.assign(fd.project(u1_func, Vv_CG))
D_proj.assign(fd.project(D1_func, Vs_CG))
outfile.write(u_proj, D_proj, qn, time=0)
# --------------------------
# 10. Time-stepping loop
# --------------------------
T = 1.0
t = 0.0
ndump = 10
dumpn = 0
energy_all = []



print(f"Starting time loop from t=0 to t={T}")
while (t < T - 0.5 * dt):
    t += dt
    uD_solver.solve()
    uD0.assign(uD1)
    qsolver.solve()
    # energy_all.append((fd.assemble(energy) - energy_0)/energy_0)
    energy = 0.5 * fd.assemble(dot(u0, u0)*dx +g*D0*D0*dx )
    #energy_all.append(energy)
    print(f"t = {t:.6f}, Energy = {energy:.6f}", dumpn, ndump)

    dumpn += 1
    if dumpn == ndump:
        energy_all.append(energy)
        check_courant(u0, dt, mesh)
        dumpn = 0
        u_proj.assign(fd.project(u1_func, Vv_CG))
        D_proj.assign(fd.project(D1_func, Vs_CG))
        outfile.write(u_proj, D_proj, qn, time=t)
        
    
# np save energy and plot energy
np.savetxt("Output0/NLRSW/energy_rsw.txt", energy_all)
np.save("Output0/NLRSW/energy.npy", energy_all)
# plt the energy
plt.plot(np.arange(len(energy_all)) * dt*ndump, energy_all, label="Energy")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy over Time")
plt.legend()
plt.savefig("Output0/NLRSW/energy_plot.png")