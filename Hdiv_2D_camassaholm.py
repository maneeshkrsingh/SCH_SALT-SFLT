from firedrake import *
import firedrake as fd
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
import numpy as np
print = PETSc.Sys.Print
# fem tools
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

# model parameters
alpha = .1
alphasq = fd.Constant(alpha**2)
dt = 0.00125
Dt = fd.Constant(dt)

# Mesh and Function Spaces
n = 64
Lx = 2* fd.pi
Ly = 2* fd.pi
mesh = fd.PeriodicRectangleMesh(n, n, Lx, Ly)


x, y = fd.SpatialCoordinate(mesh)

# set up initial condition

width = Lx / 4.0  # Width of the Gaussian
# Center positions for each component
x1c, y1c = 3 * Lx / 4, 3 * Ly / 4
x2c, y2c = Lx / 4, Ly / 4
# Gaussian components
u_expr = fd.as_vector([
    0.5 * fd.exp(-((x - x1c)**2 + (y - y1c)**2) / (2 * width**2)),
    0.5 * fd.exp(-((x - x2c)**2 + (y - y2c)**2) / (2 * width**2))
])
# u_expr = fd.as_vector([
#     0.05,
#    0.0
# ])


# #Initial condition for velocity
# x0, y0 = Lx/2, Ly/2  # ring center
# r0, sigma, A = Lx/4, Lx/15, 1.0  # radius, thickness, peak

# # --- Polar coordinates and velocity expression ---
# r = fd.sqrt((x - x0)**2 + (y - y0)**2)
# speed = A * fd.exp(-((r - r0)**2) / (2 * sigma**2))
# tangential = fd.as_vector([-(y - y0), x - x0]) / r
# u_expr = speed * tangential


# --- Function Spaces ---
deg = 1
V = fd.FunctionSpace(mesh, "RT", deg)

W  = V*V
um0 = fd.Function(W)



u0, m0 = um0.subfunctions
m0.rename("Momentum denisity")
u0.rename("Velocity")

dS = dS('everywhere', metadata = {'quadrature_degree': 4})

# make method to get bilinear form velocity
def form_viscosity(u, v,  eta = None):
    mesh = v.ufl_domain()
    if not eta:
        eta = fd.Constant(10.0)
    n = fd.FacetNormal(mesh)
    a = fd.inner(fd.grad(u), fd.grad(v))*fd.dx
    h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)
    a += (-fd.inner(2*fd.avg(fd.outer(v, n)), fd.avg(fd.grad(u)))
              - fd.inner(fd.avg(fd.grad(v)), 2*fd.avg(fd.outer(u, n)))
              + eta/h*fd.inner(2*fd.avg(fd.outer(v, n)),
               2*fd.avg(fd.outer(u, n))))*dS
    return a

# Function to check Courant number
def check_courant(u, dt, mesh):
    h = fd.CellDiameter(mesh)
    Q = fd.FunctionSpace(mesh, "DG", 0)
    cfl_expr = fd.sqrt(fd.dot(u, u)) * dt / h
    cfl = fd.project(cfl_expr, Q)
    print(f"Courant: max = {cfl.dat.data.max():.4f}, min = {cfl.dat.data.min():.4f}")

u0.interpolate(u_expr)

# Helmholtz solve to compute initial value of  m
p = fd.TestFunction(V)
q = fd.TrialFunction(V)
am = inner(p, q)*dx
Lm = inner(p,u0)*dx + alphasq*form_viscosity(u0, p)

fd.solve(am == Lm, m0, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

um1 = fd.Function(W)
um1.assign(um0)
m1, u1 = split(um1)
m0, u0 = split(um0)


h = fd.CellDiameter(mesh)

outward_normals = fd.CellNormal(mesh)
n = fd.FacetNormal(mesh)

def both(u):
    return 2*fd.avg(u)


# def perp(u):
#     return fd.cross(outward_normals, u)

def cross2d(a, b):
    return a[0]*b[1] - a[1]*b[0]

def perp(u):
    return fd.as_vector([-u[1], u[0]])

du, dm = fd.TestFunctions(W)

mh = 0.5 * (m1 + m0)
uh = 0.5 * (u1 + u0)

Upwind = 0.5 * (sign(fd.dot(uh, n)) + 1)
mh_upwind = Upwind * mh
uh_cross_dm = cross2d(uh, dm)

# F_um = (
#         inner(dm, m1 - m0) * dx
#         - Dt * inner(perp(grad(inner(dm, perp(uh)))), mh) * dx
#         + Dt * inner(both(perp(n) * inner(dm, perp(uh))), both(Upwind * mh)) * dS
#         - Dt * div(dm)*inner(uh, mh) * dx
#         + Dt * div(uh) * inner(dm, mh) * dx
#         + inner(du, uh) * dx - inner(du, mh) * dx
#         + alphasq * form_viscosity(uh, du)
# )



F_um = (
    inner(dm, m1 - m0) * dx
    + Dt * inner(perp(grad(inner(dm, perp(uh)))), mh) * dx
    - Dt *inner(both(perp(n)*inner(dm, perp(uh))), both(Upwind*mh))*dS
    - Dt * div(dm)*inner(uh, mh) * dx
    + Dt * div(uh) * inner(dm, mh) * dx
    + inner(du, u1) * dx - inner(du, m1) * dx
    + alphasq * form_viscosity(u1, du)
)


# one  more version trying out from GEM paper
# F_um = (
#         inner(dm, m1 - m0) * dx
#         - Dt*inner(perp(grad(cross2d(dm, uh))), perp(mh)) * dx
#         + Dt*jump(uh_cross_dm) * dot(perp(n('+')), mh_upwind('+'))*dS
#         - Dt * div(dm)*inner(uh, mh) * dx
#         + Dt * 2*avg(dot(uh, mh)) *jump(dm, n)* dS
#         + Dt * div(uh) * inner(dm, mh) * dx
#         + inner(du, uh) * dx - inner(du, mh) * dx
#         + alphasq * form_viscosity(uh, du)
# )

# soolver parameters
lu_parameters = {
    'snes_monitor': None,
    #'ksp_monitor': None,
    'snes_rtol': 1e-8,
    'snes_atol': 0,
    'snes_stol': 0,
    #'ksp_type': 'gmres',
    #'pc_type': 'fieldsplit',
    #'pc_fieldsplit_type': 'multiplicative'
}

s_parameters={
    'mat_type': 'aij',
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
    'snes_type': 'newtonls',             
    # 'snes_monitor': '',                  # print SNES residual norms each iteration
    # 'snes_converged_reason': '',         # print reason for convergence or failure
    # 'ksp_monitor': '',                   # print KSP residual norms
    # 'ksp_converged_reason': ''           # print linear solve reason (optional)
}

um_prob = fd.NonlinearVariationalProblem(F_um, um1)
um_solver = fd.NonlinearVariationalSolver(um_prob,
                                          solver_parameters=lu_parameters)

t = 0.0
# Create a VTK file writer for velocity
V_CG = fd.VectorFunctionSpace(mesh, "CG", 1)
u_proj = fd.Function(V_CG, name="ProjectedVelocity")
ufile = VTKFile("../CH_output0/Hdiv_CH/velocity_output.pvd", project_output=True)
u1_func, _ = um1.subfunctions  
u_proj.assign(fd.project(u1_func, V_CG))
ufile.write(u_proj, time=t)


T = 5.0
ndump = 10
dumpn = 0
t = 0.0
energy_all = []
print(f"Starting time loop from t=0 to t={T}")
while (t < T - 0.5 * dt):
    
    um_solver.solve()
    um0.assign(um1)

    energy = 0.5 * fd.assemble(dot(u0, u0)* dx + alphasq * form_viscosity(u0, u0) )
    energy_all.append(energy)
    print(f"t = {t:.6f}, Energy = {energy:.6f}")
    
    dumpn += 1
    if dumpn == ndump:
        check_courant(u0, Dt, mesh)
        dumpn = 0
        u1_func.rename("Velocity")
        u_proj.assign(fd.project(u1_func, V_CG))
        ufile.write(u_proj, time=t)

    t += dt
np.save("../CH_output0/Hdiv_CH/energy.npy", energy_all)
