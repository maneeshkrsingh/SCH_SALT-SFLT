from firedrake import *
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# fem tools
dot = fd.dot
inner = fd.inner
grad = fd.grad
transpose = fd.transpose
div = fd.div
jump = fd.jump
avg = fd.avg
outer = fd.outer

# model parameters
alpha = 1.0
alphasq = fd.Constant(alpha**2)
dt = 0.01
Dt = fd.Constant(dt)

# Mesh and Function Spaces
nx, ny = 32, 32
Ld = 20.0
mesh = fd.PeriodicRectangleMesh(nx, ny, 20.0, 20.0)

x, y = fd.SpatialCoordinate(mesh)
dx = fd.dx

deg = 1
V = fd.VectorFunctionSpace(mesh, "DG", deg)
W = fd.MixedFunctionSpace((V, V))

w0 = fd.Function(W)
m0, u0 = w0.subfunctions
m0.rename("Momentum")
u0.rename("Velocity")

# Initial velocity
u_expr = fd.as_vector([
    0.5 * fd.exp(-((x - 15.0)**2 + (y - 20.0)**2) / 8.0) +
    0.3 * fd.exp(-((x - 25.0)**2 + (y - 20.0)**2) / 8.0),
    0.0
])
u0.interpolate(u_expr)

# Helmholtz solve to compute initial m
p = fd.TestFunction(V)
m = fd.TrialFunction(V)
am = dot(p, m) * dx
Lm = (dot(p, u0) + alphasq * inner(grad(p), grad(u0))) * dx

fd.solve(am == Lm, m0, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

w1 = fd.Function(W)
w1.assign(w0)
m1, u1 = fd.split(w1)

p, q = fd.TestFunctions(W)

n = FacetNormal(mesh)
sigma_flux = Constant(1.0)
penalty = Constant(10.0 * deg**2)
h = fd.CellDiameter(mesh)
avg_h = avg(h)

def convective_rhs(u_, m_, test):
    vol_term = dot(test, dot(u_, grad(m_))) * dx

    un = 0.5 * (dot(u_('+'), n('+')) + dot(u_('-'), n('-')))
    jump_m = m_('+') - m_('-')
    avg_test = 0.5 * (test('+') + test('-'))

    flux_consistent = dot(un * jump_m, avg_test) * dS
    flux_dissipative = sigma_flux * dot(jump(m_), jump(test)) * dS

    return vol_term - flux_consistent + flux_dissipative

def helmholtz_penalty_terms(test, trial):
    penalty_term = (
        - alphasq * inner(avg(grad(test)), outer(jump(trial), n('+'))) * dS
        - alphasq * inner(outer(jump(test), n('+')), avg(grad(trial))) * dS
        + alphasq * penalty / avg_h * dot(jump(test), jump(trial)) * dS
    )
    return penalty_term

uh = 0.5 * (u1 + u0)
mh = 0.5 * (m1 + m0)

m_rhs = (
    convective_rhs(uh, mh, p) +
    dot(p, dot(transpose(grad(uh)), mh)) * dx +
    dot(p, mh * div(uh)) * dx
)

L = (
    (dot(q, u1) + alphasq * inner(grad(q), grad(u1)) - dot(q, m1)) * dx +
    helmholtz_penalty_terms(q, u1) +
    dot(p, m1 - m0) * dx + Dt * m_rhs
)

um_problem = fd.NonlinearVariationalProblem(L, w1)
um_solver = fd.NonlinearVariationalSolver(um_problem, solver_parameters={
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions
m1.rename("Momentum")
u1.rename("Velocity")

ufile = VTKFile('../CH_output0/2DCH/u_2d.pvd')
t = 0.0
ufile.write(u1, time=t)

T = 5.0
ndump = 10
dumpn = 0
print(f"Starting time loop from t=0 to t={T}")
while (t < T - 0.5 * dt):
    um_solver.solve()
    w0.assign(w1)

    dumpn += 1
    if dumpn == ndump:
        energy = 0.5 * fd.assemble((dot(u0, u0) + alphasq * inner(grad(u0), grad(u0))) * dx)
        print(f"t = {t:.2f}, Energy = {energy:.6f}")
        dumpn = 0
        ufile.write(u1, time=t)

    u0.assign(u1)
    m0.assign(m1)
    t += dt