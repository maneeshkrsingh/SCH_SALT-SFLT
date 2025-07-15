from firedrake import *
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake.output import VTKFile


alpha = 1.0
alphasq = fd.Constant(alpha**2)
dt = 0.01
Dt = fd.Constant(dt)

n = 128
mesh = fd.PeriodicRectangleMesh(n, n, 20.0, 20.0)

V = fd.VectorFunctionSpace(mesh, "CG", 1)
W = fd.MixedFunctionSpace((V, V))

# fem tools
dot = fd.dot
inner = fd.inner
grad = fd.grad
transpose = fd.transpose
div = fd.div


w0 = fd.Function(W)
m0, u0 = w0.subfunctions
m0.rename("Momentum")
u0.rename("Velocity")

x, y = fd.SpatialCoordinate(mesh)
dx = fd.dx
u_expr = fd.as_vector([
    0.5 * fd.exp(-((x - 15.0)**2 + (y - 20.0)**2) / 8.0) +
    0.3 * fd.exp(-((x - 25.0)**2 + (y - 20.0)**2) / 8.0),
    0.0
])
u0.interpolate(u_expr)

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

mh = 0.5 * (m1 + m0)
uh = 0.5 * (u1 + u0)

L = (
    (dot(q, u1) + alphasq * inner(grad(q), grad(u1)) - dot(q, m1)) * dx +
    (dot(p, m1 - m0) + Dt * (
        dot(p, dot(uh, grad(mh))) +
        dot(p, dot(transpose(grad(uh)), mh)) +
        dot(p, mh * div(uh))
    )) * dx
)

uprob = fd.NonlinearVariationalProblem(L, w1)
usolver = fd.NonlinearVariationalSolver(uprob, solver_parameters={
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

T = 50.0
ndump = 10
dumpn = 0

print(f"Starting time loop from t=0 to t={T}")

while (t < T - 0.5 * dt):
    t += dt

    energy = 0.5 * fd.assemble((dot(u0, u0) + alphasq * inner(grad(u0), grad(u0))) * dx)
    print(f"t = {t:.2f}, Energy = {energy:.6f}")

    usolver.solve()

    w0.assign(w1)

    dumpn += 1
    if dumpn == ndump:
        dumpn = 0
        ufile.write(u1, time=t)

print("Time loop finished.")
print("Output saved to u_2d.pvd. Use a visualisation tool like ParaView to view the results.")

# fig, axes = plt.subplots()
# plt.plot(fd.sqrt(dot(u1, u1)), axes=axes, cmap='viridis')
# axes.set_title("Magnitude of Velocity at T=100")
# plt.show()
