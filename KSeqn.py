from firedrake import *
import matplotlib.pyplot as plt


dt = 0.1
n = 100
mesh = PeriodicIntervalMesh(n, 40.0)
V = FunctionSpace(mesh, "HER", 3)

u0 = Function(V)
x, = SpatialCoordinate(mesh)

u0.project(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))
               + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

phi = TestFunction(V)
u1 = Function(V)
u1.assign(u0)

L = ((u1-u0)/dt * phi + (u1.dx(0)).dx(0)*(phi.dx(0)).dx(0) - u1.dx(0)* phi.dx(0) -0.5 * u1*u1*phi.dx(0)) * dx

uprob = NonlinearVariationalProblem(L, u1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})


T = 100.0
ufile = File('KS_fig/u.pvd')
t = 0.0
ufile.write(u1, time=t)
all_us = []

ndump = 10
dumpn = 0


while (t < T - 0.5*dt):
   t += dt

   usolver.solve()
   u0.assign(u1)

   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(u1, time=t)
      all_us.append(Function(u1))

try:
  fig, axes = plt.subplots()
  plot(all_us[-1], axes=axes)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)