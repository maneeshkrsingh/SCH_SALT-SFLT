from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


#set the parameters for the scheme. ::
alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.1
Dt = Constant(dt)

n = 100
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V))

w0 = Function(W)
m0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))+ 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

p = TestFunction(V)
m = TrialFunction(V)

am = p*m*dx
Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx

solve(am == Lm, m0, solver_parameters={ 'ksp_type': 'preonly','pc_type': 'lu' })


p, q = TestFunctions(W)

w1 = Function(W)
w1.assign(w0)
m1, u1 = split(w1)
m0, u0 = split(w0)

mh = 0.5*(m1 + m0)
uh = 0.5*(u1 + u0)

L = (
(q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx +
(p*(m1-m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh))*dx
)


uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={'mat_type': 'aij','ksp_type': 'preonly','pc_type': 'lu'})


m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions


T = 10.0
ufile = File('CH_fig/u.pvd')
t = 0.0
ufile.write(u1, time=t)
all_us = []

N_t = 495
# We also initialise a dump counter so we only dump every 10 timesteps. ::
ndump = 10
dumpn = 0

# Enter the timeloop. ::
#while (t < T - 0.5*dt):
for i in range(N_t):
   t += dt
    # The energy can be computed and checked. ::
   E = assemble((u0*u0 + alphasq*u0.dx(0)*u0.dx(0))*dx)
   print('timestep', i, "t = ", t, "E = ", E)
   np.save("Energy.npy",  E)

   usolver.solve()
   w0.assign(w1)

  # Finally, we check if it is time to dump the data. 

   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(u1, time=t)
      all_us.append(Function(u1))

# This solution leads to emergent peakons (peaked solitons); the left
# peakon is travelling faster than the right peakon, so they collide and
# momentum is transferred to the right peakon.
#
# At last, we call the function :func:`plot <firedrake.plot.plot>` on the final
# value to visualize it::

fig, axes = plt.subplots()
plot(all_us[-1], axes=axes)
plt.show()