from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# set the parameters
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
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

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

fx1 = Function(V)
fx2 = Function(V)
fx3 = Function(V)
fx4 = Function(V)

fx1.interpolate(0.1*sin(pi*x/8.))
fx2.interpolate(0.1*sin(2.*pi*x/8.))
fx3.interpolate(0.1*sin(3.*pi*x/8.))
fx4.interpolate(0.1*sin(4.*pi*x/8.))

R = FunctionSpace(mesh, "R", 0)

sqrt_dt = dt**0.5
dW1 = Function(R)
dW2 = Function(R)
dW3 = Function(R)
dW4 = Function(R)

# noise term
dW1.assign(np.random.normal(0.0, 1.0))
dW2.assign(np.random.normal(0.0, 1.0))
dW3.assign(np.random.normal(0.0, 1.0))
dW4.assign(np.random.normal(0.0, 1.0))
       
# finite element linear functional 
mh = 0.5*(m1 + m0)
uh = 0.5*(u1 + u0)

#SFLT type
Ln = sqrt_dt*(fx1*dW1+fx2*dW2+fx3*dW3+fx4*dW4)

L = ((q*u1*dt + alphasq*q.dx(0)*u1.dx(0)*dt+q*Ln - q*m1)*dx 
                +(p*(m1-m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh))*dx)

uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={'mat_type': 'aij','ksp_type': 'preonly','pc_type': 'lu'})

m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions

T = 10.0
ufile = File('SFLT_CH_fig/u.pvd')
t = 0.0
ufile.write(u1, time=t)
all_us = []

ndump = 10
dumpn = 0

while (t < T - 0.5*dt):
  t += dt

  E = assemble(((Ln+u0)*u0*Dt + Dt*alphasq*u0.dx(0)*u0.dx(0))*dx)
  print("t = ", t, "E = ", E)
  np.save("SFLT_Energy.npy",  E)

  usolver.solve()
  w0.assign(w1)

  dumpn += 1
  if dumpn == ndump:
    dumpn -= ndump
    ufile.write(u1, time=t)
    all_us.append(Function(u1))

fig, axes = plt.subplots()
plot(all_us[-1], axes=axes)
plt.show()