from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.output import VTKFile
# set the parameters
alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.01
Dt = Constant(dt)

n = 1000
mesh = PeriodicIntervalMesh(n, 4.0)



V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V))

w0 = Function(W)
m0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)

u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))
#u0.interpolate(0.5*2/(exp(x-10.0)+exp(-x+10.0)))



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

noise_scale = 1.5
       
mu = 0.000001 # viscosity


## Setup noise term using Matern formula
W_F = FunctionSpace(mesh, "DG", 0)
dW = Function(W_F)
dphi = TestFunction(V)
du = TrialFunction(V)

cell_area = CellVolume(mesh)
alpha_w = (1/cell_area**0.5)
kappa_inv_sq = Constant(1.0)

dU_1 = Function(V)
dU_2 = Function(V)
dU_3 = Function(V)

sp = {'ksp_type': 'preonly', 'pc_type': 'lu'}


dW.assign(np.random.normal(0.0, 1.0))

a_w = (dphi*du + kappa_inv_sq*dphi.dx(0)*du.dx(0))*dx

L_w0 = alpha_w*dphi*dW*dx
w_prob0 = LinearVariationalProblem(a_w, L_w0, dU_1)
wsolver0 = LinearVariationalSolver(w_prob0, solver_parameters=sp)

L_w1 = dphi*dU_1*dx
w_prob1 = LinearVariationalProblem(a_w, L_w1, dU_2)
wsolver1 = LinearVariationalSolver(w_prob1,solver_parameters=sp)

L_w = dphi*dU_2*dx
w_prob = LinearVariationalProblem(a_w, L_w, dU_3)
wsolver = LinearVariationalSolver(w_prob, solver_parameters=sp)


 # solve  dW --> dU0 --> dU1 --> dU
wsolver0.solve()
wsolver1.solve()
wsolver.solve()

#SFLT type
Ln = noise_scale*sqrt_dt*(fx1*dW1+fx2*dW2+fx3*dW3+fx4*dW4)

# with mathern kernel
Ln_matern = noise_scale*sqrt_dt*dU_3

# bilinear form
mh = 0.5*(m1 + m0)+ Ln # modified density with forcing noise 
uh = 0.5*(u1 + u0)

L = ((q*u1*dt + alphasq*q.dx(0)*u1.dx(0)*dt - q*m1)*dx 
      +(p*(m1-m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh + mu*p.dx(0)*mh.dx(0)))*dx)

uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=sp)

m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions

T = 10.0
ufile = VTKFile('SFLT_CH_fig/u.pvd')
t = 0.0
ufile.write(u1, time=t)
all_us = []
energies = []  # List to store energy values at each timestep
ndump = 10
dumpn = 0

while (t < T - 0.5*dt):
  t += dt
  E = assemble((u0*u0*Dt + Dt*alphasq*u0.dx(0)*u0.dx(0))*dx)
  #print("t = ", t, "E = ", E)
  energies.append(E)  # Append energy to the list
  
  usolver.solve()
  w0.assign(w1)
  dumpn += 1
  if dumpn == ndump:
    dumpn -= ndump
    ufile.write(u1, time=t)
    all_us.append(Function(u1))
np.save("SFLT_Energy.npy", np.array(energies))
fig, axes = plt.subplots()
plot(all_us[-1], axes=axes)
plt.show()
