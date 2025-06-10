from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

#set the parameters for the scheme. ::
alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.0005
Dt = Constant(dt)
L = 40.0 # length of the domain
n = 5000
mesh = PeriodicIntervalMesh(n, 40.0)

V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V))

w0 = Function(W)
m0, u0 = w0.subfunctions

x, = SpatialCoordinate(mesh)

peak_width=1/6
#u_ic = conditional(x < L/2., exp((x-L/2)/sqrt(alphasq)), exp(-(x-L/2)/sqrt(alphasq)))
u_ic = 0.5*2/(exp((x-L/4)/peak_width)+exp((-x+L/4)/peak_width))
u0.interpolate(u_ic) # initial condition 

# # Gaussian initial condition
# u_ic = 0.5*exp(-((x-10.)/2.)**2) # 'gaussian':
# u_ic = 0.5*exp(-((x-10.)/1.)**2) # 'gaussian_narrow': 
# u_ic = 0.5*exp(-((x-10.)/3.)**2) # 'gaussian_wide'

#u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.))+ 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))
#u0.interpolate(0.5*2/(exp((x-10.)/(1/6))+exp((-x+10.)/(1/1.)))) # initial condition u_0 = 0.5 sech((x-L)/(L/240))




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



# add the noise term
fx1 = Function(V)
fx2 = Function(V)
fx3 = Function(V)
fx4 = Function(V)

fx1.interpolate(0.1*sin(pi*x/8.))
fx2.interpolate(0.1*sin(2.*pi*x/8.))
fx3.interpolate(0.1*sin(3.*pi*x/8.))
fx4.interpolate(0.1*sin(4.*pi*x/8.))


# fx1.interpolate(0.1)
# fx2.interpolate(0.1)
# fx3.interpolate(0.1)
# fx4.interpolate(0.1)



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

noise_scale = 0.0
#SFLT type
Ln = noise_scale*sqrt_dt*(fx1*dW1+fx2*dW2+fx3*dW3+fx4*dW4)
###
# bilinear form
mh = 0.5*(m1 + m0)+ Ln # modified density with forcing noise 
uh = 0.5*(u1 + u0)

L = (
(q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx +
(p*(m1-m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh))*dx
)


uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={'mat_type': 'aij','ksp_type': 'preonly','pc_type': 'lu'})


m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions
# check the values of momentum density

T = 2.0
ufile = VTKFile('CH_fig/u.pvd')
mfile = VTKFile('CH_fig/m.pvd')
t = 0.0
ufile.write(u1, time=t)
all_us = []

N_t = 1000
# We also initialise a dump counter so we only dump every 10 timesteps. ::
# ndump=int(T / (1000 * dt))
# dumpn = 0
ndump = 400
dumpn = 0
energies = []  # List to store energy values at each timestep
# Enter the timeloop. ::
while (t < T - 0.5*dt):
#for i in range(N_t):
   t += dt
    # The energy can be computed and checked. ::
   E = assemble((u0*u0 + alphasq*u0.dx(0)*u0.dx(0))*dx)
   #print('timestep', i, "t = ", t, "E = ", E)
   
   energies.append(E)  # Append energy to the list
   usolver.solve()
   w0.assign(w1)

  # Finally, we check if it is time to dump the data. 
   print('time', t)
   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile.write(u1, time=t)
      mfile.write(m1, time=t)
      all_us.append(Function(u1))
np.save("Energy.npy",   np.array(energies))

# This solution leads to emergent peakons (peaked solitons); the left
# peakon is travelling faster than the right peakon, so they collide and
# momentum is transferred to the right peakon.
#
# At last, we call the function :func:`plot <firedrake.plot.plot>` on the final
# value to visualize it::

fig, axes = plt.subplots()
plot(all_us[-1], axes=axes)
plt.show()

energy = np.load('Energy.npy')
print('mvalue', np.max(m1.dat.data[:]),np.min(m1.dat.data[:]) )
plt.plot(energy)
plt.xlabel('Time Steps')
plt.ylabel('Energy')
plt.title('Energy vs Time Steps')
plt.show()
