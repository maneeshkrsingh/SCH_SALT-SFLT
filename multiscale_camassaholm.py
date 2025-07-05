from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

#set the parameters for the scheme. ::
alpha_1 = 1.0
alphasq_1 = Constant(alpha_1**2)
alpha_2 = 0.5
alphasq_2 = Constant(alpha_2**2)
alpha_3 = 0.125
alphasq_3 = Constant(alpha_3**2)
dt = 0.0025
Dt = Constant(dt)
Ld = 100.0 # length of the domain
n = 25000
mesh = PeriodicIntervalMesh(n, Ld)

V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V, V, V, V, V)) # m10, u10, m20, u20, m30, u30

w0 = Function(W)
m10, u10, m20, u20, m30, u30 = w0.subfunctions

x, = SpatialCoordinate(mesh)

peak_width=1/6
#u_ic = conditional(x < L/2., exp((x-L/2)/sqrt(alphasq)), exp(-(x-L/2)/sqrt(alphasq)))
#u_ic = 0.5*2/(exp((x-L/4)/peak_width)+exp((-x+L/4)/peak_width))

# # graded peakon
# u1_ic = conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq_1)), exp(-(x-Ld/2)/sqrt(alphasq_1)))
# u2_ic = conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq_2)), exp(-(x-Ld/2)/sqrt(alphasq_2)))
# u3_ic = conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq_3)), exp(-(x-Ld/2)/sqrt(alphasq_3)))

# # graded gaussian
# u1_ic = 0.5*exp(-((x-Ld/2.)/alpha_1)**2)
# u2_ic = 0.5*exp(-((x-Ld/2.)/alpha_2)**2)
# u3_ic = 0.5*exp(-((x-Ld/2.)/alpha_3)**2)


# one more graded gaussian Gaussian distribution u(x, 0) = (1/σ √π) e−(x−x0)2/σ 2
# u1_ic = (1/(alpha_1*sqrt(pi)))*exp(-((x-20.)/alpha_1)**2)
# u2_ic = (1/(alpha_2*sqrt(pi)))*exp(-((x-20.)/alpha_2)**2)
# u3_ic = (1/(alpha_3*sqrt(pi)))*exp(-((x-20.)/alpha_3)**2)
#u1_ic = 0.5*exp(-((x-10.)/3.)**2)


# graded peakon and  ntipeakon initial condition
# Parameters
c1 = 1.0      # Amplitude of peakon (positive)
x1 = -2.0     # Position of peakon


c2 = 1.0      # Amplitude of antipeakon (positive value; sign handled below)
x2 = 2.0      # Position of antipeakon


# Peakon + Antipeakon initial condition
u1_ic = c1 * exp(-(1/alpha_1) * abs(x - Ld/8.)) - c2 * exp(-(1/alpha_1) * abs(x - 7*Ld/8.))
u2_ic = c1 * exp(-(1/alpha_2) * abs(x - Ld/8.)) - c2 * exp(-(1/alpha_2) * abs(x - 7*Ld/8.))
u3_ic = c1 * exp(-(1/alpha_3) * abs(x - Ld/8.)) - c2 * exp(-(1/alpha_3) *abs(x - 7*Ld/8.))


# play initial condition with zero initial conditions for u2 and u3
# u1_ic = -c * exp(-alpha_1 * abs(x - Ld/4))
# u1_ic = 0.0
# u2_ic = c1 * exp(-(1/alpha_3) * abs(x - Ld/8.)) - c2 * exp(-(1/alpha_3) *abs(x - 7*Ld/8.))
# u3_ic =  0.0



u10.interpolate(u1_ic) # initial condition 
u20.interpolate(u2_ic) # initial condition 
u30.interpolate(u3_ic) # initial condition 



p = TestFunction(V)
m1 = TrialFunction(V)
m2 = TrialFunction(V)
m3 = TrialFunction(V)

am1 = p*m1*dx
Lm1 = (p*u10 + alphasq_1*p.dx(0)*u10.dx(0))*dx
solve(am1 == Lm1, m10, solver_parameters={ 'ksp_type': 'preonly','pc_type': 'lu' })

am2 = p*m2*dx
Lm2 = (p*u20 + alphasq_2*p.dx(0)*u20.dx(0))*dx
solve(am2 == Lm2, m20, solver_parameters={ 'ksp_type': 'preonly','pc_type': 'lu' })


am3 = p*m3*dx
Lm3 = (p*u30 + alphasq_3*p.dx(0)*u30.dx(0))*dx
solve(am3 == Lm3, m30, solver_parameters={ 'ksp_type': 'preonly','pc_type': 'lu' })



#p, q, r, s = TestFunctions(W) # m11, u11, m21, u21

#r, p, s, q = TestFunctions(W) # m11, u11, m21, u21, m31, u31

p1, q1, p2, q2, p3, q3 = TestFunctions(W) # m11, u11, m21, u21, m31, u31

w1 = Function(W)
w1.assign(w0)


m11, u11, m21, u21,  m31, u31 = split(w1)
m10, u10, m20, u20,  m30, u30  = split(w0)


# bilinear form
mh1 = 0.5*(m11 + m10) 
uh1 = 0.5*(u11 + u10)

mh2 = 0.5*(m21 + m20) 
uh2 = 0.5*(u21 + u20)


mh3 = 0.5*(m31 + m30) 
uh3 = 0.5*(u31 + u30)


L = (
    (q1*u11 + alphasq_1*q1.dx(0)*u11.dx(0) - q1*m11)*dx +
    (q2*u21 + alphasq_2*q2.dx(0)*u21.dx(0) - q2*m21)*dx +
    (q3*u31 + alphasq_3*q3.dx(0)*u31.dx(0) - q3*m31)*dx +
    (p1*(m11-m10) + p1*(m20-m21)  + Dt*(p1*uh1.dx(0)*(mh1-mh2) - p1.dx(0)*uh1*(mh1-mh2)))*dx +
    (p2*(m21-m20) + p2*(m30-m31) + Dt*(p2*uh1.dx(0)*(mh2-mh3) + p2*uh2.dx(0)*(mh2-mh3) - p2.dx(0)*(uh1+uh2)*(mh2-mh3)))*dx +  # <-- add '+' here
    (p3*(m31-m30)  + Dt*(p3*uh1.dx(0)*mh3 + p3*uh2.dx(0)*mh3 + p3*uh3.dx(0)*mh3 - p3.dx(0)*(uh1+uh2+uh3)*mh3))*dx
)


uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={'mat_type': 'aij','ksp_type': 'preonly','pc_type': 'lu'})


m10, u10, m20, u20, m30, u30 = w0.subfunctions
m11, u11, m21, u21, m31, u31 = w1.subfunctions
# check the values of momentum density

T = 40.0
ufile_1 = VTKFile('../CH_output0/MultiCH/anitpeakon/u1.pvd')
ufile_2 = VTKFile('../CH_output0/MultiCH/anitpeakon/u2.pvd')
ufile_3 = VTKFile('../CH_output0/MultiCH/anitpeakon/u3.pvd')
t = 0.0
ufile_1.write(u11, time=t)
ufile_2.write(u21, time=t)
ufile_3.write(u31, time=t)
all_u1s = []
all_u2s = []
all_u3s = []

#N_t = 100
# We also initialise a dump counter so we only dump every 10 timesteps. ::
# ndump=int(T / (1000 * dt))
# dumpn = 0
ndump = 40
dumpn = 0
energies_1 = []  # List to store energy values at each timestep
energies_2 = []  # List to store energy values at each timestep
energies_3 = []  # List to store energy values at each timestep
total_energy = []  # List to store total energy values at each timestep
# Enter the timeloop. ::
while (t < T - 0.5*dt):
#for i in range(N_t):
   t += dt
    
   usolver.solve()
   w0.assign(w1)
   # The energy can be computed and checked. ::
   E1 = assemble((u11*u11 + alphasq_1*u11.dx(0)*u11.dx(0))*dx)
   E2 = assemble((u21*u21 + alphasq_2*u21.dx(0)*u21.dx(0))*dx)
   E3 = assemble((u31*u31 + alphasq_3*u31.dx(0)*u31.dx(0))*dx)
   print('timestep', "t = ", t, "E1 = ", E1, "E2 = ", E2, "E3 = ", E3, 'Total Energy', E1 + E2 + E3)
   
   energies_1.append(E1)  # Append energy to the list
   energies_2.append(E2)  # Append energy to the list
   energies_3.append(E3)  # Append energy to the list
   total_energy.append(E1 + E2 + E3)  # Append total energy to the list
  # Finally, we check if it is time to dump the data. 
   print('time', t)
   dumpn += 1
   if dumpn == ndump:
      dumpn -= ndump
      ufile_1.write(u11, time=t)
      ufile_2.write(u21, time=t)
      ufile_3.write(u31, time=t)
      all_u1s.append(Function(u11))
      all_u2s.append(Function(u21))
      all_u3s.append(Function(u31))
np.save("anitpeakon_Energy_1.npy",   np.array(energies_1))
np.save("anitpeakon_Energy_2.npy",   np.array(energies_2))
np.save("anitpeakon_Energy_3.npy",   np.array(energies_3))
np.save("anitpeakon_Energy.npy",   np.array(total_energy))

# This solution leads to emergent peakons (peaked solitons); the left
# peakon is travelling faster than the right peakon, so they collide and
# momentum is transferred to the right peakon.
#
# At last, we call the function :func:`plot <firedrake.plot.plot>` on the final
# value to visualize it::

fig, axes = plt.subplots()
plot(all_u1s[0], axes=axes, color='b', label='u1')
plot(all_u2s[0], axes=axes, color='r', label='u2')
plot(all_u3s[0], axes=axes, color='g', label='u3')
axes.legend(['u1', 'u2', 'u3'])
plt.show()

# energy = np.load('Energy.npy')
# print('mvalue', np.max(m1.dat.data[:]),np.min(m1.dat.data[:]) )
# plt.plot(energy)
# plt.xlabel('Time Steps')
# plt.ylabel('Energy')
# plt.title('Energy vs Time Steps')
# plt.show()
