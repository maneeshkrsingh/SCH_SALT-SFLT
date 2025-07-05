import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# fix the seed 
np.random.seed(42)


alpha = 1.0
alphasq = Constant(alpha**2)
mu = 0.01 # viscosity term
dt = 0.0025
Dt = Constant(dt)

# simulation parameters
Ld = 40.0
n = 5000
resolutions = n
epsilon = Constant(0.01)  # small parameter for the peakon
peak_width=1/6
deltax = Ld / resolutions

# Build the mesh. 
mesh = PeriodicIntervalMesh(n, Ld)
x, = SpatialCoordinate(mesh)


# and build a :class:`mixed function space <.MixedFunctionSpace>` for the
# two variables. ::

V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V))


w0 = Function(W)
m0, u0 = w0.subfunctions



ic_dict = {'two_peaks': (0.2*2/(exp(x-403./15.*40./Ld) + exp(-x+403./15.*40./Ld))
                             + 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld))),
               'gaussian': 0.5*exp(-((x-10.)/2.)**2),
               'gaussian_narrow': 0.5*exp(-((x-10.)/1.)**2),
               'gaussian_wide': 0.5*exp(-((x-10.)/3.)**2),
               'peakon': conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq)), exp(-(x-Ld/2)/sqrt(alphasq))),
               'one_peak': 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld)),
               'proper_peak': 0.5*2/(exp(x-Ld/4)+exp(-x+Ld/4)),
               'new_peak': 0.5*2/(exp((x-Ld/4)/peak_width)+exp((-x+Ld/4)/peak_width)),
               'flat': Constant(2*pi**2/(9*40**2)),
               'fast_flat': Constant(0.1),
               'coshes': Constant(2000)*cosh((2000**0.5/2)*(x-0.75))**(-2)+Constant(1000)*cosh(1000**0.5/2*(x-0.25))**(-2),
               'd_peakon':exp(-sqrt((x-Ld/2)**2 + epsilon * deltax ** 2) / sqrt(alphasq)),
               'zero': Constant(0.0),
               'two_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) - exp(-(x+Ld/4)/sqrt(alphasq)),
                                          conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) - exp((x-3*Ld/4)/sqrt(alphasq)),
                                                      exp((x-5*Ld/4)/sqrt(alphasq)) - exp(-(x-3*Ld/4)/sqrt(alphasq)))),
                'twin_wide_gaussian': exp(-((x-10.)/3.)**2) + 0.5*exp(-((x-30.)/3.)**2),
               'twin_peakons': conditional(x < Ld/4, exp((x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                           conditional(x < Ld/2, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5* exp((x-Ld/2)/sqrt(alphasq)),
                                                       conditional(x < 3*Ld/4, exp(-(x-Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq)),
                                                                   exp((x-5*Ld/4)/sqrt(alphasq)) + 0.5 * exp(-(x-Ld/2)/sqrt(alphasq))))),
               'periodic_peakon': (conditional(x < Ld/2, 0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp((x-Ld/2)/sqrt(alphasq))
                                                                                                + exp(-Ld/sqrt(alphasq))*exp(-(x-Ld/2)/sqrt(alphasq))),
                                                         0.5 / (1 - exp(-Ld/sqrt(alphasq))) * (exp(-(x-Ld/2)/sqrt(alphasq))
                                                                                               + exp(-Ld/sqrt(alphasq))*exp((x-Ld/2)/sqrt(alphasq))))),
               'cos_bell':conditional(x < Ld/4, (cos(pi*(x-Ld/8)/(2*Ld/8)))**2, 0.0),
               'antisymmetric': 1/(exp((x-Ld/4)/Ld)+exp((-x+Ld/4)/Ld)) - 1/(exp((Ld-x-Ld/4)/Ld)+exp((Ld+x+Ld/4)/Ld))}




u0.interpolate(ic_dict['gaussian_wide']) # initial condition 



# solve for inital density m
p = TestFunction(V)
m = TrialFunction(V)

am = p*m*dx
Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx

solve(am == Lm, m0, solver_parameters={
      'ksp_type': 'preonly',
      'pc_type': 'lu'
      }
   )

# create weak form for the CH equation
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

#n = 1,3,5,7,
#sin(2*(n+1)*pi*x/Ld)
# sigmas=[0.05, 0.1, 0.2, 0.5, 1.0],


# fx1.interpolate(0.01*sin(pi*x/8.))
# fx2.interpolate(0.02*sin(3.*pi*x/8.))
# fx3.interpolate(0.05*sin(5.*pi*x/8.))
# fx4.interpolate(0.1*sin(7.*pi*x/8.))

# if constant noise is needed, uncomment the following lines
# fx1.interpolate(0.1)
# fx2.interpolate(0.1)
# fx3.interpolate(0.1)
# fx4.interpolate(0.1)



# --- Noise basis functions (Xi_functions) ---
Nxi = 5  # Number of noise modes
sigmas = [0.01, 0.02, 0.05, 0.1, 0.2]  # Amplitudes for each mode (adjust as needed)
Xi_functions = []

for i in range(Nxi):
    fx = Function(V)
    n = 2 * (i + 1) + 10  # frequencies: 12, 14, 16, 18, ...
    if (i + 1) % 2 == 1:
        fx.interpolate(sigmas[i] * sin(n * pi * x / Ld))
    else:
        fx.interpolate(sigmas[i] * cos(n * pi * x / Ld))
    Xi_functions.append(fx)

# --- Wiener processes for each mode ---
R = FunctionSpace(mesh, "R", 0)
sqrt_dt = dt**0.5
dWs = [Function(R) for _ in range(Nxi)]
for dW in dWs:
    dW.assign(np.random.normal(0.0, 1.0))



#consant noise
# noise_scale = 0.0
# Ln = noise_scale*sqrt_dt*dW1


#SFLT type
noise_scale = 0.0
Ln = noise_scale * sqrt_dt * sum(Xi_functions[i] * dWs[i] for i in range(Nxi))


#print('noise value', assemble(abs(Ln)*dx)/Ld)

#print('noise value', assemble(Ln*dx))
###
# bilinear form
# finite element with SALT
mh = 0.5*(m1 + m0)
uh = 0.5*(u1 + u0)
v = uh*dt+Ln

#SALT type
L = ((q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx 
        +(p*(m1-m0) +(p*v.dx(0)*mh -p.dx(0)*v*mh))*dx)



# Foward solver setup
uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})

# to store the solutiion's compoenents
m0, u0 = w0.subfunctions
m1, u1 = w1.subfunctions


# Final time
T = 150.0

# VTK files for output
if noise_scale == 0.0:
  ufile = VTKFile('../CH_output0/SALT/gaussian_wide/u_pure.pvd')
  mfile = VTKFile('../CH_output0/SALT/gaussian_wide/m_pure.pvd')
else: 
  ufile = VTKFile('../CH_output0/SALT/gaussian_wide/u_noise.pvd')
  mfile = VTKFile('../CH_output0/SALT/gaussian_wide/m_noise.pvd')

t = 0.0
# ufile.write(u1, time=t)
all_us = []
all_ms = []

# --- Add this for energy storage ---
energy_one = []
energy_two = []

# We also initialise a dump counter so we only dump
ndump = 100
dumpn = 0

# timestep loop
while (t < T - 0.5*dt):
   t += dt

   usolver.solve()
   w0.assign(w1)

   dumpn += 1
   if dumpn == ndump:
      # The energy can be computed and checked
      # first conservation
      # H = assemble(u1*dx)
      # print("t = ", t, "H = ", H)
      E_1 = assemble((u1*u1)*dx)
      E_2 = assemble((alphasq*u1.dx(0)*u1.dx(0))*dx)
      print("t = ", t, "E_1 = ", E_1, "E_2 = ", E_2, 'TE', E_1 + E_2)
      dumpn -= ndump
      ufile.write(u1, time=t)
      all_us.append(Function(u1))
      mfile.write(m1, time=t)
      all_ms.append(Function(m1))
      # --- Save energy and time ---
      energy_one.append(E_1)
      energy_two.append(E_2)
      print('mvalue', assemble(dt*abs(m1)*dx)/Ld )

# # After time loop, convert to numpy arrays
# energy_one = np.array(energy_one)
# energy_two = np.array(energy_two)
# # # Save energy arrays to disk
# np.save("SALT_gaussian_wide_energy_one.npy", energy_one)
# np.save("SALT_gaussian_wide_energy_two.npy", energy_two)

# try:
#   from firedrake.pyplot import plot
#   fig, axes = plt.subplots()
#   plot(all_us[-1], axes=axes)
# except Exception as e:
#   warning("Cannot plot figure. Error msg: '%s'" % e)

# # And finally show the figure::

# try:
#   plt.show()
# except Exception as e:
#   warning("Cannot show figure. Error msg: '%s'" % e)



