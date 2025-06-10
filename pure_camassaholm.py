import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# fix the seed 
np.random.seed(42)


alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.001
Dt = Constant(dt)

# simulation parameters
Ld = 40.0
n = 5000
resolutions = n
epsilon = Constant(0.01)  # small parameter for the peakon
peak_width=1/6
deltax = Ld / resolutions

# Build the mesh. 
mesh = PeriodicIntervalMesh(n, 40.0)
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




u0.interpolate(ic_dict['proper_peak']) # initial condition 



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



fx1.interpolate(0.1*sin(pi*x/8.))
fx2.interpolate(0.1*sin(2.*pi*x/8.))
fx3.interpolate(0.1*sin(3.*pi*x/8.))
fx4.interpolate(0.1*sin(4.*pi*x/8.))

# if constant noise is needed, uncomment the following lines
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


dW1.assign(np.random.normal(0.0, 1.0))
dW2.assign(np.random.normal(0.0, 1.0))
dW3.assign(np.random.normal(0.0, 1.0))
dW4.assign(np.random.normal(0.0, 1.0))



#consant noise
noise_scale = 0.0
Ln = noise_scale*sqrt_dt*dW1


#SFLT type
#noise_scale = 10.0
#Ln = noise_scale*sqrt_dt*(fx1*dW1+fx2*dW2+fx3*dW3+fx4*dW4)
#print('noise value', assemble(Ln*dx))
###
# bilinear form
mh = 0.5*(m1 + m0)+ Ln # modified density with forcing noise 
uh = 0.5*(u1 + u0)

L = (
(q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx +
(p*(m1-m0) + Dt*(p*uh.dx(0)*mh -p.dx(0)*uh*mh))*dx
)



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
T = 50.0

# VTK files for output
if noise_scale == 0.0:
  ufile = VTKFile('../CH_output0/perCH_fig/u_pure.pvd')
  mfile = VTKFile('../CH_output0/perCH_fig/m_pure.pvd')
else: 
  ufile = VTKFile('../CH_output0/perCH_fig_noise/u_noise.pvd')
  mfile = VTKFile('../CH_output0/perCH_fig_noise/m_noise.pvd')


t = 0.0
ufile.write(u1, time=t)
all_us = []
all_ms = []


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
      E = assemble((u0*u0 + alphasq*u0.dx(0)*u0.dx(0))*dx)
      print("t = ", t, "E = ", E)
      dumpn -= ndump
      ufile.write(u1, time=t)
      all_us.append(Function(u1))
      mfile.write(m1, time=t)
      all_ms.append(Function(m1))



try:
  from firedrake.pyplot import plot
  fig, axes = plt.subplots()
  plot(all_us[-1], axes=axes)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e)

# And finally show the figure::

try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e)


print('mvalue', np.max(m1.dat.data[:]),np.min(m1.dat.data[:]) )
