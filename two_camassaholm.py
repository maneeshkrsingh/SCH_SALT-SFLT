import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# fix the seed 
np.random.seed(42)


alpha = 1.0
alphasq = Constant(alpha**2)
mu = 0.01 # viscosity term
dt = 0.02
Dt = Constant(dt)

# simulation parameters
Ld = 100.0
n = 10000
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
W = MixedFunctionSpace((V, V, V))


w0 = Function(W)
m0, u0, eta0 = w0.subfunctions



ic_dict = {'two_peaks': (0.2*2/(exp(x-403./15.*40./Ld) + exp(-x+403./15.*40./Ld))
                             + 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld))),
               'gaussian': 0.5*exp(-((x-10.)/2.)**2),
               'gaussian_narrow': 0.5*exp(-((x-10.)/1.)**2),
               'gaussian_wide': 0.5*exp(-((x-10)/3.)**2),
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




#u0.interpolate(ic_dict['gaussian_wide']) # initial condition 
eta0.interpolate(ic_dict['gaussian_wide']) # initial condition
u0.interpolate(ic_dict['zero']) # initial condition 
m0.interpolate(ic_dict['zero']) # initial condition

# solve for inital density m
# p = TestFunction(V)
# m = TrialFunction(V)

# am = p*m*dx
# Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx

# solve(am == Lm, m0, solver_parameters={
#       'ksp_type': 'preonly',
#       'pc_type': 'lu'
#       }
#    )

# create weak form for the CH equation
p, q, r = TestFunctions(W)

w1 = Function(W)
w1.assign(w0)
m1, u1, eta1 = split(w1)
m0, u0, eta0 = split(w0)

# Define the variational form for the two-component Camassa-Holm equation
def get_camassa_holm_form(b, q, u1, u0, m1, m0, eta1, eta0, alphasq, mu, Dt, p, r):
    """
    Returns the variational form L for the two-component Camassa-Holm equation
    for a given value of b.
    """
    mh = 0.5*Dt*(m1+m0)
    etah = 0.5*Dt*(eta1+eta0)
    uh = 0.5 * (u1 + u0)
    if b <= 0:
        # put some stability/penalization term
        L = (
            (q*u1+alphasq*q.dx(0)*u1.dx(0)-q*m1)*dx +
            (p *(m1-m0)+(b-1)*p*uh.dx(0)*mh-p.dx(0)*uh*mh-0.5*p.dx(0)*etah**2+mu*Dt*p.dx(0)*mh.dx(0))*dx +
            (r *(eta1-eta0)-r.dx(0)*uh*etah)*dx
        )
    else:
        # b > 0, use the original form
        L = (
            (q*u1+alphasq*q.dx(0)*u1.dx(0)-q*m1)*dx +
            (p *(m1-m0)+(b-1)*p*uh.dx(0)*mh-p.dx(0)*uh*mh-0.5*p.dx(0)*etah**2+mu*Dt*p.dx(0)*mh.dx(0))*dx +
            (r *(eta1-eta0)-r.dx(0)*uh*etah)*dx
        )
    return L

# Set the value of b here
b = 3
# Use the function to get L
L = get_camassa_holm_form(b, q, u1, u0, m1, m0, eta1, eta0, alphasq, mu, Dt, p, r)
# Foward solver setup
uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
   {'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'})

# to store the solutiion's compoenents
m0, u0, eta0 = w0.subfunctions
m1, u1, eta1 = w1.subfunctions


# Final time
T = 1500.0

# VTK files for output
ufile = VTKFile('../CH_output0/TwoCH/u_pure.pvd')
mfile = VTKFile('../CH_output0/TwoCH/m_pure.pvd')
etafile = VTKFile('../CH_output0/TwoCH/eta_pure.pvd')


t = 0.0
# ufile.write(u1, time=t)
all_us = []
all_ms = []
all_etas = []

# --- Add this for energy storage ---
energy_one = []
energy_two = []
total_energy = []

# We also initialise a dump counter so we only dump
ndump = 100
dumpn = 0

# Save initial condition at t=0
ufile.write(u0, time=0.0)
mfile.write(m0, time=0.0)
etafile.write(eta0, time=0.0)
all_us.append(u0.dat.data_ro.copy())
all_ms.append(m0.dat.data_ro.copy())
all_etas.append(eta0.dat.data_ro.copy())

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
        ufile.write(u0, time=t)
        all_us.append(u0.dat.data_ro.copy())
        mfile.write(m0, time=t)
        etafile.write(eta0, time=t)
        all_ms.append(m0.dat.data_ro.copy())
        all_etas.append(eta0.dat.data_ro.copy())
        # --- Save energy and time ---
        energy_one.append(E_1)
        energy_two.append(E_2)
        total_energy.append(E_1 + E_2)
        print('mvalue', assemble(dt*abs(m1)*dx)/Ld )
        print('etavalue', assemble(dt*abs(eta1)*dx)/Ld )

# # After time loop, convert to numpy arrays
energy_one = np.array(energy_one)
energy_two = np.array(energy_two)
total_energy = np.array(total_energy)

# # Save energy arrays to disk
np.save("twoCH_energy_one.npy", energy_one)
np.save("twoCH_energy_two.npy", energy_two)
np.save("twoCH_total_energy.npy", total_energy)

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



