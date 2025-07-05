from firedrake import *
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Problem parameters
alpha = 1.0
alphasq = Constant(alpha**2)
dt = 0.0025
Dt = Constant(dt)
Ld = 40.0
n = 5000
mesh = PeriodicIntervalMesh(n, Ld)
x, = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V))

# Initial condition dictionary
ic_dict = {
    'gaussian_wide': 0.5*exp(-((x-10.)/3.)**2),
}

# Set up initial condition ONCE
w0 = Function(W)
m0, u0 = w0.subfunctions
u0.interpolate(ic_dict['gaussian_wide'])
# Solve for initial m0
p_ic = TestFunction(V)
m_ic = TrialFunction(V)
am = p_ic*m_ic*dx
Lm = (p_ic*u0 + alphasq*p_ic.dx(0)*u0.dx(0))*dx
solve(am == Lm, m0, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

# Noise basis functions (Xi_functions), defined ONCE
Nxi = 5
sigmas = [0.01, 0.02, 0.05, 0.1, 0.2]
Xi_functions = []
for i in range(Nxi):
    fx = Function(V)
    n = 2 * (i + 1) + 10
    if (i + 1) % 2 == 1:
        fx.interpolate(sigmas[i] * sin(n * pi * x / Ld))
    else:
        fx.interpolate(sigmas[i] * cos(n * pi * x / Ld))
    Xi_functions.append(fx)
R = FunctionSpace(mesh, "R", 0)
sqrt_dt = dt**0.5

# Ensemble parameters
num_ensembles = 10
T = 2.0
ndump = 100
nsteps = int(T // dt)


for ens in ProgressBar("Ensemble").iter(range(num_ensembles)):
    np.random.seed()  # Ensure different random stream for each ensemble member

    w1 = Function(W)
    w1.assign(w0)

    all_us = []
    t = 0.0
    dumpn = 0

    if ens == num_ensembles - 1:
        energy_one = []
        energy_two = []

    ufile = VTKFile(f'../CH_output0/Ensemble/SALT/gaussian_wide/u_ens{ens:03d}.pvd')

    time_steps = int(T // dt)
    for step in ProgressBar("Time step").iter(range(time_steps)):
        t = (step + 1) * dt

        # Generate new noise for each time step
        dWs = [Function(R) for _ in range(Nxi)]
        for dW in dWs:
            dW.assign(np.random.normal(0.0, 1.0))
        noise_scale = 0.1
        Ln = noise_scale * sqrt_dt * sum(Xi_functions[i] * dWs[i] for i in range(Nxi))

        # --- Use split(w1) and split(w0) for UFL expressions ---
        m1, u1 = split(w1)
        m0, u0 = split(w0)

        p, q = TestFunctions(W)
        mh = 0.5*(m1 + m0)
        uh = 0.5*(u1 + u0)
        v = uh*dt + Ln
        L = ((q*u1 + alphasq*q.dx(0)*u1.dx(0) - q*m1)*dx 
                +(p*(m1-m0) +(p*v.dx(0)*mh -p.dx(0)*v*mh))*dx)

        uprob = NonlinearVariationalProblem(L, w1)
        usolver = NonlinearVariationalSolver(uprob, solver_parameters={
            'mat_type': 'aij',
            'ksp_type': 'preonly',
            'pc_type': 'lu'
        })

        usolver.solve()
        w0.assign(w1)

        # After solve
        u1_func = w1.subfunctions[1]
        dumpn += 1
        if dumpn == ndump:
            ufile.write(u1_func, time=t)
            all_us.append(u1_func.copy(deepcopy=True))
            if ens == num_ensembles - 1:
                E_1 = assemble((u1_func*u1_func)*dx)
                E_2 = assemble((alphasq*u1_func.dx(0)*u1_func.dx(0))*dx)
                energy_one.append(E_1)
                energy_two.append(E_2)
            dumpn -= ndump

    us_array = np.array([u.dat.data[:] for u in all_us])
    np.save(f"../CH_output0/Ensemble/SALT/gaussian_wide/u_ens{ens:03d}.npy", us_array)
    if ens == num_ensembles - 1:
        np.save("../CH_output0/Ensemble/SALT/gaussian_wide/energy_one.npy", np.array(energy_one))
        np.save("../CH_output0/Ensemble/SALT/gaussian_wide/energy_two.npy", np.array(energy_two))



