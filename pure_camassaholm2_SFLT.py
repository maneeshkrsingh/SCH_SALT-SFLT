import numpy as np
import os
from firedrake import *
import matplotlib.pyplot as plt
from firedrake.output import VTKFile

# fix the seed
np.random.seed(42)

# ============================================================
# Two-Component Camassa-Holm (CH2) System
# ============================================================
#   m_t + u*m_x + 2*m*u_x = -g*rho*rho_x,   m = u - alpha^2 * u_xx
#   rho_t + (rho*u)_x = 0
# ============================================================

alpha = 1.0
alphasq = Constant(alpha**2)
g_const = 1.0  # gravity parameter (adjustable)
g = Constant(g_const)
dt = 0.0025
Dt = Constant(dt)

# simulation parameters
Ld = 40.0
n = 5000
resolutions = n
epsilon = Constant(0.01)
peak_width = 1/6
deltax = Ld / resolutions

# Build the mesh
mesh = PeriodicIntervalMesh(n, Ld)
x, = SpatialCoordinate(mesh)

# ============================================================
# Function spaces: now a triple (m, u, rho)
# ============================================================
V = FunctionSpace(mesh, "CG", 1)
W = MixedFunctionSpace((V, V, V))  # (m, u, rho)

w0 = Function(W)
m0, u0, rho0 = w0.subfunctions

# ============================================================
# Initial conditions for u
# ============================================================
ic_dict = {
    'two_peaks': (0.2*2/(exp(x-403./15.*40./Ld) + exp(-x+403./15.*40./Ld))
                 + 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld))),
    'gaussian': 0.5*exp(-((x-10.)/2.)**2),
    'gaussian_narrow': 0.5*exp(-((x-10.)/1.)**2),
    'gaussian_wide': 0.5*exp(-((x-10.)/3.)**2),
    'peakon': conditional(x < Ld/2., exp((x-Ld/2)/sqrt(alphasq)),
                          exp(-(x-Ld/2)/sqrt(alphasq))),
    'one_peak': 0.5*2/(exp(x-203./15.*40./Ld)+exp(-x+203./15.*40./Ld)),
    'proper_peak': 0.5*2/(exp(x-Ld/4)+exp(-x+Ld/4)),
    'new_peak': 0.5*2/(exp((x-Ld/4)/peak_width)+exp((-x+Ld/4)/peak_width)),
    'periodic_peakon': (conditional(x < Ld/2,
        0.5/(1 - exp(-Ld/sqrt(alphasq))) * (exp((x-Ld/2)/sqrt(alphasq))
            + exp(-Ld/sqrt(alphasq))*exp(-(x-Ld/2)/sqrt(alphasq))),
        0.5/(1 - exp(-Ld/sqrt(alphasq))) * (exp(-(x-Ld/2)/sqrt(alphasq))
            + exp(-Ld/sqrt(alphasq))*exp((x-Ld/2)/sqrt(alphasq))))),
    'cos_bell': conditional(x < Ld/4, (cos(pi*(x-Ld/8)/(2*Ld/8)))**2, 0.0),
}

# Initial conditions for rho (density / free-surface elevation)
rho0_background = 1.0  # background constant density
rho_ic_dict = {
    'constant': Constant(rho0_background),
    'small_perturbation': rho0_background + 0.1*exp(-((x - Ld/2)/3.0)**2),
    'gaussian_bump': rho0_background + 0.2*exp(-((x - Ld/2)/2.0)**2),
    'co_located': rho0_background + 0.1*exp(-((x - 10.)/3.)**2),  # same center as gaussian_wide u IC
    'offset': rho0_background + 0.1*exp(-((x - 20.)/3.)**2),  # density bump offset from velocity
    'two_bumps': rho0_background + 0.1*exp(-((x - 10.)/2.)**2) + 0.05*exp(-((x - 30.)/2.)**2),
}

# Choose initial conditions
u_ic_choice = 'gaussian_wide'
rho_ic_choice = 'small_perturbation'

u0.interpolate(ic_dict[u_ic_choice])
rho0.interpolate(rho_ic_dict[rho_ic_choice])

print(f"IC: u = {u_ic_choice}, rho = {rho_ic_choice}")

# ============================================================
# Solve for initial value of m = u - alpha^2 * u_xx
# ============================================================
p = TestFunction(V)
m = TrialFunction(V)

am = p*m*dx
Lm = (p*u0 + alphasq*p.dx(0)*u0.dx(0))*dx

solve(am == Lm, m0, solver_parameters={
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

# ============================================================
# Create weak form for the CH2 system
# ============================================================
p, q, r = TestFunctions(W)  # test functions for (m, u, rho)

w1 = Function(W)
w1.assign(w0)
m1, u1, rho1 = split(w1)
m0, u0, rho0 = split(w0) # to set up time difference

# ============================================================
# Noise (SFLT type) — applied to momentum equation only
# ============================================================
Nxi = 5
sigmas = [0.01, 0.02, 0.05, 0.1, 0.2]
Xi_functions = []

for i in range(Nxi):
    fx = Function(V)
    nn = 2 * (i + 1) + 10  # frequencies: 12, 14, 16, 18, 20
    if (i + 1) % 2 == 1:
        fx.interpolate(sigmas[i] * sin(nn * pi * x / Ld))
    else:
        fx.interpolate(sigmas[i] * cos(nn * pi * x / Ld))
    Xi_functions.append(fx)

R = FunctionSpace(mesh, "R", 0)
sqrt_dt = dt**0.5
dWs = [Function(R) for _ in range(Nxi)]
for dW in dWs:
    dW.assign(np.random.normal(0.0, 1.0))

# SFLT noise for momentum equation
noise_scale = 0.025
Ln = noise_scale * sqrt_dt * sum(Xi_functions[i] * dWs[i] for i in range(Nxi))

# Optional: noise for continuity equation (set to 0 for deterministic density transport)
noise_scale_rho = 0.0  # set > 0 for stochastic density transport
Ln_rho = Constant(0.0)
if noise_scale_rho > 0:
    dWs_rho = [Function(R) for _ in range(Nxi)]
    for dW in dWs_rho:
        dW.assign(np.random.normal(0.0, 1.0))
    Ln_rho = noise_scale_rho * sqrt_dt * sum(Xi_functions[i] * dWs_rho[i] for i in range(Nxi))

print('noise value (momentum)', assemble(abs(Ln)*dx)/Ld)

# ============================================================
# Semi-implicit Crank-Nicolson weak form
# ============================================================
# Time-averaged quantities
mh = 0.5*Dt*(m1 + m0) + Ln       # modified momentum density (with noise)
uh = 0.5*(u1 + u0)
rhoh = 0.5*(rho1 + rho0)

# --- Equation 1: m_t + u*m_x + 2*m*u_x = -g*rho*rho_x ---
# Weak form of momentum:
#   <p, m1-m0> + <p, uh_x * mh> - <p_x, uh * mh>  (advection in conservative-like form)
#   + <p, g * rhoh * rhoh_x>  (pressure term)  = 0
#
# Note: u*m_x + 2*m*u_x = (u*m)_x + m*u_x, so weak form with IBP:
#   <p, m1-m0> + <p*uh_x, mh> - <p_x, uh*mh> + <p, g*rhoh*rhoh_x> = 0

F_momentum = (
    p*(m1 - m0)*dx
    + (p*uh.dx(0)*mh - p.dx(0)*uh*mh)*dx
    + p * g * rhoh * rhoh.dx(0) * Dt * dx  # pressure coupling (scaled by Dt for consistency)
)

# --- Equation 2: Helmholtz relation m = u - alpha^2 * u_xx ---
F_helmholtz = (
    q*u1*dx + alphasq*q.dx(0)*u1.dx(0)*dx - q*m1*dx
)

# --- Equation 3: rho_t + (rho*u)_x = 0 ---
# Weak form: <r, rho1-rho0> - <r_x, rhoh*uh>*Dt = 0  (IBP of divergence form)
# For periodic BCs, boundary terms vanish.
F_continuity = (
    r*(rho1 - rho0)*dx
    - r.dx(0) * rhoh * uh * Dt * dx
)

# Combined system
L = F_momentum + F_helmholtz + F_continuity

# ============================================================
# Forward solver setup
# ============================================================
uprob = NonlinearVariationalProblem(L, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

# Extract subfunctions for output
m0, u0, rho0 = w0.subfunctions
m1, u1, rho1 = w1.subfunctions

# ============================================================
# Time stepping
# ============================================================
T = 50.0
t = 0.0

# Output directory — all outputs go here
output_dir = f'CH2_output/{u_ic_choice}_rho_{rho_ic_choice}'
os.makedirs(output_dir, exist_ok=True)

# VTK output
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)

if noise_scale == 0.0:
    ufile = VTKFile(os.path.join(vtk_dir, 'u_pure.pvd'))
    mfile = VTKFile(os.path.join(vtk_dir, 'm_pure.pvd'))
    rhofile = VTKFile(os.path.join(vtk_dir, 'rho_pure.pvd'))
else:
    ufile = VTKFile(os.path.join(vtk_dir, 'u_noise1.pvd'))
    mfile = VTKFile(os.path.join(vtk_dir, 'm_noise1.pvd'))
    rhofile = VTKFile(os.path.join(vtk_dir, 'rho_noise1.pvd'))

# uniform grid for saving snapshots: x_j = j*h, j = 0, ..., N-1
N_save = 2000
x_save = np.linspace(0, Ld, N_save, endpoint=False)

# storage
all_us = []
all_ms = []
all_rhos = []
energy_one = []
energy_two = []
mass_rho = []
energy_rho = []
times = []

ndump = 100
dumpn = 0

# Time loop
while (t < T - 0.5*dt):
    t += dt

    for dW in dWs:
        dW.assign(np.random.normal(0.0, 1.0))
    if noise_scale_rho > 0:
        for dW in dWs_rho:
            dW.assign(np.random.normal(0.0, 1.0))

    usolver.solve()
    w0.assign(w1)

    dumpn += 1
    if dumpn == ndump:
        E_1 = assemble((u1*u1)*dx)
        E_2 = assemble((alphasq*u1.dx(0)*u1.dx(0))*dx)
        M_rho = assemble(rho1*dx)
        E_rho = assemble(rho1*rho1*dx)
        H_total = E_1 + E_2 + g_const * E_rho

        print(f"t = {t:.4f}, E_u_L2 = {E_1:.6f}, E_ux_L2 = {E_2:.6f}, "
              f"M_rho = {M_rho:.6f}, E_rho = {E_rho:.6f}, H_total = {H_total:.6f}")

        dumpn -= ndump
        ufile.write(u1, time=t)
        mfile.write(m1, time=t)
        rhofile.write(rho1, time=t)

        energy_one.append(E_1)
        energy_two.append(E_2)
        mass_rho.append(M_rho)
        energy_rho.append(E_rho)
        times.append(t)

        # save u, m, rho evaluated at x_save
        all_us.append(np.array([u1.at(xj) for xj in x_save]))
        all_ms.append(np.array([m1.at(xj) for xj in x_save]))
        all_rhos.append(np.array([rho1.at(xj) for xj in x_save]))

# save everything
np.save(os.path.join(output_dir, "x_eval.npy"), x_save)
np.save(os.path.join(output_dir, "ch2_times.npy"), np.array(times))
np.save(os.path.join(output_dir, "u_snapshots.npy"), np.array(all_us))
np.save(os.path.join(output_dir, "m_snapshots.npy"), np.array(all_ms))
np.save(os.path.join(output_dir, "rho_snapshots.npy"), np.array(all_rhos))
np.save(os.path.join(output_dir, "ch2_energy_one.npy"), np.array(energy_one))
np.save(os.path.join(output_dir, "ch2_energy_two.npy"), np.array(energy_two))
np.save(os.path.join(output_dir, "ch2_mass_rho.npy"), np.array(mass_rho))
np.save(os.path.join(output_dir, "ch2_energy_rho.npy"), np.array(energy_rho))

print(f"Done. All data saved to {output_dir}/")
print(f"Run:  python postprocess_ch2.py {output_dir}")
