import numpy as np
import os
from firedrake import *
from firedrake.output import VTKFile

# fix the seed
np.random.seed(42)

# ============================================================
# Two-Component Camassa-Holm (CH2) — Dam-Break Problem
# ============================================================
# Pure CH2 (α₂ = 0):
#   m_t + u m_x + 2 m u_x = -g ρ̄ ρ̄_x,   m = u - α₁² u_xx
#   ρ̄_t + (ρ̄ u)_x = 0
#
# K1 = (1 - α₁² ∂_xx)^{-1}  (Helmholtz inverse, handled by the mixed form)
# K2 = Identity  (since α₂ = 0)
#
# Case 1: α₁ = 0.3, α₂ = 0, a = 4, L = 12π, g = 1
# Dam-break IC: u(x,0) = 0,  ρ̄(x,0) = 1 + tanh(x+a) - tanh(x-a)
#
# Physics: The density gradient ρ̄_x drives velocity via pressure -g ρ̄ ρ̄_x.
# Peakons (cusps in u) and shockpeakons (jumps in ρ̄) emerge at FINITE TIME
# from smooth initial data — this is the singular behaviour we want to capture.
# ============================================================

# --- Physical parameters (matching the paper) ---
alpha1 = 0.3
alpha1sq = Constant(alpha1**2)
g_const = 1.0
g = Constant(g_const)

# --- Domain: periodic [-L, L], L = 12π ---
L = 12.0 * np.pi  # ≈ 37.7
domain_length = 2 * L  # full periodic domain

# Dam-break width
a_dam = 4.0

# --- Discretization ---
n_elem = 6000        # fine mesh to resolve peakon cusps
dt = 0.0005          # small dt — peakon formation needs temporal resolution
Dt = Constant(dt)
T_final = 5.0        # long enough to see multiple peakon emissions

print(f"h = {domain_length/n_elem:.5f}, dt = {dt}, CFL-like = {dt*3/(domain_length/n_elem):.3f}")

# --- Mesh ---
mesh = PeriodicIntervalMesh(n_elem, domain_length)
x, = SpatialCoordinate(mesh)
xs = x - L  # shifted coordinate: [-L, L]

# ============================================================
# Function spaces
# ============================================================
V_cg = FunctionSpace(mesh, "CG", 1)
V_dg = FunctionSpace(mesh, "DG", 0)  # for gradient computation
W = MixedFunctionSpace((V_cg, V_cg, V_cg))  # (m, u, ρ̄)

w0 = Function(W)
m0_sub, u0_sub, rho0_sub = w0.subfunctions

# ============================================================
# Dam-break initial conditions (eq 5.2)
# ============================================================
u0_sub.assign(0.0)
rho_ic = 1.0 + tanh(xs + a_dam) - tanh(xs - a_dam)
rho0_sub.interpolate(rho_ic)

print(f"\nDam-break IC: α₁ = {alpha1}, a = {a_dam}, L = 12π ≈ {L:.4f}, g = {g_const}")
print(f"ρ̄(x,0) = 1 + tanh(x+{a_dam}) - tanh(x-{a_dam})")
print(f"ρ̄ range: [{1.0 + np.tanh(-a_dam) - np.tanh(a_dam):.4f}, "
      f"{1.0 + np.tanh(a_dam) - np.tanh(-a_dam):.4f}]")

# ============================================================
# Initial m = u - α₁² u_xx (= 0 since u₀ = 0)
# ============================================================
p_test = TestFunction(V_cg)
m_trial = TrialFunction(V_cg)
am = p_test * m_trial * dx
Lm = (p_test * u0_sub + alpha1sq * p_test.dx(0) * u0_sub.dx(0)) * dx
solve(am == Lm, m0_sub, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

# ============================================================
# Weak form — Crank-Nicolson
# ============================================================
p, q, r = TestFunctions(W)

w1 = Function(W)
w1.assign(w0)
m1, u1, rho1 = split(w1)
m0, u0, rho0 = split(w0)

uh = 0.5 * (u1 + u0)
mh = 0.5 * Dt * (m1 + m0)
rhoh = 0.5 * (rho1 + rho0)

# Momentum: m_t + u m_x + 2 m u_x + g ρ̄ ρ̄_x = 0
F_momentum = (
    p * (m1 - m0) * dx
    + (p * uh.dx(0) * mh - p.dx(0) * uh * mh) * dx
    + p * g * rhoh * rhoh.dx(0) * Dt * dx
)

# Helmholtz: m = u - α₁² u_xx
F_helmholtz = (
    q * u1 * dx + alpha1sq * q.dx(0) * u1.dx(0) * dx - q * m1 * dx
)

# Continuity: ρ̄_t + (ρ̄ u)_x = 0
# CG weak form with IBP: ∫ r(ρ1-ρ0) dx - ∫ r_x (ρh*uh) Dt dx = 0
F_continuity = (
    r * (rho1 - rho0) * dx
    - r.dx(0) * rhoh * uh * Dt * dx
)

F_total = F_momentum + F_helmholtz + F_continuity

# ============================================================
# Solver
# ============================================================
uprob = NonlinearVariationalProblem(F_total, w1)
usolver = NonlinearVariationalSolver(uprob, solver_parameters={
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
})

m0_sub, u0_sub, rho0_sub = w0.subfunctions
m1_sub, u1_sub, rho1_sub = w1.subfunctions

# ============================================================
# Output directory
# ============================================================
output_dir = f'CH2_output/dambreak_a1_{alpha1}_a{a_dam}_T{T_final}'
os.makedirs(output_dir, exist_ok=True)
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)

ufile = VTKFile(os.path.join(vtk_dir, 'u.pvd'))
mfile = VTKFile(os.path.join(vtk_dir, 'm.pvd'))
rhofile = VTKFile(os.path.join(vtk_dir, 'rho.pvd'))

# ============================================================
# Storage for space-time plots
# ============================================================
n_plot = 2000  # high resolution for sharp features
x_plot = np.linspace(0, domain_length, n_plot + 1)[:-1]
xs_plot = x_plot - L

# Storage
rho_spacetime = []
u_spacetime = []
ux_spacetime = []    # ∂u/∂x — key peakon indicator
rhox_spacetime = []  # ∂ρ̄/∂x — key shockpeakon indicator
m_spacetime = []     # momentum m = u - α₁² u_xx
time_snapshots = []

# Diagnostics
energy_kin = []
mass_rho_arr = []
energy_pot = []
ux_max = []          # ||u_x||_∞ — blows up at peakon formation
rhox_max = []        # ||ρ̄_x||_∞ — blows up at shockpeakon formation
momentum_arr = []    # ∫u dx
times_diag = []

# Helper for DG gradient projection — set up solvers ONCE
ux_dg = Function(V_dg)
rhox_dg = Function(V_dg)

v_dg = TestFunction(V_dg)
ux_trial = TrialFunction(V_dg)

# ux projection: solve  ∫ v * ux_dg dx = ∫ v * u1.dx(0) dx
a_dg = v_dg * ux_trial * dx
L_ux = v_dg * u1_sub.dx(0) * dx
ux_prob = LinearVariationalProblem(a_dg, L_ux, ux_dg)
ux_solver = LinearVariationalSolver(ux_prob,
    solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

# rhox projection: solve  ∫ v * rhox_dg dx = ∫ v * rho1.dx(0) dx
L_rhox = v_dg * rho1_sub.dx(0) * dx
rhox_prob = LinearVariationalProblem(a_dg, L_rhox, rhox_dg)
rhox_solver = LinearVariationalSolver(rhox_prob,
    solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

def evaluate_at_points(func, x_pts):
    """Evaluate a 1D Firedrake function at an array of x-coordinates."""
    vals = np.zeros(len(x_pts))
    for i, xp in enumerate(x_pts):
        try:
            vals[i] = func.at(xp)
        except Exception:
            vals[i] = np.nan
    return vals

# ============================================================
# Time stepping
# ============================================================
t = 0.0
ndump = 40       # snapshot every ndump*dt = 0.02 time units
ndump_vtk = 200  # VTK every ndump_vtk*dt = 0.1 time units
dumpn = 0
step = 0

# Store initial condition
rho_vals = evaluate_at_points(rho0_sub, x_plot)
u_vals = evaluate_at_points(u0_sub, x_plot)
rho_spacetime.append(rho_vals.copy())
u_spacetime.append(u_vals.copy())
ux_spacetime.append(np.zeros_like(u_vals))
rhox_spacetime.append(np.gradient(rho_vals, xs_plot))
m_spacetime.append(evaluate_at_points(m0_sub, x_plot))
time_snapshots.append(0.0)

print(f"\nStarting time integration to T = {T_final}...")
print(f"Snapshots every {ndump*dt:.4f} time units ({int(T_final/(ndump*dt))} total)")
print("-" * 80)

while t < T_final - 0.5*dt:
    t += dt
    step += 1

    usolver.solve()
    w0.assign(w1)

    dumpn += 1
    if dumpn == ndump:
        dumpn = 0

        # --- Compute gradients by DG projection ---
        ux_solver.solve()
        rhox_solver.solve()

        # Store space-time data
        rho_vals = evaluate_at_points(rho1_sub, x_plot)
        u_vals = evaluate_at_points(u1_sub, x_plot)
        ux_vals = evaluate_at_points(ux_dg, x_plot)
        rhox_vals = evaluate_at_points(rhox_dg, x_plot)
        m_vals = evaluate_at_points(m1_sub, x_plot)

        rho_spacetime.append(rho_vals.copy())
        u_spacetime.append(u_vals.copy())
        ux_spacetime.append(ux_vals.copy())
        rhox_spacetime.append(rhox_vals.copy())
        m_spacetime.append(m_vals.copy())
        time_snapshots.append(t)

        # Diagnostics
        E_u_L2 = assemble(u1_sub**2 * dx)
        E_ux_L2 = assemble(alpha1sq * u1_sub.dx(0)**2 * dx)
        E_kin = E_u_L2 + E_ux_L2
        M_rho = assemble(rho1_sub * dx)
        E_rho = assemble(rho1_sub**2 * dx)
        E_pot = g_const * E_rho
        H_total = E_kin + E_pot
        ux_inf = np.nanmax(np.abs(ux_vals))
        rhox_inf = np.nanmax(np.abs(rhox_vals))
        M_u = assemble(u1_sub * dx)

        energy_kin.append(E_kin)
        mass_rho_arr.append(M_rho)
        energy_pot.append(E_pot)
        ux_max.append(ux_inf)
        rhox_max.append(rhox_inf)
        momentum_arr.append(M_u)
        times_diag.append(t)

    # VTK output (less frequent)
    if step % ndump_vtk == 0:
        ufile.write(u1_sub, time=t)
        mfile.write(m1_sub, time=t)
        rhofile.write(rho1_sub, time=t)

    # Progress
    if step % 2000 == 0:
        # use latest diagnostics if available
        if times_diag:
            print(f"t = {t:.4f}, E_u_L2 = {E_u_L2:.6f}, E_ux_L2 = {E_ux_L2:.6f}, "
                  f"M_rho = {M_rho:.6f}, E_rho = {E_rho:.6f}, H_total = {H_total:.6f} | "
                  f"||u_x||_∞ = {ux_max[-1]:.2f}, ||ρ̄_x||_∞ = {rhox_max[-1]:.2f}")
        else:
            print(f"t = {t:.4f}/{T_final}")

print(f"\nDone. {len(time_snapshots)} snapshots stored.")

# ============================================================
# Convert to arrays and save
# ============================================================
rho_spacetime = np.array(rho_spacetime)
u_spacetime = np.array(u_spacetime)
ux_spacetime = np.array(ux_spacetime)
rhox_spacetime = np.array(rhox_spacetime)
m_spacetime = np.array(m_spacetime)
time_snapshots = np.array(time_snapshots)

energy_kin = np.array(energy_kin)
mass_rho_arr = np.array(mass_rho_arr)
energy_pot = np.array(energy_pot)
ux_max = np.array(ux_max)
rhox_max = np.array(rhox_max)
momentum_arr = np.array(momentum_arr)
times_diag = np.array(times_diag)

for name, arr in [("rho_spacetime", rho_spacetime), ("u_spacetime", u_spacetime),
                   ("ux_spacetime", ux_spacetime), ("rhox_spacetime", rhox_spacetime),
                   ("m_spacetime", m_spacetime), ("time_snapshots", time_snapshots),
                   ("xs_plot", xs_plot), ("energy_kin", energy_kin),
                   ("mass_rho", mass_rho_arr), ("energy_pot", energy_pot),
                   ("ux_max", ux_max), ("rhox_max", rhox_max),
                   ("momentum", momentum_arr), ("times_diag", times_diag)]:
    np.save(os.path.join(output_dir, f"{name}.npy"), arr)

print(f"Data saved to {output_dir}/")
print(f"Run:  python dambreak_postprocess_ch2.py {output_dir}")