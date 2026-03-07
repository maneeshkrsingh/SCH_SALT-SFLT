"""
Post-processing for CH2 simulations
=====================================
Works for both Gaussian IC and Dam-Break output.
Auto-detects file naming convention.

Usage:
    python postprocess_ch2.py <output_dir> [n_lines]

Examples:
    python postprocess_ch2.py CH2_output/gaussian_wide_rho_small_perturbation 80
    python postprocess_ch2.py CH2_output/dambreak_a1_0.3_a4.0_T5.0 80
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os, sys

try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except Exception:
    pass


# ==========================================================
# Helpers
# ==========================================================
def safe_load(data_dir, *names):
    """Try loading from multiple possible filenames."""
    for name in names:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def waterfall_3d(x, data, times, ax,
                 n_lines=80, color='k', linewidth=0.5):
    """One 3D line per time slice — real physical values on z-axis."""
    Nt = len(data)
    if Nt <= n_lines:
        indices = np.arange(Nt)
    else:
        indices = np.linspace(0, Nt - 1, n_lines, dtype=int)
    for idx in indices:
        ax.plot(x, np.full_like(x, times[idx]), data[idx],
                c=color, linewidth=linewidth, solid_capstyle='round')


def setup_ax(ax, title='', elev=55, azim=270, box_aspect=None):
    if box_aspect is None:
        box_aspect = [2, 3, 2]
    ax.set_box_aspect(box_aspect)
    ax.set_title(title, pad=-40, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_xlabel('x', labelpad=2)
    ax.set_ylabel('t', labelpad=2)
    ax.tick_params(axis='x', pad=-3, labelsize=8)
    ax.tick_params(axis='y', pad=-3, labelsize=8)
    ax.tick_params(axis='z', pad=-2, labelsize=7)


# ==========================================================
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python postprocess_ch2.py <output_dir> [n_lines]")
        sys.exit(1)

    data_dir = sys.argv[1]
    n_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    # --- Load (auto-detect naming) ---
    x     = safe_load(data_dir, "x_eval.npy", "xs_plot.npy")
    times = safe_load(data_dir, "ch2_times.npy", "time_snapshots.npy")
    u     = safe_load(data_dir, "u_snapshots.npy", "u_spacetime.npy")
    m     = safe_load(data_dir, "m_snapshots.npy", "m_spacetime.npy")
    rho   = safe_load(data_dir, "rho_snapshots.npy", "rho_spacetime.npy")

    if x is None or times is None or u is None:
        print("ERROR: could not find snapshot files.")
        sys.exit(1)

    has_rho = rho is not None
    rho_pert = (rho - 1.0) if has_rho else None

    # Optional gradient/diagnostic arrays
    ux_st    = safe_load(data_dir, "ux_spacetime.npy")
    rhox_st  = safe_load(data_dir, "rhox_spacetime.npy")
    ux_max   = safe_load(data_dir, "ux_max.npy")
    rhox_max = safe_load(data_dir, "rhox_max.npy")
    t_diag   = safe_load(data_dir, "times_diag.npy")
    E_kin    = safe_load(data_dir, "ch2_energy_one.npy", "energy_kin.npy")
    E_pot    = safe_load(data_dir, "ch2_energy_rho.npy", "energy_pot.npy")
    E2       = safe_load(data_dir, "ch2_energy_two.npy")
    M_rho    = safe_load(data_dir, "ch2_mass_rho.npy", "mass_rho.npy")
    Mom      = safe_load(data_dir, "momentum.npy")

    # auto x-limits: use full range, or zoom for dam-break
    is_dambreak = np.min(x) < -5  # dam-break has x ∈ [-L, L]
    x_lim = (np.min(x), np.max(x))
    if is_dambreak:
        x_lim = (-25, 25)

    print(f"Loaded: u {u.shape}, t ∈ [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"  u ∈ [{u.min():.4f}, {u.max():.4f}]")
    if has_rho:
        print(f"  ρ̄ ∈ [{rho.min():.4f}, {rho.max():.4f}]")
    print(f"  Mode: {'dam-break' if is_dambreak else 'Gaussian IC'}")

    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # ==========================================================
    # 1. Waterfall: velocity u — three viewing angles
    # ==========================================================
    for tag, elev, azim, ba in [('', 55, 270, [2,3,2]),
                                 ('_top', 75, 270, [2,3,1.5]),
                                 ('_low', 30, 250, [2,3,2.5])]:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        waterfall_3d(x, u, times, ax, n_lines=n_lines, color='k')
        setup_ax(ax, title='Velocity u', elev=elev, azim=azim, box_aspect=ba)
        ax.set_zlabel('u', labelpad=2)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_velocity{tag}.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved waterfall_velocity{tag}")

    # ==========================================================
    # 2. Waterfall: density ρ̄ − 1
    # ==========================================================
    if has_rho:
        for tag, elev, azim, ba in [('', 55, 270, [2,3,2]),
                                     ('_top', 75, 270, [2,3,1.5])]:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            waterfall_3d(x, rho_pert, times, ax, n_lines=n_lines, color='#E67E22')
            setup_ax(ax, title='Density ρ̄ − 1', elev=elev, azim=azim, box_aspect=ba)
            ax.set_zlabel('ρ̄ − 1', labelpad=2)
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            for ext in ['jpg', 'png']:
                fig.savefig(os.path.join(fig_dir, f'waterfall_density{tag}.{ext}'),
                            dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved waterfall_density{tag}")

    # ==========================================================
    # 3. Overlay: u (black) + ρ̄−1 (orange)
    # ==========================================================
    if has_rho:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        waterfall_3d(x, rho_pert, times, ax, n_lines=n_lines, color='#E67E22')
        waterfall_3d(x, u, times, ax, n_lines=n_lines, color='k')
        setup_ax(ax, title='u (black) and ρ̄−1 (orange)')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_overlay.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved waterfall_overlay")

    # ==========================================================
    # 4. Side-by-side u and ρ̄−1
    # ==========================================================
    if has_rho:
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        waterfall_3d(x, u, times, ax1, n_lines=n_lines, color='k')
        setup_ax(ax1, title='Velocity u'); ax1.set_zlabel('u')

        ax2 = fig.add_subplot(122, projection='3d')
        waterfall_3d(x, rho_pert, times, ax2, n_lines=n_lines, color='#E67E22')
        setup_ax(ax2, title='Density ρ̄ − 1'); ax2.set_zlabel('ρ̄ − 1')

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_sidebyside.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved waterfall_sidebyside")

    # ==========================================================
    # 5. Space-time contour: ρ̄ and u
    # ==========================================================
    X_g, T_g = np.meshgrid(x, times)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    if has_rho:
        pcm1 = a1.pcolormesh(X_g, T_g, rho, shading='auto', cmap='hot')
        a1.set_title('Density ρ̄(x,t)'); plt.colorbar(pcm1, ax=a1)
    else:
        pcm1 = a1.pcolormesh(X_g, T_g, m, shading='auto', cmap='hot')
        a1.set_title('Momentum m(x,t)'); plt.colorbar(pcm1, ax=a1)

    pcm2 = a2.pcolormesh(X_g, T_g, u, shading='auto', cmap='hot')
    a2.set_title('Velocity u(x,t)'); plt.colorbar(pcm2, ax=a2)

    for a in [a1, a2]:
        a.set_xlabel('x'); a.set_ylabel('t')
        a.set_xlim(x_lim)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'spacetime_contour.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved spacetime_contour.png")

    # ==========================================================
    # 6. Gradient space-time (if available)
    # ==========================================================
    if ux_st is not None and rhox_st is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

        vmax_r = np.nanpercentile(np.abs(rhox_st), 99)
        a1.pcolormesh(X_g, T_g, rhox_st, shading='auto',
                      cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r)
        a1.set_title('∂ρ̄/∂x — shockpeakon')

        vmax_u = np.nanpercentile(np.abs(ux_st), 99)
        a2.pcolormesh(X_g, T_g, ux_st, shading='auto',
                      cmap='RdBu_r', vmin=-vmax_u, vmax=vmax_u)
        a2.set_title('∂u/∂x — peakon')

        for a in [a1, a2]:
            a.set_xlabel('x'); a.set_ylabel('t')
            a.set_xlim(x_lim)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'spacetime_gradients.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Saved spacetime_gradients.png")

    # ==========================================================
    # 7. Peakon formation: ||u_x||_∞ and ||ρ̄_x||_∞
    # ==========================================================
    if ux_max is not None and t_diag is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

        a1.plot(t_diag, ux_max, 'b-', linewidth=1.5)
        a1.set_xlabel('t'); a1.set_ylabel('||∂u/∂x||_∞')
        a1.set_title('Velocity gradient blowup')
        a1.set_yscale('log'); a1.grid(True, alpha=0.3)

        if rhox_max is not None:
            a2.plot(t_diag, rhox_max, 'r-', linewidth=1.5)
            a2.set_xlabel('t'); a2.set_ylabel('||∂ρ̄/∂x||_∞')
            a2.set_title('Density gradient blowup')
            a2.set_yscale('log'); a2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'peakon_formation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved peakon_formation.png")

    # ==========================================================
    # 8. Conservation diagnostics
    # ==========================================================
    if E_kin is not None:
        td = t_diag if t_diag is not None else np.linspace(0, times[-1], len(E_kin))
        n_d = min(len(E_kin), len(td))
        td = td[:n_d]

        ncols = 3 if Mom is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))

        H = E_kin[:n_d] + (E_pot[:n_d] if E_pot is not None else 0)
        if E2 is not None:
            H = H + E2[:n_d]
        axes[0].plot(td, H, 'k-', lw=1.5)
        axes[0].set_title('Hamiltonian H'); axes[0].set_xlabel('t')
        axes[0].grid(True, alpha=0.3)

        if M_rho is not None:
            axes[1].plot(td, M_rho[:n_d], 'r-', lw=1.5)
            axes[1].set_title('Mass ∫ρ̄ dx'); axes[1].set_xlabel('t')
            axes[1].grid(True, alpha=0.3)

        if Mom is not None and ncols == 3:
            axes[2].plot(td, Mom[:n_d], 'b-', lw=1.5)
            axes[2].set_title('Momentum ∫u dx'); axes[2].set_xlabel('t')
            axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, 'conservation.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("Saved conservation.png")

    # ==========================================================
    # 9. Profile snapshots
    # ==========================================================
    n_snaps = 8
    idx_s = np.linspace(0, len(times)-1, n_snaps, dtype=int)

    has_grads = (ux_st is not None) and (rhox_st is not None)
    nrows = 2 if has_grads else 1

    fig, axes = plt.subplots(nrows, 2, figsize=(14, 5*nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]  # make 2D

    for i in idx_s:
        axes[0,1].plot(x, u[i], lw=1, label=f't={times[i]:.2f}')
    axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('u')
    axes[0,1].set_title('Velocity u'); axes[0,1].legend(fontsize=7, ncol=2)
    axes[0,1].set_xlim(x_lim); axes[0,1].grid(True, alpha=0.3)

    if has_rho:
        for i in idx_s:
            axes[0,0].plot(x, rho[i], lw=1, label=f't={times[i]:.2f}')
        axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('ρ̄')
        axes[0,0].set_title('Density ρ̄'); axes[0,0].legend(fontsize=7, ncol=2)
        axes[0,0].set_xlim(x_lim); axes[0,0].grid(True, alpha=0.3)

    if has_grads:
        for i in idx_s[1:]:
            axes[1,0].plot(x, rhox_st[i], lw=1, label=f't={times[i]:.2f}')
        axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('∂ρ̄/∂x')
        axes[1,0].set_title('Density gradient'); axes[1,0].legend(fontsize=7, ncol=2)
        axes[1,0].set_xlim(x_lim); axes[1,0].grid(True, alpha=0.3)

        for i in idx_s[1:]:
            axes[1,1].plot(x, ux_st[i], lw=1, label=f't={times[i]:.2f}')
        axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('∂u/∂x')
        axes[1,1].set_title('Velocity gradient'); axes[1,1].legend(fontsize=7, ncol=2)
        axes[1,1].set_xlim(x_lim); axes[1,1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'profiles.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved profiles.png")

    # ==========================================================
    print(f"\nAll figures in: {os.path.abspath(fig_dir)}/")
    for f in sorted(os.listdir(fig_dir)):
        sz = os.path.getsize(os.path.join(fig_dir, f)) / 1024
        print(f"  {f:42s}  ({sz:.0f} KB)")
