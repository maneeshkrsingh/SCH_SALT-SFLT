"""
Post-processing for CH / CH2 simulations
=========================================
3D waterfall plot — same technique as KS notebook.

Usage:
    python postprocess_ch2.py <output_dir> [n_lines]

Example:
    python postprocess_ch2.py CH2_output/gaussian_wide_rho_small_perturbation 80
"""


"""
Post-processing for CH / CH2 simulations
=========================================
3D waterfall plot — same technique as KS notebook,
tuned for CH2 amplitude range.

Usage:
    python postprocess_ch2.py <output_dir> [n_lines]

Example:
    python postprocess_ch2.py CH2_output/gaussian_wide_rho_small_perturbation 80
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# optional: use scienceplots if available
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except Exception:
    pass


def waterfall_3d(x, data, times, ax,
                 n_lines=80, color='k', linewidth=0.5):
    """
    Waterfall plot: one 3D line per time slice.
    z-axis shows real physical values — no artificial scaling.
    """
    Nt = len(data)
    if Nt <= n_lines:
        indices = np.arange(Nt)
    else:
        indices = np.linspace(0, Nt - 1, n_lines, dtype=int)

    for idx in indices:
        ax.plot(x, np.full_like(x, times[idx]), data[idx],
                c=color, linewidth=linewidth, solid_capstyle='round')


def setup_waterfall_axes(ax, title='', elev=55, azim=270,
                         box_aspect=None):
    """Common axis setup for waterfall figures."""
    ax.set_xlabel('x', labelpad=2)
    ax.set_ylabel('t', labelpad=2)
    ax.set_zlabel('u', labelpad=2)
    if box_aspect is None:
        box_aspect = [2, 3, 1.2]   # taller z than KS default
    ax.set_box_aspect(box_aspect)
    ax.set_title(title, pad=-40, fontsize=11)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.tick_params(axis='x', pad=-3, labelsize=8)
    ax.tick_params(axis='y', pad=-3, labelsize=8)
    ax.tick_params(axis='z', pad=-2, labelsize=7)


# ================================================================
if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Usage: python postprocess_ch2.py <output_dir> [n_lines]")
        sys.exit(1)

    data_dir = sys.argv[1]
    n_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    # load data
    x = np.load(os.path.join(data_dir, "x_eval.npy"))
    times = np.load(os.path.join(data_dir, "ch2_times.npy"))
    u = np.load(os.path.join(data_dir, "u_snapshots.npy"))

    has_rho = os.path.exists(os.path.join(data_dir, "rho_snapshots.npy"))
    if has_rho:
        rho = np.load(os.path.join(data_dir, "rho_snapshots.npy"))

    print(f"Loaded: u {u.shape}, t in [{times[0]:.2f}, {times[-1]:.2f}]")
    print(f"  u range: [{np.min(u):.4f}, {np.max(u):.4f}]")
    if has_rho:
        print(f"  rho range: [{np.min(rho):.4f}, {np.max(rho):.4f}]")

    fig_dir = os.path.join(data_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # No artificial scaling — z-axis shows real physical values
    # Visibility controlled purely through box_aspect and view angle

    # ============================================================
    # Figure 1: Velocity — standard view
    # ============================================================
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    waterfall_3d(x, u, times, ax, n_lines=n_lines,
                 color='k', linewidth=0.5)
    setup_waterfall_axes(ax, title='Velocity u', elev=55, azim=270,
                         box_aspect=[2, 3, 2])
    ax.set_zlabel('u', labelpad=2)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ['jpg', 'png']:
        fig.savefig(os.path.join(fig_dir, f'waterfall_velocity.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved waterfall_velocity")

    # ============================================================
    # Figure 1b: Velocity — top-down view (like KS notebook)
    # ============================================================
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    waterfall_3d(x, u, times, ax, n_lines=n_lines,
                 color='k', linewidth=0.5)
    setup_waterfall_axes(ax, title='Velocity u (top view)',
                         elev=75, azim=270,
                         box_aspect=[2, 3, 1.5])
    ax.set_zlabel('u', labelpad=2)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ['jpg', 'png']:
        fig.savefig(os.path.join(fig_dir, f'waterfall_velocity_top.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved waterfall_velocity_top")

    # ============================================================
    # Figure 1c: Velocity — low angle (dramatic, shows peaks)
    # ============================================================
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    waterfall_3d(x, u, times, ax, n_lines=n_lines,
                 color='k', linewidth=0.5)
    setup_waterfall_axes(ax, title='Velocity u (low angle)',
                         elev=30, azim=250,
                         box_aspect=[2, 3, 2.5])
    ax.set_zlabel('u', labelpad=2)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ext in ['jpg', 'png']:
        fig.savefig(os.path.join(fig_dir, f'waterfall_velocity_low.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved waterfall_velocity_low")

    # ============================================================
    # Figure 2: Density (rho - 1) — standard view
    # ============================================================
    if has_rho:
        rho_pert = rho - 1.0

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        waterfall_3d(x, rho_pert, times, ax, n_lines=n_lines,
                     color='#E67E22', linewidth=0.5)
        setup_waterfall_axes(ax, title='Density ρ̄ − 1', elev=55, azim=270,
                             box_aspect=[2, 3, 2])
        ax.set_zlabel('ρ̄ − 1', labelpad=2)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_density.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved waterfall_density")

    # ============================================================
    # Figure 3: Both overlaid — u (black) + rho-1 (orange)
    # ============================================================
    if has_rho:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        waterfall_3d(x, rho_pert, times, ax, n_lines=n_lines,
                     color='#E67E22', linewidth=0.5)
        waterfall_3d(x, u, times, ax, n_lines=n_lines,
                     color='k', linewidth=0.5)
        setup_waterfall_axes(ax, title='u (black) and ρ̄−1 (orange)',
                             elev=55, azim=270,
                             box_aspect=[2, 3, 2])

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_overlay.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved waterfall_overlay")

    # ============================================================
    # Figure 4: Side-by-side u and rho-1
    # ============================================================
    if has_rho:
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        waterfall_3d(x, u, times, ax1, n_lines=n_lines,
                     color='k', linewidth=0.5)
        setup_waterfall_axes(ax1, title='Velocity u',
                             elev=55, azim=270,
                             box_aspect=[2, 3, 2])
        ax1.set_zlabel('u', labelpad=2)

        ax2 = fig.add_subplot(122, projection='3d')
        waterfall_3d(x, rho_pert, times, ax2, n_lines=n_lines,
                     color='#E67E22', linewidth=0.5)
        setup_waterfall_axes(ax2, title='Density ρ̄ − 1',
                             elev=55, azim=270,
                             box_aspect=[2, 3, 2])
        ax2.set_zlabel('ρ̄ − 1', labelpad=2)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ['jpg', 'png']:
            fig.savefig(os.path.join(fig_dir, f'waterfall_sidebyside.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved waterfall_sidebyside")

    # ============================================================
    # Verification: 2D profiles at selected times
    # ============================================================
    check_times = [0.25, 7.25, 14.25, 25.0, 35.0, 50.0]
    colors_check = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    fig_check, ax_check = plt.subplots(figsize=(10, 4))
    for tc, cc in zip(check_times, colors_check):
        idx = np.argmin(np.abs(times - tc))
        ax_check.plot(x, u[idx], color=cc, linewidth=1.2,
                      label=f't = {times[idx]:.2f}')
    ax_check.set_xlabel('x')
    ax_check.set_ylabel('u')
    ax_check.set_title('Verification: u(x, t) at selected times')
    ax_check.legend(fontsize=8, ncol=3)
    ax_check.grid(True, alpha=0.3)
    fig_check.tight_layout()
    fig_check.savefig(os.path.join(fig_dir, 'verify_profiles.png'),
                      dpi=150, bbox_inches='tight')
    plt.close(fig_check)
    print("Saved verify_profiles.png")
