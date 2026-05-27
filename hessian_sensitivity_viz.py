"""
Hessian Eigenvalue Sensitivity Analysis — 4 separate figures
Assumption: λ₁ > λ₂ ≥ 0  (example: λ₁=9, λ₂=1)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Shared setup ──────────────────────────────────────────────────────────────
lambda1 = 9.0
lambda2 = 1.0

theta = np.radians(35)
v1 = np.array([ np.cos(theta),  np.sin(theta)])
v2 = np.array([-np.sin(theta),  np.cos(theta)])

V = np.column_stack([v1, v2])
H = V @ np.diag([lambda1, lambda2]) @ V.T

def quadratic_loss(w1, w2):
    w = np.array([w1, w2])
    return 0.5 * w @ H @ w

BG       = '#ffffff'
PANEL_BG = '#ffffff'
C_RED    = '#c0392b'
C_BLUE   = '#1a5fa8'
C_GREEN  = '#1a7a3c'
C_AMBER  = '#b07000'
C_GRAY   = '#555555'
GRID_CLR = '#e0e0e0'
TXT      = '#111111'

TITLE_SUFFIX = (r'$\quad[\lambda_1 > \lambda_2 \geq 0$'
                r'$;\ \ \text{example:}\ \lambda_1{=}9,\ \lambda_2{=}1]$')

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 : Loss landscape
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
fig1.patch.set_facecolor(BG)
style_ax(ax1)

x = np.linspace(-2.2, 2.2, 400)
y = np.linspace(-2.2, 2.2, 400)
X, Y = np.meshgrid(x, y)
Z    = np.vectorize(quadratic_loss)(X, Y)

levels = np.linspace(0.05, 18, 22)
cf = ax1.contourf(X, Y, Z, levels=levels, cmap='plasma', alpha=0.82)
ax1.contour(X, Y, Z, levels=levels, colors='white', alpha=0.20, linewidths=0.4)

def draw_arrow(ax, vec, color, label, tag, tip_scale, label_offset):
    tip  = vec * tip_scale
    tail = -vec * tip_scale * 0.25
    ax.annotate('', xy=tip, xytext=tail,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=2.5, mutation_scale=18))
    tx = tip + np.asarray(label_offset)
    ax.text(*tx, f'$\\mathbf{{v}}_{label}$\n({tag})',
            color=color, fontsize=9.5, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, alpha=0.95))

draw_arrow(ax1, v1, C_RED,  '1', 'large $\\lambda_1$', tip_scale=1.4,
           label_offset=[0.30, 0.02])
draw_arrow(ax1, v2, C_BLUE, '2', 'small $\\lambda_2$', tip_scale=1.6,
           label_offset=[0.05, 0.26])

# w* — gold star, simple white label (no heavy box)
ax1.plot(0, 0, '*', color='#FFD700', markersize=15, zorder=6,
         markeredgecolor='black', markeredgewidth=0.8)
ax1.text(0.10, -0.22, '$w^*$', color='white', fontsize=11, fontweight='bold')

ax1.annotate('Tight contours\n(high curvature $\\lambda_1$)', xy=v1 * 0.55,
             xytext=(0.5, 1.6), color=C_RED, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
             bbox=dict(boxstyle='round', fc='white', ec=C_RED, alpha=0.9))
ax1.annotate('Flat contours\n(low curvature $\\lambda_2$)', xy=-v2 * 0.75,
             xytext=(0.7, -1.8), color=C_BLUE, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2),
             bbox=dict(boxstyle='round', fc='white', ec=C_BLUE, alpha=0.9))

cb = plt.colorbar(cf, ax=ax1, label='$L(\\delta w)$', shrink=0.85)
cb.ax.yaxis.label.set_color(C_GRAY)
cb.ax.tick_params(colors=C_GRAY)
ax1.set_xlim(-2.2, 2.2); ax1.set_ylim(-2.2, 2.2)
ax1.set_xlabel('$\\delta w_1$', color=C_GRAY)
ax1.set_ylabel('$\\delta w_2$', color=C_GRAY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Loss Landscape  $L(\\delta w)=\\frac{1}{2}\\delta w^\\top H\\,\\delta w$\n'
              'around optimum $w^*$  ' + TITLE_SUFFIX,
              color=TXT, fontsize=9, pad=8)
ax1.set_aspect('equal')

fig1.tight_layout()
fig1.savefig('panel1_loss_landscape.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved → panel1_loss_landscape.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 : Algorithm flowchart
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7.5, 6.5))
fig2.patch.set_facecolor(BG)
style_ax(ax2)
ax2.set_xlim(0, 10); ax2.set_ylim(1.9, 10.4)
ax2.axis('off')

ax2.set_title('Algorithm: Hessian Vector Product via Two-Pass Backpropagation\n'
              'Layer Sensitivity $= \\lambda_{\\max}$  '
              '(largest eigenvalue of the loss Hessian w.r.t. layer weights)',
              color=TXT, fontsize=10, pad=10)

steps = [
    (5.0, 9.20, 'Draw random $v$,  normalize $v \\leftarrow v/\\|v\\|_2$',              C_GRAY),
    (5.0, 7.90, '1st backpropagation:  $g_i = \\partial L / \\partial W_i$',            C_AMBER),
    (5.0, 6.60, 'Inner product:  $gv = g_i^\\top v$   (scalar)',                        C_GREEN),
    (5.0, 5.30, '2nd backpropagation:  $Hv = \\partial(gv) / \\partial W_i$',          C_RED),
    (5.0, 4.00, 'Normalize:  $v \\leftarrow Hv / \\|Hv\\|_2$',                         C_BLUE),
    (5.0, 2.70, 'Repeat $\\times k$  →  $\\lambda = v^\\top Hv \\to \\lambda_{\\max}$', C_GREEN),
]

BOX = dict(boxstyle='round,pad=0.35', facecolor='#f4f6ff', alpha=0.95)
for sx, sy, stxt, sec in steps:
    ax2.text(sx, sy, stxt, ha='center', va='center', color=TXT, fontsize=9.5,
             bbox={**BOX, 'edgecolor': sec})

for k in range(len(steps) - 1):
    ax2.annotate('', xy=(5, steps[k+1][1]+0.42), xytext=(5, steps[k][1]-0.42),
                 arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.4))

loop_top    = steps[2][1] + 0.46
loop_bottom = steps[4][1] - 0.46
ax2.annotate('', xy=(1.2, loop_top), xytext=(1.2, loop_bottom),
             arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.6))
ax2.text(0.45, (loop_top + loop_bottom) / 2, 'Power\niteration\nloop',
         ha='center', va='center', color=C_RED, fontsize=8, rotation=90)

fig2.tight_layout()
fig2.savefig('panel2_algorithm.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved → panel2_algorithm.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 : 1-D loss profiles
# ══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8.5, 5.5))
fig3.patch.set_facecolor(BG)
style_ax(ax3)

t       = np.linspace(-2, 2, 300)
loss_v1 = 0.5 * lambda1 * t ** 2
loss_v2 = 0.5 * lambda2 * t ** 2

ax3.plot(t, loss_v1, color=C_RED,  lw=2.5,
         label='Along $v_1$  (large $\\lambda_1$)  — steep')
ax3.plot(t, loss_v2, color=C_BLUE, lw=2.5,
         label='Along $v_2$  (small $\\lambda_2$)  — flat')

delta_ann = 1.3
dL1_ann = 0.5 * lambda1 * delta_ann ** 2   # 7.605
dL2_ann = 0.5 * lambda2 * delta_ann ** 2   # 0.845

# λ₁ label — formula + value
ax3.annotate('', xy=(delta_ann, dL1_ann), xytext=(delta_ann + 0.12, dL1_ann + 1.6),
             arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))
ax3.text(delta_ann + 0.15, dL1_ann + 1.6,
         f'$\\frac{{1}}{{2}}\\lambda_1\\delta^2 = {dL1_ann:.1f}$',
         color=C_RED, fontsize=10, va='center',
         bbox=dict(boxstyle='round', fc='white', ec=C_RED, alpha=0.92))

# λ₂ label — arrow starts from far right so label is clearly visible
ax3.annotate('', xy=(delta_ann, dL2_ann), xytext=(2.3, dL2_ann + 1.8),
             arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2))
ax3.text(2.33, dL2_ann + 1.8,
         f'$\\frac{{1}}{{2}}\\lambda_2\\delta^2 = {dL2_ann:.1f}$',
         color=C_BLUE, fontsize=10, va='center',
         bbox=dict(boxstyle='round', fc='white', ec=C_BLUE, alpha=0.92))

ax3.axvline( delta_ann, color=C_GRAY, ls=':', lw=1.2)
ax3.axvline(-delta_ann, color=C_GRAY, ls=':', lw=1.2)
ax3.text(delta_ann + 0.04, 0.3, f'$\\delta = {delta_ann}$',
         color=C_GRAY, fontsize=9, va='bottom')

# Gap annotation — at same δ=1.3 as the individual labels so values are consistent
# Arrow placed just left of the dotted line; label further left
ax3.annotate('', xy=(delta_ann - 0.12, dL1_ann),
             xytext=(delta_ann - 0.12, dL2_ann),
             arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=2.0))
ax3.text(delta_ann - 0.15, (dL1_ann + dL2_ann) / 2,
         f'$\\frac{{1}}{{2}}(\\lambda_1-\\lambda_2)\\delta^2 = {dL1_ann - dL2_ann:.1f}$',
         color=C_GREEN, fontsize=9, va='center', ha='right',
         bbox=dict(boxstyle='round', fc='white', ec=C_GREEN, alpha=0.92))

ax3.set_xlim(-2, 2.95); ax3.set_ylim(-0.5, 20)
ax3.set_xlabel('Perturbation magnitude $\\delta$', color=C_GRAY)
ax3.set_ylabel('Loss change  $L(\\delta) = \\frac{1}{2}\\lambda\\,\\delta^2$', color=C_GRAY)
ax3.set_title('1-D Loss along Each Eigenvector  ($\\lambda_1 > \\lambda_2$)\n'
              'Steeper parabola = larger $\\lambda$ = more sensitive  '
              + TITLE_SUFFIX,
              color=TXT, fontsize=9, pad=8)
ax3.legend(fontsize=8.5, facecolor='white', edgecolor=GRID_CLR,
           labelcolor=TXT, loc='upper left')
ax3.grid(True, color=GRID_CLR, lw=0.6)

fig3.tight_layout()
fig3.savefig('panel3_1d_profiles.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved → panel3_1d_profiles.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 : Power iteration convergence
# ══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(9.0, 6.0))
fig4.patch.set_facecolor(BG)
style_ax(ax4)

# Run power iteration
np.random.seed(7)
v_curr = np.random.randn(2)
v_curr /= np.linalg.norm(v_curr)

n_iter = 12
angle_errors       = []
rayleigh_quotients = []

for _ in range(n_iter):
    cos_a = abs(v_curr @ v1)
    angle_errors.append(np.degrees(np.arccos(np.clip(cos_a, 0, 1))))
    rayleigh_quotients.append(v_curr @ H @ v_curr)
    v_curr = H @ v_curr
    v_curr /= np.linalg.norm(v_curr)

angle_errors.append(np.degrees(np.arccos(np.clip(abs(v_curr @ v1), 0, 1))))
rayleigh_quotients.append(v_curr @ H @ v_curr)
iters = np.arange(n_iter + 1)

# Left axis: θ_k (log)
l_angle, = ax4.semilogy(
    iters, [max(a, 1e-12) for a in angle_errors],
    color=C_AMBER, lw=2.5, marker='o', markersize=5,
    label='$\\theta_k$: angle between $v_k$ and $v_1$')

ax4.set_ylabel('$\\theta_k$  (degrees, log)\n'
               '$\\theta_k \\to 0$ means $v_k$ has converged to $v_1$',
               color=C_AMBER, fontsize=9)
ax4.tick_params(axis='y', colors=C_AMBER, labelsize=8)
ax4.spines['left'].set_edgecolor(C_AMBER)

# Right axis: Rayleigh quotient (linear)
ax4b = ax4.twinx()
ax4b.set_facecolor(PANEL_BG)

l_rq, = ax4b.plot(iters, rayleigh_quotients,
                  color=C_BLUE, lw=2.5, marker='s', markersize=5, ls='--',
                  label='Rayleigh quotient  $r_k = v_k^\\top H\\,v_k$')
ax4b.axhline(lambda1, color=C_BLUE, ls=':', lw=1.2, alpha=0.40)
ax4b.text(7.6, lambda1 + 0.22, '$\\lambda_1 = \\lambda_{\\max}$',
          color=C_BLUE, fontsize=8.5)

ax4b.set_ylabel('Rayleigh quotient  $r_k = v_k^\\top H\\,v_k$\n'
                '$r_k \\to \\lambda_1$ as $\\theta_k \\to 0$',
                color=C_BLUE, fontsize=9)
ax4b.tick_params(axis='y', colors=C_BLUE, labelsize=8)
ax4b.spines['right'].set_edgecolor(C_BLUE)
ax4b.set_ylim(0, lambda1 * 1.3)

ax4.set_xlabel('Power iteration step  $k$', color=C_GRAY)
ax4.set_title('Power Iteration Convergence\n'
              '$\\theta_k \\to 0$  and  $r_k \\to \\lambda_1$  as  $v_k$ aligns with $v_1$  '
              + TITLE_SUFFIX,
              color=TXT, fontsize=9, pad=8)

ax4.legend(handles=[l_angle, l_rq], fontsize=8.5, facecolor='white',
           edgecolor=GRID_CLR, labelcolor=TXT, loc='lower left')
ax4.grid(True, color=GRID_CLR, lw=0.6)

# ── Geometric inset (old two-arrow style) — shifted LEFT to avoid data lines ─
# Blank space: k=2-7, lower portion of log scale (θ already near 0 there)
ax_geo = ax4.inset_axes([0.24, 0.10, 0.28, 0.52])
ax_geo.set_facecolor('#f6f8ff')
ax_geo.set_xlim(-0.05, 1.25)
ax_geo.set_ylim(-0.15, 1.15)
ax_geo.set_aspect('equal')
ax_geo.axis('off')
for sp in ax_geo.spines.values():
    sp.set_visible(False)

# quarter-circle arc
arc = np.linspace(0, np.pi / 2, 80)
ax_geo.plot(np.cos(arc) * 0.85, np.sin(arc) * 0.85, color=GRID_CLR, lw=1.2)

# true v1 at 20°
a1 = np.radians(20)
ax_geo.annotate('', xy=(np.cos(a1), np.sin(a1)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=2.2))
ax_geo.text(np.cos(a1) + 0.05, np.sin(a1) + 0.02,
            '$v_1$', color=C_RED, fontsize=9, fontweight='bold')

# current iterate v_k at 62°
ak = np.radians(62)
ax_geo.annotate('', xy=(np.cos(ak), np.sin(ak)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=C_AMBER, lw=2.2,
                                linestyle='dashed'))
ax_geo.text(np.cos(ak) - 0.22, np.sin(ak) + 0.03,
            '$v_k$', color=C_AMBER, fontsize=9, fontweight='bold')

# θ_k arc
arc_gap = np.linspace(a1, ak, 40)
ax_geo.plot(np.cos(arc_gap) * 0.38, np.sin(arc_gap) * 0.38,
            color=C_GREEN, lw=2)
mid = (a1 + ak) / 2
ax_geo.text(np.cos(mid) * 0.50, np.sin(mid) * 0.50,
            '$\\theta_k$', color=C_GREEN, fontsize=10, fontweight='bold')

# convergence arrow along the outer arc
ax_geo.annotate('', xy=(np.cos(a1 + 0.10) * 0.85, np.sin(a1 + 0.10) * 0.85),
                xytext=(np.cos(ak - 0.10) * 0.85, np.sin(ak - 0.10) * 0.85),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=1.2,
                                connectionstyle='arc3,rad=0.3'))

ax_geo.text(0.50, -0.12, '$\\theta_k \\to 0$ as $k \\to \\infty$',
            ha='center', color=C_GRAY, fontsize=7.5)
ax_geo.set_title('$\\theta_k$ = angle between $v_k$ and $v_1$',
                 fontsize=7.5, color=TXT, pad=2)

fig4.tight_layout()
fig4.savefig('panel4_convergence.png', dpi=160, bbox_inches='tight', facecolor=BG)
print("Saved → panel4_convergence.png")

plt.show()
