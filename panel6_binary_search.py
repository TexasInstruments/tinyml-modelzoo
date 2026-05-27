"""
Binary Search Bitwidth Selection — algorithm explanation figure
Classification: metric vs bitwidth line chart
"""

import numpy as np
import matplotlib.pyplot as plt

BG       = '#ffffff'
PANEL_BG = '#ffffff'
C_RED    = '#c0392b'
C_BLUE   = '#1a5fa8'
C_GREEN  = '#1a7a3c'
C_AMBER  = '#b07000'
C_GRAY   = '#555555'
GRID_CLR = '#e0e0e0'
TXT      = '#111111'

# ── Simulated calibration metrics ─────────────────────────────────────────────
bits_cls = np.array([4,    5,    6,    7,    8   ])
acc_cls  = np.array([90.5, 95.4, 96.2, 97.5, 98.3])
threshold_cls = 95.0

def smooth(x, y, n=400):
    coeffs = np.polyfit(x, y, deg=min(4, len(x)-1))
    xf = np.linspace(x[0], x[-1], n)
    return xf, np.polyval(coeffs, xf)

xf_cls, yf_cls = smooth(bits_cls, acc_cls)

# ── Binary search data ────────────────────────────────────────────────────────
# Classification [4,8]: Iter1 try 6 PASS→[4,5], Iter2 try 4 FAIL→[5,5], Iter3 try 5 PASS → select 5
bs_cls = [
    dict(bit=6, metric=96.2, passed=True),
    dict(bit=4, metric=90.5, passed=False),
    dict(bit=5, metric=95.4, passed=True),
]
final_cls = 5

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax_cls = plt.subplots(1, 1, figsize=(7, 5.8))
fig.patch.set_facecolor(BG)

ax_cls.set_facecolor(PANEL_BG)
ax_cls.tick_params(colors=TXT, labelsize=9)
for sp in ax_cls.spines.values():
    sp.set_edgecolor(GRID_CLR)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT : metric vs bitwidth line chart  (classification)
# ══════════════════════════════════════════════════════════════════════════════
ylim = (88, 100)

ax_cls.fill_between(xf_cls, threshold_cls, ylim[1], alpha=0.07, color=C_GREEN)
ax_cls.fill_between(xf_cls, ylim[0], threshold_cls, alpha=0.07, color=C_RED)

ax_cls.text(7.85, (ylim[1] + threshold_cls) / 2,
            'Meets\nthreshold', ha='right', va='center',
            color=C_GREEN, fontsize=8, style='italic', alpha=0.9)
ax_cls.text(7.85, (ylim[0] + threshold_cls) / 2,
            'Below\nthreshold', ha='right', va='center',
            color=C_RED, fontsize=8, style='italic', alpha=0.9)

ax_cls.plot(xf_cls, yf_cls, color=C_BLUE, lw=2.5, zorder=3,
            label='Calibration accuracy')
ax_cls.axhline(threshold_cls, color=C_RED, ls='--', lw=1.8, zorder=4,
               label=f'Threshold  (95.0 %)')

label_pt_offsets = [(18, 34), (20, -44), (-100, 22)]
for i, (step, (dx, dy)) in enumerate(zip(bs_cls, label_pt_offsets)):
    bit, metric, passed = step['bit'], step['metric'], step['passed']
    col    = C_GREEN if passed else C_RED
    result = 'PASS' if passed else 'FAIL'

    ax_cls.axvline(bit, color=col, ls=':', lw=1.2, alpha=0.55, zorder=2)
    ax_cls.plot(bit, metric, 'o', color=col, markersize=12,
                zorder=6, markeredgecolor='white', markeredgewidth=1.2)
    ax_cls.text(bit, metric, str(i+1), ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=7)
    ax_cls.annotate(
        f'Iter {i+1}: {bit}-bit\n{metric:.1f} %  →  {result}',
        xy=(bit, metric), xytext=(dx, dy), textcoords='offset points',
        fontsize=8, color=col, va='center',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.92),
        arrowprops=dict(arrowstyle='->', color=col, lw=1.1), zorder=8,
    )

ax_cls.axvline(final_cls, color=C_AMBER, lw=2.5, zorder=5,
               label=f'Selected: {final_cls}-bit  (minimum valid)')
ax_cls.text(final_cls + 0.08, ylim[0] + (ylim[1]-ylim[0]) * 0.06,
            f'{final_cls}-bit\nselected', color=C_AMBER,
            fontsize=8.5, fontweight='bold', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C_AMBER, alpha=0.92))

ax_cls.set_xlim(bits_cls[0] - 0.5, bits_cls[-1] + 0.5)
ax_cls.set_ylim(*ylim)
ax_cls.set_xticks(bits_cls)
ax_cls.set_xticklabels([f'{b}-bit' for b in bits_cls], fontsize=8.5)
ax_cls.set_xlabel('Target average bitwidth', color=C_GRAY, fontsize=9)
ax_cls.set_ylabel('Calibration accuracy (%)', color=C_GRAY, fontsize=9)
ax_cls.set_title('Classification\n'
                 'Search range [4, 8],   threshold = 95 % accuracy',
                 color=TXT, fontsize=10, pad=8)
ax_cls.legend(fontsize=8.5, facecolor='white', edgecolor=GRID_CLR,
              labelcolor=TXT, loc='lower right')
ax_cls.grid(True, color=GRID_CLR, lw=0.6)

# ══════════════════════════════════════════════════════════════════════════════
fig.suptitle('Automatic Avg. Bitwidth Selection via Binary Search',
             color=TXT, fontsize=11, fontweight='bold')

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig('panel6_binary_search.png', dpi=160,
            bbox_inches='tight', facecolor=BG)
print("Saved → panel6_binary_search.png")
plt.show()
