"""
Greedy Bit Allocation Algorithm Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

BG       = '#ffffff'
PANEL_BG = '#ffffff'
C_RED    = '#c0392b'
C_BLUE   = '#1a5fa8'
C_GREEN  = '#1a7a3c'
C_AMBER  = '#b07000'
C_GRAY   = '#555555'
GRID_CLR = '#e0e0e0'
TXT      = '#111111'

fig = plt.figure(figsize=(13, 8.5))
fig.patch.set_facecolor(BG)

gs = GridSpec(1, 2, figure=fig, width_ratios=[1.7, 1.0], wspace=0.06)
ax  = fig.add_subplot(gs[0])   # flowchart
axR = fig.add_subplot(gs[1])   # example outcome

# ── shared style ──────────────────────────────────────────────────────────────
for a in [ax, axR]:
    a.set_facecolor(PANEL_BG)
    a.tick_params(colors=TXT, labelsize=8.5)
    for sp in a.spines.values():
        sp.set_edgecolor(GRID_CLR)

# ══════════════════════════════════════════════════════════════════════════════
# Left panel : Flowchart
# ══════════════════════════════════════════════════════════════════════════════
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

fig.suptitle('Greedy Bit Allocation Based on Hessian Sensitivity\n'
             'Sensitive layers are assigned more bits; insensitive layers are quantized more aggressively',
             color=TXT, fontsize=11, fontweight='bold', y=0.98)

BOX = dict(boxstyle='round,pad=0.42', alpha=0.95)

def fbox(ax, x, y, txt, fc, ec, fs=9):
    ax.text(x, y, txt, ha='center', va='center', fontsize=fs, color=TXT,
            bbox={**BOX, 'facecolor': fc, 'edgecolor': ec})

def farrow(ax, x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=C_GRAY, lw=1.4))

cx = 5.0   # horizontal centre of flowchart

# ── Step 1 ────────────────────────────────────────────────────────────────────
fbox(ax, cx, 9.30,
     'Step 1:  Compute $\\lambda_{\\max}$ per layer\n'
     '(Hessian eigenvalue via power iteration)',
     '#f4f6ff', C_GRAY)

farrow(ax, cx, 8.88, 8.53)

# ── Step 2 ────────────────────────────────────────────────────────────────────
fbox(ax, cx, 8.18,
     'Step 2:  Set total bit budget\n'
     '$B\\ =$ target_avg_bw $\\times \\sum_i n_i$',
     '#fff8e8', C_AMBER)

farrow(ax, cx, 7.75, 7.40)

# ── Step 3 ────────────────────────────────────────────────────────────────────
fbox(ax, cx, 7.05,
     'Step 3:  Initialise all layers to 2-bit\n'
     'bits_used $= 2 \\times \\sum_i n_i$',
     '#f0f8ff', C_BLUE)

farrow(ax, cx, 6.62, 6.28)

# ── Greedy loop boundary ──────────────────────────────────────────────────────
loop_rect = FancyBboxPatch(
    (0.40, 2.00), 9.20, 4.10,
    boxstyle='round,pad=0.12',
    fill=False, edgecolor=C_RED, linewidth=1.6, linestyle='--')
ax.add_patch(loop_rect)

ax.text(0.62, 6.22,
        'while  bits_used $<$ B :',
        color=C_RED, fontsize=8.5, fontweight='bold', va='bottom')

# ── Inner step A : compute ratio ──────────────────────────────────────────────
fbox(ax, cx, 5.25,
     'For each layer $i$,  compute efficiency of upgrading  '
     '$b_{old} \\to b_{new}$:\n'
     '$reduction_i = \\lambda_{max}^{(i)} \\times n_i \\times'
     '\\left(\\frac{1}{4^{b_{old}}} -'
     ' \\frac{1}{4^{b_{new}}}\\right)$   (quantisation error saved)\n'
     '$cost_i = (b_{new} - b_{old}) \\times n_i$   (extra bits needed)\n'
     '$ratio_i = reduction_i \\;/\\; cost_i$   (error saved per bit spent)',
     '#fff4f4', C_RED, fs=8.5)

farrow(ax, cx, 4.00, 3.72)

# ── Inner step B : upgrade ────────────────────────────────────────────────────
fbox(ax, cx, 3.45,
     'Upgrade the layer with the highest ratio\n'
     '(most sensitive layers upgraded first)\n'
     'Update  bits_used',
     '#f0fff4', C_GREEN)

# loop-back arrow on the left
ax.annotate('', xy=(0.70, 5.7), xytext=(0.70, 2.58),
            arrowprops=dict(arrowstyle='<->', color=C_RED, lw=1.6))
ax.text(0.35, 4.25, 'repeat', color=C_RED, fontsize=8,
        rotation=90, ha='center', va='center')

farrow(ax, cx, 3.05, 2.70)

# ── Step 5 : Output ───────────────────────────────────────────────────────────
fbox(ax, cx, 2.35,
     'Output:  final bitwidth per layer\n'
     '(higher $\\lambda_{\\max}$ → higher bitwidth assigned)',
     '#f0fff4', C_GREEN)

# ══════════════════════════════════════════════════════════════════════════════
# Right panel : Example outcome
# ══════════════════════════════════════════════════════════════════════════════
BITS_COLORS = {2: '#d62728', 4: '#e07a00', 8: '#1a7a3c', 32: '#1a5fa8'}

# Simulated 8-layer model (sorted by sensitivity descending for clarity)
layer_names = ['conv1', 'conv2', 'bn1', 'conv3', 'fc1', 'bn2', 'conv4', 'fc2']
sensitivities = [8.5,    6.8,    5.2,    3.9,    2.8,   2.1,   1.5,    1.2]
assigned_bits = [32,      8,      8,      4,      4,     4,     2,      2  ]

y_pos = np.arange(len(layer_names))
bar_colors = [BITS_COLORS[b] for b in assigned_bits]

axR.barh(y_pos, sensitivities, color=bar_colors, alpha=0.78,
         height=0.58, edgecolor='white', linewidth=0.6)

# bitwidth label at end of each bar
for i, (s, b) in enumerate(zip(sensitivities, assigned_bits)):
    axR.text(s + 0.15, i, f'{b}-bit',
             va='center', ha='left', fontsize=8.5, fontweight='bold',
             color=BITS_COLORS[b])

axR.set_yticks(y_pos)
axR.set_yticklabels(layer_names, fontsize=8.5, color=TXT)
axR.set_xlabel('$\\lambda_{\\max}$  (sensitivity)', color=C_GRAY, fontsize=9)
axR.set_xlim(0, 12.5)
axR.set_title('Example output\n(higher $\\lambda_{\\max}$ → more bits)',
              color=TXT, fontsize=9.5, pad=6)
axR.grid(True, axis='x', color=GRID_CLR, lw=0.6)
axR.invert_yaxis()   # most sensitive at top

# legend
legend_patches = [mpatches.Patch(color=c, label=f'{b}-bit', alpha=0.78)
                  for b, c in sorted(BITS_COLORS.items(), reverse=True)]
axR.legend(handles=legend_patches, fontsize=8, loc='lower right',
           facecolor='white', edgecolor=GRID_CLR, labelcolor=TXT,
           title='Assigned\nbitwidth', title_fontsize=7.5)

fig.savefig('panel5_bit_allocation.png', dpi=160, bbox_inches='tight',
            facecolor=BG)
print("Saved → panel5_bit_allocation.png")
plt.show()
