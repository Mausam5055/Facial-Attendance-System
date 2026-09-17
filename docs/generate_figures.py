"""
Script to generate / update technical diagrams for the LaTeX project report.
Note: Figure 3 (system_architecture.png) and Figure 4 (recognition_pipeline.png)
are high-resolution graphic diagrams and should remain preserved.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('docs/images', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'


def generate_embedding_space():
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Outer container card
    card = patches.FancyBboxPatch(
        (1.5, 1.5), 97, 97,
        boxstyle='round,pad=1.0,rounding_size=1.5',
        facecolor='#FFFFFF',
        edgecolor='#E2E8F0',
        lw=1.5
    )
    ax.add_patch(card)

    # Title Badge at Top
    ax.add_patch(patches.FancyBboxPatch(
        (5, 87), 90, 8.5,
        boxstyle='round,pad=0.4,rounding_size=0.8',
        facecolor='#0F172A',
        edgecolor='#0F172A',
        lw=0
    ))
    ax.text(
        50, 91.2,
        r'128-Dimensional Face Embedding Metric Space & Euclidean Decision Boundary ($\theta = 0.60$)',
        fontsize=10.5,
        fontweight='bold',
        ha='center',
        va='center',
        color='#FFFFFF'
    )

    # Decision Geometry Center & Radius
    cx, cy = 34, 48
    radius = 26

    # Outer Rejection Zone
    ax.add_patch(patches.Circle((cx, cy), radius + 8, facecolor='#FEF2F2', edgecolor='none', alpha=0.45))

    # Acceptance Zone
    accept_circle = patches.Circle(
        (cx, cy), radius,
        facecolor='#F0FDF4',
        edgecolor='#16A34A',
        lw=2.0,
        linestyle='--'
    )
    ax.add_patch(accept_circle)

    # Acceptance Region Label
    ax.text(
        cx, cy + radius - 3.5,
        r'ACCEPTANCE DISK: $d(u, v) < \theta$ ($\theta = 0.60$)',
        fontsize=8.5,
        fontweight='bold',
        ha='center',
        color='#15803D'
    )

    # 1. Enrolled Reference Point (Center)
    ax.plot(cx, cy, marker='o', markersize=12, color='#1E40AF', zorder=5)
    ax.text(
        cx, cy - 5.5,
        r'Enrolled Reference ($v_{\mathrm{ref}}$)' + '\n[Mausam Kar : 24BAI10284]',
        fontsize=8.5,
        fontweight='bold',
        ha='center',
        color='#1E3A8A'
    )

    # 2. Live Query Face Point (Inside Boundary)
    qx, qy = cx + 11, cy + 10
    ax.plot(qx, qy, marker='o', markersize=9, color='#16A34A', zorder=5)
    ax.plot([cx, qx], [cy, qy], color='#16A34A', lw=2.2, zorder=4)
    ax.text(
        (cx + qx) / 2 - 4.5, (cy + qy) / 2 + 1.5,
        'd = 0.235',
        fontsize=8.5,
        fontweight='bold',
        color='#15803D'
    )

    # Query Match Callout Card
    ax.add_patch(patches.FancyBboxPatch(
        (65, 58), 30, 22,
        boxstyle='round,pad=0.6,rounding_size=1.0',
        facecolor='#DCFCE7',
        edgecolor='#86EFAC',
        lw=1.5
    ))
    ax.text(80, 75.0, r'Live Query Face ($u_{\mathrm{query}}$)', fontsize=9.0, fontweight='bold', ha='center', color='#14532D')
    ax.text(80, 69.5, r'$d = 0.235 < 0.60$', fontsize=8.5, ha='center', color='#166534')
    ax.text(80, 64.5, r'Status: $\mathbf{MATCH}$ (Present $\checkmark$)', fontsize=8.5, fontweight='bold', ha='center', color='#15803D')
    ax.text(80, 60.0, 'Confidence Score: 76.5%', fontsize=8.0, ha='center', color='#166534')

    # Connector arrow
    ax.annotate(
        '',
        xy=(qx + 1.5, qy + 1.5),
        xytext=(65, 69),
        arrowprops=dict(arrowstyle='->', color='#16A34A', lw=1.5)
    )

    # 3. Imposter / Unknown Face Point
    ix, iy = cx + 33, cy + 16
    ax.plot(ix, iy, marker='o', markersize=9, color='#DC2626', zorder=5)
    ax.plot([cx, ix], [cy, iy], color='#DC2626', lw=1.8, linestyle=':', zorder=4)
    ax.text(
        cx + 17, cy + 10.5,
        'd = 0.890',
        fontsize=8.5,
        fontweight='bold',
        color='#B91C1C'
    )

    # Imposter Reject Callout Card
    ax.add_patch(patches.FancyBboxPatch(
        (65, 28), 30, 22,
        boxstyle='round,pad=0.6,rounding_size=1.0',
        facecolor='#FEE2E2',
        edgecolor='#FCA5A5',
        lw=1.5
    ))
    ax.text(80, 45.0, r'Unregistered Face ($w_{\mathrm{imposter}}$)', fontsize=9.0, fontweight='bold', ha='center', color='#991B1B')
    ax.text(80, 39.5, r'$d = 0.890 \geq 0.60$', fontsize=8.5, ha='center', color='#991B1B')
    ax.text(80, 34.5, r'Status: $\mathbf{REJECT}$ (Unknown $\times$)', fontsize=8.5, fontweight='bold', ha='center', color='#B91C1C')
    ax.text(80, 30.0, 'Attendance Log: Suppressed', fontsize=8.0, ha='center', color='#7F1D1D')

    # Connector arrow
    ax.annotate(
        '',
        xy=(ix + 1.0, iy - 1.0),
        xytext=(65, 39),
        arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5)
    )

    # Bottom Formula Card
    ax.add_patch(patches.FancyBboxPatch(
        (5, 5), 90, 15,
        boxstyle='round,pad=0.5,rounding_size=0.8',
        facecolor='#F8FAFC',
        edgecolor='#CBD5E1',
        lw=1.2
    ))
    ax.text(
        50, 15.0,
        r'$\mathbf{Euclidean\ Distance:}\ d(u, v) = \|u - v\|_2 = \sqrt{\sum_{k=1}^{128} (u_k - v_k)^2}$',
        fontsize=8.5,
        ha='center',
        color='#0F172A'
    )
    ax.text(
        50, 9.0,
        r'$\mathbf{Classification\ Rule:}\ \hat{y} = \arg\min_j d(u, v_j)\ \ \text{if}\ \min_j d(u, v_j) < 0.60\ \ \text{else\ "Unknown"}\qquad \mathbf{Confidence:}\ S(u, v) = 1.0 - d(u, v)$',
        fontsize=8.0,
        ha='center',
        color='#334155'
    )

    plt.tight_layout()
    plt.savefig('docs/images/embedding_space.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Successfully generated clean, spacious docs/images/embedding_space.png')


if __name__ == '__main__':
    generate_embedding_space()
