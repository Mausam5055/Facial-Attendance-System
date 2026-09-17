import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs('docs/images', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# 1. SYSTEM ARCHITECTURE DIAGRAM (Modern 3-Tier Enterprise Design)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Presentation Layer (Layer 1)
ax.add_patch(patches.FancyBboxPatch((3, 68), 94, 28, boxstyle='round,pad=0.8,rounding_size=1.5', facecolor='#F5F7FF', edgecolor='#4F46E5', linewidth=2.0))
ax.add_patch(patches.FancyBboxPatch((3, 91), 94, 5, boxstyle='round,pad=0.3,rounding_size=1.0', facecolor='#4F46E5', edgecolor='#4F46E5', linewidth=0))
ax.text(50, 93.5, '1. PRESENTATION LAYER (Streamlit Web & Kiosk Interface · app.py)', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#FFFFFF')

ui_boxes = [
    ('Kiosk Scanner', ['Live webcam & snapshot', 'Bounding box match overlay', "Today's check-ins feed", 'Multi-face auto detection'], 5.2, 70.5, 21),
    ('Face Enrollment', ['Single-face quality check', 'Crop preview & thumbnail', 'Registered roster list', 'Instant disk & DB sync'], 28.7, 70.5, 21),
    ('Reports & Analytics', ['Present vs Absent roster', 'Dynamic KPI metrics', 'Altair visual chart feed', 'CSV export serialization'], 52.2, 70.5, 21),
    ('Diagnostics & Audit', ['SQLite storage footprint', 'Backend engine status', 'Real-time audit log stream', 'Search & severity filter'], 75.7, 70.5, 21)
]
for title, items, x, y, w in ui_boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, 18.5, boxstyle='round,pad=0.4,rounding_size=0.8', facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2))
    ax.text(x + w/2, y + 15.2, title, fontsize=9.2, fontweight='bold', ha='center', color='#312E81')
    ax.plot([x + 1.5, x + w - 1.5], [y + 13.5, y + 13.5], color='#E2E8F0', lw=1.0)
    for idx, item in enumerate(items):
        ax.text(x + 1.2, y + 10.5 - idx * 2.7, f"• {item}", fontsize=7.4, color='#334155')

# Service Layer (Layer 2)
ax.add_patch(patches.FancyBboxPatch((3, 35.5), 94, 27.5, boxstyle='round,pad=0.8,rounding_size=1.5', facecolor='#F0FDF4', edgecolor='#16A34A', linewidth=2.0))
ax.add_patch(patches.FancyBboxPatch((3, 58), 94, 5, boxstyle='round,pad=0.3,rounding_size=1.0', facecolor='#16A34A', edgecolor='#16A34A', linewidth=0))
ax.text(50, 60.5, '2. SERVICE LAYER (Computer Vision & Core Domain Logic · services/)', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#FFFFFF')

svc_boxes = [
    ('Module 1: FaceService', ['Haar Cascade / HOG face detection', '128-D metric-space face embeddings', 'Euclidean distance math & bbox logic', 'Deterministic spatial projection fallback'], 5.2, 37.5, 29),
    ('Module 2: EnrollmentService', ['Filesystem reference photo storage', 'Database CRUD on known_faces', 'Zero / multiple face rejection guard', '1:N Euclidean candidate search engine'], 36.2, 37.5, 28),
    ('Module 3: AttendanceService', ['Strict single-mark daily deduplication', 'Unknown identity guard (θ = 0.60)', 'Dynamic daily roster calculation', 'CSV export & manual admin override'], 66.2, 37.5, 30.5)
]
for title, items, x, y, w in svc_boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, 18.5, boxstyle='round,pad=0.4,rounding_size=0.8', facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2))
    ax.text(x + w/2, y + 15.2, title, fontsize=9.2, fontweight='bold', ha='center', color='#14532D')
    ax.plot([x + 1.5, x + w - 1.5], [y + 13.5, y + 13.5], color='#E2E8F0', lw=1.0)
    for idx, item in enumerate(items):
        ax.text(x + 1.2, y + 10.5 - idx * 2.7, f"• {item}", fontsize=7.4, color='#334155')

# Data Layer (Layer 3)
ax.add_patch(patches.FancyBboxPatch((3, 3), 94, 27.5, boxstyle='round,pad=0.8,rounding_size=1.5', facecolor='#F8FAFC', edgecolor='#475569', linewidth=2.0))
ax.add_patch(patches.FancyBboxPatch((3, 25.5), 94, 5, boxstyle='round,pad=0.3,rounding_size=1.0', facecolor='#475569', edgecolor='#475569', linewidth=0))
ax.text(50, 28.0, '3. DATA LAYER (Persistence, Image Repository & Audit Logging · database/, data/, logs/)', fontsize=11.5, fontweight='bold', ha='center', va='center', color='#FFFFFF')

data_boxes = [
    ('SQLite Database (attendance.db)', ['known_faces (id, name, photo_path)', 'attendance (id, name, date, time, conf)', 'Composite unique index on (name, date)', 'Atomic transactions & ACID safety'], 5.2, 5.0, 31),
    ('Image Store (known_faces_images/)', ['Per-member reference directories', 'ref_<timestamp>.jpg reference photos', 'avatar.jpg (80×80 face thumbnail)', 'Raw image file validation & persistence'], 38.2, 5.0, 30),
    ('Audit Trail (logs/app.log)', ['Formatted audit timestamps & levels', 'Match confidence & execution timings', 'Traceability & runtime diagnostics', 'Rolling operational log stream'], 70.2, 5.0, 26.5)
]
for title, items, x, y, w in data_boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, 18.5, boxstyle='round,pad=0.4,rounding_size=0.8', facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.2))
    ax.text(x + w/2, y + 15.2, title, fontsize=9.2, fontweight='bold', ha='center', color='#0F172A')
    ax.plot([x + 1.5, x + w - 1.5], [y + 13.5, y + 13.5], color='#E2E8F0', lw=1.0)
    for idx, item in enumerate(items):
        ax.text(x + 1.2, y + 10.5 - idx * 2.7, f"• {item}", fontsize=7.4, color='#334155')

# Connecting Inter-Layer Arrows
for arrow_x in [15.7, 50.0, 84.3]:
    # Presentation <-> Service
    ax.annotate('', xy=(arrow_x, 63.3), xytext=(arrow_x, 67.8), arrowprops=dict(arrowstyle='<->', color='#4F46E5', lw=2.2, mutation_scale=14))
    # Service <-> Data
    ax.annotate('', xy=(arrow_x, 30.8), xytext=(arrow_x, 35.3), arrowprops=dict(arrowstyle='<->', color='#16A34A', lw=2.2, mutation_scale=14))

plt.tight_layout()
plt.savefig('docs/images/system_architecture.png', bbox_inches='tight', dpi=300)
plt.close()

# -------------------------------------------------------------
# 2. RECOGNITION PIPELINE FLOWCHART
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

ax.text(50, 96, 'Biometric Face Recognition & Attendance Pipeline', fontsize=14, fontweight='bold', ha='center', color='#1E293B')

steps = [
    ('1. Input Acquisition', 'Webcam / Video Stream / Photo Upload\nRGB 3-Channel Image Frame', 10, 80, 80, 9, '#EFF6FF', '#3B82F6'),
    ('2. Preprocessing & Downsampling', 'Downscale frame by factor 0.25 (FRAME_SCALE)\nConvert BGR to RGB / Grayscale', 10, 67, 80, 9, '#F0FDF4', '#22C55E'),
    ('3. Face Detection Engine', 'Haar Feature Cascade (OpenCV) / HOG (dlib)\nExtract Bounding Box Coordinates (top, right, bottom, left)', 10, 54, 80, 9, '#FEF3C7', '#F59E0B'),
    ('4. Face Alignment & Embedding Extraction', 'Normalize facial region to standard crop\nExtract 128-Dimensional Deep / Deterministic Embedding Vector', 10, 41, 80, 9, '#F3E8FF', '#A855F7'),
    ('5. Euclidean Distance Matching Engine', 'Compute distance d(u, v) = sqrt(sum((u_i - v_i)^2)) against enrolled set\nFind candidate with min_dist; verify against threshold theta = 0.60', 10, 28, 80, 9, '#FCE7F3', '#EC4899'),
    ('6. Deduplication & Logging Decision', 'Check SQLite attendance table for (name, today_date)\nIf not present: INSERT INTO attendance; else: Duplicate Suppressed', 10, 15, 80, 9, '#ECFDF5', '#10B981'),
    ('7. Output & Visualization', 'Draw color-coded bounding boxes & confidence scores on UI\nDisplay live toast notification and refresh attendance statistics', 10, 2, 80, 9, '#F1F5F9', '#64748B')
]

for title, desc, x, y, w, h, bg, border in steps:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.5', facecolor=bg, edgecolor=border, linewidth=2))
    ax.text(x + w/2, y + h - 2.8, title, fontsize=10, fontweight='bold', ha='center', color='#0F172A')
    ax.text(x + w/2, y + 2, desc, fontsize=8, ha='center', color='#334155', linespacing=1.2)

for y_arr in [80, 67, 54, 41, 28, 15]:
    ax.annotate('', xy=(50, y_arr - 3.8), xytext=(50, y_arr), arrowprops=dict(facecolor='#1E293B', edgecolor='#1E293B', width=1.5, headwidth=6, headlength=6))

plt.tight_layout()
plt.savefig('docs/images/recognition_pipeline.png', bbox_inches='tight', dpi=300)
plt.close()

# -------------------------------------------------------------
# 3. HAAR CASCADE & HOG FEATURE DIAGRAM
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('(A) Haar-like Feature Prototypes (Viola-Jones)', fontsize=11, fontweight='bold', pad=10)

ax1.add_patch(patches.Rectangle((1, 6.5), 1.5, 2.5, facecolor='white', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((2.5, 6.5), 1.5, 2.5, facecolor='black', edgecolor='black', lw=1.5))
ax1.text(2.5, 5.7, 'Two-Rectangle (Edge)', fontsize=8, ha='center')

ax1.add_patch(patches.Rectangle((5.5, 6.5), 1, 2.5, facecolor='white', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((6.5, 6.5), 1, 2.5, facecolor='black', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((7.5, 6.5), 1, 2.5, facecolor='white', edgecolor='black', lw=1.5))
ax1.text(7, 5.7, 'Three-Rectangle (Line)', fontsize=8, ha='center')

ax1.add_patch(patches.Rectangle((1.5, 1.8), 1.5, 1.5, facecolor='white', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((3, 1.8), 1.5, 1.5, facecolor='black', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((1.5, 0.3), 1.5, 1.5, facecolor='black', edgecolor='black', lw=1.5))
ax1.add_patch(patches.Rectangle((3, 0.3), 1.5, 1.5, facecolor='white', edgecolor='black', lw=1.5))
ax1.text(3, -0.4, 'Four-Rectangle (Diagonal)', fontsize=8, ha='center')

ax1.text(7, 2.2, r'Integral Image Formula:' + '\n' + r'$II(x,y) = \sum_{x^\prime \leq x, y^\prime \leq y} I(x^\prime,y^\prime)$' + '\n' + r'$\Delta = \sum \text{Pixels}_{\text{white}} - \sum \text{Pixels}_{\text{black}}$',
         fontsize=8.5, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F5F9', edgecolor='#64748B'))

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('(B) Histogram of Oriented Gradients (HOG)', fontsize=11, fontweight='bold', pad=10)

for i in range(4):
    for j in range(4):
        ax2.add_patch(patches.Rectangle((1 + i*1.8, 2.5 + j*1.8), 1.8, 1.8, facecolor='#F8FAFC', edgecolor='#94A3B8', lw=1))
        angles = [0, 45, 90, 135]
        theta = np.radians(angles[(i+j)%4])
        dx, dy = 0.5 * np.cos(theta), 0.5 * np.sin(theta)
        cx, cy = 1 + i*1.8 + 0.9, 2.5 + j*1.8 + 0.9
        ax2.arrow(cx - dx/2, cy - dy/2, dx, dy, head_width=0.15, head_length=0.15, fc='#4F46E5', ec='#4F46E5')

ax2.text(4.6, 1.2, r'Gradient: $\nabla I = (\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y})^T, \theta = \arctan(\frac{\partial I / \partial y}{\partial I / \partial x})$' + '\n9-bin orientation histogram per 8x8 pixel cell',
         fontsize=8.5, ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='#EEF2FF', edgecolor='#6366F1'))

plt.tight_layout()
plt.savefig('docs/images/haar_and_hog_diagram.png', bbox_inches='tight', dpi=300)
plt.close()

# -------------------------------------------------------------
# 4. 128-D EMBEDDING SPACE & EUCLIDEAN MATCHING
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

circle_pos = patches.Circle((4, 5.2), 2.4, facecolor='#EFF6FF', edgecolor='#3B82F6', lw=2, linestyle='--')
ax.add_patch(circle_pos)
ax.text(4, 7.9, r'Threshold Margin ($\theta = 0.60$)', fontsize=9, ha='center', color='#1D4ED8', fontweight='bold')

ax.plot(4, 5.2, 'o', markersize=10, color='#1E40AF')
ax.text(4, 4.4, 'Enrolled Reference ($v_{ref}$)\n[Mausam Kar]', fontsize=8.5, ha='center', fontweight='bold', color='#1E40AF')

ax.plot(5.2, 5.9, 'o', markersize=9, color='#16A34A')
ax.text(5.4, 6.3, 'Query Face ($u_{query}$)\n' + r'$d = 0.235 < 0.60 \rightarrow$ MATCH (Present)', fontsize=8, color='#15803D', fontweight='bold')
ax.plot([4, 5.2], [5.2, 5.9], 'k-', lw=1.5)

ax.plot(8.5, 7.5, 'o', markersize=9, color='#DC2626')
ax.text(8.5, 8.2, 'Imposter / Unknown ($w$)\n' + r'$d = 0.890 \geq 0.60 \rightarrow$ REJECT', fontsize=8, ha='center', color='#B91C1C', fontweight='bold')
ax.plot([4, 8.5], [5.2, 7.5], 'r--', lw=1.5)

ax.text(5, 0.9, r'Euclidean Distance: $d(u, v) = \sqrt{\sum_{i=1}^{128} (u_i - v_i)^2}$' + '\n' + r'Match Decision: $\text{Identity} = \arg\min_k d(u, v_k) \quad \text{if } \min d < 0.60 \text{ else "Unknown"}$',
        fontsize=9, ha='center', bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC', edgecolor='#64748B'))

plt.tight_layout()
plt.savefig('docs/images/embedding_space.png', bbox_inches='tight', dpi=300)
plt.close()
print('All technical diagrams generated successfully!')
