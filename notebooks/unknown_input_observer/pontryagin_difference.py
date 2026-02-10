# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog

plt.style.use(['default', './style.mplstyle'])

def get_h_rep(vertices):
    hull = ConvexHull(vertices)
    # A * x <= b
    # hull.equations is [normal, offset] -> normal * x + offset <= 0
    # normal * x <= -offset
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]
    return A, b

def pontryagin_difference(v_A, v_B):
    # Get H-rep of A: H_A * x <= K_A
    H_A, K_A = get_h_rep(v_A)
    
    # Calculate support function h_B(H_A_i) = max(H_A_i * y) for y in vertices of B
    # For each constraint i in A
    s = []
    for i in range(len(H_A)):
        support_val = np.max(np.dot(v_B, H_A[i]))
        s.append(support_val)
    
    s = np.array(s)
    K_diff = K_A - s
    
    return H_A, K_diff

def h_to_v(H, K):
    # 1. Find the Chebyshev center to get a point strictly inside
    # We want to maximize 'r' such that H*x + r*||H_i|| <= K
    norms = np.linalg.norm(H, axis=1).reshape(-1, 1)
    
    # Objective: minimize -r (which maximizes r)
    # Variables are [x1, x2, r]
    c = np.array([0, 0, -1]) 
    
    # Constraints: [H, norms] * [x, r]^T <= K
    A_lp = np.hstack([H, norms])
    res = linprog(c, A_ub=A_lp, b_ub=K, bounds=(None, None), method='highs')
    
    if not res.success or res.x[2] <= 1e-9:
        print("Difference set is likely empty or a lower dimension.")
        return None
    
    interior_point = res.x[:2]

    # 2. Compute Intersection with the strictly interior point
    # Format: [normal, -offset] -> n*x - offset <= 0
    halfspaces = np.column_stack((H, -K))
    try:
        hs = HalfspaceIntersection(halfspaces, interior_point)
        # Return vertices of the resulting intersection
        return hs.intersections
    except Exception as e:
        print(f"Qhull error even with interior point: {e}")
        return None

# Define Polytope A (Square)
# v_A = np.array([[-12, -12], [12, -12], [12, 12], [-12, 12]])
v_A = np.array([[-15, -15], [15, -15], [15, 15], [-15, 15]])

# Define Polytope B (Triangle)
# v_B = np.array([[-6, -6], [6, -6], [6, 6], [-6, 6]])
v_B = np.array([[-9, -9], [9, -9], [9, 9], [-9, 9]])
# v_B = np.array([[-12, -12], [5, -5], [9, 9]])

# Compute Difference
H_diff, K_diff = pontryagin_difference(v_A, v_B)
v_diff = h_to_v(H_diff, K_diff)

# Visualization
plt.figure(figsize=(8, 8))

# Plot A
poly_A = plt.Polygon(v_A, fill=True, color='lightblue', alpha=0.3, label='Set $\mathcal{D}$')
plt.gca().add_patch(poly_A)

# Plot B (at origin)
poly_B = plt.Polygon(v_B, fill=True, color='orange', alpha=0.5, label='Set $\mathcal{R}$ (at origin)')
plt.gca().add_patch(poly_B)

# Plot A - B
if v_diff is not None:
    hull_diff = ConvexHull(v_diff)
    v_diff_sorted = v_diff[hull_diff.vertices]
    poly_diff = plt.Polygon(v_diff_sorted, fill=False, edgecolor='blue', linestyle='--', linewidth=2, alpha=0.6, label='$\mathcal{D} \ominus \mathcal{R}$ (Pontryagin Difference)')
    plt.gca().add_patch(poly_diff)
    
    # Illustrate the property: if x in A-B, then x + B subset A
    # Let's pick a vertex of the difference and show x + B
    sample_x = v_diff_sorted[0]
    v_B_shifted = v_B + sample_x
    poly_B_shifted = plt.Polygon(v_B_shifted, fill=False, edgecolor='red', linestyle='--', linewidth=2, label='$x + \mathcal{R} \in \mathcal{D}$ (for $x \in \mathcal{D} \ominus \mathcal{R}$)')
    plt.gca().add_patch(poly_B_shifted)

plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
# plt.title('Pontryagin Difference of Polytopes $\mathcal{$')
plt.savefig('.figures/pontryagin_difference.png')
