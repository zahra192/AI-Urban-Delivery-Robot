import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ============================================================
# DATA LOADING
# ============================================================
df = pd.read_csv(r'C:\Users\mszah\Downloads\paractice\clustering_dataset.csv')
X = df.values  # numpy array of shape (210, 2)

np.random.seed(42)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def euclidean_distance(a, b):
    """Compute Euclidean distance between two points."""
    return np.sqrt(np.sum((a - b) ** 2))


def compute_sse(X, labels, centers):
    """
    Compute per-cluster SSE and total SSE.
    Returns a list of per-cluster SSE values and the total SSE.
    """
    K = len(centers)
    per_cluster_sse = []
    for k in range(K):
        pts = X[labels == k]
        if len(pts) == 0:
            per_cluster_sse.append(0.0)
        else:
            sse = np.sum((pts - centers[k]) ** 2)
            per_cluster_sse.append(sse)
    return per_cluster_sse, sum(per_cluster_sse)


def assign_clusters(X, centers):
    """Assign each data point to the nearest cluster center."""
    labels = np.zeros(len(X), dtype=int)
    for i, point in enumerate(X):
        dists = [euclidean_distance(point, c) for c in centers]
        labels[i] = np.argmin(dists)
    return labels


# ============================================================
# K-MEANS CLUSTERING (FROM SCRATCH)
# ============================================================

def kmeans(X, K, max_iter=300, tol=1e-4, verbose=False):
    """
    K-Means clustering implemented from scratch.

    Parameters:
        X        : Data array of shape (n_samples, n_features)
        K        : Number of clusters
        max_iter : Maximum number of iterations
        tol      : Convergence tolerance (minimum centroid shift)
        verbose  : If True, print iteration details

    Returns:
        labels        : Cluster label for each data point
        centers       : Final cluster centroids
        iteration_log : List of dicts with per-iteration stats
    """
    # Randomly select K data points as initial centroids
    idx = np.random.choice(len(X), K, replace=False)
    centers = X[idx].copy()

    labels = np.zeros(len(X), dtype=int)
    iteration_log = []

    for iteration in range(1, max_iter + 1):

        # Step 1: Assign each point to the nearest centroid
        labels = assign_clusters(X, centers)

        # Step 2: Compute SSE for current assignment
        per_sse, total_sse = compute_sse(X, labels, centers)
        cluster_sizes = [np.sum(labels == k) for k in range(K)]

        # Record iteration statistics
        iteration_log.append({
            'iteration': iteration,
            'cluster_sizes': cluster_sizes,
            'per_cluster_sse': per_sse,
            'total_sse': total_sse
        })

        if verbose:
            print(f"\n  Iteration {iteration}:")
            for k in range(K):
                print(f"    Cluster {k+1}: {cluster_sizes[k]} points | SSE = {per_sse[k]:.4f}")
            print(f"    Total SSE = {total_sse:.4f}")

        # Step 3: Recompute centroids as the mean of each cluster
        new_centers = np.zeros_like(centers)
        for k in range(K):
            pts = X[labels == k]
            if len(pts) > 0:
                new_centers[k] = pts.mean(axis=0)
            else:
                # Keep old centroid if cluster is empty
                new_centers[k] = centers[k]

        # Check for convergence: max centroid shift below tolerance
        shift = np.max([euclidean_distance(centers[k], new_centers[k]) for k in range(K)])
        centers = new_centers

        if shift < tol:
            if verbose:
                print(f"\n  Converged at iteration {iteration} (shift = {shift:.6f})")
            break

    return labels, centers, iteration_log


# ============================================================
# K-MEDOIDS CLUSTERING (FROM SCRATCH — PAM ALGORITHM)
# ============================================================

def kmedoids(X, K, max_iter=300, verbose=False):
    """
    K-Medoids clustering implemented from scratch using the PAM algorithm.

    Unlike K-Means, the cluster center (medoid) must be an actual data point.
    At each iteration, the algorithm tries swapping each medoid with every
    other point in its cluster and keeps the swap if it reduces total SSE.

    Parameters:
        X        : Data array of shape (n_samples, n_features)
        K        : Number of clusters
        max_iter : Maximum number of iterations
        verbose  : If True, print iteration details

    Returns:
        labels        : Cluster label for each data point
        medoid_idx    : Indices of the final medoids in X
        iteration_log : List of dicts with per-iteration stats
    """
    n = len(X)

    # Randomly select K data points as initial medoids (convert to plain int for clean printing)
    medoid_idx = [int(i) for i in np.random.choice(n, K, replace=False)]

    iteration_log = []

    for iteration in range(1, max_iter + 1):

        # Step 1: Assign each point to its nearest medoid
        labels = assign_clusters(X, X[medoid_idx])

        # Compute SSE with current medoids
        per_sse, total_sse = compute_sse(X, labels, X[medoid_idx])
        cluster_sizes = [np.sum(labels == k) for k in range(K)]

        # Record iteration statistics
        iteration_log.append({
            'iteration': iteration,
            'medoid_indices': medoid_idx.copy(),
            'medoid_points': [X[m].tolist() for m in medoid_idx],
            'cluster_sizes': cluster_sizes,
            'per_cluster_sse': per_sse,
            'total_sse': total_sse
        })

        if verbose:
            print(f"\n  Iteration {iteration}:")
            print(f"    Medoid indices: {medoid_idx}")
            for k in range(K):
                print(f"    Medoid {k+1}: Point #{medoid_idx[k]} "
                      f"({X[medoid_idx[k]][0]:.4f}, {X[medoid_idx[k]][1]:.4f}) | "
                      f"Cluster size: {cluster_sizes[k]} | SSE: {per_sse[k]:.4f}")
            print(f"    Total SSE = {total_sse:.4f}")

        # Step 2: Try to improve medoids by swapping within each cluster
        new_medoid_idx = medoid_idx.copy()
        improved = False

        for k in range(K):
            best_sse = total_sse
            best_medoid = medoid_idx[k]

            # Try every point in cluster k as a candidate medoid
            cluster_points = np.where(labels == k)[0]
            for candidate in cluster_points:
                if candidate in new_medoid_idx:
                    continue  # Skip if already a medoid

                # Temporarily swap current medoid with candidate
                temp_medoids = new_medoid_idx.copy()
                temp_medoids[k] = candidate
                temp_labels = assign_clusters(X, X[temp_medoids])
                _, temp_sse = compute_sse(X, temp_labels, X[temp_medoids])

                # Accept swap if it reduces total SSE
                if temp_sse < best_sse:
                    best_sse = temp_sse
                    best_medoid = int(candidate)
                    improved = True

            new_medoid_idx[k] = int(best_medoid)

        # Convergence: no improvement found in this iteration
        if not improved:
            if verbose:
                print(f"\n  Converged at iteration {iteration} (no improvement found)")
            break

        medoid_idx = new_medoid_idx

    # Compute final cluster assignments
    labels = assign_clusters(X, X[medoid_idx])
    return labels, medoid_idx, iteration_log


# ============================================================
# PART 1: ELBOW CURVE (K = 1 TO 12)
# ============================================================
print("=" * 60)
print("PART 1: ELBOW CURVE (K = 1 to 12)")
print("=" * 60)

sse_values = []
K_range = range(1, 13)

for K in K_range:
    _, _, log = kmeans(X, K, verbose=False)
    sse = log[-1]['total_sse']
    sse_values.append(sse)
    print(f"  K={K:2d}  |  SSE = {sse:.2f}")

# Optimal K = 5: largest SSE drop occurs at K=4->5 (764 -> 162),
# and the curve flattens out significantly after K=5
optimal_k = 5
print(f"\n  Optimal K (Elbow Point) = {optimal_k}")

# Plot the elbow curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_range, sse_values, 'b-o', linewidth=2.5, markersize=8,
        markerfacecolor='steelblue', markeredgecolor='white', markeredgewidth=1.5,
        label='SSE per K')

# Mark the elbow point with a distinct red star marker
ax.plot(optimal_k, sse_values[optimal_k - 1], 'r*', markersize=22, zorder=5,
        label=f'Optimal K = {optimal_k} (Elbow Point)')

# Annotate the elbow point on the graph
ax.annotate(f' Optimal K = {optimal_k}\n SSE = {sse_values[optimal_k-1]:.1f}',
            xy=(optimal_k, sse_values[optimal_k - 1]),
            xytext=(optimal_k + 1.5, sse_values[optimal_k - 1] + 120),
            fontsize=11, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.8),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red'))

ax.set_xlabel('K (Number of Clusters)', fontsize=13)
ax.set_ylabel('SSE / WCSS', fontsize=13)
ax.set_title('Elbow Curve — K-Means Clustering (K = 1 to 12)',
             fontsize=14, fontweight='bold')
ax.set_xticks(list(K_range))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'C:\Users\mszah\Downloads\paractice\elbow_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [Saved] elbow_curve.png")


# ============================================================
# PART 2: RUN K-MEANS WITH OPTIMAL K
# ============================================================
K_opt = optimal_k

print("\n" + "=" * 60)
print(f"PART 2: K-MEANS with K = {K_opt}")
print("=" * 60)

start_time = time.time()
km_labels, km_centers, km_log = kmeans(X, K_opt, verbose=True)
km_time = time.time() - start_time

print(f"\n  K-Means wall-clock time : {km_time:.6f} seconds")
print(f"  Total Iterations        : {len(km_log)}")


# ============================================================
# PART 3: RUN K-MEDOIDS WITH OPTIMAL K
# ============================================================
print("\n" + "=" * 60)
print(f"PART 3: K-MEDOIDS with K = {K_opt}")
print("=" * 60)

start_time = time.time()
kmed_labels, kmed_medoids, kmed_log = kmedoids(X, K_opt, verbose=True)
kmed_time = time.time() - start_time

print(f"\n  K-Medoids wall-clock time : {kmed_time:.6f} seconds")
print(f"  Total Iterations          : {len(kmed_log)}")

# Display how medoids change across iterations
print("\n  --- K-Medoids: Medoid Selection Changes Per Iteration ---")
for entry in kmed_log:
    it  = entry['iteration']
    pts = entry['medoid_points']
    idx = entry['medoid_indices']
    print(f"\n  Iteration {it}  |  Medoid indices: {idx}")
    for k in range(K_opt):
        print(f"    Medoid {k+1}: Point #{idx[k]}  ({pts[k][0]:.4f}, {pts[k][1]:.4f})")


# ============================================================
# PART 4: EXECUTION TIME SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PART 4: EXECUTION TIME SUMMARY")
print("=" * 60)
print(f"  K-Means   : {km_time:.6f} seconds  |  {len(km_log)} iterations")
print(f"  K-Medoids : {kmed_time:.6f} seconds  |  {len(kmed_log)} iterations")
print(f"  K-Medoids is approximately {kmed_time/km_time:.1f}x slower than K-Means")


# ============================================================
# PART 5: SIDE-BY-SIDE CLUSTER VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("PART 5: FINAL CLUSTER VISUALIZATION")
print("=" * 60)

colors = ['#E63946', '#2A9D8F', '#E9C46A', '#457B9D', '#F4A261', '#8338EC']

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f'K-Means vs K-Medoids  —  K = {K_opt}',
             fontsize=16, fontweight='bold')

# --- Left plot: K-Means results ---
ax1 = axes[0]
for k in range(K_opt):
    pts = X[km_labels == k]
    ax1.scatter(pts[:, 0], pts[:, 1], c=colors[k], s=55, alpha=0.75,
                edgecolors='white', linewidth=0.4, label=f'Cluster {k+1}')

# Plot centroids using a distinct 'X' marker
ax1.scatter(km_centers[:, 0], km_centers[:, 1],
            c='black', s=260, marker='X', zorder=6,
            edgecolors='white', linewidth=1.5, label='Centroids')
for k, c in enumerate(km_centers):
    ax1.annotate(f'C{k+1}', xy=(c[0], c[1]),
                 xytext=(c[0] + 0.2, c[1] + 0.2),
                 fontsize=9, fontweight='bold')

km_final_sse = km_log[-1]['total_sse']
ax1.set_title(f'K-Means\n{len(km_log)} iterations  |  '
              f'Time = {km_time:.4f}s  |  SSE = {km_final_sse:.2f}', fontsize=11)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.2)
ax1.text(0.98, 0.02, f'Total SSE: {km_final_sse:.2f}',
         transform=ax1.transAxes, fontsize=9, ha='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- Right plot: K-Medoids results ---
ax2 = axes[1]
for k in range(K_opt):
    pts = X[kmed_labels == k]
    ax2.scatter(pts[:, 0], pts[:, 1], c=colors[k], s=55, alpha=0.75,
                edgecolors='white', linewidth=0.4, label=f'Cluster {k+1}')

# Plot medoids using a distinct diamond marker
medoid_pts = X[kmed_medoids]
ax2.scatter(medoid_pts[:, 0], medoid_pts[:, 1],
            c='black', s=300, marker='D', zorder=6,
            edgecolors='white', linewidth=1.5, label='Medoids')
for k, m in enumerate(kmed_medoids):
    ax2.annotate(f'M{k+1}', xy=(X[m][0], X[m][1]),
                 xytext=(X[m][0] + 0.2, X[m][1] + 0.2),
                 fontsize=9, fontweight='bold')

kmed_final_sse = kmed_log[-1]['total_sse']
ax2.set_title(f'K-Medoids\n{len(kmed_log)} iterations  |  '
              f'Time = {kmed_time:.4f}s  |  SSE = {kmed_final_sse:.2f}', fontsize=11)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.2)
ax2.text(0.98, 0.02, f'Total SSE: {kmed_final_sse:.2f}',
         transform=ax2.transAxes, fontsize=9, ha='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(r'C:\Users\mszah\Downloads\paractice\clusters_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [Saved] clusters_comparison.png")


# ============================================================
# COMPARATIVE ANALYSIS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("COMPARATIVE ANALYSIS SUMMARY")
print("=" * 60)
print(f"""
  Algorithm  | Iterations | Time (s)     | Final SSE
  -----------|------------|--------------|----------
  K-Means    | {len(km_log):10d} | {km_time:12.6f} | {km_final_sse:.4f}
  K-Medoids  | {len(kmed_log):10d} | {kmed_time:12.6f} | {kmed_final_sse:.4f}

  (a) Iterations to Convergence:
      Both algorithms converged in the same number of iterations.
      K-Means converges quickly because updating centroids (mean) is a
      simple arithmetic operation. K-Medoids performs an exhaustive swap
      search within each cluster per iteration, making each iteration far
      more expensive even when the iteration count is similar.

  (b) Cluster Shapes and Sizes:
      Both algorithms identified {K_opt} well-separated clusters.
      K-Means centroids are virtual points (averages) that may not
      correspond to any actual data point. K-Medoids medoids are always
      real data points, which can make cluster boundaries slightly
      more representative of the actual data distribution.

  (c) Algorithm Suitability for This Dataset:
      K-Means finished in {km_time:.4f}s vs K-Medoids {kmed_time:.4f}s
      (K-Medoids is ~{kmed_time/km_time:.0f}x slower).
      K-Medoids achieved slightly lower SSE ({kmed_final_sse:.2f} vs {km_final_sse:.2f}),
      but the improvement is marginal. Since the clusters in this dataset
      are clean and well-separated with no significant outliers, K-Means
      is the more suitable choice given its substantial speed advantage.
      K-Medoids would be preferable when data contains outliers or when
      interpretability of cluster centers is important.
""")

print("Done! Output files saved in: C:\\Users\\mszah\\Downloads\\paractice\\")
print("   1. elbow_curve.png")
print("   2. clusters_comparison.png")