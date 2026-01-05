import numpy as np


def get_mask_indices(
    H,
    W,
    rows,
    cols,
    pts: np.ndarray
) -> np.ndarray:
    """
    Return grid mask indices for a set of pixel points.

    Masks are ordered left-to-right, top-to-bottom.

    Args:
        pts (np.ndarray): (N, 2) array of pixel coordinates (u, v)
        H (int): image height
        W (int): image width
        rows (int): number of grid rows
        cols (int): number of grid columns

    Returns:
        np.ndarray: (N,) array of mask indices
    """
    
    pts = np.asarray(pts)
    u = pts[:,0]
    v = pts[:,1]

    cell_h = H / rows
    cell_w = W / cols

    row_idx = np.floor(v / cell_h).astype(int)
    col_idx = np.floor(u / cell_w).astype(int)

    # clip to valid range (important for border pixels)
    row_idx = np.clip(row_idx, 0, rows - 1)
    col_idx = np.clip(col_idx, 0, cols - 1)

    return row_idx * cols + col_idx

def fit_ground_plane_ransac(
    pts: np.ndarray,
    n_iters: int = 200,
    dist_thresh: float = 0.05,
    expected_normal: np.ndarray = np.array([0, -1, 0]), 
    angle_thresh: float = 0.95, # Cosine similarity (approx 25 degrees)
):
    """
    Fits a plane, that aligns with normal.
    """
    X = pts.T  # (N,3)
    N = X.shape[0]
    
    # Need at least 3 points
    if N < 3:
        return np.array([0, 1, 0]), 0.0, np.zeros(N, dtype=bool)

    best_inliers = np.zeros(N, dtype=bool)
    best_count = 0
    best_model = (np.array([0, -1, 0]), 0.0) # Default fallback

    for _ in range(n_iters):
        # 1. Sample
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = X[idx]

        # 2. Model
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n = n / norm

        # Angle Constraint
        # Check if normal is roughly parallel to UP vector
        # abs() handles both Up and Down directions
        if abs(np.dot(n, expected_normal)) < angle_thresh:
            continue
        d = -n @ p1

        # 3. Evaluate
        dists = np.abs(X @ n + d)
        inliers = dists < dist_thresh
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (n, d)

    # If no valid plane found
    if best_count < 3:
        return best_model[0], best_model[1], best_inliers

    # 4. Refinement 
    X_in = X[best_inliers]
    centroid = X_in.mean(axis=0)
    # Centering improves numerical stability
    _, _, Vt = np.linalg.svd(X_in - centroid)
    n_refined = Vt[-1]
    
    # Ensure normal points in same direction as expected_normal (optional)
    if np.dot(n_refined, expected_normal) < 0:
        n_refined = -n_refined
        
    d_refined = -n_refined @ centroid

    return n_refined, d_refined, best_inliers

def estimate_ground_height(pts_3d, car_height):
    if pts_3d.shape[1] < 10:
        print("ERROR")
        return None

    # Calculate median depth to scale threshold
    d_ref = np.median(np.linalg.norm(pts_3d, axis=0))
    
    # Relaxed threshold slightly
    dist_thresh = 0.005 * d_ref 

    expected_up = np.array([0, 1, 0]) 
    n, d, inliers = fit_ground_plane_ransac(
        pts_3d, 
        dist_thresh=dist_thresh,
        expected_normal=expected_up
    )

    # Safety check: if empty mask
    if inliers.sum() == 0:
        print("No inliers")
        return None

    # Relaxed ratio check 
    inlier_ratio = inliers.sum() / pts_3d.shape[1]
    # d is distance from origin to plane.
    height = abs(d) #of the points in the world coordinate
    print(inlier_ratio, car_height-height)
    if inlier_ratio > 0.35 and 0.15 < abs(car_height-height)< 1.0:    
        return height, inliers
    else: 
        print("NOT ENOUGH INLIERS TO ESTIMATE A GOOD PLANE ")
        return None, None