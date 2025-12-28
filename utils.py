import numpy as np

def get_weighted_y(pts: np.ndarray, eps :float = 1e-6) -> float: 
    """
    Compute a robust, scale-invariant estimate of the ground Y coordinate.

    Vertical outliers are rejected using MAD-based filtering, then the remaining
    Y values are averaged with inverse-depth weighting so that closer points
    contribute more.

    Args:
        pts (np.ndarray): 3D points of shape (3, N) as [X; Y; Z].
        eps (float): Numerical stability constant.

    Returns:
        float: Distance-weighted ground Y estimate.
    """
    Y = pts[1, :]
    Z = pts[2, :]
    y_med = np.median(Y)
    mad = np.median(np.abs(Y - y_med)) + eps

    mask = np.abs(Y - y_med) / mad < 3.0

    weights = 1.0 / (Z[mask] + eps)
    y_est = np.sum(weights * Y[mask]) / np.sum(weights)

    return y_est

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