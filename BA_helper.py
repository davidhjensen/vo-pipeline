import numpy as np
import cv2

from scipy.sparse import lil_matrix


def as_lk_points(x: np.ndarray) -> np.ndarray:
        """
        Convert keypoints to OpenCV LK format:
        (N,1,2), float32, contiguous
        """
        if x.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        if x.ndim == 2 and x.shape[1] == 2:
            x = x[:, None, :]

        return np.ascontiguousarray(x, dtype=np.float32)


def build_window_data(S: dict) -> tuple[int, list[tuple[int, int]], np.ndarray]:
    """
    Build data structures for sliding window BA
    Args:
        S: current state with observation history
    Returns:
        n_landmarks: number of unique landmarks in the window
        obs_list: list of (frame_idx, landmark_idx) observations
        obs_pixels: (N, 2) array of observed pixel coordinates
    """
    # Get all unique landmark IDs currently being tracked in the window
    active_ids = S["ids"]         
    # Map global IDs to local indices [0...M-1] for the optimizer to work with
    id_to_idx = {id_: i for i, id_ in enumerate(active_ids)}
    n_landmarks = len(active_ids)
    
    obs_list = [] # List of (pose_idx, landmark_idx)
    obs_pixels = []
    
    # Loop through history deque to find observations of our active landmarks
    for f_idx, (pixels, ids) in enumerate(S["obs_history"]):
        for p, id_ in zip(pixels, ids):
            if id_ in id_to_idx:   # only consider landmarks still being tracked
                obs_list.append((f_idx, id_to_idx[id_]))
                obs_pixels.append(p[0]) # (u, v) coordinates
                
    return n_landmarks, obs_list, np.array(obs_pixels)




def compute_rep_err(x_vec, window_poses: list, n_landmarks: int, obs_map: dict, k: np.ndarray) -> np.ndarray:
    """
    Args:
        x_vec: parameters optimized by the solver
        window_poses: list of original (3,4) poses (for anchor/indexing)
        n_landmarks: total unique landmarks in the window
        obs_map: dict {frame_idx: {'ids': [...], 'pixels': [...]}}
        k: camera matrix
    """
    poses, landmarks = unpack_params(x_vec, window_poses, n_landmarks)
    residuals = []

    # Get the indices of frames in the current window (e.g., [0, 1, 2, 3, 4])
    window_indices = sorted(obs_map.keys())

    for f_idx in window_indices:
        pose = poses[f_idx]
        visible_ids = obs_map[f_idx]['ids']
        observed_pixels = obs_map[f_idx]['pixels']
        
        projected = project_points(landmarks[:, visible_ids], pose, k)
        
        # Error calculation
        err = (observed_pixels - projected).flatten()
        residuals.append(err)

    return np.concatenate(residuals)




def project_points(X: np.ndarray, T: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Projects 3D points X into camera with pose T
    Args:
        X: (3, K) landmark coordinates
        T: (3, 4) camera pose [R|t]
    Returns:
        pixels: (K, 2) projected pixel coordinates
    """
    # Transform to camera frame
    X_hom = np.vstack((X, np.ones(X.shape[1])))
    P_cam = T @ X_hom # (3, K)
    
    # Project to image plane
    P_img = k @ P_cam
    
    pixels = P_img[:2] / P_img[2]
    return pixels.T # (K, 2)



def pack_params(window_poses: list[np.ndarray], window_landmarks: np.ndarray) -> np.ndarray:
    """
    Packs the window poses and landmarks into a single parameter vector for optimization.
    Args:
        window_poses: List of (3, 4) matrices in the sliding window
        window_landmarks: (3, M) matrix of landmarks optimized in this window
    Returns:
        x_vec: 1D array of packed parameters
    """
    pose_params = []
    for i in range(1, len(window_poses)):
        R = window_poses[i][:, :3]
        t = window_poses[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)
        pose_params.append(np.hstack((rvec.flatten(), t.flatten())))
    
    # Flatten landmarks (3 * M)
    landmark_params = window_landmarks.flatten()
    
    return np.concatenate((*pose_params, landmark_params))

def unpack_params(x_vec: np.ndarray, window_poses: list[np.ndarray], n_landmarks: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Unpacks the x_vec back into structured poses and landmarks.
    Args:
        x_vec: 1D array of packed parameters
        window_poses: List of (3, 4) matrices in the sliding window
        n_landmarks: Number of landmarks
    Returns:
        poses: Dictionary mapping frame index to (3, 4) pose matrix
        landmarks: (3, M) matrix of landmarks
    """
    n_active_poses = len(window_poses) - 1
    poses = {0: window_poses[0]} # Fix the first pose as the anchor
    
    # Extract poses
    for i in range(n_active_poses):
        idx = i * 6
        rvec = x_vec[idx:idx+3]
        t = x_vec[idx+3:idx+6]
        R, _ = cv2.Rodrigues(rvec)
        poses[i+1] = np.hstack((R, t.reshape(3, 1)))
        
    # Extract landmarks
    l_start = n_active_poses * 6
    landmarks = x_vec[l_start:].reshape(3, n_landmarks)
    
    return (poses, landmarks)


def get_jac_sparsity(n_poses: int, n_landmarks: int, obs_list: list[tuple[int, int]]) -> lil_matrix:
    """
    Compute the Jacobian sparsity pattern for sliding window BA
    Args:
        n_poses: number of poses in the window
        n_landmarks: number of landmarks in the window
        obs_list: list of (frame_idx, landmark_idx) observations
    Returns:
        sparsity: lil_matrix indicating non-zero structure of the Jacobian
    """
    n_vars = (n_poses - 1) * 6 + n_landmarks * 3 # We fix the first pose
    n_residuals = len(obs_list) * 2
    sparsity = lil_matrix((n_residuals, n_vars), dtype=int)
    
    for i, (f_idx, l_idx) in enumerate(obs_list):
        res_idx = i * 2
        # If not the fixed anchor pose (index 0)
        if f_idx > 0:
            var_idx = (f_idx - 1) * 6
            sparsity[res_idx:res_idx+2, var_idx:var_idx+6] = 1
        
        # Link to landmark parameters
        l_var_idx = ((n_poses - 1) * 6) + (l_idx * 3)
        sparsity[res_idx:res_idx+2, l_var_idx:l_var_idx+3] = 1
        
    return sparsity