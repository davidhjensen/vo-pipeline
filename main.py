import os
from glob import glob

from typing import Tuple, Dict

import cv2
import numpy as np
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from visualization import initTrajectoryPlot, updateTrajectoryPlot, draw_optical_flow
# Dataset -> 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
DATASET = 3

# Define dataset paths
# (Set these variables before running)
kitti_path = "kitti/kitti05/kitti"
malaga_path = "malaga/malaga-urban-dataset-extract-07"
parking_path = "parking/parking"
own_dataset_path = "VAMR_Rome_dataset"

if DATASET == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
    img_dir = os.path.join(kitti_path, '05/image_0')
    images = sorted(glob(os.path.join(img_dir, '*.png')))
    last_frame = 4540
    K = np.array([
        [7.18856e+02, 0, 6.071928e+02],
        [0, 7.18856e+02, 1.852157e+02],
        [0, 0, 1]
    ])
    ground_truth = np.loadtxt(os.path.join(kitti_path, 'poses', '05.txt'))
    ground_truth = ground_truth[:, [-9, -1]]  # same as MATLAB(:, [end-8 end])

elif DATASET == 1:
    assert 'malaga_path' in locals(), "You must define malaga_path"
    img_dir = os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images')
    images = sorted(glob(os.path.join(img_dir, '*.jpg')))
    images = images[0::2]   # left
    last_frame = len(images)
    K = np.array([
        [621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]
    ])
    ground_truth = None
    
elif DATASET == 2:
    assert 'parking_path' in locals(), "You must define parking_path"
    img_dir = os.path.join(parking_path, 'images')
    images = sorted(glob(os.path.join(img_dir, '*.png')))
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=",", usecols=(0, 1, 2))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    
elif DATASET == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"
    img_dir=os.path.join(own_dataset_path, 'images')
    images = sorted(glob(os.path.join(img_dir, '*.png')))
    last_frame = len(images)
    K = np.array([
        [1.05903465e+03, 0.00000000e+00, 6.29060709e+02],
        [0.00000000e+00, 1.06306400e+03, 3.28563696e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])
    ground_truth = None

else:
    raise ValueError("Invalid dataset index")

# Paramaters for Shi-Tomasi corners
if DATASET == 0: 
    feature_params = dict( maxCorners = 60,
                        qualityLevel = 0.01,
                        minDistance = 10,
                        blockSize = 7)

    # Parameters for LKT
    lk_params = dict( winSize  = (21, 21),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.001))
    
    # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
    new_feature_min_squared_diff = 4
    rows_roi_corners = 2
    cols_roi_corners = 4

elif DATASET == 1: 
    feature_params = dict( maxCorners = 60,
                        qualityLevel = 0.05,
                        minDistance = 10,
                        blockSize = 9 )

    # Parameters for LKT
    lk_params = dict( winSize  = (21, 21),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
    new_feature_min_squared_diff = 4
    rows_roi_corners = 3
    cols_roi_corners = 3

elif DATASET == 3: 
    feature_params = dict( maxCorners = 60,
                        qualityLevel = 0.05,
                        minDistance = 10,
                        blockSize = 9 )
    # Parameters for LKT
    lk_params = dict( winSize  = (21, 21),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
    new_feature_min_squared_diff = 4
    rows_roi_corners = 3
    cols_roi_corners = 3


# Next keyframe to use for bootstrapping
KITTI_BS_KF = 3
MALAGA_BS_KF = 5
PARKING_BS_KF = 3
CUSTOM_BS_KF = 5

# Number of rows and columns to divide image into for feature detection and number of features to track in each cell
KITTI_ST_ROWS, KITTI_ST_COLS, KITTI_NUM_FEATURES = 2, 4, 20
MALAGA_ST_ROWS, MALAGA_ST_COLS, MALAGA_NUM_FEATURES = 2, 4, 20
PARKING_ST_ROWS, PARKING_ST_COLS, PARKING_NUM_FEATURES = 2, 4, 20
CUSTOM_ST_ROWS, CUSTOM_ST_COLS, CUSTOM_NUM_FEATURES = 2, 4, 20


class VO_Params():
    bs_kf_1: str # path to first keyframe used for bootstrapping dataset
    bs_kf_2: str # path to second keyframe used for bootstrapping dataset
    feature_masks: list[np.ndarray] # mask image into regions for feature tracking 
    shi_tomasi_params: dict
    klt_params: dict
    k: np.ndarray # camera intrinsics matrix
    start_idx: int # index of the frame to start continous operation at (2nd bootstrap keyframe index)
    new_feature_min_squared_diff: float # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
    rows_roi_corners :int
    cols_roi_corners :int
    abs_eig_min: float =1e-2
    
    
    # ADD NEW PARAMS HERE

    if DATASET == 0: 
        alpha: float = 0.02
    elif DATASET == 1: 
        alpha :float = 0.02
    elif DATASET == 3: 
        alpha :float = 0.02

    def __init__(self, bs_kf_1, bs_kf_2, shi_tomasi_params, klt_params, k, start_idx, new_feature_min_squared_diff):
        self.bs_kf_1 = bs_kf_1
        self.bs_kf_2 = bs_kf_2
        self.feature_masks = self.get_feature_masks(bs_kf_1, rows_roi_corners, cols_roi_corners)
        self.shi_tomasi_params = shi_tomasi_params
        self.klt_params = klt_params
        self.k = k
        self.start_idx = start_idx
        self.new_feature_min_squared_diff = new_feature_min_squared_diff
    
    def get_feature_masks(self, img_path, rows, cols) -> list[np.ndarray]:
        """Generate masks for each cell in a grid

        Args:
            img_path (str): path to an image with dimensions to be used for masking
            rows (int): number of rows to split the image into
            cols (int): number of columns to split the image into

        Returns:
            list[np.ndarray]: a list of masks for each grid cell
        """
        # get image shape
        img = cv2.imread(img_path)
        H, W = img.shape[:2]

        # get boundries of the cells
        row_boundries = np.linspace(0, H, rows + 1, dtype=int)
        col_boundries = np.linspace(0, W, cols + 1, dtype=int)

        # create masks left to right, top to bottom
        masks = []
        for row in range(rows):
            for col in range(cols):
                mask = np.zeros((H, W), dtype="uint8")
                r_s, r_e = row_boundries[[row, row + 1]]
                c_s, c_e = col_boundries[[col, col + 1]]
                mask[r_s:r_e, c_s:c_e] = 255
                masks.append(mask)

        return masks

if DATASET == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
    bs_kf_1 = images[0]
    bs_kf_2 = images[KITTI_BS_KF]
    start_idx = KITTI_BS_KF
    # ADD NEW PARAMS HERE

elif DATASET == 1:
    assert 'malaga_path' in locals(), "You must define malaga_path"
    bs_kf_1 = images[0]
    bs_kf_2 = images[MALAGA_BS_KF]
    start_idx = MALAGA_BS_KF
    # ADD NEW PARAMS HERE

elif DATASET == 2:
    assert 'parking_path' in locals(), "You must define parking_path"
    bs_kf_1 = images[0]
    bs_kf_2 = images[PARKING_BS_KF]
    start_idx = PARKING_BS_KF
    # ADD NEW PARAMS HERE

elif DATASET == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"
    bs_kf_1 = images[0]
    bs_kf_2 = images[CUSTOM_BS_KF]
    start_idx = CUSTOM_BS_KF
    # ADD NEW PARAMS HERE

else:
    raise ValueError("Invalid dataset index")

# ADD NEW PARAMS HERE TO THE INIT
params = VO_Params(bs_kf_1, bs_kf_2, feature_params, lk_params, K, start_idx, new_feature_min_squared_diff)
class Pipeline():

    params: VO_Params

    def __init__(self, params: VO_Params):

        self.params = params

        return

    def extractFeaturesBootstrap(self):
        """
        Step 1 (Initialization): detect Shi-Tomasi corners on a grid using feature masks.

        Returns:
            st_corners (np.ndarray): (N, 1, 2) float32 corners for KLT tracking.
        """
        st_corners = np.empty((0, 1, 2), dtype=np.float32)
        img_grayscale = cv2.imread(self.params.bs_kf_1, cv2.IMREAD_GRAYSCALE)
        for n, mask in enumerate(self.params.feature_masks):
            features = cv2.goodFeaturesToTrack(img_grayscale, mask=mask, **self.params.shi_tomasi_params)
            # If no corners are found in this region, skip it
            if features is None: 
                print(f"No features found for mask {n+1}!")
                continue
            # Warn if very few features were found in this region (not necessarily an error)
            if features.shape[0] < 10:
                #print(f"Only {features.shape[0]} features found for mask {n+1}!")
                pass
            st_corners = np.vstack((st_corners, features))
        return st_corners

    def trackForwardBootstrap(self, st_corners_kf_1):
        """
        Module 2-Initialization: track keypoints from bs_kf_1 to bs_kf_2 with KLT across intermediate frames.

        Args:
            st_corners_kf_1 (np.ndarray): (N,1,2) keypoints in bs_kf_1.

        Returns:
            initial_points[still_detected] (np.ndarray): (N,1,2) points in bs_kf_1 that were tracked to the next keyframe.
            points[still_detected] (np.ndarray): (M,1,2) tracked points in bs_kf_2.
        """
        img_bs_kf_1_index=images.index(self.params.bs_kf_1)
        img_bs_kf_2_index=images.index(self.params.bs_kf_2)
        still_detected=np.ones(st_corners_kf_1.shape[0],dtype=bool)
        points = self.as_lk_points(st_corners_kf_1.copy())
        initial_points = st_corners_kf_1.copy()
        #Track keypoints frame-by-frame from first bs frame to second bs frame
        for i in range(img_bs_kf_1_index, img_bs_kf_2_index):
            current_image=cv2.imread(images[i],cv2.IMREAD_GRAYSCALE)
            next_image=cv2.imread(images[i+1],cv2.IMREAD_GRAYSCALE)
            nextPts,status,error=cv2.calcOpticalFlowPyrLK(current_image,next_image,points, None, **self.params.klt_params)
            points=nextPts
            status=status.flatten()
            still_detected=still_detected & (status==1)

        # Keep only points that were successfully tracked throughout
        return initial_points[still_detected], points[still_detected]

    def findRelativePose(self, points1, points2):
        #F mat using ransac
        fundamental_matrix, inliers =cv2.findFundamentalMat(points1,points2,cv2.FM_RANSAC,ransacReprojThreshold=1.0)

        #using boolean vector
        inliers = inliers.ravel().astype(bool)

        #compute the essential matrix
        E= K.T@ fundamental_matrix@K

        #recover the relative camera pose
        _,R,t,_=cv2.recoverPose(E,points1[inliers],points2[inliers],K)

        return np.hstack((R, t)), points1[inliers, :, :], points2[inliers, :, :]

    def bootstrapPointCloud(self, H: np.ndarray, points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
        """Bootstrap the initial 3D point cloud using least squares assuming the first frame is the origin

        Args:
            params (VO_Params): params object for the dataset being used
            H (np.ndarray): homographic transformation from bootstrap keyframe 1 to 2
            points_1 (np.ndarray): keypoints detected in bootstrap keyframe 1
            points_2 (np.ndarray): keypoints tracked in bootstrap keyframe 2

        Returns:
            np.ndarray: [3 x k] array of triangulated points
        """
        # projection matrices
        proj_1 = self.params.k @ np.hstack([np.eye(3), np.zeros((3,1))])
        proj_2 = self.params.k @ H
        
        # reshaping the points into 2xK
        p1 = points_1.reshape(-1,2).T
        p2 = points_2.reshape(-1,2).T

        # triangulate homogeneous coordinates using DLT
        points_homo = cv2.triangulatePoints(proj_1, proj_2, p1, p2)

        # convert back to 3D
        points_3d = (points_homo[:3, :]/points_homo[3, :])

        return points_3d

    def bootstrapState(self, P_i: np.ndarray, X_i: np.ndarray) -> dict[str : np.ndarray]:
        """
            Builds the initial state taking the 2D keypoints and their respective 3D points generated by the bootstrap
            
            Args:
                P_i: kx1x2 matrix of 2D keypoints found from the first two frames
                X_i: 3xk matrix containing the 3D projections of the keypoints
            
            Returns:
                dictionary of dictionaries, where every variable of interest is a key and its value is the matrix (e.g. S["P"] returns a kx1x2 matrix of keypoints)  
        """
        S : dict[str : np.ndarray] = {}
        assert P_i.shape[0] == X_i.shape[1], "2D keypoints number of rows should match the 3D keypoints number of columns"
        
        S["P"] = P_i    
        S["X"] = X_i    
        S["C"] = np.empty((0,1,2))
        S["F"] = np.empty_like(S["C"])
        S["T"] = np.empty((0,12))

        return S

    def trackForward(self, state: dict[str:np.ndarray], img_1: np.ndarray, img_2: np.ndarray) -> Tuple[dict[str:np.ndarray], np.ndarray, np.ndarray]:
        """
        Track 2D keypoints from img_1 to img_2 using KLT optical flow

        Args:
            img_1 (np.ndarray): first image (grayscale)
            img_2 (np.ndarray): second image (grayscale)
            points_1 (np.ndarray): keypoints in img_1 to be tracked
        Returns:
            dict[str: np.ndarray]: updated state
            np.ndarray: keypoints from the previous frame that were successfully tracked forward
            np.ndarray: candidate keypoints from the previous frame that were successfully tracked forward
        """
        # NOTE: might make sense to replace assert with ifs since maybe even if we have no points nor candidates we still want to return them empty again
        # FIRST WE TRACK "ESTABILISHED KEYPOINTS"
        points_2D = self.as_lk_points(state["P"])
        assert points_2D.shape[0] > 0, "There are no keypoints here, we can't track them forward"    

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_1, nextImg=img_2, prevPts=points_2D, nextPts=None, **self.params.klt_params)
        status = status.flatten()   # we are going to use them as booleans so maybe we should cast them with astype (?)
        
        # update the state with the current points
        state["P"] = current_points[status == 1]
        state["X"] = state["X"][:, status == 1]  # only get the ones with "true" status and slice them as 3xk
        

        # THEN WE TRACK CANDIDATES - but in the first frame there are no candidates to track other than the established points
        # Therefore 
        candidates = self.as_lk_points(state["C"])
        if candidates.shape[0] != 0:
            assert candidates.dtype == np.float32
            assert candidates.ndim == 3 and candidates.shape[1:] == (1, 2)
            current_cands, status_cands, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_1, nextImg=img_2, prevPts=candidates, nextPts=None, **self.params.klt_params)
            status_cands = status_cands.flatten() # same as above
            
            state["C"] = current_cands[status_cands == 1]

            # initial observations, still have some doubts on these two
            state["F"] = state["F"][status_cands == 1]
            state["T"] = state["T"][status_cands == 1]

        return state, points_2D[status == 1], candidates[status_cands == 1]

    def estimatePose(self, state: dict[str:np.ndarray]) -> tuple[dict[str:np.ndarray], np.ndarray, np.ndarray]:
        """
        Estimate camera pose using PnP RANSAC and update state to keep only inliers
        
        Args:
            params (VO_Params): parameters for the VO pipeline
            state (dict): current state that also contains 2D keypoints and 3D points
        
        Returns:
            tuple[dict, np.ndarray, np.ndarray]: updated state with only inliers for P and X and camera pose as 3x4 matrix and index of inliers
        """
        
        pts_2D = state["P"][:, 0, :]
        pts_3D = state["X"]
        pts_3D = pts_3D.T # according to documentation we need them as kx3 not 3xk
        K = self.params.k

        assert state["X"].shape[1] == state["P"].shape[0], "The number of 2D points does not match the number of 3D correspondencies"

        success, r_vec, t_vec, inliers_idx =  cv2.solvePnPRansac(
            objectPoints=pts_3D,
            imagePoints=pts_2D,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=5.0,
            confidence=0.99,
            iterationsCount=100
        )

        if not success:
            print("Pose estimation failed")
            return ({}, np.zeros((3,4))) # maybe we could raise an error instead of returning this
        
        # r_vec needs to be converted into a 3x3
        R, _ = cv2.Rodrigues(r_vec)    
        T_w2c = np.hstack((R, t_vec))
        # now, inliers are the indices in pts_2d and pts_3d corresponding to the inliers; it is a 2D since openCV returns it as such, so we need to convert it in a 1D array to use np features
        inliers_idx = inliers_idx.flatten()
        new_state = state.copy()
        # by doing the following thing the idea is that we are only keeping the inliers
        new_state["P"] = state["P"][inliers_idx]
        new_state["X"] = state["X"][:, inliers_idx] # slice this since we want it as a 3xk

        return new_state, T_w2c, inliers_idx
    

    def tryTriangulating(self, params: VO_Params, state: dict[str:np.ndarray], cur_pose: np.ndarray) -> dict[str:np.ndarray]:
        """
        Triangulate new points based on the bearing angle threshold to ensure sufficient baseline without relying on scale (ambiguous)
        
        Args:
            params (VO_Params): parameters for the VO pipeline
            state (dict): current state that also contains 2D keypoints and 3D points
            cur_pose (np.ndarray): pose of current frame (3x4 matrix) 
        Returns:
            tuple[dict, np.ndarray]: updated state with only inliers for P and X and camera pose as 3x4 matrix
        """
        
        if state["C"].shape[0] == 0:
            return state

        pts_2D_cur = state["C"]  #(m,1,2)
        pts_2D_first_obs = state["F"] #(m,1,2)
        pts_2D_cur = pts_2D_cur[:, 0, :]  #(m,2)
        pts_2D_first_obs = pts_2D_first_obs[:, 0, :]  #(m,2)
        m = pts_2D_cur.shape[0]
        if pts_2D_first_obs.shape[0] != m: 
            print("ERROR in shapes of tracked keypoints")

        pts_2D_cur_hom = np.column_stack((pts_2D_cur, np.ones(m)))
        pts_2D_first_obs_hom = np.column_stack((pts_2D_first_obs, np.ones(m)))
        K = params.k
        K_inv = np.linalg.inv(K)
        n_pts_2D_cur = K_inv@pts_2D_cur_hom.T #(3,m)
        n_pts_2D_first_obs = K_inv@pts_2D_first_obs_hom.T #(3,m)
        
        first_poses_mat = state["T"].copy()  #flattened on C order (m,12)
        first_poses_mat_tensor = first_poses_mat.reshape(m, 3, 4) #(m,3,4)
        R_f_obs = first_poses_mat_tensor[:, :, :-1] #(m,3,3)

        #Project all the candidate points that have been tracked up to now
        R_inv = cur_pose[:, :-1].T

        cur_to_kp = R_inv @ n_pts_2D_cur #(3,m) each is a vector in the world's coordinate frame
        cur_to_kp = cur_to_kp.T #(m,3)

        #Project all the points at their first observation
        n_pts_2D_first_obs = n_pts_2D_first_obs.T[:, :, None] #add dimension for batching (m,3,1)
        R_f_obs_inv = R_f_obs.transpose(0,2,1) #(m,3,3)

        pts_3d_first_obs = R_f_obs_inv @ n_pts_2D_first_obs #(m,3,1)
        first_to_kp = pts_3d_first_obs.squeeze(-1) #(m,3) each is a vector in the world's coordinate frame
        
        #Compute bearing angle between the projections
        cur_bearings = cur_to_kp / np.linalg.norm(cur_to_kp, axis=1, keepdims=True)
        first_bearings = first_to_kp / np.linalg.norm(first_to_kp, axis=1, keepdims=True)

        # Compute angle between rays
        cos_alpha = np.sum(cur_bearings * first_bearings, axis=1)
        cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        #Find indices of where alpha exceeds the threshold
        idx = np.where(alpha > params.alpha)[0]
        not_idx = np.where(alpha <= params.alpha)[0]
        
        #Extract the corresponding matrices
        poses = first_poses_mat[idx]
        valid_pts_2D_first = pts_2D_first_obs[idx, :]
        valid_pts_2D_cur = pts_2D_cur[idx, :]
        
        rounded_poses = np.round(poses, decimals=5)
        
        #Find the indices where the transformation is the same as well as the unique poses
        unique_poses, inverse_idx = np.unique(rounded_poses, axis=0, return_inverse=True)

        #Regroup those having the same camera pose
        groups = [np.where(inverse_idx == i)[0] for i in range(len(unique_poses))] #the indices in the rounded poses for which the transformation is the same

        #Current projection matrix
        proj_2 = K @ cur_pose
        
        for i, g in enumerate(groups):

            c_1_pose = unique_poses[i].reshape(3,4)
            proj_1 = K@c_1_pose
            valid_pts_1 = valid_pts_2D_first[g].T #(2,k)
            valid_pts_2 = valid_pts_2D_cur[g].T #(2,k)
            points_homo = cv2.triangulatePoints(proj_1, proj_2, valid_pts_1, valid_pts_2)
            
            # convert back to 3D
            points_3d = (points_homo[:3, :]/points_homo[3, :]) #(3,k)
            valid_3d_pts, mask = self.cheirality_check(points_3d, c_1_pose, cur_pose ) #(3,j)

            pixel_coords = valid_pts_2[:, mask].T  #(j,2)
            pixel_coords = pixel_coords[:, None, :] #Add dimension for consistency (j,1,2)
            state["P"] = np.concatenate((state["P"], pixel_coords), axis=0) #(n+j,1 ,2)
            state["X"] = np.concatenate((state["X"], valid_3d_pts), axis = 1) #(3, n+j)
        
        #Update the candidate set removing the now triangulated points 
        state["C"] = pts_2D_cur[not_idx, None, :]
        state["F"] = pts_2D_first_obs[not_idx, None, :]
        state["T"] = first_poses_mat[not_idx, :]
        
        return state

    def cheirality_check(self, points_3d, Pi_1, Pi_2): 
        """
        Checks whether the newly triangulated points are in front of both cameras

        Args:
            points_3d (np.ndarray) a (3,k) vector containing the newly triangulated points 
            Pi_1: the 3x4 homogeneous transformation matrix of the first camera
            Pi_2: the 3x4 homogeneous transformation matrix of the second camera
        Returns:
            tuple[valid_pts (np.ndarray), mask]: (3, j) posiitve-depth points and mask of valid points 
        """

        #Transform from world coordinates into camera coordinates
        points_3d_hom = np.vstack((points_3d, np.ones(points_3d.shape[1])))
        p_3d_1 = Pi_1 @ points_3d_hom #(3,k)
        p_3d_2 = Pi_2 @ points_3d_hom #(3,k)

        mask = (p_3d_1[2,:]>0) & (p_3d_2[2,:]>0)

        valid_pts = points_3d[:,mask] #(3,j)
        return valid_pts, mask


    def extractFeaturesOperation(self, img_grayscale):
        """
        Step 1 (Initialization): detect Shi-Tomasi corners on a grid using feature masks to find new candidate keypoints.

        Args:
            img_grayscale (np.ndarray): current frame in grayscale (H x W).

        Returns:
            potential_kp_candidates (np.ndarray): (N, 1, 2) float32 corners for KLT tracking.
        """
        potential_kp_candidates = np.empty((0, 1, 2), dtype=np.float32)
        eig = cv2.cornerMinEigenVal(img_grayscale, blockSize=7, ksize=3)
        for n, mask in enumerate(self.params.feature_masks):
            
            # Shi–Tomasi corner detector: use a threshold on the minimum eigenvalue to avoid low texture regions
            eig_roi = eig[mask > 0]
            if eig_roi.size == 0:
                continue

            # if even the best corner in the ROI is too weak -> return nothing
            if float(eig_roi.max()) < self.params.abs_eig_min:
                continue
            
            features = cv2.goodFeaturesToTrack(img_grayscale, mask=mask, **self.params.shi_tomasi_params)
            
            # If no corners are found in this region, skip it
            if features is None: 
                print(f"No features found for mask {n+1}!")
                continue
            
            potential_kp_candidates = np.vstack((potential_kp_candidates, features))
        
        return potential_kp_candidates

    def keypoints2set(keypoints: np.ndarray) -> set:
        """Convert numpy keypoint list [k x 1 x 2] to a set of keypoints

        Args:
            keypoints (np.ndarray): keypoint array

        Returns:
            set: keypoint set
        """
        return set([(row[0][0], row[0][1]) for row in keypoints.tolist()])

    def set2keypoints(keypoint_set: set) -> np.ndarray:
        """Convert keypoint set (u, v) to a numpy array [k x 1 x 2]

        Args:
            keypoint_set (set): keypoint set

        Returns:
            np.ndarray: keypoint array with shape [k x 1 x 2]
        """
        return np.array([[[keypoint[0], keypoint[1]]] for keypoint in keypoint_set])

    def addNewFeatures(self, S: dict, potential_candidate_features: np.ndarray, cur_pose: np.ndarray) -> dict:
        """Given an array of features, update S with featurees that are not already being tracked

        Args:
            S (dict): state
            potential_candidate_features (np.ndarray): features extracted from current frame
            cur_pose (np.ndarray): pose of current frame

        Returns:
            dict: updated state
        """
        S_new = S.copy()

        # setup
        cur = np.vstack((S["P"], S["C"]))[:, 0, :]
        new = potential_candidate_features[:, 0, :]

        # calculate the squared differences between every point pair (rows corrispond to new features, cols to cur features)
        dists = np.linalg.norm((new[:, None, :] - cur[None, :, :]), axis=2)

        # for every new point, find the distance to the closest current point
        min_dists = np.min(dists, axis=1)

        # create a mask, keeping only features that are greater than a minimum distance from any already tracked feature
        new_features_mask = np.where(min_dists > self.params.new_feature_min_squared_diff, True, False)

        # mask the potential new features so only unique ones are kept
        new_features = potential_candidate_features[new_features_mask, :, :]

        # append new features to current points, first observed points, and first observed camera pose
        S_new["C"] = np.vstack((S["C"], new_features))
        S_new["F"] = np.vstack((S["F"], new_features))
        S_new["T"] = np.vstack((S["T"], cur_pose.flatten()[None, :].repeat(new_features.shape[0], axis=0)))
        return S_new

    
    @staticmethod
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
    


# Create instance of pipeline
pipeline = Pipeline(params)

# extract features from the first image of the dataset
bootstrap_features_kf_1 = pipeline.extractFeaturesBootstrap()

# track extracted features forward to the next keyframe in the dataset
bootstrap_tracked_features_kf_1, bootstrap_tracked_features_kf_2 = pipeline.trackForwardBootstrap(bootstrap_features_kf_1)

# calculate the homographic transformation between the first two keyframes
homography, ransac_features_kf_1, ransac_features_kf_2 = pipeline.findRelativePose(bootstrap_tracked_features_kf_1, bootstrap_tracked_features_kf_2)

# triangulate features from the first two keyframes to generate initial 3D point cloud
bootstrap_point_cloud = pipeline.bootstrapPointCloud(homography, ransac_features_kf_1, ransac_features_kf_2)

# generate initial state
S = pipeline.bootstrapState(P_i=ransac_features_kf_2, X_i=bootstrap_point_cloud)

# ploting setup
est_path = []
# initialize previous image
last_image = cv2.imread(images[params.start_idx], cv2.IMREAD_GRAYSCALE)

# first “flow” image to show (just grayscale -> bgr)
first_vis = cv2.cvtColor(last_image, cv2.COLOR_GRAY2BGR)

total_frames = last_frame - params.start_idx
plot_state = initTrajectoryPlot(ground_truth, first_flow_bgr=first_vis, total_frames=total_frames)

R_cw = homography[:3, :3]
t_cw = homography[:3, 3]
R_wc = R_cw.T
t_wc = - R_wc @ t_cw
est_path.append([t_wc[0], t_wc[2]])
theta = -(scipy.spatial.transform.Rotation.from_matrix(R_wc).as_euler("xyz")[1] + np.pi/2)

#Initialize candidate set with the second keyframe
potential_candidate_features = pipeline.extractFeaturesOperation(last_image)

# find which features are not currently tracked and add them as candidate features
S = pipeline.addNewFeatures(S, potential_candidate_features, homography)
frame_counter = params.start_idx +1

for i in range(params.start_idx + 1, last_frame):
    frame_counter+=1
    # read in next image
    image_path = images[i]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: could not read {image_path}")
        continue

    # track keypoints forward one frame
    S, last_features, last_candidates = pipeline.trackForward(S, last_image, image)
    
    # plot tracked keypoints (pre-ransac) and candidate keypoints
    img_to_show = draw_optical_flow(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), last_candidates, S["C"], (0, 0, 255), 1, .15)

    # estimate pose, only keeping inliers from PnP with RANSAC
    S, pose, inliers_idx = pipeline.estimatePose(S)

    # plot inlier keypoints
    img_to_show = draw_optical_flow(img_to_show, last_features[inliers_idx], S["P"], (0, 255, 0), 1, .15)

    # attempt triangulating candidate keypoints, only adding ones with sufficient baseline
    S = pipeline.tryTriangulating(params, S, pose)

    # find features in current frame
    potential_candidate_features = pipeline.extractFeaturesOperation(image)

    # find which features are not currently tracked and add them as candidate features
    S = pipeline.addNewFeatures(S, potential_candidate_features, pose)
    n_inliers = len(inliers_idx)

    # plot current pose
    R_cw = pose[:3, :3]
    t_cw = pose[:3, 3]
    R_wc = R_cw.T
    t_wc = - R_wc @ t_cw
    est_path.append([t_wc[0], t_wc[2]])
    theta = -(scipy.spatial.transform.Rotation.from_matrix(R_wc).as_euler("xyz")[1] +np.pi/2)
    
    updateTrajectoryPlot(
        plot_state, 
        np.asarray(est_path), 
        theta, 
        S["X"],
        S["P"].shape[0], 
        flow_bgr=img_to_show,
        frame_idx=frame_counter,
        n_inliers=n_inliers,
    )

    # update last image
    last_image = image
    
    # debugging prints
    if last_features[inliers_idx].shape[0] < 50:
        print("***************************")
        print(f"# Keypoints Tracked: {last_features.shape[0]}\n"
              f"#Candidates Tracked: {last_candidates.shape[0]}\n"
              f"#Inliers for RANSAC: {last_features[inliers_idx].shape[0]}\n"
              f"#New Keypoints Added: {S['P'].shape[0] - last_features[inliers_idx].shape[0]}")

cv2.destroyAllWindows()