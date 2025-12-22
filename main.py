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

# Dataset -> 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
DATASET = 2

# Define dataset paths
# (Set these variables before running)
kitti_path = "kitti/kitti05/kitti"
malaga_path = "malaga/malaga-urban-dataset-extract-07"
parking_path = "parking"
# own_dataset_path = "/path/to/own_dataset"

if DATASET == 0:
    assert 'kitti_path' in locals(), "You must define kitti_path"
    img_dir = os.path.join(kitti_path, '05/image_0')
    images = glob(os.path.join(img_dir, '*.png'))
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
    images = sorted(glob(os.path.join(img_dir, '*.png')))
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
    images = glob(os.path.join(img_dir, '*.png'))
    last_frame = 598
    K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=",", usecols=(0, 1, 2))
    ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
    ground_truth = ground_truth[:, [-9, -1]]
    
elif DATASET == 3:
    # Own Dataset
    # TODO: define your own dataset and load K obtained from calibration of own camera
    assert 'own_dataset_path' in locals(), "You must define own_dataset_path"

else:
    raise ValueError("Invalid dataset index")

# Paramaters for Shi-Tomasi corners
feature_params = dict( maxCorners = 30,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for LKT
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
new_feature_min_squared_diff = 5


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
    # ADD NEW PARAMS HERE
    alpha: float = 0.11

    def __init__(self, bs_kf_1, bs_kf_2, shi_tomasi_params, klt_params, k, start_idx, new_feature_min_squared_diff):
        self.bs_kf_1 = bs_kf_1
        self.bs_kf_2 = bs_kf_2
        self.feature_masks = self.get_feature_masks(bs_kf_1, 3, 3)
        self.shi_tomasi_params = shi_tomasi_params
        self.klt_params = klt_params
        self.k = k
        self.start_idx = start_idx
        self.new_feature_min_squared_diff = new_feature_min_squared_diff
        # ADD NEW PARAMS HERE
    
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

    def ransacHomography(self, points1, points2):
        """
        Estimate relative pose between two keyframes using RANSAC.

        Args:
            points1, points2: corresponding 2D points (N,2) or (N,1,2)

        Returns:
            H (3x4): relative transformation matrix
            points1[inliers] (N, 1, 2): inlier points from first keyframe
            points2[inliers] (N, 1, 2): inlier points from second keyframe
        """
        #F mat using ransac
        fundamental_matrix, inliers =cv2.findFundamentalMat(points1,points2,cv2.FM_RANSAC,ransacReprojThreshold=1.0)

        #using boolean vector
        inliers = inliers.ravel().astype(bool)

        #compute the essential matrix
        E= K.T@ fundamental_matrix@K

        #recover the relative camera pose
        _,R,t,mask_pose=cv2.recoverPose(E,points1[inliers],points2[inliers],K)

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

        # triangulate homogeneous coordinates using DLT
        points_homo = cv2.triangulatePoints(proj_1, proj_2, points_1, points_2)

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
        # print("2d points shape")
        # print(P_i.shape)
        # print("£d points shape")
        # print(X_i.shape)  
        S["C"] = np.empty((0,1,2))
        S["F"] = np.empty_like(S["C"])
        S["T"] = np.empty((0,12))

        return S

    def trackForward(self, state: dict[str:np.ndarray], img_1: np.ndarray, img_2: np.ndarray) -> dict[str:np.ndarray]:
        """
        Track 2D keypoints from img_1 to img_2 using KLT optical flow

        Args:
            img_1 (np.ndarray): first image (grayscale)
            img_2 (np.ndarray): second image (grayscale)
            points_1 (np.ndarray): keypoints in img_1 to be tracked
        Returns:
            np.ndarray: tracked keypoints in img_2
        """
        # NOTE: might make sense to replace assert with ifs since maybe even if we have no points nor candidates we still want to return them empty again
        # FIRST WE TRACK "ESTABILISHED KEYPOINTS"
        new_state = state.copy()
        points_2D = self.as_lk_points(state["P"])
        assert points_2D.shape[0] > 0, "There are no keypoints here, we can't track them forward"    

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_1, nextImg=img_2, prevPts=points_2D, nextPts=None, **self.params.klt_params)
        status = status.flatten()   # we are going to use them as booleans so maybe we should cast them with astype (?)
        
        # update the state with the current points
        new_state["P"] = current_points[status == 1]
        new_state["X"] = state["X"][:, status == 1]  # only get the ones with "true" status and slice them as 3xk

        # THEN WE TRACK CANDIDATES - but in the first frame there are no candidates to track other than the established points
        # Therefore 
        candidates = self.as_lk_points(state["C"])
        if candidates.shape[0] != 0:
            assert candidates.dtype == np.float32
            assert candidates.ndim == 3 and candidates.shape[1:] == (1, 2)
            current_cands, status_cands, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_1, nextImg=img_2, prevPts=candidates, nextPts=None, **self.params.klt_params)
            status_cands = status_cands.flatten() # same as above
            
            new_state["C"] = current_cands[status_cands == 1]

            # initial observations, still have some doubts on these two
            new_state["F"] = state["F"][status_cands == 1]
            new_state["T"] = state["T"][status_cands == 1]

        return new_state

    def estimatePose(self, state: dict[str:np.ndarray]) -> tuple[dict[str:np.ndarray], np.ndarray]:
        """
        Estimate camera pose using PnP RANSAC and update state to keep only inliers
        
        Args:
            params (VO_Params): parameters for the VO pipeline
            state (dict): current state that also contains 2D keypoints and 3D points
        
        Returns:
            tuple[dict, np.ndarray]: updated state with only inliers for P and X and camera pose as 3x4 matrix
        """
        
        pts_2D = state["P"][:, 0, :]
        pts_3D = state["X"]
        pts_3D = pts_3D.T # according to documentation we need them as kx3 not 3xk
        K = self.params.k
        
        # print("Shape of 2D points")
        # print(pts_2D.shape)
        # print("Shape of 3D points")
        # print(pts_3D.shape)
        success, r_vec, t_vec, inliers_idx =  cv2.solvePnPRansac(
            objectPoints=pts_3D,
            imagePoints=pts_2D,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=2.0,
            confidence=0.999,
            iterationsCount=100
        )

        if not success:
            print("Pose estimation failed")
            return ({}, np.zeros((3,4))) # maybe we could raise an error instead of returning this
        
        # r_vec needs to be converted into a 3x3
        R, _ = cv2.Rodrigues(r_vec)    
        camera_pose = np.hstack((R, t_vec))
        
        # now, inliers are the indices in pts_2d and pts_3d corresponding to the inliers; it is a 2D since openCV returns it as such, so we need to convert it in a 1D array to use np features
        inliers_idx = inliers_idx.flatten()
        new_state = state.copy()
        # by doing the following thing the idea is that we are only keeping the inliers
        new_state["P"] = state["P"][inliers_idx]
        new_state["X"] = state["X"][:, inliers_idx] # slice this since we want it as a 3xk
        # print("Shape of 2D points")
        # print(new_state["P"].shape)
        # print("Shape of 3D points")
        # print(new_state["X"].shape)
        return (new_state, camera_pose)
    

    
    def tryTriangulating(self, state: dict[str:np.ndarray], cur_pose: np.ndarray) -> dict[str:np.ndarray]:
        """
        Triangulate new points based on the bearing angle threshold to ensure sufficient baseline without relying on scale (ambiguous)
        
        Args:
            params (VO_Params): parameters for the VO pipeline
            state (dict): current state that also contains 2D keypoints and 3D points
            cur_pose (np.ndarray): pose of current frame (3x4 matrix) 
        Returns:
            tuple[dict, np.ndarray]: updated state with only inliers for P and X and camera pose as 3x4 matrix
        """
        new_state = state.copy()
        #print(new_state)
        if new_state["C"].shape[0] == 0:
            return state

        pts_2D_cur = state["C"]  #(m,1,2)
        pts_2D_first_obs = state["F"] #(m,1,2)
        pts_2D_cur = pts_2D_cur[:, 0, :]  #(m,2)
        pts_2D_first_obs = pts_2D_first_obs[:, 0, :]  #(m,2)
        m = pts_2D_cur.shape[0]
        if pts_2D_first_obs.shape[0] != m: 
            print("ERROR in shapes of tracked keypoints")
        print(f"2d_cur shape: {pts_2D_cur.shape}\n2d_first shape: {pts_2D_first_obs.shape}\n")

        pts_2D_cur_hom = np.column_stack((pts_2D_cur, np.ones(m)))
        pts_2D_first_obs_hom = np.column_stack((pts_2D_first_obs, np.ones(m)))
        K = self.params.k
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
            proj_1 = K@(unique_poses[i].reshape(3,4))
            valid_pts_1 = valid_pts_2D_first[g].T #(2,k)
            valid_pts_2 = valid_pts_2D_cur[g].T #(2,k)
            points_homo = cv2.triangulatePoints(proj_1, proj_2, valid_pts_1, valid_pts_2)
            # convert back to 3D
            points_3d = (points_homo[:3, :]/points_homo[3, :]) #(3,k)
            
            pixel_coords = valid_pts_2.T  #(k,2)
            pixel_coords = pixel_coords[:, None, :] #Add dimension for consistency (k,1,2)
            new_state["P"] = np.concatenate((state["P"], pixel_coords), axis=0) #(n+k,1 ,2)
            new_state["X"] = np.concatenate((state["X"], points_3d), axis = 1) #(3, n+k)
        
        #Update the candidate set removing the now triangulated points 
        new_state["C"] = pts_2D_cur[not_idx, None, :]
        new_state["F"] = pts_2D_first_obs[not_idx, None, :]
        new_state["T"] = first_poses_mat[not_idx, :]
        
        return new_state


    def extractFeaturesOperation(self, img_grayscale):
        """
        Step 1 (Initialization): detect Shi-Tomasi corners on a grid using feature masks to find new candidate keypoints.

        Args:
            img_grayscale (np.ndarray): current frame in grayscale (H x W).

        Returns:
            potential_kp_candidates (np.ndarray): (N, 1, 2) float32 corners for KLT tracking.
        """
        potential_kp_candidates = np.empty((0, 1, 2), dtype=np.float32)
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

    plt.ion()

    def initTrajectoryPlot(self, gt_path: np.ndarray, arrow_len: float = 0.3) -> Dict[str, object]:
        """
        Initialize trajectory plot with ground truth and placeholders
        for estimated trajectory and camera orientation.

        Args:
            gt_path (np.ndarray): Ground truth path [n x 2]
            arrow_len (float): Length of orientation arrow (meters)

        Returns:
            dict: plot state dictionary
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        if DATASET in [0, 2]:
            # ---- Ground truth (time-colored) ----
            x_gt, y_gt = gt_path[:, 0], gt_path[:, 1]
            t = np.arange(len(gt_path))

            points = np.stack((x_gt, y_gt), axis=1).reshape(-1, 1, 2)
            segments = np.concatenate((points[:-1], points[1:]), axis=1)

            norm = mpl.colors.Normalize(vmin=t.min(), vmax=t.max())
            lc = LineCollection(segments, cmap="viridis", norm=norm, linewidth=2.5)
            lc.set_array(t)

            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("Time step (GT)")

        # ---- Estimated trajectory ----
        est_line, = ax.plot([], [], "r-", linewidth=2, label="VO estimate")
        est_point, = ax.plot([], [], "ro", markersize=5)

        # ---- Orientation arrow ----
        heading_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="->",
            linewidth=2,
            color="red",
            mutation_scale=15,
        )
        ax.add_patch(heading_arrow)

        # ---- Formatting ----
        ax.set_title("Ground Truth vs VO Estimated Trajectory")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        ax.autoscale()
        plt.tight_layout()
        plt.show()

        return {
            "fig": fig,
            "ax": ax,
            "est_line": est_line,
            "est_point": est_point,
            "heading_arrow": heading_arrow,
            "arrow_len": arrow_len,
        }

    def updateTrajectoryPlot(self, plot_state: Dict[str, object], est_path: np.ndarray, theta: float) -> None:
        """
        Update estimated trajectory and camera orientation arrow.

        Args:
            plot_state (dict): Plot state returned by initTrajectoryPlot
            est_path (np.ndarray): Estimated path [k x 2]
            theta (float): Current yaw angle (radians)
        """
        x = est_path[:, 0]
        y = est_path[:, 1]


        # Update trajectory
        plot_state["est_line"].set_data(x, y)
        plot_state["est_point"].set_data([x[-1]], [y[-1]])

        # Update orientation arrow
        x0, y0 = x[-1], y[-1]
        L = plot_state["arrow_len"]
        x1 = x0 + L * np.cos(theta)
        y1 = y0 + L * np.sin(theta)

        plot_state["heading_arrow"].set_positions((x0, y0), (x1, y1))

        plt.pause(0.001)

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
homography, ransac_features_kf_1, ransac_features_kf_2 = pipeline.ransacHomography(bootstrap_tracked_features_kf_1, bootstrap_tracked_features_kf_2)

# triangulate features from the first two keyframes to generate initial 3D point cloud
bootstrap_point_cloud = pipeline.bootstrapPointCloud(homography, ransac_features_kf_1, ransac_features_kf_2)

# generate initial state
S = pipeline.bootstrapState(P_i=ransac_features_kf_2, X_i=bootstrap_point_cloud)
print("Bootstrapped state")
#print(S)

# ploting setup
plot_state = pipeline.initTrajectoryPlot(ground_truth)
est_path = []
theta = 0.0

# initialize previous image
last_image = cv2.imread(images[params.start_idx], cv2.IMREAD_GRAYSCALE)
cv2.imshow("tracking...", last_image)

#Initialize candidate set with the second keyframe
potential_candidate_features = pipeline.extractFeaturesOperation(last_image)

# find which features are not currently tracked and add them as candidate features
S = pipeline.addNewFeatures(S, potential_candidate_features, homography)
print("Added new features")
#print(S)

for i in range(params.start_idx + 1, last_frame + 1):
    #print(i)
    # read in next image
    image_path = images[i]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: could not read {image_path}")
        continue

    # track keypoints forward one frame
    S = pipeline.trackForward(S, last_image, image)
    #print(S)
    # estimate pose, only keeping inliers from PnP with RANSAC
    S, pose = pipeline.estimatePose(S)

    # attempt triangulating candidate keypoints, only adding ones with sufficient baseline
    S = pipeline.tryTriangulating(params, S, pose)

    # find features in current frame
    potential_candidate_features = pipeline.extractFeaturesOperation(image)

    # find which features are not currently tracked and add them as candidate features
    S = pipeline.addNewFeatures(S, potential_candidate_features, pose)

    # plot current pose
    est_path.append(-1*pose[:2, 3])
    theta = scipy.spatial.transform.Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")[1]
    print(est_path[-1])
    pipeline.updateTrajectoryPlot(plot_state, np.asarray(est_path), theta - np.pi)

    # update last image
    last_image = image

    # pause for 0.01 seconds
    cv2.imshow("tracking...", last_image)
    cv2.waitKey(10)

cv2.destroyAllWindows()