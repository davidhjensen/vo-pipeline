import os
from glob import glob
import time

from typing import Tuple

import cv2
import numpy as np
import scipy
from scipy.optimize import least_squares
from collections import deque

from visualization import * 
from BA_helper import as_lk_points, pack_params, get_jac_sparsity, compute_rep_err, unpack_params_T
from GD_helper import get_mask_indices, estimate_ground_height, fit_ground_plane_ransac

##-------------------GLOBAL VARIABLES------------------##
# Dataset -> 0: KITTI, 1: Malaga, 2: Parking, 3: Own Dataset
DATASET = 1

class D:
    KITTI = 0
    MALAGA = 1
    PARKING = 2
    CUSTOM = 3

# Next keyframe to use for bootstrapping
KITTI_BS_KF = 3
MALAGA_BS_KF = 5
PARKING_BS_KF = 5
CUSTOM_BS_KF = 5

# Define dataset paths
# (Set these variables before running)
kitti_path = "kitti/kitti05/kitti"
malaga_path = "malaga/malaga-urban-dataset-extract-07"
parking_path = "parking/parking"
own_dataset_path = "VAMR_Rome_dataset/VAMR_Rome_dataset"


match DATASET:
    case D.KITTI:
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

    ##------------------PARAMETERS FOR KITTI------------------##
        # Shi-Tomasi corner parameters

        feature_params = dict(  maxCorners = 100,
                                qualityLevel = 0.01,
                                minDistance = 10)
        
        feature_params_BA = dict(  maxCorners = 100,
                                qualityLevel = 0.005,
                                minDistance = 10)
        
        feature_params_gd_detection = dict( maxCorners = 100,
                                        qualityLevel = 0.005,
                                        minDistance = 3,
                                        blockSize = 3)
        #RANSAC PARAMETERS 
        ransac_params = dict(   cameraMatrix=K,
                                distCoeffs=None,
                                reprojectionError=2.0, 
                                flags=cv2.SOLVEPNP_P3P,
                                confidence=0.99,
                                iterationsCount=2000)
        
        ransac_params_BA = dict( cameraMatrix=K,
                                distCoeffs=None,
                                reprojectionError=2.0, 
                                flags=cv2.SOLVEPNP_P3P,
                                confidence=0.98,
                                iterationsCount=2000)
        
        # Parameters for LK
        lk_params = dict(   winSize  = (21, 21),
                            maxLevel = 2, 
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        
        lk_params_BA = dict( winSize  = (21, 21),
                            maxLevel = 2, 
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        
        # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
        new_feature_min_squared_diff = 2
        rows_roi_corners = 2
        cols_roi_corners = 4
        rows_roi_corners_bs = 3
        cols_roi_corners_bs = 5
        
        # Bootstrapping parameters
        bs_kf_1 = images[0]
        bs_kf_2 = images[KITTI_BS_KF]
        start_idx = KITTI_BS_KF
        
        # Bundle adjustment parameters
        window_size = 10

        alpha : float = 0.02
        abs_eig_min : float = 0
        min_features : int = 30
        min_features_BA : int = 60


    case D.MALAGA:
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

    ##------------------PARAMETERS FOR MALAGA------------------##
        # Shi-Tomasi corner parameters
        feature_params_BA = dict(  maxCorners = 100,
                                qualityLevel = 0.01,
                                minDistance = 10,
                                blockSize = 7 )

        #RANSAC PARAMETERS 
        ransac_params_BA = dict(   cameraMatrix=K,
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_P3P,
                                reprojectionError=2.0,
                                confidence=0.99,
                                iterationsCount=2000)

        # Parameters for LKT
        lk_params_BA = dict(   winSize  = (21, 21),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.001))
        
        feature_params = dict(  maxCorners = 100,
                                qualityLevel = 0.01,
                                minDistance = 10,
                                blockSize = 7 )
        feature_params_gd_detection = dict( maxCorners = 100,
                                        qualityLevel = 0.005,
                                        minDistance = 3,
                                        blockSize = 3)
        #RANSAC PARAMETERS 
        ransac_params = dict(   cameraMatrix=K,
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_P3P,
                                reprojectionError=2.0,
                                confidence=0.99,
                                iterationsCount=2000)

        # Parameters for LKT
        lk_params = dict(   winSize  = (21, 21),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.001))
        
        # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
        new_feature_min_squared_diff = 4
        rows_roi_corners = 3
        cols_roi_corners = 3
        rows_roi_corners_bs = 3
        cols_roi_corners_bs = 5
        
        # Bootstrapping parameters
        bs_kf_1 = images[0]
        bs_kf_2 = images[MALAGA_BS_KF]
        start_idx = MALAGA_BS_KF
        
        # Bundle adjustment parameters
        window_size = 5

        alpha : float = 0.02
        abs_eig_min : float = 1e-2
        min_features : int = 50
        min_features_BA : int = 50
        
    case D.PARKING:
        assert 'parking_path' in locals(), "You must define parking_path"
        img_dir = os.path.join(parking_path, 'images')
        images = sorted(glob(os.path.join(img_dir, '*.png')))
        last_frame = 598
        K = np.loadtxt(os.path.join(parking_path, 'K.txt'), delimiter=",", usecols=(0, 1, 2))
        ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
        
    ##------------------PARAMETERS FOR PARKING------------------##
        # Shi-Tomasi corner parameters    
        # Paramaters for Shi-Tomasi corners
        feature_params = dict( maxCorners = 150,
                            qualityLevel = 0.05,
                            minDistance = 7,
                            blockSize = 7 )
        
        feature_params_gd_detection = dict( maxCorners = 100,
                                        qualityLevel = 0.005,
                                        minDistance = 3,
                                        blockSize = 3)
        #RANSAC PARAMETERS 
        ransac_params = dict(   cameraMatrix=K,
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_P3P,
                                reprojectionError=2.0,
                                confidence=0.99,
                                iterationsCount=1000)

        # Parameters for LKT
        lk_params = dict( winSize  = (21, 21),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001))
        
        # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
        new_feature_min_squared_diff = 5
        rows_roi_corners = 3
        cols_roi_corners = 3
        rows_roi_corners_bs = 3
        cols_roi_corners_bs = 5
        
        # Bootstrapping parameters
        bs_kf_1 = images[0]
        bs_kf_2 = images[PARKING_BS_KF]
        start_idx = PARKING_BS_KF
        
        # Bundle adjustment parameters
        window_size = 5

        alpha : float = 0.05
        abs_eig_min : float = 1e-5
        
    case D.CUSTOM:
        # Own Dataset
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
        
    ##------------------PARAMETERS FOR CUSTOM------------------##
        # Shi-Tomasi corner parameters    
        feature_params = dict(  maxCorners = 60,
                                qualityLevel = 0.05,
                                minDistance = 10,
                                blockSize = 9 )
        feature_params_gd_detection = dict( maxCorners = 100,
                                        qualityLevel = 0.005,
                                        minDistance = 3,
                                        blockSize = 3)
        
        #RANSAC PARAMETERS 
        ransac_params = dict(   cameraMatrix=K,
                                distCoeffs=None,
                                flags=cv2.SOLVEPNP_P3P,
                                reprojectionError=5.0,
                                confidence=0.99,
                                iterationsCount=100)
        # Parameters for LKT
        lk_params = dict(   winSize  = (21, 21),
                            maxLevel = 2,
                            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
        
        # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
        new_feature_min_squared_diff = 4
        rows_roi_corners = 3
        cols_roi_corners = 3
        rows_roi_corners_bs = 3
        cols_roi_corners_bs = 5
        
        # Bootstrapping parameters
        bs_kf_1 = images[0]
        bs_kf_2 = images[CUSTOM_BS_KF]
        start_idx = CUSTOM_BS_KF
        
        # Bundle adjustment parameters
        window_size = 10

        alpha : float = 0.02
        abs_eig_min : float = 1e-2
        
    case _:
        raise ValueError("Invalid dataset index")

class VO_Params():
    bs_kf_1 : str # path to first keyframe used for bootstrapping dataset
    bs_kf_2 : str # path to second keyframe used for bootstrapping dataset
    rows_roi_corners_bs : int # number of rows to split image into for ground and feature detection in bootstrapping
    cols_roi_corners_bs : int # number of cols to split image into for ground and feature detection in bootstrapping
    rows_roi_corners : int # number of rows to split image into for feature detection
    cols_roi_corners : int # number of cols to split image into for feature detection
    feature_masks_bs: list[np.ndarray] # mask image into regions for feature tracking 
    feature_masks : list[np.ndarray] # mask image into regions for feature tracking 
    shi_tomasi_params_bs : dict # cv2 parameters for Shi-Tomasi corners in bootstrapping phase
    shi_tomasi_params : dict # cv2 parameters for Shi-Tomasi corners
    klt_params : dict # cv2 parameters for KLT tracking
    ransac_params : dict #cv2 params for ransac
    k : np.ndarray # camera intrinsics matrix
    start_idx: int # index of the frame to start continous operation at (2nd bootstrap keyframe index)
    new_feature_min_squared_diff: float # min squared diff in pxl from a new feature to the nearest existing feature for the new feature to be added
    min_features : int
    def __init__(self, bs_kf_1, bs_kf_2, shi_tomasi_params, shi_tomasi_params_bs, klt_params, ransac_params, k, start_idx, new_feature_min_squared_diff, window_size, alpha, abs_eig_min, min_features):
        self.bs_kf_1 = bs_kf_1
        self.bs_kf_2 = bs_kf_2
        self.feature_masks_bs = self.get_feature_masks(bs_kf_1, rows_roi_corners_bs, cols_roi_corners_bs)
        self.feature_masks = self.get_feature_masks(bs_kf_1, rows_roi_corners, cols_roi_corners)
        self.shi_tomasi_params_bs = shi_tomasi_params_bs
        self.shi_tomasi_params = shi_tomasi_params
        self.ransac_params = ransac_params
        self.klt_params = klt_params
        self.k = k
        self.start_idx = start_idx
        self.new_feature_min_squared_diff = new_feature_min_squared_diff
        self.rows_roi_corners_bs = rows_roi_corners_bs
        self.cols_roi_corners_bs = cols_roi_corners_bs
        self.rows_roi_corners = rows_roi_corners
        self.cols_roi_corners = cols_roi_corners
        self.window_size = window_size

        self.idx_ground = self.rows_roi_corners_bs * self.cols_roi_corners_bs -(np.floor(self.cols_roi_corners_bs/2).astype(int)+1)
        ground_list = [self.idx_ground-1, self.idx_ground, self.idx_ground+1]
        self.idx_ground_set = set(ground_list)
        self.approx_car_height = 1.65
        self.alpha = alpha
        self.abs_eig_min = abs_eig_min
        self.min_features = min_features

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
        self.H, self.W = img.shape[:2]

        # get boundries of the cells
        row_boundries = np.linspace(0, self.H, rows + 1, dtype=int)
        col_boundries = np.linspace(0, self.W, cols + 1, dtype=int)

        # create masks left to right, top to bottom
        masks = []
        for row in range(rows):
            for col in range(cols):
                mask = np.zeros((self.H, self.W), dtype="uint8")
                r_s, r_e = row_boundries[[row, row + 1]]
                c_s, c_e = col_boundries[[col, col + 1]]
                mask[r_s:r_e, c_s:c_e] = 255
                masks.append(mask)

        return masks

class Pipeline():

    params: VO_Params

    def __init__(self, params: VO_Params, use_sliding_window_BA: bool = False, use_scale :bool = False):
        self.params = params
        self.use_sliding_BA : bool = use_sliding_window_BA  # whether we want to use sliding_window BA or not, by default, it's false
        self.next_id : int = 0  # this parameter is only used for sliding BA and is used to keep track of landmarks through operation (cause landmarks might decrease/increase)
        self.full_trajectory = []
        self.last_scale = 1
        self.use_scale : bool = use_scale
        self.visualize = False

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

            st_corners = np.vstack((st_corners, features))
        return st_corners
    
    def extractFeaturesGD(self, img):
        """
        Step 1 (Initialization): detect Shi-Tomasi corners on a grid using feature masks.

        Returns:
            st_corners (np.ndarray): (N, 1, 2) float32 corners for KLT tracking.
        """
        st_corners = np.empty((0, 1, 2), dtype=np.float32)
        if img is None:
            img_grayscale = cv2.imread(self.params.bs_kf_1, cv2.IMREAD_GRAYSCALE)
        else: 
            img_grayscale = img
        for n, mask in enumerate(self.params.feature_masks_bs):
            if n in self.params.idx_ground_set: 
                features = cv2.goodFeaturesToTrack(img_grayscale, mask=mask, **self.params.shi_tomasi_params_bs)
            else:
                continue
            
            # If no corners are found in this region, skip it
            if features is None: 
                print(f"No features found for mask {n+1}!")
                continue

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
        img_bs_kf_1_index = images.index(self.params.bs_kf_1)
        img_bs_kf_2_index = images.index(self.params.bs_kf_2)
        still_detected = np.ones(st_corners_kf_1.shape[0],dtype=bool)
        points = as_lk_points(st_corners_kf_1.copy())
        initial_points = st_corners_kf_1.copy()
        #Track keypoints frame-by-frame from first bs frame to second bs frame
        for i in range(img_bs_kf_1_index, img_bs_kf_2_index):
            current_image = cv2.imread(images[i],cv2.IMREAD_GRAYSCALE)
            next_image = cv2.imread(images[i+1],cv2.IMREAD_GRAYSCALE)
            nextPts, status, _ = cv2.calcOpticalFlowPyrLK(current_image,next_image,points, None, **self.params.klt_params)
            points = nextPts
            status = status.flatten()
            still_detected = still_detected & (status==1)

        # Keep only points that were successfully tracked throughout
        return initial_points[still_detected], points[still_detected]

    def findRelativePose(self, points1, points2):
        """
        Step 3 (Initialization): estimate the relative pose between the two bootstrap keyframes using the 8-point algorithm with RANSAC.
        Args:
            points1 (np.ndarray): (N,1,2) keypoints in bs_kf_1.
            points2 (np.ndarray): (N,1,2) corresponding keypoints in bs_kf_2.
        Returns:
            np.ndarray: (3,4) relative pose [R|t] from bs_kf_1 to bs_kf_2.
            np.ndarray: (M,1,2) inlier keypoints in bs_kf_1.
            np.ndarray: (M,1,2) inlier keypoints in bs_kf_2.
        """
        #F mat using ransac
        fundamental_matrix, inliers = cv2.findFundamentalMat(points1,points2,cv2.FM_RANSAC,ransacReprojThreshold=1.0)

        #using boolean vector
        inliers = inliers.ravel().astype(bool)

        #compute the essential matrix
        E= K.T @ fundamental_matrix@K

        #recover the relative camera pose
        _,R,t,_ = cv2.recoverPose(E,points1[inliers],points2[inliers],K)
        
        return np.hstack((R, t)), points1[inliers, :, :], points2[inliers, :, :]

    def bootstrapPointCloud(self, H: np.ndarray, points_1: np.ndarray, points_2: np.ndarray) -> np.ndarray:
        """Bootstrap the initial 3D point cloud using least squares assuming the first frame is the origin

        Args:
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

    def bootstrapState(self, P_1: np.ndarray, P_2: np.ndarray, X_2: np.ndarray, homography: np.ndarray) -> dict[str : np.ndarray]:
        """
            Initializes the state after the previous steps.
            Args:
                P_1: keypoints in the original (first) frame
                P_2: keypoints in the second frame selected for bootstrapping
                X_2: current landmarks from the second frame
                homography: relative transformation between first and second frame selected for bootstrapping
            Returns:
                dict: the state in the form of a dictionary where each state-string variable is the key to get the value
                      e.g.: S["P"] returns the current frame 2D landmarks
        """
        S : dict[str : np.ndarray] = {}
        assert P_2.shape[0] == X_2.shape[1], "2D keypoints number of rows should match the 3D keypoints number of columns"
        assert P_1.shape[0] == P_2.shape[0], "2D keypoints from frame 1 and 2 MUST be the same"
        
        S["P"] = P_2    # these are the "current" keypoints from frame 2    
        S["X"] = X_2    
        S["C"] = np.empty((0,1,2))
        S["F"] = np.empty_like(S["C"])
        S["T"] = np.empty((0,12))

        if self.use_sliding_BA:
            n_pts = X_2.shape[1] # we get the landmark number
            S["ids"] = np.arange(n_pts)     # initialize this as the range going from 0 to n_pts-1
            self.next_id = n_pts
            # only created if we are using BA
            S["P_history"] = deque(maxlen=self.params.window_size)
            S["P_history"].append((P_1.copy(), S["ids"].copy()))   # this is a list of tuples containing kx1x2 keypoints and the IDs associated to the landmarks for tracking through frames
            S["P_history"].append((P_2.copy(), S["ids"].copy()))  
            
            S["pose_history"] = deque(maxlen=self.params.window_size)
            S["pose_history"].append(np.hstack((np.eye(3), np.zeros((3,1))))) # frame zero, so the origin
            S["pose_history"].append(homography)  # frame one, the homography from frame 0 to frame 1 we found earlier

        return S

    def trackForward(self, state: dict[str:np.ndarray], img_1: np.ndarray, img_2: np.ndarray) -> Tuple[dict[str:np.ndarray], np.ndarray, np.ndarray]:
        """
        Track 2D keypoints from img_1 to img_2 using KLT optical flow

        Args:
            state (dict): current state
            img_1 (np.ndarray): first image (grayscale)
            img_2 (np.ndarray): second image (grayscale)
        Returns:
            dict[str: np.ndarray]: updated state
            np.ndarray: keypoints from the previous frame that were successfully tracked forward
            np.ndarray: candidate keypoints from the previous frame that were successfully tracked forward
        """
        # FIRST WE TRACK "ESTABILISHED KEYPOINTS"
        points_2D = as_lk_points(state["P"])
        assert points_2D.shape[0] > 0, "There are no keypoints here, we can't track them forward"    

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(prevImg=img_1, nextImg=img_2, prevPts=points_2D, nextPts=None, **self.params.klt_params)
        status = status.flatten()   # we are going to use them as booleans so maybe we should cast them with astype (?)
        
        # update the state with the current points
        state["P"] = current_points[status == 1]
        state["X"] = state["X"][:, status == 1]  # only get the ones with "true" status and slice them as 3xk
        
        if self.use_sliding_BA:
            state["ids"] = state["ids"][status == 1]    # filter new ids
                        
        # THEN WE TRACK CANDIDATES - but in the first frame there are no candidates to track other than the established points
        # Therefore 
        candidates = as_lk_points(state["C"])
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
            **self.params.ransac_params
        )

        if not success:
            print("Pose estimation failed")
            return ({}, np.zeros((3,4)), np.empty((0,0))) # maybe we could raise an error instead of returning this
        
        # r_vec needs to be converted into a 3x3
        R, _ = cv2.Rodrigues(r_vec)    
        T_w2c = np.hstack((R, t_vec))
        # now, inliers are the indices in pts_2d and pts_3d corresponding to the inliers; it is a 2D since openCV returns it as such, so we need to convert it in a 1D array to use np features
        inliers_idx = inliers_idx.flatten()
        new_state = state.copy()
        # by doing the following thing the idea is that we are only keeping the inliers
        new_state["P"] = state["P"][inliers_idx]
        new_state["X"] = state["X"][:, inliers_idx] # slice this since we want it as a 3xk
                
        if self.use_sliding_BA:
            new_state["ids"] = state["ids"][inliers_idx] # update also the "valid" indices
            new_state["pose_history"].append(T_w2c)

        return new_state, T_w2c, inliers_idx
    

    def tryTriangulating(self, state: dict[str:np.ndarray], cur_pose: np.ndarray) -> dict[str:np.ndarray]:
        """
        Triangulate new points based on the bearing angle threshold to ensure sufficient baseline without relying on scale (ambiguous)
        
        Args:
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
        K = self.params.k
        K_inv = np.linalg.inv(K)
        n_pts_2D_cur = K_inv @ pts_2D_cur_hom.T #(3,m)
        n_pts_2D_first_obs = K_inv @ pts_2D_first_obs_hom.T #(3,m)
        
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
        idx = np.where(alpha > self.params.alpha)[0]
        not_idx = np.where(alpha <= self.params.alpha)[0]
        
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
            proj_1 = K @ c_1_pose
            valid_pts_1 = valid_pts_2D_first[g].T #(2,k)
            valid_pts_2 = valid_pts_2D_cur[g].T #(2,k)
            points_homo = cv2.triangulatePoints(proj_1, proj_2, valid_pts_1, valid_pts_2)
            
            # convert back to 3D
            points_3d = (points_homo[:3, :]/points_homo[3, :]) #(3,k)
            valid_3d_pts, mask = self.cheirality_check(points_3d, c_1_pose, cur_pose ) #(3,j)

            pixel_coords = valid_pts_2[:, mask].T  #(j,2)
            pixel_coords = pixel_coords[:, None, :] #Add dimension for consistency (j,1,2)
            state["P"] = np.concatenate((state["P"], pixel_coords), axis=0) #(n+j,1 ,2)
            state["X"] = np.concatenate((state["X"], valid_3d_pts), axis=1) #(3, n+j)
            
            if self.use_sliding_BA:
                new_ids = np.arange(self.next_id, self.next_id + valid_3d_pts.shape[1])     # get the new ids
                state["ids"] = np.concatenate((state["ids"], new_ids))  # add them to the state
                self.next_id += len(new_ids)    # update the next_id
        
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

        # Transform from world coordinates into camera coordinates
        points_3d_hom = np.vstack((points_3d, np.ones(points_3d.shape[1])))
        p_3d_1 = Pi_1 @ points_3d_hom #(3,k)
        p_3d_2 = Pi_2 @ points_3d_hom #(3,k)

        mask = (p_3d_1[2,:]>0) & (p_3d_2[2,:]>0)

        valid_pts = points_3d[:,mask] #(3,j)

        return valid_pts, mask
    def build_exclusion_mask(self, keypoints_list, radius=8):
        """
        Create a mask that excludes pixels around existing keypoints.

        keypoints_list: list of (N,1,2) arrays (e.g. [S["P"], S["C"]])
        """
        H, W = self.params.H, self.params.W
        excl = np.zeros((H, W), dtype=np.uint8)

        for kp_arr in keypoints_list:
            if kp_arr.size == 0:
                continue
            pts = kp_arr.reshape(-1, 2).astype(int)
            for u, v in pts:
                cv2.circle(excl, (u, v), radius, 255, -1)

        return excl

    def extractFeaturesOperation(self, img_grayscale, S):
        """
        Step 1 (Initialization): detect Shi-Tomasi corners on a grid using feature masks to find new candidate keypoints.

        Args:
            img_grayscale (np.ndarray): current frame in grayscale (H x W).
        Returns:
            potential_kp_candidates (np.ndarray): (N, 1, 2) float32 corners for KLT tracking.
        """
        excl_mask = self.build_exclusion_mask(
            [S["P"], S["C"]],
            radius=self.params.new_feature_min_squared_diff
        )
        potential_kp_candidates = np.empty((0, 1, 2), dtype=np.float32)
        eig = cv2.cornerMinEigenVal(img_grayscale, blockSize=7, ksize=3)
        feature_list = []
        min_features = self.params.min_features
        
        for n, mask in enumerate(self.params.feature_masks):
            effective_mask = cv2.bitwise_and(
                mask,
                cv2.bitwise_not(excl_mask)
            )
            # Shi–Tomasi corner detector: use a threshold on the minimum eigenvalue to avoid low texture regions
            eig_roi = eig[effective_mask > 0]
            if eig_roi.size == 0:
                continue

            # if even the best corner in the ROI is too weak -> return nothing
            if float(eig_roi.max()) < self.params.abs_eig_min:
                continue
            
            features = cv2.goodFeaturesToTrack(img_grayscale, mask=effective_mask, **self.params.shi_tomasi_params)
            
            # If no corners are found in this region, skip it
            if features is None: 
                print(f"No features found for mask {n+1}!")
                continue
            
            num_features = features.shape[0]
            if num_features < min_features and num_features > 15: 
                min_features = num_features 
            
            feature_list.append(features)

        for feat in feature_list: 
            features_to_track = feat[:min_features, :, :]
            potential_kp_candidates = np.vstack((potential_kp_candidates, features_to_track))
        
        return potential_kp_candidates

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
    
    def groundDetection(self, current_keypoints:np.ndarray, current_3d_landmarks: np.ndarray):
        
        current_keypoints = np.squeeze(current_keypoints,axis=1)
        idx_pts = get_mask_indices(self.params.H, self.params.W, self.params.rows_roi_corners_bs, self.params.cols_roi_corners_bs, current_keypoints)
        gd_mask = (idx_pts == self.params.idx_ground) | (idx_pts == self.params.idx_ground+1) | (idx_pts == self.params.idx_ground-1)
        
        gd_points = current_3d_landmarks[:, gd_mask]
        if gd_points.shape[1] > 0: 
            h, inliers = estimate_ground_height(gd_points, self.params.approx_car_height)
            if h is not None:
                print(f"Height: {h}")
                scale = self.params.approx_car_height / h
                self.last_scale = scale
            else: 
                self.last_scale=1
        
        return self.last_scale, gd_mask, inliers

    def slidingWindowRefinement(self, S: dict) -> dict:
        """
        Perform sliding window bundle adjustment to refine camera poses and 3D landmarks in the current window.
        
        Args:
            S (dict): current state containing pose history, landmark history, and observations.
        Returns:
            dict: updated state with refined poses and landmarks.
        """
        # Check if BA is enabled and window is full, but we could also start when we have like 5 or 6 (that is a minimal change though, especially since drift grows with time, so if we start later it is ok)
        if not self.use_sliding_BA or len(S["pose_history"]) < self.params.window_size:
            return S

        # We only optimize landmarks that are currently in our 'current and valid' set (S["X"])
        active_ids = S["ids"]
        id_to_idx = {id_: i for i, id_ in enumerate(active_ids)}
        n_landmarks = len(active_ids)
        
        window_poses = list(S["pose_history"])
                
        active_ids = S["ids"]
        id_to_idx = {id_: i for i, id_ in enumerate(active_ids)}
        n_landmarks = len(active_ids)
        
        # Build a flat list of all 2D observations in the window
        obs_map = {}  # (frame_idx, landmark_idx) -> pixel (u, v)
        
        for f_idx, (pixels, ids) in enumerate(S["P_history"]):
            local_ids = []
            local_pixels = []
            for p, pt_id in zip(pixels, ids):
                if pt_id in id_to_idx:
                    local_ids.append(id_to_idx[pt_id])
                    local_pixels.append(p[0]) # Extract (u, v)
            
            # Fill the dictionary for this frame
            obs_map[f_idx] = {
                'ids': np.array(local_ids),
                'pixels': np.array(local_pixels)
            }
        
        obs_list_for_sparsity = []
        for f_idx, data in obs_map.items():
            for l_idx in data['ids']:
                obs_list_for_sparsity.append((f_idx, l_idx))
                
        # Pack variables and Sparsity Mask
        x0 = pack_params(window_poses, S["X"])
        A = get_jac_sparsity(len(window_poses), n_landmarks, obs_list_for_sparsity)  # needed for scipy least square

        # Huber norm is used to ignore KLT tracking outliers
        res = least_squares(
            compute_rep_err, x0, 
            jac_sparsity=A,
            args=(window_poses, n_landmarks, obs_map, self.params.k),
            loss='huber', f_scale=1.0, method='trf', ftol=1e-3
        )

        # Update State with Refined Values
        new_poses, new_X, new_S = unpack_params_T(res.x, window_poses, n_landmarks, S)
        
        # Update current state
        S = new_S
        S["X"] = new_X  # refined landmarks
        # current_pose = new_poses[len(window_poses) - 1]

        # Update history deques with refined values
        for i in range(len(S["pose_history"])):
            S["pose_history"][i] = new_poses[i]
            
        return S
    
    def updateFullTraj(self, window_poses): 
        
        full_traj = self.full_trajectory.copy()
        
        #Convert into world Frame:
        window_poses_list = list(window_poses)
        local_traj = []
        for pose in window_poses_list:
            R_cw = pose[:3, :3]
            t_cw = pose[:3, 3]
            R_wc = R_cw.T
            t_wc = - R_wc @ t_cw
            forward_vec = R_wc[:, 2]
            theta = np.arctan2(forward_vec[0], forward_vec[2])
            theta = theta - (np.pi)/2 if DATASET != D.MALAGA else theta
            state_to_plot = (np.array([t_wc[0], t_wc[2]]), theta)
            local_traj.append(state_to_plot)
        
        start_idx = max(0, len(full_traj)-len(local_traj))
        full_traj[start_idx:start_idx+len(local_traj)]=local_traj
        
        return full_traj 
    
    def pipeline_init(self, img):
        """
        Initialize the VO pipeline by bootstrapping from the first two keyframes.
        
        Returns:
            tuple[dict, np.ndarray]: initial state and homographic transformation between the first two keyframes.
        """
        # extract features from the first image of the dataset
        bootstrap_features_kf_1 = self.extractFeaturesBootstrap()

        # track extracted features forward to the next keyframe in the dataset
        bootstrap_tracked_features_kf_1, bootstrap_tracked_features_kf_2 = self.trackForwardBootstrap(bootstrap_features_kf_1)

        # calculate the homographic transformation between the first two keyframes
        homography, ransac_features_kf_1, ransac_features_kf_2 = self.findRelativePose(bootstrap_tracked_features_kf_1, bootstrap_tracked_features_kf_2)

        # triangulate features from the first two keyframes to generate initial 3D point cloud
        bootstrap_point_cloud = self.bootstrapPointCloud(homography, ransac_features_kf_1, ransac_features_kf_2)
        
        # generate initial state
        S = pipeline.bootstrapState(P_1=ransac_features_kf_1,P_2=ransac_features_kf_2, X_2=bootstrap_point_cloud, homography=homography)

        gd_features_kf_1 = self.extractFeaturesGD(img=None)
        bs_gd_tracked_features_kf_1, bs_gd_tracked_features_kf_2 = self.trackForwardBootstrap(gd_features_kf_1)
        gd_point_cloud = self.bootstrapPointCloud(homography, bs_gd_tracked_features_kf_1, bs_gd_tracked_features_kf_2)
        scale=1
        if self.use_scale and DATASET in [D.KITTI]: 
            scale, gd_mask, inliers = pipeline.groundDetection(bs_gd_tracked_features_kf_2, gd_point_cloud)

            if self.visualize and inliers is not None : 

                gd_fit_2 = bs_gd_tracked_features_kf_2[gd_mask, :, :]
                gd_fit_2 = gd_fit_2[inliers, :, :]
                gd_point_cloud = gd_point_cloud[:, gd_mask]

                #DRAW ALL FEATURES IN BLUE 
                vis = draw_new_features(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), ransac_features_kf_2, color=(255, 255, 0))

                #DRAW ESTIMATED GROUND IN RED 
                vis = draw_new_features(vis, bs_gd_tracked_features_kf_2, color=(0, 0, 255))

                #DRAW INLIERS GROUND IN GREEN 
                vis = draw_new_features(vis, gd_fit_2, color=(0, 255, 0))

                # Recover plane parameters again for visualization
                n, d, _ = fit_ground_plane_ransac(
                    gd_point_cloud,
                    dist_thresh=0.001 * np.median(np.linalg.norm(gd_point_cloud, axis=0)),
                    expected_normal=np.array([0, 1, 0])
                    )

                vis = draw_plane_on_image(vis, n, d, K, homography)
                cv2.imshow("Ground detection", vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                visualize_ground_plane(
                    X_all=bootstrap_point_cloud,
                    X_ground=gd_point_cloud,
                    inliers=inliers,
                    n=n,
                    d=d,
                    title="Bootstrap ground plane fit"
                )
        
        return (S, homography, scale)
    


plot_same_window : bool = True     # splits the visualization into two windows for poor computers like mine

# create instance of pipeline
use_sliding_window_BA : bool = False   # boolean to decide if BA is used or not
use_scale : bool = True
# create instance of parameters
if use_sliding_window_BA: 
    params = VO_Params(bs_kf_1, 
                    bs_kf_2, 
                    feature_params_BA, 
                    feature_params_gd_detection, 
                    lk_params_BA, 
                    ransac_params_BA, 
                    K, 
                    start_idx, 
                    new_feature_min_squared_diff, 
                    window_size, 
                    alpha, 
                    abs_eig_min, 
                    min_features_BA)
else: 
    params = VO_Params(
                    bs_kf_1, 
                    bs_kf_2, 
                    feature_params, 
                    feature_params_gd_detection, 
                    lk_params, 
                    ransac_params, 
                    K, 
                    start_idx, 
                    new_feature_min_squared_diff, 
                    window_size, 
                    alpha, 
                    abs_eig_min, 
                    min_features)
    
pipeline = Pipeline(params = params, use_sliding_window_BA = use_sliding_window_BA, use_scale=use_scale)

img = cv2.imread(params.bs_kf_2, cv2.IMREAD_GRAYSCALE)
# generate initial state
S, homography, scale = pipeline.pipeline_init(img)

# initialize previous image
last_image = cv2.imread(images[params.start_idx], cv2.IMREAD_GRAYSCALE)

# first “flow” image to show (just grayscale -> bgr)
first_vis = cv2.cvtColor(last_image, cv2.COLOR_GRAY2BGR)

total_frames = last_frame - params.start_idx

if plot_same_window:
    plot_state = initTrajectoryPlot(ground_truth, first_flow_bgr=first_vis, total_frames=total_frames, rows=params.rows_roi_corners, cols=params.cols_roi_corners)
else:
    plot_state = initTrajectoryPlotNoFlow(ground_truth, first_flow_bgr=first_vis, total_frames=total_frames, rows=params.rows_roi_corners, cols=params.cols_roi_corners)

R_cw = homography[:3, :3]
t_cw = homography[:3, 3]
R_wc = R_cw.T
t_wc = - R_wc @ t_cw
forward_vec = R_wc[:, 2]
theta = np.arctan2(forward_vec[0], forward_vec[2])
theta = theta - (np.pi)/2 if DATASET != D.MALAGA else theta
state_to_plot = (np.array([t_wc[0], t_wc[2]]), theta)
pipeline.full_trajectory.append(state_to_plot)

#Initialize candidate set with the second keyframe
potential_candidate_features = pipeline.extractFeaturesOperation(last_image, S)

# find which features are not currently tracked and add them as candidate features
S = pipeline.addNewFeatures(S, potential_candidate_features, homography)
frame_counter = params.start_idx + 1

for i in range(params.start_idx + 1, last_frame):
    start = time.time()
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

    # perform sliding window bundle adjustment to refine pose and landmarks
    if use_sliding_window_BA:
        dummy=(np.zeros(2), 0)
        pipeline.full_trajectory.append(dummy)
        S["P_history"].append((S["P"].copy(), S["ids"].copy()))  # append the new keypoints
        S = pipeline.slidingWindowRefinement(S)
        pose = list(S["pose_history"])[-1]
        pipeline.full_trajectory = pipeline.updateFullTraj(S["pose_history"])
    else: 
        R_cw = pose[:3, :3]
        t_cw = pose[:3, 3]
        R_wc = R_cw.T
        t_wc = - R_wc @ t_cw
        forward_vec = R_wc[:, 2]
        theta = np.arctan2(forward_vec[0], forward_vec[2])
        theta = theta - (np.pi)/2 if DATASET != D.MALAGA else theta
        state_to_plot = (np.array([t_wc[0], t_wc[2]]), theta)
        pipeline.full_trajectory.append(state_to_plot)

    # plot inlier keypoints
    img_to_show = draw_optical_flow(img_to_show, last_features[inliers_idx], S["P"], (0, 255, 0), 1, .15)

    # find features in current frame
    potential_candidate_features = pipeline.extractFeaturesOperation(image, S)

    # find which features are not currently tracked and add them as candidate features
    S = pipeline.addNewFeatures(S, potential_candidate_features, pose)

    # attempt triangulating candidate keypoints, only adding ones with sufficient baseline
    S = pipeline.tryTriangulating(S, pose)

    n_inliers = len(inliers_idx) 

    # update last image
    last_image = image
    
    # debugging prints
    if last_features[inliers_idx].shape[0] < 50:
        print("***************************")
        print(f"# Keypoints Tracked: {last_features.shape[0]}\n"
              f"#Candidates Tracked: {last_candidates.shape[0]}\n"
              f"#Inliers for RANSAC: {last_features[inliers_idx].shape[0]}\n"
              f"#New Keypoints Added: {S['P'].shape[0] - last_features[inliers_idx].shape[0]}")
    end = time.time()
    fps = 1/(end-start)
    if plot_same_window:
        updateTrajectoryPlotBA(
        plot_state, 
        pipeline.full_trajectory,
        S["X"],
        S["P"].shape[0], 
        gt=ground_truth,
        flow_bgr=img_to_show,
        frame_idx=frame_counter,
        n_inliers=n_inliers, 
        scale=scale, 
        fps=fps
    )
        
    else:
        updateTrajectoryPlotNoFlowBA(
            plot_state, 
            pipeline.full_trajectory,
            S["X"],
            S["P"].shape[0], 
            # flow_bgr=img_to_show,
            frame_idx=frame_counter,
            n_inliers=n_inliers
        )
        
        cv2.imshow("tracking...", img_to_show)
        cv2.waitKey(10)

cv2.destroyAllWindows()
