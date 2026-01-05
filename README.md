# Visual Odometry Pipeline
The repo contains a full-stack implementation in Python utilizing OpenCV functions. A general overview of the pipeline is provided below - for a more detailed description, see the project [report](). *ADD LINK TO REPORT*

[![Watch the video](https://raw.githubusercontent.com/yourusername/yourrepository/main/assets/thumbnail.jpg)](https://raw.githubusercontent.com/yourusername/yourrepository/main/assets/video.mp4)
*ADD ACTUAL VIDEO AND THUMBNAIL TO REPO AND UPDATE THESE LINKS*

## Overview

### Initialization
Before continuous operation, an initial set of 2D-3D point correspondences is found by detecting features in the first frame of the dataset, tracking them forward until the baseline is sufficiently large, and then using the 2D-2D correspondences to estimate the fundamental matrix. The rotation and translation can be recovered and used to triangulate 3D landmarks from the 2D features. A gound plane is estimated using RANSAC and used to initialize the scale based on the average height of a car. These correspondences are used to initialize a Markovian state used in continous operation.

### Continous Operation
The problem is cast into a Markovian process where the current state depends only on the previous state, previous frame, and current frame: $S_{k} = f(S_{k-1}, I_{k-1}, I_k)$. The state contains the information required by the function $f$ to:
- Track features from the previous frame to the current frame
- Estimate the current pose
- Triangulate new landmarks
- Apply sliding-window bundle adjustment to refine pose and landmark estimates

### Algorithms
Listed below are the algorithms used for the main functionality:
- **Feature extraction**: *Shi-Tomasi corner detector*
- **Feature tracking**: *Kincade-Lucas-Tomasi (KLT) tracker*
- **Fundamental matrix estimation**: *8-Point algorithm with RANSAC*
- **Triangulation**: *Least Squares*
- **Ground-plane estimation**: *plane with RANSAC*
- **Pose estimation**: P3P with RANSAC

### Datasets
Four datasets are used for testing:
- [Malaga 07](https://www.mrpt.org/MalagaUrbanDataset#:~:text=malaga%2Durban%2Ddataset%2Dextract%2D07.zip)
- [KITTI 05](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [Parking](https://www.google.com) *ADD LINK!*
- [Custom](https://www.google.com) *ADD LINK!*

They can be downloaded, extracted, and copied into the corrisponding folders in the repo to validate functionality.

## Development Environment
The project uses a Conda environment named: vo_env
To see details, check out `env.yml` in the root directory.

### Creating the Environment

From the repository root, run:

`conda env create -f env.yml`

If the environment already exists, update it with:

`conda env update -f env.yml --prune`

### Using the Project in VS Code

Always open the project through the workspace file:

- Open VS Code
- File -> Open Workspace from File
- Select vo.code-workspace

If VS Code prompts for a Python interpreter, select: `conda: vo_env`

### Verifying the Installation
Open a VS Code terminal and run:

`python -c "import cv2; print(cv2.__version__)"`

An OpenCV version 4.8 or higher should be printed.

### Contributor Notes
- Do not install OpenCV using pip inside this environment
- Always open the project via vo.code-workspace
- Update environment.yml if dependencies change
- Re-run the update command after pulling changes
