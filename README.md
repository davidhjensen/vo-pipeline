# Visual Odometry Pipeline
Full-stack VO pipeline tested on standard CV datasets as well as custom dataset

## Development Environment

### Conda Environment Definition

The project uses a Conda environment named: vo_env
To see details, check out `env.yml` in the root directory.

### VS Code Workspace Configuration

#### Creating the Environment

From the repository root, run:

`conda env create -f env.yml`

If the environment already exists, update it with:

`conda env update -f env.yml --prune`

### Using the Project in VS Code

Always open the project through the workspace file:

- Open VS Code
- File -> Open Workspace from File
- Select vo.code-workspace

If VS Code prompts for a Python interpreter:
Select: conda: vo_env

### Verifying the Installation

Open a VS Code terminal and run:

`python -c "import cv2; print(cv2.version)"`

An OpenCV version 4.8 or higher should be printed.

### Contributor Notes

- Do not install OpenCV using pip inside this environment
- Always open the project via vo.code-workspace
- Update environment.yml if dependencies change
- Re-run the update command after pulling changes
