import cv2
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

def initTrajectoryPlot(
    gt_path: np.ndarray,
    arrow_len: float = 0.3,
    first_flow_bgr: np.ndarray | None = None,
    total_frames: int | None = None,
):
    plt.ion()
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    ax_flow   = fig.add_subplot(gs[0, :])   # <-- ONE axis, spans whole top
    ax_kp     = fig.add_subplot(gs[1, 0])
    ax_local  = fig.add_subplot(gs[1, 1])
    ax_global = fig.add_subplot(gs[1, 2])

    H, W = first_flow_bgr.shape[:2]

    # ---------- GLOBAL TRAJECTORY ----------
    if (gt_path is not None) and (len(gt_path) > 0):
        x_gt, y_gt = gt_path[:, 0], gt_path[:, 1]
        #ax_global.plot(x_gt, y_gt, color="gray", lw=2, label="GT")

        # start centered around GT extents
        xmin, xmax = float(np.min(x_gt)), float(np.max(x_gt))
        ymin, ymax = float(np.min(y_gt)), float(np.max(y_gt))
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

        pad = 10.0
        span = max((xmax - xmin), (ymax - ymin)) + 2 * pad
        span = max(span, 250.0)  # minimum view
        ax_global.set_xlim(cx - span / 2, cx + span / 2)
        ax_global.set_ylim(cy - span / 2, cy + span / 2)
    else:
        span = 250.0
        ax_global.set_xlim(-span/2, span/2)
        ax_global.set_ylim(-span/2, span/2)

    est_line, = ax_global.plot([], [], "r-", lw=2, label="VO")
    est_point, = ax_global.plot([], [], "ro")

    heading_arrow = FancyArrowPatch((0, 0), (0, 0),
                                    arrowstyle="->",
                                    color="red",
                                    linewidth=2,
                                    mutation_scale=15)
    ax_global.add_patch(heading_arrow)

    ax_global.set_title("Global trajectory")
    ax_global.set_aspect("equal", adjustable="box")
    ax_global.grid(True)
    ax_global.legend()

    # ---------- LOCAL TRAJECTORY ----------
    local_traj, = ax_local.plot([], [], "b-", lw=2, label="Last 20 poses")
    map_scatter = ax_local.scatter([], [], s=6, c="black", alpha=0.4)
    ax_local.set_title("Local window (x-z)")
    ax_local.set_aspect("equal", adjustable="box")
    ax_local.grid(True)
    ax_local.legend()

    # ---------- OPTICAL FLOW IMAGE ----------
    if first_flow_bgr is None:
        # fallback placeholder
        first_flow_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        first_flow_rgb = cv2.cvtColor(first_flow_bgr, cv2.COLOR_BGR2RGB)

    flow_im = ax_flow.imshow(first_flow_rgb)
    ax_flow.set_axis_off()
    # kflow pixel coordinate space
    ax_flow.set_xlim(0, W)
    ax_flow.set_ylim(H, 0)

    # ---------- KEYPOINTS OVER TIME ----------
    kp_line, = ax_kp.plot([], [], "-o", ms=3, lw=2, label="Tracked (P)")
    inl_line, = ax_kp.plot([], [], "-o", ms=3, lw=2, label="Inliers (PnP)")
    ax_kp.set_title("Keypoints over time")
    ax_kp.set_xlabel("Frame")
    ax_kp.set_ylabel("Count")
    ax_kp.grid(True)
    ax_kp.legend()

    init_window = 50   # frames to show initially
    ax_kp.set_xlim(0, init_window)

    plt.tight_layout()
    plt.show()

    return {
        "fig": fig,
        "ax_global": ax_global,
        "ax_local": ax_local,
        "ax_flow": ax_flow,
        "ax_kp": ax_kp,
        "est_line": est_line,
        "est_point": est_point,
        "heading_arrow": heading_arrow,
        "local_traj": local_traj,
        "map_scatter": map_scatter,
        "flow_im": flow_im,
        "kp_line": kp_line,
        "inl_line": inl_line,
        "arrow_len": arrow_len,
        "frames": [],
        "kp_hist": [],
        "inl_hist": [],
        "cand_hist": [],
        "total_frames": total_frames,
        # global bounds that expand only when needed
        "gxlim": list(ax_global.get_xlim()),
        "gylim": list(ax_global.get_ylim()),
    }


def updateTrajectoryPlot(
    plot_state,
    est_path: np.ndarray,
    theta: float,
    pts3d: np.ndarray,
    n_keypoints: int,
    flow_bgr: np.ndarray | None = None,
    frame_idx: int | None = None,
    n_inliers: int | None = None,
):
    # ---------- GLOBAL ----------
    x, y = est_path[:, 0], est_path[:, 1]
    plot_state["est_line"].set_data(x, y)
    plot_state["est_point"].set_data([x[-1]], [y[-1]])

    x0, y0 = float(x[-1]), float(y[-1])
    L = plot_state["arrow_len"]
    plot_state["heading_arrow"].set_positions(
        (x0, y0), (x0 + L*np.cos(theta), y0 + L*np.sin(theta))
    )

    # expand global limits only when needed (no recenter)
    axg = plot_state["ax_global"]
    xmin, xmax = plot_state["gxlim"]
    ymin, ymax = plot_state["gylim"]
    span_x = xmax - xmin
    span_y = ymax - ymin

    pad_x = max(10.0, 0.08 * span_x)
    pad_y = max(10.0, 0.08 * span_y)

    changed = False
    if x0 < xmin + pad_x:
        xmin = x0 - pad_x; changed = True
    elif x0 > xmax - pad_x:
        xmax = x0 + pad_x; changed = True
    if y0 < ymin + pad_y:
        ymin = y0 - pad_y; changed = True
    elif y0 > ymax - pad_y:
        ymax = y0 + pad_y; changed = True

    if changed:
        axg.set_xlim(xmin, xmax)
        axg.set_ylim(ymin, ymax)
        plot_state["gxlim"] = [xmin, xmax]
        plot_state["gylim"] = [ymin, ymax]

    # ---------- LOCAL ----------
    k = min(20, len(est_path))
    x_loc = x[-k:]
    y_loc = y[-k:]
    plot_state["local_traj"].set_data(x_loc, y_loc)

    if pts3d.size > 0:
        plot_state["map_scatter"].set_offsets(np.column_stack((pts3d[0, :], pts3d[2, :])))

    axl = plot_state["ax_local"]
    cx, cy = x_loc[-1], y_loc[-1]
    dx = np.max(np.abs(x_loc - cx)) if k > 1 else 0.0
    dy = np.max(np.abs(y_loc - cy)) if k > 1 else 0.0
    d = max(dx, dy)

    m_min = 0.05
    m = max(m_min, 3.0 * d)
    m = min(m, 20.0)
    axl.set_xlim(cx - m, cx + m)
    axl.set_ylim(cy - m, cy + m)

    # ---------- OPTICAL FLOW IMAGE ----------
    if flow_bgr is not None:
        plot_state["flow_im"].set_data(cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB))

    # ---------- KEYPOINT TIME SERIES ----------
    if frame_idx is not None:
        plot_state["frames"].append(len(plot_state["frames"]))
        plot_state["kp_hist"].append(int(n_keypoints))
        plot_state["inl_hist"].append(int(n_inliers) if n_inliers is not None else np.nan)
        
        f = plot_state["frames"]
        plot_state["kp_line"].set_data(f, plot_state["kp_hist"])
        plot_state["inl_line"].set_data(f, plot_state["inl_hist"])

        axk = plot_state["ax_kp"]
        x_now = f[-1]
        xmin, xmax = axk.get_xlim()
        if x_now > xmax:
            axk.set_xlim(xmin, x_now + 10)
        
        axk.relim()
        axk.autoscale_view(scaley=True, scalex=False)

        # frame counter in the title bar
        tot = plot_state.get("total_frames", None)
        if tot is None:
            plot_state["fig"].suptitle(f"Frame {frame_idx}")
        else:
            plot_state["fig"].suptitle(f"Frame {frame_idx} / {tot}")

    plt.pause(0.001)

def draw_optical_flow(
    img: np.ndarray,
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    color=(0, 255, 0),
    thickness=1,
    tipLength=0.3,
) -> np.ndarray:
    """
    Draw arrows from previous to current 2D points.

    Args:
        img (np.ndarray): BGR image to draw on
        pts_prev (np.ndarray): (N, 1, 2) previous points
        pts_curr (np.ndarray): (N, 1, 2) current points
        color (tuple): BGR color
        thickness (int): line thickness
        tipLength (float): arrow tip size

    Returns:
        np.ndarray: image with optical flow arrows
    """
    vis = img.copy()

    # Remove singleton dimension → (N, 2)
    p0 = pts_prev.reshape(-1, 2)
    p1 = pts_curr.reshape(-1, 2)

    for (x0, y0), (x1, y1) in zip(p0, p1):
        pt0 = (int(round(x0)), int(round(y0)))
        pt1 = (int(round(x1)), int(round(y1)))

        cv2.arrowedLine(
            vis,
            pt0,
            pt1,
            color,
            thickness,
            tipLength=tipLength,
        )

    return vis

def draw_new_features(
    img: np.ndarray,
    new_pts: np.ndarray,
    color=(0, 0, 255),
    radius: int = 2,
    thickness: int = -1,
) -> np.ndarray:
    """
    Draw newly detected 2D features as small circles.

    Args:
        img (np.ndarray): BGR image to draw on
        new_pts (np.ndarray): (N, 1, 2) or (N, 2) array of points
        color (tuple): BGR color (default red)
        radius (int): circle radius in pixels
        thickness (int): -1 for filled circle

    Returns:
        np.ndarray: image with drawn circles
    """
    vis = img.copy()

    if new_pts.size == 0:
        return vis

    pts = new_pts.reshape(-1, 2)

    for x, y in pts:
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            radius,
            color,
            thickness,
        )

    return vis