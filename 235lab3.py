"""
CDS/ME-235 Lab 3 - Vision-Based Grasping & Tower of Hanoi
Uses FK/IK from 235lab2_Steven.py

Structure:
  - ArucoDetector             (Part 1)
  - print_inlab_tag_detections(Part 1 in-lab)
  - get_grasp_pose()          (Part 2)
  - compute_and_print_inlab_grasp() (Part 2 in-lab)
  - HanoiSolver               (Part 3)
  - Tests at the bottom
"""

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import time

# ---------------------------------------------------------------------------
# Import kinematics from Lab 2
# (assumes 235lab2_Steven.py is in the same directory or on PYTHONPATH)
# ---------------------------------------------------------------------------
from lab235lab2_Steven import (
    FK, IK, SafetyCheck, DHModifiedToClassical,
    _DH_CL, _make_T6t,
    RUN_ON_ROBOT, AJ, VJ, AL, VL, POS_SAFE, UR_IP,
)

# ---------------------------------------------------------------------------
# Camera intrinsics (given in lab spec)
# ---------------------------------------------------------------------------
K = np.array([
    [1698.75,    0.0,    1115.55],
    [   0.0,  1695.98,   751.98],
    [   0.0,     0.0,      1.0 ],
], dtype=np.float64)

DIST = np.array(
    [-0.00670872, -0.1481124, -0.00250596, 0.00299921, -1.68711031],
    dtype=np.float64
)

ARUCO_TAG_SIZE = 0.02   # 2 cm tag width (meters)

# Transform from frame 5 to camera (given in lab spec, Part 3)
T5C = np.array([
    [1, 0, 0,  0.0   ],
    [0, 1, 0, -0.1016],
    [0, 0, 1,  0.0848],
    [0, 0, 0,  1.0   ],
], dtype=np.float64)

# Camera index for cv2.VideoCapture - change if needed
CAMERA_INDEX = 0


# ---------------------------------------------------------------------------
# Camera pose helper (spec: pos=(0, -0.4, 0.4), RPY=(pi, 0, pi))
# ---------------------------------------------------------------------------
def _make_camera_pose():
    """
    Build the 4x4 camera pose in the robot base frame as specified in the lab.
    Position: (0.0, -0.4, 0.4), RPY: (pi, 0, pi).
    Used for both Part 1 and Part 2 in-lab tasks.
    """
    def Rx(a): return np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    def Rz(a): return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    T              = np.eye(4)
    T[:3, :3]      = Rz(np.pi) @ Ry(0.0) @ Rx(np.pi)
    T[:3,  3]      = [0.0, -0.4, 0.4]
    return T


# ---------------------------------------------------------------------------
# 3D tag corner definition for solvePnP
# ---------------------------------------------------------------------------
def _tag_object_points(tag_size):
    """
    3D corners of an ArUco tag in its own local frame (meters).
    Origin at tag center, z=0 plane, corners in order:
      top-left, top-right, bottom-right, bottom-left.
    """
    h = tag_size / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: rotation vector -> 4x4 homogeneous transform
# ---------------------------------------------------------------------------
def _rvec_tvec_to_T(rvec, tvec):
    """Convert OpenCV rvec/tvec to a 4x4 rigid body transform."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


def _T_to_rvec_tvec(T):
    """Inverse of _rvec_tvec_to_T."""
    R    = T[:3, :3]
    rvec, _ = cv2.Rodrigues(R)
    tvec    = T[:3, 3].reshape(3, 1)
    return rvec, tvec


def rpy_from_R(R):
    """Extract roll-pitch-yaw (XYZ convention) from a 3x3 rotation matrix."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.arctan2( R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = np.arctan2( R[1,0], R[0,0])
    else:   # gimbal lock
        roll  = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw   = 0.0
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Part 1 - ArucoDetector
# ---------------------------------------------------------------------------

class ArucoDetector:
    """
    Detects ArUco markers (DICT_ARUCO_ORIGINAL) in a frame and returns
    their 6-DOF pose relative to the camera.
    """

    def __init__(self,
                 camera_matrix: np.ndarray = K,
                 dist_coeffs:   np.ndarray = DIST,
                 tag_size:      float      = ARUCO_TAG_SIZE):
        self.K        = camera_matrix
        self.dist     = dist_coeffs
        self.tag_size = tag_size
        self._obj_pts = _tag_object_points(tag_size)

        # Use the original ArUco dictionary as required by the spec
        self._dict     = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self._params   = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(self._dict, self._params)

    # -----------------------------------------------------------------------
    def _estimate_pose(self, corners_single):
        """
        Estimate pose of a single marker given its 4 image corners (shape (4,2)).
        Uses cv2.solvePnP directly since estimatePoseSingleMarkers was removed
        in newer OpenCV versions.

        Returns (rvec, tvec) as (3,1) arrays.
        """
        img_pts = corners_single.reshape(4, 2).astype(np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self._obj_pts,
            img_pts,
            self.K,
            self.dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("solvePnP failed for marker corners")
        return rvec, tvec

    # -----------------------------------------------------------------------
    def find_tags(self, frame: np.ndarray):
        """
        Detect ArUco markers in *frame* (BGR or grayscale numpy array).

        Returns
        -------
        list of (id: int, T_cam_tag: np.ndarray shape (4,4))
            Each entry is the rigid body transform that maps points in the
            TAG frame into the CAMERA frame (i.e. T_camera_from_tag).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if frame.ndim == 3 else frame.copy()

        corners, ids, _ = self._detector.detectMarkers(gray)

        results = []
        if ids is None:
            return results

        for i, marker_id in enumerate(ids.ravel()):
            rvec, tvec = self._estimate_pose(corners[i])
            T_cam_tag  = _rvec_tvec_to_T(rvec, tvec)
            results.append((int(marker_id), T_cam_tag))

        return results

    # -----------------------------------------------------------------------
    def draw_tags(self, frame: np.ndarray, detections=None):
        """
        Annotate *frame* with detected markers and their axes.
        If *detections* is None, find_tags() is called automatically.
        Returns a copy of the frame with overlays.
        """
        out = frame.copy()
        if detections is None:
            detections = self.find_tags(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if frame.ndim == 3 else frame.copy()
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(out, corners, ids)
            for i in range(len(ids)):
                rvec, tvec = _T_to_rvec_tvec(detections[i][1])
                cv2.drawFrameAxes(out, self.K, self.dist,
                                  rvec, tvec, self.tag_size * 1.5)
        return out


# ---------------------------------------------------------------------------
# Part 1 in-lab - print tag detections
# ---------------------------------------------------------------------------

def print_inlab_tag_detections(image_path: str, camera_pose_T: np.ndarray = None):
    """
    Part 1 In-lab (section 3a) and Lab Report Question 4a:
    Detect and print all tag IDs and their transforms from the in-lab test image.

    Prints both:
      - T_cam_tag : transform of tag in camera frame (raw detection)
      - T_base_tag: transform of tag in robot base frame (if camera_pose_T provided)

    Parameters
    ----------
    image_path    : path to the in-lab test image
    camera_pose_T : (4,4) camera pose in robot base frame. If None, only
                    the camera-frame transforms are printed.
    """
    if not os.path.exists(image_path):
        print(f"  ERROR: image not found at {image_path}")
        return []

    frame      = cv2.imread(image_path)
    det        = ArucoDetector()
    detections = det.find_tags(frame)

    print(f"\n--- Part 1 In-lab: Tag Detections ({image_path}) ---")
    if not detections:
        print("  No tags detected.")
        return []

    for tag_id, T_cam_tag in detections:
        print(f"\n  Tag ID: {tag_id}")
        print(f"  T_cam_tag (tag in camera frame):")
        print(np.round(T_cam_tag, 6))
        rpy = rpy_from_R(T_cam_tag[:3, :3])
        print(f"  position (m) : {T_cam_tag[:3, 3].round(6)}")
        print(f"  RPY (rad)    : {rpy.round(6)}")

        if camera_pose_T is not None:
            T_base_tag = camera_pose_T @ T_cam_tag
            print(f"  T_base_tag (tag in robot base frame):")
            print(np.round(T_base_tag, 6))
            rpy_base = rpy_from_R(T_base_tag[:3, :3])
            print(f"  position (m) : {T_base_tag[:3, 3].round(6)}")
            print(f"  RPY (rad)    : {rpy_base.round(6)}")

    # Save annotated image for visual confirmation
    annotated = det.draw_tags(frame, detections)
    out_path  = image_path.rsplit(".", 1)[0] + "_annotated." + image_path.rsplit(".", 1)[1]
    cv2.imwrite(out_path, annotated)
    print(f"\n  Annotated image saved -> {out_path}")

    return detections


# ---------------------------------------------------------------------------
# Part 2 - Grasp pose
# ---------------------------------------------------------------------------

def get_grasp_pose(T_ArUco_base: np.ndarray, d: float) -> np.ndarray:
    """
    Compute the gripper pose (in robot base frame) needed to grasp a
    rectangular block whose ArUco marker is always in the BOTTOM-LEFT corner
    of the block face (so the block center is in the +x/+y quadrant of the
    ArUco frame).

    Parameters
    ----------
    T_ArUco_base : (4,4)
        Transform from robot base to the ArUco tag (i.e. T_base_ArUco).
    d : float
        Width of the rectangular prism (meters).

    Returns
    -------
    T_bg : (4,4)
        Transform from robot base to the gripper center, positioned
        above the block center at a safe pre-grasp standoff, with the
        gripper z-axis pointing DOWN and open fingers aligned with the
        block x-axis.

    Notes
    -----
    * Tag is in the bottom-left, so block center is at (+d/2, +d/2, 0)
      in the ArUco frame (flat on table -> no z offset from ArUco plane).
    * Block thickness ~5 cm; gripper pre-grasp standoff is 5 cm above
      the top face so fingers clear the block before closing.
    * Gripper center is 20 cm along the tool z-axis (lab spec section 3.2c).
    * The gripper z-axis must point DOWN (toward the table) so that
      closing the fingers squeezes the block sides.
    """
    BLOCK_THICKNESS = 0.05    # 5 cm (spec section 3.2)
    STANDOFF        = 0.15    # 5 cm pre-grasp clearance
    GRIPPER_OFFSET  = 0.20    # 20 cm flange -> gripper center (spec)

    # -----------------------------------------------------------------------
    # 1. Block center in ArUco frame
    #    Tag is bottom-left; center is +d/2 along x and y.
    #    The tag lies flat (z ~0 in robot frame), so z of center in
    #    ArUco frame is also ~0.
    # -----------------------------------------------------------------------
    p_center_aruco = np.array([d / 2.0, d / 2.0, 0.0, 1.0])
    p_center_base  = T_ArUco_base @ p_center_aruco   # (4,)

    # -----------------------------------------------------------------------
    # 2. Gripper orientation
    #    We want the gripper z-axis pointing DOWN (world -z) and grabbing
    #    along the block x-axis (spec recommends grabbing along x-axis).
    #    In robot base frame:
    #      gripper_z = [0, 0, -1]  (pointing down)
    #      gripper_x aligned with ArUco x (block side to grab)
    #      gripper_y = gripper_z x gripper_x
    # -----------------------------------------------------------------------
    aruco_x_base = T_ArUco_base[:3, 0]
    aruco_x_base = aruco_x_base / np.linalg.norm(aruco_x_base)

    gripper_z = np.array([0.0, 0.0, -1.0])   # pointing down
    gripper_x = aruco_x_base                  # grab along block x-axis
    gripper_y = np.cross(gripper_z, gripper_x)
    gripper_y /= np.linalg.norm(gripper_y)
    # Re-orthogonalize x
    gripper_x = np.cross(gripper_y, gripper_z)
    gripper_x /= np.linalg.norm(gripper_x)

    R_gripper = np.column_stack([gripper_x, gripper_y, gripper_z])

    # -----------------------------------------------------------------------
    # 3. Gripper position
    #    block_top_z  = tag_z + block_thickness
    #    flange_z     = block_top_z + standoff
    #    gripper_z    = flange_z - gripper_offset   (tool-z points down)
    # -----------------------------------------------------------------------
    table_z         = p_center_base[2]
    block_top_z     = table_z + BLOCK_THICKNESS
    gripper_z_coord = block_top_z + STANDOFF + GRIPPER_OFFSET

    p_gripper = np.array([
        p_center_base[0],
        p_center_base[1],
        gripper_z_coord,
    ])

    # -----------------------------------------------------------------------
    # 4. Assemble T_bg
    # -----------------------------------------------------------------------
    T_bg = np.eye(4)
    T_bg[:3, :3] = R_gripper
    T_bg[:3,  3] = p_gripper
    return T_bg


# ---------------------------------------------------------------------------
# Part 2 in-lab - print grasp transforms
# ---------------------------------------------------------------------------

def compute_and_print_inlab_grasp(detections, camera_pose_T, block_width=0.08):
    """
    Part 2 In-lab (section 3a) and Lab Report Question 4a:
    Given detections from the in-lab test image, compute and print the grasp
    transform for each detected tag.

    Parameters
    ----------
    detections    : list of (id, T_cam_tag) from ArucoDetector.find_tags()
    camera_pose_T : (4,4) camera pose in robot base frame
    block_width   : float, 8 cm per spec section 3a
    """
    print("\n--- Part 2 In-lab: Grasp Transforms ---")
    for tag_id, T_cam_tag in detections:
        T_base_tag = camera_pose_T @ T_cam_tag

        print(f"\n  Tag ID {tag_id}:")
        print(f"  T_base_tag (tag in base frame):")
        print(np.round(T_base_tag, 6))

        T_grasp = get_grasp_pose(T_base_tag, d=block_width)
        print(f"  T_bg (grasp pose, base -> gripper center):")
        print(np.round(T_grasp, 6))


# ---------------------------------------------------------------------------
# Part 3 - HanoiSolver
# ---------------------------------------------------------------------------

# Hanoi sequence generator (recursive, classic 3-peg)
def _hanoi_moves(n, source=0, target=2, auxiliary=1):
    """Return ordered list of (from_peg, to_peg) moves for n disks."""
    if n == 0:
        return []
    moves  = _hanoi_moves(n-1, source, auxiliary, target)
    moves += [(source, target)]
    moves += _hanoi_moves(n-1, auxiliary, target, source)
    return moves


class HanoiSolver:
    """
    Solves the Tower of Hanoi with a UR10e robot using vision.

    Assumes:
      - Camera is mounted at frame 5 with transform T5C (see top of file).
      - Towers are ArUco IDs 0, 1, 2.
      - Blocks  are ArUco IDs 3, 4, 5 (widths 0.06, 0.10, 0.12 m).
      - Blocks start stacked on tower 0, ordered largest-at-bottom.
    """

    # Known block widths (id -> width in meters, spec section 4.2b)
    BLOCK_WIDTHS = {3: 0.06, 4: 0.10, 5: 0.12}

    # Tower side length (spec section 4.2b)
    TOWER_SIZE = 0.20   # 20 cm x 20 cm

    # Approximate height of each block - used to estimate z of stacked blocks.
    # Tune these in lab if needed.
    BLOCK_HEIGHTS = {5: 0.04, 4: 0.04, 3: 0.04}

    # Scan joint configs: positions that give the camera good coverage.
    # Adjust in lab based on actual workspace.
    SCAN_CONFIGS = [
    	np.array([ 30, -113, 111, -85, -93, 0.0]),
    	np.array([ 41, -105, 122, -103, -94, 0.0]),
    	np.array([ 55, -118, 131, -104, -87, 0.0]),
    	np.array([ 46.4, -85, 101, -108, -86, 0.0]),
    	np.array([ 81, -96, 115, -108, -86, 0.0]),
    	np.array([ 96, -121, 120, -85, -87, 0.0]),
    	np.array([ 96, -93, 110, -107, -87, 0.0]),
    ]
    
    SCAN_CONFIGS = np.radians(SCAN_CONFIGS)

    # Safe travel height above the table (meters, base frame z)
    TRAVEL_Z = 0.50

    def __init__(self,
                 detector: ArucoDetector = None,
                 rob=None,
                 gripper=None):
        self.detector    = detector or ArucoDetector()
        self.rob         = rob        # urx Robot object (None = offline)
        self.gripper     = gripper    # RobotiqGripper object (None = offline)
        self.tower_poses = {}         # id -> T_base_tag
        self.block_poses = {}         # id -> T_base_tag
        # Stack state: tower_id -> list of block_ids (bottom first)
        self.stack = {0: [5, 4, 3], 1: [], 2: []}

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_joint_angles(self):
        """Read current joint angles (or return safe config offline)."""
        if self.rob:
            return np.array(self.rob.getj())
        return POS_SAFE.copy()

    def _movej(self, theta):
        """Move joints to *theta* (radians)."""
        if self.rob:
            self.rob.movej(theta, AJ, VJ, wait=True)
            
    def _movel(self, pose, rpy):
        """Move joints to pose (xyz) and orientation (rpy)."""
        if self.rob:
            self.rob.movel((pose[0], pose[1], pose[2], rpy[0], rpy[1], rpy[2]), AL, VL, relative=False, wait=True)

    def _get_frame(self):
        """
        Capture one frame from the robot's wrist camera.
        Offline: returns None (callers must handle).
        """
        if self.rob is None:
            return None
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None
        
        # Verify the camera actually returned the requested resolution.
        h, w = frame.shape[:2]
        if h != 1536 or w != 2304:
            raise RuntimeError(
                f"Camera returned {w}x{h} but K was calibrated for 2304x1536. "
                f"Tag localization will be incorrect at this resolution."
            )
        
        return frame

    def _fk_frame5(self, theta):
        """
        Compute the 4x4 transform from the base frame to FRAME 5
        (NOT the tool - camera is attached at frame 5, not the flange).
        """
        T = np.eye(4)
        for i in range(5):    # joints 1-5 only
            a, alpha, d, off = _DH_CL[i]
            ct, st = np.cos(theta[i] + off), np.sin(theta[i] + off)
            ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([
                [ct, -st*ca,  st*sa,  a*ct],
                [st,  ct*ca, -ct*sa,  a*st],
                [0.0, sa,     ca,     d   ],
                [0.0, 0.0,    0.0,    1.0 ],
            ])
            T = T @ Ti
        return T

    def _camera_to_base(self, theta, T_cam_tag):
        """
        Convert a tag pose in the camera frame to the robot base frame.

        T_base_tag = T_base_5 @ T5C @ T_cam_tag
        """
        T_base_5   = self._fk_frame5(theta)
        T_base_tag = T_base_5 @ T5C @ T_cam_tag
        return T_base_tag

    # -----------------------------------------------------------------------
    # Part 3a - marker_search
    # -----------------------------------------------------------------------

    def marker_search(self):
        """
        Sweep the robot to several scan configurations, capture frames,
        detect ArUco tags, and store their poses in the base frame.

        Populates self.tower_poses (IDs 0-2) and self.block_poses (IDs 3-5).
        Returns (tower_poses, block_poses).
        """
        if self.rob is None:
            print("[marker_search] Offline mode - returning empty dicts.")
            return {}, {}

        for config in self.SCAN_CONFIGS:
            self._movej(config)
            time.sleep(0.3)    # let robot settle

            frame = self._get_frame()
            if frame is None:
                continue

            theta      = self._get_joint_angles()
            detections = self.detector.find_tags(frame)

            for tag_id, T_cam_tag in detections:
                T_base_tag = self._camera_to_base(theta, T_cam_tag)
                if tag_id in (0, 1, 2):
                    if tag_id in self.tower_poses:
                        print(f"   TOWER {tag_id} already exists in self.tower_poses. Continuing...")
                        continue
                    self.tower_poses[tag_id] = T_base_tag
                    print(f"  Found TOWER {tag_id}: pos={T_base_tag[:3,3].round(4)}")
                    print(f"  T_base_tag:\n{np.round(T_base_tag, 6)}")
                elif tag_id in (3, 4, 5):
                    if tag_id in self.block_poses:
                        print(f"  BLOCK {tag_id} already exists in self.block_poses. Continuing...")
                        continue
                    self.block_poses[tag_id] = T_base_tag
                    print(f"  Found BLOCK {tag_id}: pos={T_base_tag[:3,3].round(4)}")
                    print(f"  T_base_tag:\n{np.round(T_base_tag, 6)}")

            if len(self.tower_poses) >= 3 and len(self.block_poses) >= 3:
                print("  All tags found - stopping scan.")
                print(f"  self.tower_poses length: {len(self.tower_poses)}")
                print(f"  self.block_poses length: {len(self.block_poses)}")
                break

        return self.tower_poses, self.block_poses

    # -----------------------------------------------------------------------
    # Grasp / place helpers
    # -----------------------------------------------------------------------

    def _block_stack_z(self, tower_id):
        """
        Estimate z of the top of the current stack on tower_id.
        Uses known block heights rather than noisy ArUco z.
        """
        z_table = 0.0
        # if tower_id in self.tower_poses:
        #     z_table = self.tower_poses[tower_id][2, 3]
        z = z_table
        for bid in self.stack[tower_id]:
            z += self.BLOCK_HEIGHTS[bid]
        return z

    def _pick_block(self, block_id, tower_id):
        """
        Move to the block, grasp it, and lift it to TRAVEL_Z.
        """
        if self.rob is None:
            print(f"  [offline] pick block {block_id} from tower {tower_id}")
            return

        T_aruco = self.block_poses[block_id]
        d       = self.BLOCK_WIDTHS[block_id]

        # Adjust z using stacked height estimate
        T_aruco_adj       = T_aruco.copy()
        T_aruco_adj[2, 3] = (self._block_stack_z(tower_id)
                             - self.BLOCK_HEIGHTS.get(block_id, 0.04))

        T_pre = get_grasp_pose(T_aruco_adj, d)
        print(f"  Pre-grasp pose (block {block_id}):\n{np.round(T_pre, 6)}")

        # IK -> move above block
        #sols = IK(T_pre)
        #if not sols:
            #raise RuntimeError(f"No IK solution for pre-grasp pose (block {block_id})")
        #theta_pre = DHModifiedToClassical(sols[0])

        # Open gripper
        self.gripper.open(speed=128, force=10)

        # self._movej(theta_pre)
        pre_pose = T_pre[:3, 3]
        rpy = rpy_from_R(T_pre[:3, :3])
        print(rpy)
        self._movel(pre_pose, rpy)
        print("Moved to pre-grasp position")

        # Lower to grasp height (remove standoff)
        # T_grasp       = T_pre.copy()
        # T_grasp[2, 3] -= 0.05
        # sols_g = IK(T_grasp)
        # if sols_g:
        #     self._movej(DHModifiedToClassical(sols_g[0]))
        grasp_pose = pre_pose.copy()
        grasp_pose[2] -= 0.15
        self._movel(grasp_pose, rpy)

        # Close gripper
        self.gripper.close(speed=200, force=200)
        time.sleep(0.5)

        # Lift to travel height
        # T_lift       = T_grasp.copy()
        # T_lift[2, 3] = self.TRAVEL_Z
        # sols_l = IK(T_lift)
        # if sols_l:
        #     self._movej(DHModifiedToClassical(sols_l[0]))
        lift_pose = grasp_pose.copy()
        lift_pose[2] = self.TRAVEL_Z
        self._movel(lift_pose, rpy)

    def _place_block(self, block_id, tower_id):
        """
        Move to tower, lower block, release, and retract.
        """
        if self.rob is None:
            print(f"  [offline] place block {block_id} on tower {tower_id}")
            return

        T_tower = self.tower_poses[tower_id]
        d       = self.BLOCK_WIDTHS[block_id]

        # Place at the top of the current stack
        T_place       = T_tower.copy()
        T_place[2, 3] = self._block_stack_z(tower_id)

        T_pre = get_grasp_pose(T_place, d)
        print(f"  Place pose (block {block_id} -> tower {tower_id}):\n{np.round(T_pre, 6)}")

        # sols = IK(T_pre)
        # if not sols:
        #     raise RuntimeError(
        #         f"No IK solution for place pose (block {block_id}, tower {tower_id})"
        #     )
        # self._movej(DHModifiedToClassical(sols[0]))
        pre_pose = T_pre[:3, 3]
        print(f"T_pre: {T_pre}")
        print(f"pre-pose: {pre_pose}")
        rpy = rpy_from_R(T_pre[:3, :3])
        print(f"rpy: {rpy}")
        self._movel(pre_pose, rpy)

        # Lower
        # T_down       = T_pre.copy()
        # T_down[2, 3] -= 0.05
        # sols_d = IK(T_down)
        # if sols_d:
        #     self._movej(DHModifiedToClassical(sols_d[0]))
        down_pose = pre_pose.copy()
        down_pose[2] -= 0.05
        print(f"down_pose: {down_pose}")
        self._movel(down_pose, rpy)

        # Release
        self.gripper.open(speed=128, force=10)
        time.sleep(0.3)

        # Retract back to pre-grasp height
        # self._movej(DHModifiedToClassical(sols[0]))
        self._movel(pre_pose, rpy)

    # -----------------------------------------------------------------------
    # Part 3b - solve_tower_of_hanoi
    # -----------------------------------------------------------------------

    def solve_tower_of_hanoi(self, n_disks=3):
        """
        Executes the Tower of Hanoi solution, moving n_disks from peg 0 to peg 2.
        Expects marker_search() to have been called first.
        """
        moves = _hanoi_moves(n_disks)
        print(f"Tower of Hanoi: {len(moves)} moves for {n_disks} disks.")

        for step_num, (src, dst) in enumerate(moves):
            block_id = self.stack[src][-1]   # top of source stack
            print(f"\n  Step {step_num+1}: move block {block_id}  peg {src} -> peg {dst}")

            self._pick_block(block_id, src)
            self.stack[src].pop()

            self._place_block(block_id, dst)
            self.stack[dst].append(block_id)

        print("Hanoi complete!")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_aruco_detection_practice_image(image_path="aruco_detection_test_practice.png"):
    """
    Part 1 Lab Prep test: run ArucoDetector on the provided practice image.
    Expected:
      - IDs 0 and 5 detected
      - ID 0: position ~(0.157, -0.100, 0.542), rotation ~pi about y
      - ID 5: position ~(0.087, -0.084, 0.489), rotation ~pi about y
    """
    print("\n--- test_aruco_detection_practice_image ---")
    if not os.path.exists(image_path):
        print(f"  SKIP: {image_path} not found.")
        return True

    frame   = cv2.imread(image_path)
    det     = ArucoDetector()
    results = det.find_tags(frame)

    if not results:
        print("  FAIL: no tags detected.")
        return False

    expected_positions = {
        0: np.array([ 0.157,  -0.100,  0.542]),
        5: np.array([ 0.0865, -0.0836, 0.489]),
    }

    found_ids = {r[0] for r in results}
    print(f"  Detected IDs: {sorted(found_ids)}")
    all_ok = True

    for tag_id, T in results:
        pos = T[:3, 3]
        rpy = rpy_from_R(T[:3, :3])
        print(f"  ID {tag_id}:")
        print(f"    position (m) : {pos.round(4)}")
        print(f"    RPY (rad)    : {rpy.round(4)}")

        if tag_id in expected_positions:
            exp = expected_positions[tag_id]
            err = np.linalg.norm(pos - exp)
            ok  = err < 0.015   # 1.5 cm tolerance
            print(f"    pos error vs expected: {err*100:.2f} cm  -> {'PASS' if ok else 'WARN'}")
            if not ok:
                all_ok = False

    # Save annotated image next to input
    annotated = det.draw_tags(frame, results)
    out_path  = image_path.replace(".png", "_annotated.png")
    cv2.imwrite(out_path, annotated)
    print(f"  Annotated image saved -> {out_path}")
    return all_ok


def test_aruco_synthetic():
    """
    Render a synthetic ArUco marker into a blank image and confirm the
    detector finds it. Pose accuracy is not checked since a flat-on marker
    with no perspective gives poor 3D cues - we just verify detection runs
    without crashing and returns the right ID.
    """
    print("\n--- test_aruco_synthetic ---")

    img_size   = 800
    frame      = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
    marker_img = np.zeros((200, 200), dtype=np.uint8)
    aruco.generateImageMarker(aruco_dict, 0, 200, marker_img)
    frame[300:500, 300:500, :] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

    det     = ArucoDetector()
    results = det.find_tags(frame)

    if results and results[0][0] == 0:
        print(f"  PASS: detected ID {results[0][0]}, pos={results[0][1][:3,3].round(4)}")
        return True
    else:
        print(f"  FAIL: expected ID 0, got {[r[0] for r in results]}")
        return False


def test_get_grasp_pose_practice():
    """
    Part 2 Lab Prep test: compute grasp poses for both tags in the practice image
    using the camera pose given in the spec.

    Spec camera pose: position (0.0, -0.4, 0.4), RPY (pi, 0, pi) w.r.t. base frame.
    Tag poses from spec (in camera frame):
      ID 0: pos (0.157, -0.100, 0.542)
      ID 5: pos (0.0865, -0.0836, 0.489)
      Both with rotation ~pi about y-axis.

    With the camera pointing straight down, the tags end up below the camera
    origin in base-frame z, so we check that the gripper z offset matches
    the expected value of BLOCK_THICKNESS + STANDOFF - GRIPPER_OFFSET.
    """
    print("\n--- test_get_grasp_pose_practice ---")

    T_cam_base = _make_camera_pose()

    def Ry(a): return np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])

    def make_T_cam_tag(pos):
        T = np.eye(4)
        T[:3, 3]  = pos
        T[:3, :3] = Ry(np.pi)
        return T

    tags = {
        0: make_T_cam_tag([0.157,  -0.100,  0.542]),
        5: make_T_cam_tag([0.0865, -0.0836, 0.489]),
    }

    all_ok = True
    for tag_id, T_cam_tag in tags.items():
        T_base_tag = T_cam_base @ T_cam_tag
        print(f"\n  Tag {tag_id} in base frame:")
        print(f"    position: {T_base_tag[:3,3].round(4)}")

        T_grasp = get_grasp_pose(T_base_tag, d=0.06)
        print(f"  Grasp pose for block {tag_id}:")
        print(f"    position: {T_grasp[:3,3].round(4)}")
        print(f"    rotation:\n{T_grasp[:3,:3].round(4)}")

        pointing_down = T_grasp[2, 2] < -0.9
        expected_z    = T_base_tag[2, 3] + 0.05 + 0.05 - 0.20
        z_err         = abs(T_grasp[2, 3] - expected_z)
        z_ok          = z_err < 0.01

        print(f"    gripper z-axis pointing down  : {pointing_down}")
        print(f"    gripper z within 1 cm of expected ({expected_z:.4f} m): "
              f"{z_ok}  (err={z_err*100:.2f} cm)")

        ok = pointing_down and z_ok
        print(f"    -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False

    return all_ok


def test_grasp_pose_orientation():
    """
    Unit test: verify get_grasp_pose returns a proper rotation matrix
    and that the gripper z-axis points downward in the base frame.
    """
    print("\n--- test_grasp_pose_orientation ---")

    T_aruco        = np.eye(4)
    T_aruco[:3, 3] = [0.3, -0.5, 0.0]
    d              = 0.10

    T_bg = get_grasp_pose(T_aruco, d)
    R    = T_bg[:3, :3]

    det           = np.linalg.det(R)
    orthogonal    = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    pointing_down = T_bg[2, 2] < -0.9

    print(f"  det(R) = {det:.6f}  (expect 1.0)")
    print(f"  R orthogonal          : {orthogonal}")
    print(f"  gripper z points down : {pointing_down}")
    ok = orthogonal and abs(det - 1) < 1e-5 and pointing_down
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_hanoi_sequence():
    """
    Verify the Hanoi move sequence is correct for 3 disks (should be 7 moves).
    Simulate the stack state and verify all blocks reach peg 2 in the correct order.
    """
    print("\n--- test_hanoi_sequence ---")
    moves = _hanoi_moves(3)
    print(f"  Move count: {len(moves)}  (expect 7)")

    stack = {0: [5, 4, 3], 1: [], 2: []}   # bottom-first
    for step, (src, dst) in enumerate(moves):
        assert stack[src], f"Step {step}: source peg {src} is empty!"
        top = stack[src].pop()
        if stack[dst]:
            assert top < stack[dst][-1], \
                f"Step {step}: placed wider block {top} on narrower {stack[dst][-1]}!"
        stack[dst].append(top)
        print(f"  Move {step+1}: block {top}  peg {src} -> peg {dst}  |  stacks: {stack}")

    ok = (stack[2] == [5, 4, 3] and not stack[0] and not stack[1])
    print(f"  Final state correct: {ok}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_camera_to_base_frame():
    """
    Test _fk_frame5 and _camera_to_base by placing the robot at a known
    config and checking that the result matches a manual computation.
    """
    print("\n--- test_camera_to_base_frame ---")
    solver = HanoiSolver()

    theta    = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    T_base_5 = solver._fk_frame5(theta)
    print(f"  T_base_5 position: {T_base_5[:3,3].round(4)}")

    # A tag 0.3 m in front of the camera (along camera z)
    T_cam_tag       = np.eye(4)
    T_cam_tag[2, 3] = 0.3

    T_base_tag = solver._camera_to_base(theta, T_cam_tag)
    print(f"  T_base_tag position: {T_base_tag[:3,3].round(4)}")

    # Verify by manually computing expected position
    expected = T_base_5 @ T5C @ T_cam_tag
    err = np.linalg.norm(T_base_tag[:3, 3] - expected[:3, 3])
    ok  = err < 1e-9
    print(f"  Matches manual computation (err={err:.2e}): {'PASS' if ok else 'FAIL'}")
    return ok


def test_rvec_tvec_roundtrip():
    """Test that _rvec_tvec_to_T and _T_to_rvec_tvec are inverse operations."""
    print("\n--- test_rvec_tvec_roundtrip ---")

    rng       = np.random.default_rng(42)
    angle     = rng.uniform(0.1, np.pi - 0.1)
    axis      = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    rvec_orig = (angle * axis).reshape(3, 1)
    tvec_orig = rng.uniform(-0.5, 0.5, (3, 1))

    T                  = _rvec_tvec_to_T(rvec_orig, tvec_orig)
    rvec_rec, tvec_rec = _T_to_rvec_tvec(T)

    R_orig, _ = cv2.Rodrigues(rvec_orig)
    R_rec,  _ = cv2.Rodrigues(rvec_rec)
    rot_ok    = np.allclose(R_orig,            R_rec,           atol=1e-9)
    trans_ok  = np.allclose(tvec_orig.ravel(), tvec_rec.ravel(), atol=1e-9)

    ok = rot_ok and trans_ok
    print(f"  Rotation match    : {rot_ok}")
    print(f"  Translation match : {trans_ok}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_lab3_tests(practice_image="aruco_detection_test_practice.png"):
    """Run all Lab 3 tests."""
    print("-" * 60)
    print("LAB 3 TEST SUITE")
    print("-" * 60)

    tests = [
        ("aruco_detection_practice_image", lambda: test_aruco_detection_practice_image(practice_image)),
        ("aruco_synthetic",                test_aruco_synthetic),
        ("get_grasp_pose_practice",        test_get_grasp_pose_practice),
        ("grasp_pose_orientation",         test_grasp_pose_orientation),
        ("hanoi_sequence",                 test_hanoi_sequence),
        ("camera_to_base_frame",           test_camera_to_base_frame),
        ("rvec_tvec_roundtrip",            test_rvec_tvec_roundtrip),
    ]

    results = []
    for name, t in tests:
        try:
            r = t()
            results.append((name, r if r is not None else True))
        except Exception as e:
            print(f"  EXCEPTION in {name}: {e}")
            results.append((name, False))

    # Part 1 in-lab dry run: print tag transforms for practice image
    # In the actual lab, this will be called with the real in-lab image.
    T_cam_base = _make_camera_pose()
    print_inlab_tag_detections(practice_image, camera_pose_T=T_cam_base)

    # Part 2 in-lab dry run: print grasp transforms for practice image
    # In the actual lab, this will use the real in-lab image detections.
    if os.path.exists(practice_image):
        frame      = cv2.imread(practice_image)
        det        = ArucoDetector()
        detections = det.find_tags(frame)
        compute_and_print_inlab_grasp(detections, T_cam_base, block_width=0.08)
    else:
        print("  SKIP: practice image not found for grasp pose dry run.")

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)
    passed = sum(1 for _, r in results if r)
    for name, r in results:
        print(f"  {'PASS' if r else 'FAIL'}  {name}")
    print(f"\n{passed}/{len(results)} tests passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_lab3_tests("aruco_detection_test_practice.png")

    if RUN_ON_ROBOT:
        import urx
        from pyrobotiqur import RobotiqGripper

        rob_conn = urx.Robot(UR_IP)
        rob_conn.set_tcp((0, 0, 0, 0, 0, 0))
        rob_conn.set_payload(1.5, (0, 0, 0.07))
        time.sleep(0.2)
        rob_conn.movej(POS_SAFE, AJ, VJ, wait=True)

        # Initialize and activate gripper
        g = RobotiqGripper(UR_IP, port=63352, timeout=20.0)
        g.connect()
        g.activate()

        # Camera pose (same for both Part 1 and Part 2 in-lab, per spec)
        T_cam_base = _make_camera_pose()

        # Part 1 in-lab (section 3a) + Lab Report Question 4a:
        # swap in the actual filename provided in lab
        inlab_detections = print_inlab_tag_detections(
            "aruco_detection_test.png",
            camera_pose_T=T_cam_base,
        )

        # Part 2 in-lab (section 3a) + Lab Report Question 4a:
        compute_and_print_inlab_grasp(
            inlab_detections,
            camera_pose_T=T_cam_base,
            block_width=0.08,
        )

        # Part 3: Tower of Hanoi
        # detector = ArucoDetector()
        # solver   = HanoiSolver(detector=detector, rob=rob_conn, gripper=g)

        # print("\n--- Searching for markers ---")
        # solver.marker_search()

        # print("\n--- Solving Tower of Hanoi ---")
        # solver.solve_tower_of_hanoi(n_disks=3)

        # rob_conn.movej(POS_SAFE, AJ, VJ, wait=True)
        rob_conn.close()
        print("Done.")
