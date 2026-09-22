"""
controller.py  --  Autonomous precision landing on a moving ArUco platform
==========================================================================

Pipeline (runs at 240 Hz, called by local_arena.py):

  camera frame ─► ArUco corners ─► ray-cast to marker plane (uses drone pose)
        ─► world-frame rover position ─► Kalman filter (pos, vel, acc)
        ─► relative PD + acceleration feed-forward  (outer loop, world frame)
        ─► desired thrust vector  ─► desired roll / pitch
        ─► attitude PD (inner loop) ─► thrust mixer ─► 4 motor forces [N]

Conventions (verified against local_arena.py)
  * World: x,y horizontal, z up.  Body: x forward, y left, z up (thrust axis).
  * Motor layout (body frame):  f1 (+L,+L)  f2 (-L,+L)  f3 (+L,-L)  f4 (-L,-L)
        tau_x =  L (f1 + f2 - f3 - f4)
        tau_y =  L (-f1 + f2 - f3 + f4)
        tau_z =  0   <-- the arena applies pure vertical forces at points, so
                         there is NO yaw authority.  Yaw simply stays constant.
  * Camera: mounted 0.1 m below the base, looks along body -z, image "up" is
    body +x, so image-right = body -y.  Vertical FOV = 60 deg (PyBullet
    computeProjectionMatrixFOV), square pixels.
"""

import math
import numpy as np
import cv2


# ══════════════════════════════════════════════════════════════════════════
#  Physical / geometric constants
# ══════════════════════════════════════════════════════════════════════════
G = 9.81
MASS = 1.2                 # 1.0 chassis + 4 x 0.05 rotor pods
ARM = 0.25                 # motor lever arm in x and y  (L in the arena)
IXX = 0.0175               # approx. composite roll/pitch inertia [kg m^2]
DT = 1.0 / 240.0
F_MAX = 12.0               # per-motor thrust limit [N]  (T/W ~ 4)

CAM_FOV_DEG = 60.0         # vertical FOV
CAM_OFFSET_BODY = np.array([0.0, 0.0, -0.1])

Z_MARKER = 0.102           # world height of the ArUco plane
Z_CONTACT = 0.144          # base-link height when resting on the plate
Z_CUT = 0.156              # cut motors when the base drops below this


# ══════════════════════════════════════════════════════════════════════════
#  Tunables
# ══════════════════════════════════════════════════════════════════════════
# outer loop (relative position / velocity of drone w.r.t. rover)
WN_XY, ZETA_XY = 3.6, 1.0           # -> Kp = wn^2, Kd = 2 zeta wn
A_MAX_XY = 3.5                      # horizontal accel limit [m/s^2]
TILT_MAX = math.radians(25.0)       # tilt limit while flying
TILT_MAX_LAND = math.radians(12.0)  # tilt limit near the ground

# inner loop (attitude)
WN_ATT, ZETA_ATT = 26.0, 0.9

# vertical
TRACK_ALT = 3.0
SEARCH_ALT = 4.5
KZ = 1.5                            # altitude P  (-> velocity command)
KVZ = 6.0                           # vertical velocity P (-> accel)
VZ_MAX = 1.5

# descent profile  v_down(h) = min( V_TOUCH + K_DESC*h , V_DESC_MAX )
V_TOUCH = 0.25
K_DESC = 2.5
V_DESC_MAX = 1.5

# state machine
SYNC_ERR = 0.12                     # [m]   XY error to allow descent
SYNC_VEL = 0.25                     # [m/s] relative speed to allow descent
SYNC_TIME = 0.5                     # [s]   both must hold this long
GATE_LO, GATE_HI = 0.10, 0.30       # descent speed fades to 0 between these
LOST_TO_SEARCH = 0.6                # [s] no marker -> SEARCH
BLIND_LOST = 0.03                   # [s] lost while low -> BLIND_DESCEND
BLIND_MAX_ALT = 1.8                 # only go blind below this height
ABORT_ERR = 0.8                     # [m] XY error while visible -> back to TRACK


# ══════════════════════════════════════════════════════════════════════════
#  Rover Kalman filter  (constant-acceleration model, white-jerk process noise)
# ══════════════════════════════════════════════════════════════════════════
class RoverKF:
    """x = [pos, vel, acc] for X and Y (shared covariance, model is identical)."""

    def __init__(self, q_jerk=3.0):
        self.q = q_jerk
        self.x = np.zeros((3, 2))
        self.P = np.diag([1.0, 4.0, 4.0])
        self.init = False

    def reset(self, pos_xy):
        self.x = np.zeros((3, 2))
        self.x[0] = pos_xy
        self.P = np.diag([0.02 ** 2, 2.0 ** 2, 2.0 ** 2])
        self.init = True

    def predict(self, dt, damp_acc=False):
        F = np.array([[1, dt, 0.5 * dt * dt],
                      [0, 1, dt],
                      [0, 0, 1.0]])
        if damp_acc:                      # long dropout: don't trust a(t)
            F[2, 2] = 0.985
        Q = self.q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                               [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                               [dt ** 3 / 6, dt ** 2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_xy, sigma):
        S = self.P[0, 0] + sigma ** 2
        innov = z_xy - self.x[0]
        # adaptive innovation gate (grows automatically during dropouts)
        if np.linalg.norm(innov) > 6.0 * math.sqrt(S) + 0.25:
            return False
        K = self.P[:, 0] / S
        self.x = self.x + np.outer(K, innov)
        H = np.array([[1.0, 0.0, 0.0]])
        I_KH = np.eye(3) - np.outer(K, H[0])
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * sigma ** 2
        return True


# ══════════════════════════════════════════════════════════════════════════
#  Flight controller
# ══════════════════════════════════════════════════════════════════════════
class FlightController:

    # STATES  (same ids as the prototype)
    SEARCH = 0
    TRACK = 1
    DESCEND = 2
    BLIND_DESCEND = 3
    LANDED = 4
    STATE_NAMES = ["SEARCH", "TRACK", "DESCEND", "BLIND_DESCEND", "LANDED"]

    def __init__(self, debug=False):
        self.debug = debug
        self.dt = DT
        self.t = 0.0

        self.mass = MASS
        self.gravity = G
        self.hover_force = self.mass * self.gravity / 4.0

        # ---- vision ------------------------------------------------------
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._dict_ids = [cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_4X4_50,
                          cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_6X6_250,
                          cv2.aruco.DICT_7X7_50, cv2.aruco.DICT_ARUCO_ORIGINAL,
                          cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_5X5_250]
        self._detectors = [cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(d), params) for d in self._dict_ids]
        self._det_idx = 0            # 6X6_50 first (default), others as fallback
        self._locked = False         # dictionary + id locked after 1st success
        self._target_id = None
        self._miss_frames = 0
        self.marker_yaw = 0.0        # world yaw of the marker edge (diagnostic)

        # ---- estimation --------------------------------------------------
        self.kf = RoverKF()
        self.last_seen = -1e9
        self._prev_R = None
        self.omega_b = np.zeros(3)

        # ---- state machine ----------------------------------------------
        self.state = self.SEARCH
        self._sync_timer = 0.0
        self._anchor = None          # spiral-search centre
        self._spiral_theta = 0.0

        self._log_t = 0.0

    # ------------------------------------------------------------------
    #  small math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def quat_to_rot(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

    @staticmethod
    def rot_to_euler(R):
        """ZYX (roll about x, pitch about y, yaw about z) - same as PyBullet."""
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        yaw = math.atan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    @staticmethod
    def quat_to_euler(q):
        return FlightController.rot_to_euler(FlightController.quat_to_rot(q))

    def _body_rates(self, R):
        """Angular velocity in the body frame from consecutive attitudes."""
        if self._prev_R is None:
            self._prev_R = R
            return np.zeros(3)
        Rrel = self._prev_R.T @ R
        M = 0.5 * (Rrel - Rrel.T)
        self._prev_R = R
        return np.array([M[2, 1], M[0, 2], M[1, 0]]) / self.dt

    # ------------------------------------------------------------------
    #  [A] VISION
    # ------------------------------------------------------------------
    def _detect_corners(self, frame):
        """Return (4,2) float array of marker corners in pixels, or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if not self._locked:
            # first ~0.25 s use the default dictionary, afterwards sweep others
            if self._miss_frames > 60:
                self._det_idx = (self._det_idx + 1) % len(self._detectors)
        corners, ids, _ = self._detectors[self._det_idx].detectMarkers(gray)
        if ids is None or len(ids) == 0:
            self._miss_frames += 1
            return None
        ids = ids.flatten()
        if self._target_id is not None and self._target_id in ids:
            k = int(np.where(ids == self._target_id)[0][0])
        else:
            areas = [cv2.contourArea(c.reshape(4, 2).astype(np.float32)) for c in corners]
            k = int(np.argmax(areas))
        if not self._locked:
            self._locked = True
            self._target_id = int(ids[k])
        self._miss_frames = 0
        return corners[k].reshape(4, 2).astype(float)

    def _pixels_to_plane(self, uv, pos, R, w, h):
        """Ray-cast pixels onto the horizontal plane z = Z_MARKER (world frame).
        Exact for any drone attitude, so no small-angle tilt compensation."""
        f = (h / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        u, v = uv[:, 0], uv[:, 1]
        # camera looks along body -z; image-up = body +x; image-right = body -y
        d_body = np.stack([-(v - cy) / f, -(u - cx) / f, -np.ones_like(u)], axis=1)
        d_world = d_body @ R.T
        cam = pos + R @ CAM_OFFSET_BODY
        dz = d_world[:, 2]
        if np.any(dz > -1e-3):
            return None, None
        s = (Z_MARKER - cam[2]) / dz
        pts = cam + s[:, None] * d_world
        return pts, float(np.mean(s))

    def _vision(self, frame, pos, R):
        """Returns (rover_xy_world, sigma) or None."""
        corners = self._detect_corners(frame)
        if corners is None:
            return None
        h, w = frame.shape[:2]
        pts, depth = self._pixels_to_plane(corners, pos, R, w, h)
        if pts is None or depth <= 0:
            return None
        centre = pts[:, :2].mean(axis=0)
        e01 = pts[1, :2] - pts[0, :2]
        self.marker_yaw = math.atan2(e01[1], e01[0])
        f = (h / 2.0) / math.tan(math.radians(CAM_FOV_DEG) / 2.0)
        sigma = 0.6 * depth / f + 0.004        # ~0.6 px of corner noise
        return centre, sigma

    # ------------------------------------------------------------------
    #  [B] helpers for reference generation
    # ------------------------------------------------------------------
    def _spiral_ref(self, centre):
        """Archimedean spiral at ~1.5 m/s around `centre` (search pattern)."""
        b = 0.35
        th = self._spiral_theta
        r = b * th
        self._spiral_theta += self.dt * 1.5 / (b * math.sqrt(1 + th * th))
        p = centre + r * np.array([math.cos(th), math.sin(th)])
        return p

    @staticmethod
    def _descent_profile(z):
        """Smooth exponential-to-contact profile.  Returns (vz_des, az_ff)."""
        h = max(z - Z_CONTACT, 0.0)
        v = V_TOUCH + K_DESC * h
        if v >= V_DESC_MAX:
            return -V_DESC_MAX, 0.0
        return -v, K_DESC * v           # speed shrinks at rate K*v

    # ------------------------------------------------------------------
    #  MAIN ENTRY
    # ------------------------------------------------------------------
    def calculate_forces(self, camera_frame, drone_pos, drone_ori, drone_vel):
        """Called by the arena 240x per second.  Returns f1..f4 in Newtons."""
        dt = self.dt
        self.t += dt
        pos = np.asarray(drone_pos, dtype=float)
        vel = np.asarray(drone_vel, dtype=float)
        R = self.quat_to_rot(drone_ori)
        roll, pitch, yaw = self.rot_to_euler(R)
        self.omega_b = self._body_rates(R)

        # ── Emergency motor cutoff (latched) ──────────────────────────
        if self.state == self.LANDED:
            return 0.0, 0.0, 0.0, 0.0
        if self.state in (self.DESCEND, self.BLIND_DESCEND) and pos[2] <= Z_CUT:
            self.state = self.LANDED
            return 0.0, 0.0, 0.0, 0.0

        # ── [A] vision + [B] state estimation ─────────────────────────
        meas = self._vision(camera_frame, pos, R)
        lost_for = self.t - self.last_seen
        if self.kf.init:
            self.kf.predict(dt, damp_acc=(lost_for > 0.6))
        if meas is not None:
            z_xy, sigma = meas
            if not self.kf.init or lost_for > 4.0:
                self.kf.reset(z_xy)
                self.last_seen = self.t
            elif self.kf.update(z_xy, sigma):
                self.last_seen = self.t
        visible = (self.t - self.last_seen) < 1.5 * dt
        lost_for = self.t - self.last_seen

        p_r, v_r, a_r = self.kf.x[0].copy(), self.kf.x[1].copy(), self.kf.x[2].copy()
        e_xy_vec = p_r - pos[:2]
        e_xy = float(np.linalg.norm(e_xy_vec))
        ev = float(np.linalg.norm(v_r - vel[:2]))

        # ── state machine ─────────────────────────────────────────────
        if self.state == self.SEARCH:
            if visible and self.kf.init:
                self.state = self.TRACK
                self._sync_timer = 0.0

        elif self.state == self.TRACK:
            if lost_for > LOST_TO_SEARCH:
                self.state = self.SEARCH
                self._spiral_theta = 0.0
                self._anchor = None
            else:
                ok = visible and e_xy < SYNC_ERR and ev < SYNC_VEL \
                    and abs(pos[2] - TRACK_ALT) < 0.4
                self._sync_timer = self._sync_timer + dt if ok else 0.0
                if self._sync_timer >= SYNC_TIME:
                    self.state = self.DESCEND

        elif self.state == self.DESCEND:
            if lost_for > BLIND_LOST:
                if pos[2] < BLIND_MAX_ALT:
                    self.state = self.BLIND_DESCEND      # marker out of FOV
                elif lost_for > 0.3:
                    self.state = self.TRACK              # lost while still high
                    self._sync_timer = 0.0
            elif e_xy > ABORT_ERR:
                self.state = self.TRACK
                self._sync_timer = 0.0

        # BLIND_DESCEND is committed: it only ends with touchdown

        # ── horizontal reference ──────────────────────────────────────
        if self.state == self.SEARCH:
            if self.kf.init and lost_for < 3.0:
                p_ref, v_ref, a_ref = p_r, v_r, np.zeros(2)  # chase last estimate
            else:
                if self._anchor is None:
                    self._anchor = p_r.copy() if self.kf.init else pos[:2].copy()
                p_ref = self._spiral_ref(self._anchor)
                v_ref, a_ref = np.zeros(2), np.zeros(2)
        else:
            p_ref, v_ref, a_ref = p_r, v_r, a_r

        kp, kd = WN_XY ** 2, 2 * ZETA_XY * WN_XY
        a_xy = a_ref + kp * (p_ref - pos[:2]) + kd * (v_ref - vel[:2])

        # ── vertical reference ────────────────────────────────────────
        if self.state in (self.DESCEND, self.BLIND_DESCEND):
            vz_des, az_ff = self._descent_profile(pos[2])
            # fade descent speed if the XY error is too large (landing cone)
            s = float(np.clip((GATE_HI - e_xy) / (GATE_HI - GATE_LO), 0.0, 1.0))
            vz_des *= s
            az_ff *= s
        else:
            z_ref = SEARCH_ALT if self.state == self.SEARCH else TRACK_ALT
            vz_des = float(np.clip(KZ * (z_ref - pos[2]), -VZ_MAX, VZ_MAX))
            az_ff = 0.0
        a_z = az_ff + KVZ * (vz_des - vel[2])

        # ── outer loop -> thrust vector -> desired attitude ───────────
        a_z = max(a_z, -0.8 * G)                       # never ask for free-fall
        tilt_max = TILT_MAX_LAND if pos[2] < 0.8 else TILT_MAX
        a_h_max = min(A_MAX_XY, (G + a_z) * math.tan(tilt_max))
        n = np.linalg.norm(a_xy)
        if n > a_h_max:
            a_xy = a_xy * (a_h_max / n)

        a_vec = np.array([a_xy[0], a_xy[1], G + a_z])
        b3 = R[:, 2]
        T = self.mass * float(a_vec @ b3)              # thrust along current z_b
        T = float(np.clip(T, 0.1 * self.mass * G, 4 * F_MAX))

        cy, sy = math.cos(yaw), math.sin(yaw)
        ax_h = cy * a_vec[0] + sy * a_vec[1]           # yaw-aligned frame
        ay_h = -sy * a_vec[0] + cy * a_vec[1]
        pitch_des = math.atan2(ax_h, a_vec[2])
        roll_des = math.atan2(-ay_h, math.hypot(ax_h, a_vec[2]))

        # ── inner loop: attitude PD  (torque = I * desired ang. accel) ──
        wn, z = WN_ATT, ZETA_ATT
        alpha_x = wn * wn * (roll_des - roll) - 2 * z * wn * self.omega_b[0]
        alpha_y = wn * wn * (pitch_des - pitch) - 2 * z * wn * self.omega_b[1]
        tau_x, tau_y = IXX * alpha_x, IXX * alpha_y

        # ── mixer (yaw torque is not producible in this arena) ────────
        dx = tau_x / (4 * ARM)
        dy = tau_y / (4 * ARM)
        d = np.array([dx - dy, dx + dy, -dx - dy, -dx + dy])
        base = T / 4.0
        base = float(np.clip(base, -d.min(), F_MAX - d.max())) \
            if (F_MAX - d.max()) > (-d.min()) else base
        f = np.clip(base + d, 0.0, F_MAX)

        if self.debug and self.t - self._log_t > 0.5:
            self._log_t = self.t
            print(f"[{self.t:5.2f}] {self.STATE_NAMES[self.state]:13s} "
                  f"z={pos[2]:.2f} e={e_xy:.3f} ev={ev:.2f} vis={visible} "
                  f"rp=({math.degrees(roll):.1f},{math.degrees(pitch):.1f})")
        return float(f[0]), float(f[1]), float(f[2]), float(f[3])