# Autonomous Precision Landing on a Dynamic Platform

Intra IIT Tech Meet 1.0 | Track: UAV Dynamics & Autonomous Systems

A single-file, vision-based flight controller (`controller.py`) that finds a moving ground rover with a downward camera, tracks it, descends onto it and cuts the motors at touch-down. It runs at 240 Hz inside the provided PyBullet arena and returns the four rotor forces in Newtons.

## Contents of the submission

| File | Purpose |
|---|---|
| `controller.py` | The submitted controller (`FlightController.calculate_forces`). |
| `eval_landing.py` | Auxiliary batch evaluator used for the report. It does not modify the controller or the arena. |
| `README.md` | This file. |
| `Landing_Controller_Report_Final.pdf` | Technical report (maths, block diagrams, vision pipeline, state machine, tuning, edge cases, validation). |
| Demo video | Link below. |

## Demo video

> **[PASTE DEMO VIDEO LINK HERE]**

## Dependencies

Required:

- Python 3
- NumPy
- OpenCV **with `cv2.aruco`**: `opencv-contrib-python`, version **4.7 or newer** (the controller uses the `cv2.aruco.ArucoDetector` API, and the arena uses `cv2.aruco.generateImageMarker`)
- PyBullet

Optional (only for `eval_landing.py`): `matplotlib`, for the telemetry figure.

Tested with:

| Package | Version |
|---|---|
| Python | 3.14.7 |
| NumPy | 2.5.3 |
| OpenCV | 5.0.0 (`cv2.aruco` available) |
| PyBullet | API version 202010061 |
| Platform | macOS-26.6.2-arm64-arm-64bit-Mach-O |

## Setup

```bash
conda create --name pybullet_env
conda activate pybullet_env
pip install numpy opencv-contrib-python pybullet matplotlib
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"   # must print True
```

Do not install `opencv-python` alongside `opencv-contrib-python`; the two conflict and `cv2.aruco` may go missing.

## Running

Place `controller.py` in the same folder as the arena script (`local_arena_wind.py` here), then run the arena:

```bash
python local_arena_wind.py
```

This opens the PyBullet GUI and the drone camera window. The run ends with `[LOCAL TEST SUCCESS]` or `[LOCAL TEST FAILED]`.

### Optional: batch evaluation

```bash
python eval_landing.py --arena local_arena_wind            # 20 nominal runs + 3 stress scenarios
python eval_landing.py --runs 40 --workers 4               # more runs, parallel
```

Outputs are written to `./eval_out/` (`results.csv`, `summary.json`, `report_tables.md`, `run0_telemetry.png`). The evaluator runs headless with the `tiny` camera renderer and randomises start position, rover phase and wind seed.

## Solution overview

Pipeline, executed every 1/240 s:

1. **Vision.** Grayscale, then `cv2.aruco.ArucoDetector` with sub-pixel corner refinement. Eight predefined dictionaries are supported; the dictionary and marker ID lock on the first detection.
2. **2D to 3D mapping.** Each of the four corners is ray-cast onto the marker plane using the drone's full attitude and camera offset, so the position is exact at any tilt (no small-angle shortcut).
3. **State estimation.** Attitude comes from the telemetry quaternion, and body rates from consecutive rotation matrices. A constant-acceleration Kalman filter turns the noisy marker positions into rover position, velocity and acceleration. It includes an adaptive innovation gate and dropout handling.
4. **Control.** Cascaded loops: position P, velocity PI, then acceleration for the outer loop; attitude P, then rate PI for the inner loop. Gains come from pole placement on (ωn, ζ). Outer loop: ωn = 3.6 rad/s, ζ = 1. Inner loop: ωn = 26 rad/s, ζ = 0.9. Rover velocity and acceleration are fed forward, and each integrator uses conditional-integration anti-windup.
5. **Mixing.** Thrust and roll/pitch torques map to the four rotor forces, with attitude given priority over thrust (motor limit 0 to 12 N). The arena has no yaw torque, so yaw is not commanded.
6. **Landing state machine.** `SEARCH` (spiral) → `TRACK` (synchronised tracking) → `DESCEND` → `LANDED` (emergency motor cut-off). `BLIND_DESCEND` is an added, committed sub-phase for the last metres if the marker leaves the camera's view.
7. **Descent profile.** `v_down(h) = min(0.25 + 2.5 h, 1.5)` m/s, which approaches about 0.28 m/s at the cut-off height (`Z_CUT` = 0.156 m). A landing-cone gate slows the descent when the horizontal error grows.

## Compliance with the problem statement

| Requirement (PS) | Where addressed |
|---|---|
| Detect the ArUco marker with `cv2.aruco` on the downward RGB feed | `_detect_corners`, `_vision` |
| Transform pixel data and telemetry into 3D relative tracking metrics | `_pixels_to_plane`, `RoverKF` |
| Cascaded control outputting four rotor forces in Newtons | `calculate_forces` |
| State machine: Search → Tracking → Descent → Motor cut-off | `FlightController` states |
| Impact velocity above -0.65 m/s and chassis tilt at most 30° | Descent profile and tilt clamps |
| Only `controller.py` is required for the arena | The arena imports `FlightController` from it |

## Results and validation

**In the provided arena.** With the default arena settings the controller tracks the rover and lands successfully with the motors cut at touch-down. The representative default-arena run is shown as Figure 13 in the report (contact speed and tilt within the limits).

**In a more rigorous evaluation.** The report evaluates the controller in a harder setting than the provided arena: randomised start position, rover phase and wind seed, plus stress scenarios (far start, camera dropout, low-altitude blackout). There it succeeds in most runs but **not always**:

| Scenario | Success |
|---|---|
| Nominal (randomised) | 19/20 |
| Far start (marker not visible) | 5/6 |
| 1 s camera dropout during tracking | 5/6 |
| Permanent blackout below 1.0 m | 5/6 |

Nominal touch-down speed averaged 0.28 m/s (worst 0.60 m/s against the 0.65 m/s limit), and chassis tilt at touch-down was at most 17.7°. The remaining misses were not root-caused within the available time; details and the worst-case figures are in Section 10 of the report.

## Known limitations

- The arena applies only vertical point forces, so yaw is uncontrolled (the controller works in a yaw-aligned frame).
- Arena constants (marker and contact heights, camera pose, inertia) are hard-coded in `controller.py`.
- Under a permanent camera blackout near the ground, the blind descent relies on the Kalman prediction alone, and contact speed can exceed the limit (worst case 1.23 m/s in the stress test).