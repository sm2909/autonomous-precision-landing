# Autonomous Precision Landing Controller — Architecture Overview

## 1. Problem Statement

A quadrotor must locate a moving ground rover carrying an ArUco marker, track it,
descend, and land on it — using only an onboard downward-facing camera and its own
state (position, orientation, velocity). The controller runs at **240 Hz** and must
be robust to marker dropouts, gusts/steady disturbances, and actuator saturation.

The system is organized as a **perception → estimation → decision → control → mixing**
pipeline, executed once per control tick:

```
Camera frame → ArUco detection → Ray-cast to plane → Rover position (world)
      → Kalman Filter (pos, vel, acc) → State Machine
      → Outer loop: Position → Velocity → Acceleration (cascade, PI)
      → Desired thrust vector → Desired roll/pitch (tilt-limited)
      → Inner loop: Attitude → Rate (cascade, PI)
      → Motor mixer → f1, f2, f3, f4
```

Each stage is described below, along with the reasoning behind its design.

---

## 2. Vision: ArUco Detection and Ray-Casting

**What it does:** Detects the ArUco marker in the camera frame and converts its
pixel corners into a world-frame XY position of the rover.

- **Multi-dictionary detector bank.** The marker's dictionary/ID isn't known in
  advance, so the controller tries several standard ArUco dictionaries
  (`6X6_50`, `4X4_50`, `5X5_50`, etc.), starting with the most common one. Once a
  marker is found, its dictionary and ID are **locked** — this avoids wasted
  detection attempts and prevents accidentally latching onto a different marker
  later.
- **Largest-contour tie-break.** If multiple markers are visible before locking,
  the one with the largest pixel area (closest/most reliable) is chosen.
- **Exact ray-casting instead of small-angle approximation.** Each corner pixel
  is converted into a 3D ray in the body frame (using the known vertical FOV and
  camera mounting), rotated into the world frame by the drone's current attitude,
  and intersected with the known marker plane `z = Z_MARKER`. Because this uses
  the full rotation matrix rather than a flat-ground/small-tilt approximation, the
  position estimate stays accurate even while the drone is banking hard to chase
  the rover — which matters because aggressive tracking is exactly when tilt is
  largest.
- **Marker edge → yaw diagnostic.** The orientation of one marker edge is kept as
  `marker_yaw`, useful for debugging/telemetry (not used for control, since the
  airframe has no yaw actuation — see §6).
- **Measurement noise model.** The per-frame measurement noise `sigma` is modeled
  as growing with ray depth (`0.6·depth/f`), i.e. the farther away / smaller the
  marker appears, the noisier the pixel-to-world conversion is. This feeds
  directly into the Kalman filter's trust in each measurement.

---

## 3. State Estimation: Rover Kalman Filter

**What it does:** Fuses noisy, sometimes-missing vision measurements into a smooth
estimate of the rover's position, velocity, and acceleration.

- **Constant-acceleration model with white-jerk noise.** State `x = [pos, vel, acc]`
  per axis. This is the standard choice for tracking a maneuvering target: it is
  the simplest model that still lets the filter anticipate rover acceleration
  (e.g., the rover turning or speeding up) rather than always lagging behind a
  purely constant-velocity assumption.
- **Adaptive innovation gate.** A measurement is rejected if it's more than
  `6·√S + 0.25` m from the prediction. This is a soft, self-scaling outlier
  rejection: `S` (innovation covariance) grows automatically during dropouts, so
  the gate widens rather than permanently rejecting valid measurements after a
  period of being lost.
- **Acceleration damping during long dropouts (`damp_acc`).** When the marker
  has been unseen for a while, the acceleration state's self-transition is damped
  toward zero. Without this, a stale acceleration estimate would keep extrapolating
  the rover's position further and further from where it actually is — damping
  makes the filter fall back toward constant-velocity behavior, which is safer
  than trusting old acceleration data indefinitely.
- **Reset on long loss / re-acquisition.** If the marker reappears after a long
  gap (`> 4 s`), the filter is reset rather than fused — the old estimate is
  considered too stale to be useful as a prior.

**Why a Kalman filter at all, instead of using raw vision directly:** raw
per-frame marker position is noisy and can vanish for a frame; the KF supplies
(a) a smoothed position, (b) a *velocity* estimate the tracking loop needs to
match speeds with the rover, and (c) an *acceleration* estimate used as
feed-forward so the drone doesn't always control from behind a moving target.

---

## 4. State Machine

**What it does:** Governs *which behavior* is active — searching, tracking,
descending, or landed — and the safe transitions between them.

| State | Purpose |
|---|---|
| `SEARCH` | No confident lock on the rover; fly a spiral search pattern (or chase the last known estimate if recently lost) at a high, safe altitude. |
| `TRACK` | Marker visible; match rover position/velocity at a constant altitude, waiting for **sync** (small position + velocity error, held for a minimum time) before committing to descent. |
| `DESCEND` | Actively losing altitude while continuing to track XY, using a speed profile that slows near contact. |
| `BLIND_DESCEND` | Marker briefly out of view *while already low and already descending* — commit to finishing the landing using the KF's short-term prediction rather than aborting. |
| `LANDED` | Terminal state — motors cut. |

Design rationale for each transition:

- **`SYNC_ERR` / `SYNC_VEL` / `SYNC_TIME` gate before descending.** Requiring the
  position *and* velocity error to be small *and* to stay small for `0.5 s`
  prevents starting a descent on a lucky single-frame close pass, which would
  likely be immediately violated and force an abort — wasting time and inducing
  oscillation.
- **`BLIND_DESCEND` is commit-only** ("only ends with touchdown"). Once low and
  descending, aborting back to `TRACK`/`SEARCH` (climbing back up) is *more*
  dangerous than finishing a short blind descent using the KF's short-horizon
  prediction, especially since the marker is most likely to leave the frame
  right before touchdown (camera geometry/FOV shrinks the visible ground area
  as altitude drops). The `BLIND_MAX_ALT` gate ensures this only happens close
  to the ground, where the KF prediction error over a fraction of a second is
  small.
- **`ABORT_ERR` while visible → back to `TRACK`.** If the marker *is* visible
  but the tracking error has grown large during descent (e.g. rover made a
  sharp turn), the safer choice is to re-establish tracking at altitude rather
  than continuing to chase downward with a large lateral offset.
- **`LOST_TO_SEARCH` / re-lock timers** prevent the controller from getting stuck
  in `TRACK` indefinitely on a permanently lost marker, and prevent trusting a
  KF estimate that is far too stale to reset from (see §3).
- **Emergency cutoff (`Z_CUT`)** is checked *before* the rest of the pipeline
  runs, independent of the state machine's own logic, as a hard floor: any time
  the vehicle is low and descending and touches the cutoff height, motors are
  cut immediately. This is a safety backstop that doesn't rely on any control
  loop behaving correctly.

---

## 5. Outer Loop: Position → Velocity → Acceleration Cascade (Horizontal)

**What it does:** Converts the rover's estimated XY position/velocity (relative
to the drone) into a desired horizontal acceleration.

- **Cascade instead of a single PD term.** The loop is split into an explicit
  **P** stage (position error → velocity command) and a **PI** stage (velocity
  error → acceleration command), rather than one PD term directly on position
  error. The gains are derived algebraically from an equivalent single-stage PD
  design (`kp = ωₙ²`, `kd = 2ζωₙ`) via the identity `kp_pos = ωₙ/(2ζ)`,
  `kp_vel = 2ζωₙ` — so in the *linear, unsaturated* regime this cascade behaves
  identically to the original PD. The benefit of splitting it is that it exposes
  an explicit **velocity command**, which is:
  - **saturable on its own** (`V_CMD_MAX_XY`), giving a clean, interpretable speed
    limit independent of how large the position error is, and
  - a natural place to add an **integral term** (`KI_VEL_XY`).
- **Why an integral term at all.** A constant-ish disturbance (steady wind/thrust
  bias) produces a standing error under pure P/PD control, because proportional
  gain only produces force in proportion to *existing* error. An integral on the
  velocity-error signal accumulates and cancels a *sustained* bias, driving that
  standing offset to zero rather than only shrinking it.
- **Feed-forward from the KF's acceleration estimate** (`a_ref`) lets the drone
  start matching the rover's acceleration immediately rather than waiting for a
  velocity error to build up — important against a rover that speeds up or turns.

---

## 6. Vertical Channel: Altitude → Velocity → Acceleration Cascade

Same P→PI cascade structure as §5, applied to altitude, with two special pieces:

- **Descent speed profile** `v_down(h) = min(V_TOUCH + K_DESC·h, V_DESC_MAX)`:
  descent speed grows with height above the contact plane and saturates at a
  maximum, giving a smooth, fast-then-gentle approach that slows automatically
  as the vehicle nears the platform (`h → 0 ⇒ v → V_TOUCH`), instead of a fixed
  descent rate that would either be too slow high up or too fast at contact.
- **Landing-cone gating (`GATE_LO`–`GATE_HI` on `e_xy`)**: the commanded descent
  speed is faded toward zero as horizontal tracking error grows. This stops the
  drone from plunging downward while still laterally misaligned — descent only
  proceeds at full speed once the vehicle is well-centered over the platform.

---

## 7. From Acceleration Command to Desired Attitude

**What it does:** Turns the desired horizontal + vertical acceleration into a
thrust magnitude and a desired roll/pitch.

- **Two independent saturation limits are combined**, `min(A_MAX_XY, (g + a_z)·tan(tilt_max))`:
  a hard acceleration cap, and a tilt-derived cap that shrinks automatically if
  vertical acceleration demand is large (less "thrust budget" left over for
  tilt). This keeps the *physically achievable* combination of vertical and
  horizontal acceleration consistent, rather than treating the two axes as
  independent.
- **Reduced tilt limit near the ground** (`TILT_MAX_LAND` vs `TILT_MAX`): large
  bank angles close to the platform risk a rotor strike or a large lateral
  excursion with little altitude margin to recover, so tilt authority is
  deliberately reduced during the final approach.
- **Desired roll/pitch from the full acceleration vector via `atan2`**, in the
  yaw-aligned frame, rather than a small-angle linearization — this keeps
  the desired-attitude computation exact even at the edge of the tilt envelope,
  where a small-angle approximation would be least accurate.
- **Hard clamp on `roll_des`/`pitch_des`** is kept as a second, independent line
  of defense on top of the acceleration-based limit above — since the
  acceleration-based limit assumes thrust stays close to `mass·g`, which the
  integral/feed-forward terms can violate slightly under gusts; the direct
  clamp guarantees the attitude command itself is never unsafe.

---

## 8. Inner Loop: Attitude → Rate Cascade

Mirrors the outer loop's cascade philosophy (§5), now for roll and pitch:

- **P stage:** attitude error → body-rate command, saturated at `RATE_MAX`.
- **PI stage:** rate error → angular-acceleration command, with an integral term
  (`KI_RATE`) that rejects a sustained torque bias — e.g. a persistent wind
  moment or a slight mass-asymmetry — the same way `KI_VEL_XY`/`KIZ` reject
  translational bias.
- **No yaw control.** The mixer geometry (see §9) produces `tau_z = 0` for any
  motor allocation, since the arena only applies vertical point forces at each
  motor — physically, this airframe simply has no yaw authority, so no yaw loop
  is implemented; yaw is left to drift/stay constant rather than being
  (uselessly) commanded.

---

## 9. Mixer

**What it does:** Converts total thrust `T` and desired roll/pitch torques
(`tau_x`, `tau_y`) into four individual motor forces, respecting the motor
layout:

```
f1 (+L,+L)   f2 (-L,+L)
f3 (+L,-L)   f4 (-L,-L)
tau_x =  L (f1 + f2 − f3 − f4)
tau_y = -L (f1 − f2 + f3 − f4)
```

- **Differential allocation (`dx`, `dy`) added on top of an equal base thrust**
  (`T/4`) directly implements this layout: each motor gets `base ± dx ± dy`
  according to its position, so the four forces jointly produce the requested
  thrust *and* torque without needing an explicit matrix inversion.
- **Base thrust re-centered before clamping** (`base = clip(base, -d.min(), F_MAX - d.max())`
  when feasible): rather than clamping each motor force to `[0, F_MAX]`
  independently (which would distort the torque ratio between motors), the
  *shared* base level is shifted first so that, if possible, all four motors
  fit within limits while preserving the requested torque split exactly. Only
  if that's not enough does the final `np.clip` clamp individual motors (at
  which point a slight torque/thrust distortion is unavoidable — this is the
  physical actuator limit, not a modeling shortcut).

---

## 10. Anti-Windup Strategy (applied uniformly at every integral)

Every PI stage (`vel_int_xy`, `vel_int_z`, `rate_int`) uses the **same two-layer**
anti-windup design:

1. **Conditional integration:** the integral is only *committed* on a control
   tick where the loop *downstream* of it did **not** have to saturate (tilt
   limit, free-fall floor, or motor/mixer limit). The new integral value is
   always computed as a *trial* first; it's discarded (not applied) on a
   saturated tick.
2. **Hard clamp:** independently of (1), each integral state is also clamped to
   a fixed range (`I_VEL_XY_MAX`, `I_Z_MAX`, `I_RATE_MAX`) as a second line of
   defense.

**Why both:** conditional integration prevents the classic windup failure mode
— an integrator accumulating error while its output is being clipped anyway,
then producing a large overshoot once the saturation clears. The hard clamp is
a cheap, independent safety net that limits worst-case integral authority even
if some saturation condition is checked slightly too late or missed (e.g., due
to the *combined* nonlinear effect of multiple saturations interacting). Using
both is a standard "belt and suspenders" defense for control loops that must
run unattended in a real-time loop.

---

## 11. Safety Backstops (independent of the above pipeline)

- **`Z_CUT` motor cutoff**, checked at the very top of `calculate_forces`,
  before vision/estimation/control run — guarantees motors are cut the instant
  the vehicle is confirmed low enough to be on/at the platform, regardless of
  what state the rest of the pipeline is in.
- **Free-fall floor** (`a_z ≥ -0.8·g`) prevents the vertical loop from ever
  commanding an acceleration close to unpowered free-fall, which would leave
  no margin for the mixer to recover thrust in time.
- **Global thrust clamp** (`T ∈ [0.1·mg, 4·F_MAX]`) and **per-motor clamp**
  (`f ∈ [0, F_MAX]`) enforce actuator physics as the final stage no matter
  what the upstream loops requested.

---

## 12. Summary of Design Principles

| Principle | Where applied |
|---|---|
| Prefer exact geometry over small-angle approximations | Ray-casting (§2), attitude command via `atan2` (§7) |
| Cascade control for interpretable saturation + easy PI insertion | Horizontal (§5), vertical (§6), attitude (§8) |
| Integral action *only* for rejecting sustained disturbances, never unconditionally | All three PI stages, with two-layer anti-windup (§10) |
| Defense in depth: independent, redundant safety limits | Attitude clamp after accel-based tilt limit (§7); hard clamp *and* conditional integration (§10); state-machine BLIND_DESCEND gate *and* hard Z_CUT (§4, §11) |
| Only implement authority the airframe physically has | No yaw loop, since `tau_z = 0` by construction (§8) |
| Estimator hedges against stale information | Innovation gating and acceleration damping during dropouts (§3) |

This layered structure — exact geometry, a maneuvering-target estimator, a
gated state machine, cascaded PI control with disciplined anti-windup, and a
torque-preserving mixer with redundant safety limits — is what lets the
controller track and land on a moving, sometimes briefly-invisible target
robustly at 240 Hz.