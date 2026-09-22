## Rover-Following Drone Controller: Mental Model

### 1. High-Level Architecture

```text
Camera
  ↓
State Estimation
  ↓
Position / Velocity Error
  ↓
Position Controller (PD)
  ↓
Desired Acceleration
  ↓
Drone Physics
  ↓
Desired Roll / Pitch
  ↓
Attitude Controller (PD)
  ↓
Desired Torques
  ↓
Motor Mixer
  ↓
Motor Forces
```

---

# 2. Core Physical Insight

A drone cannot generate horizontal force directly.

It only generates thrust along its body z-axis.

```text
Need horizontal acceleration
        ↓
Tilt drone
        ↓
Tilt thrust vector
        ↓
Horizontal force appears
```

Thus:

```text
Position Error
      ↓
Desired Acceleration
      ↓
Desired Tilt
      ↓
Motor Forces
```

is the central idea behind the entire controller.

---

# 3. State Estimation Stack

### Camera gives:

```text
ArUco pixel coordinates
```

### Ray Casting gives:

```text
World position of rover
```

using:

* Drone position
* Drone orientation (roll/pitch/yaw)
* Camera FOV
* Marker pixel coordinates

Tilt compensation is necessary because:

```text
Marker at image center
≠
Marker directly below drone
```

when drone is tilted.

---

### Kalman Filter gives:

State:

$$
[x,\; v_x,\; a_x]
$$

$$
[y,\; v_y,\; a_y]
$$

Predict:

```text
Where should rover be according to physics?
```

Update:

```text
Where does camera say rover is?
```

Output:

```text
Rover Position
Rover Velocity
Rover Acceleration
```

---

# 4. Position Controller (Outer Loop)

Compute:

$$
e_p = p_{rover}-p_{drone}
$$

$$
e_v = v_{rover}-v_{drone}
$$

PD law:

$$
a_{des}
=
K_p e_p
+
K_d e_v
$$

Interpretation:

### P term

```text
How far am I from rover?
```

### D term

```text
Am I approaching too fast?
```

Output:

```text
Desired acceleration
```

not pitch or motor forces.

---

# 5. Acceleration → Desired Tilt

Using drone physics:

$$
F = ma
$$

Need horizontal acceleration:

$$
a_x
$$

Need horizontal force:

$$
F_x = ma_x
$$

Need thrust vector tilted:

$$
\tan(\theta)
\approx
\frac{a_x}{g}
$$

Thus:

```text
Desired Acceleration
      ↓
Desired Roll/Pitch
```

---

# 6. Attitude Controller (Inner Loop)

Inputs:

```text
Desired Roll
Desired Pitch
```

Telemetry provides:

```text
Actual Roll
Actual Pitch
Angular Velocity
```

Compute:

$$
e_\theta
=
\theta_{des}
-
\theta
$$

PD law:

$$
\tau
=
K_p e_\theta
+
K_d e_{\dot\theta}
$$

Output:

```text
Desired Torques
```

---

# 7. Motor Mixer

Convert:

```text
Total Thrust
Roll Torque
Pitch Torque
```

into:

```text
f1 f2 f3 f4
```

which are sent to the motors.

---

# 8. Controller Hierarchy

```text
Position Error
      ↓
Position PD
      ↓
Desired Acceleration

Desired Acceleration
      ↓
Physics
      ↓
Desired Roll/Pitch

Attitude Error
      ↓
Attitude PD
      ↓
Desired Torque

Desired Torque
      ↓
Mixer
      ↓
Motor Forces
```

This is called a **cascaded controller**.

---

# 9. Landing / Descent Logic

### TRACK

```text
Track rover in x,y
Maintain constant altitude
```

---

### DESCEND

```text
Track rover in x,y
Reduce altitude
```

Only the z-reference changes.

---

### BLIND_DESCEND

Marker lost.

Use Kalman Filter prediction:

```text
Last Position
Last Velocity
Last Acceleration
```

to continue estimating rover motion.

Continue same controller.

---

# 10. Descent Velocity Profiling

Bad:

```text
Always descend at -0.4 m/s
```

Good:

$$
v_z = -k h
$$

where

$$
h = altitude
$$

Example:

```text
5m → -0.5 m/s
1m → -0.5 m/s
0.5m → -0.25 m/s
0.1m → -0.05 m/s
```

Why?

```text
Far away:
Prioritize descending

Near touchdown:
Prioritize accurate x,y tracking
```

More time to eliminate final landing error.

---

# One-Line Summary

```text
Estimate rover position/velocity from camera,
compute desired acceleration using PD control,
convert acceleration into desired attitude,
use another PD controller to achieve that attitude,
and mix thrust/torques into motor forces while descending safely onto the rover.
```
