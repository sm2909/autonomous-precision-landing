PITCH
f1 = hover_force + delta
f2 = hover_force + delta
f3 = hover_force - delta
f4 = hover_force - delta

error in x increases in + direction

---

ROLL

f1 = hover_force + delta
f2 = hover_force - delta
f3 = hover_force + delta
f4 = hover_force - delta

error in y increases in + direction


## Robustness to wind:
Position → Velocity → Acceleration hierarchy
Rate controller
Integral action
Anti-windup
Tilt limits