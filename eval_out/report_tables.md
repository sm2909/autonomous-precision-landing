## Section 10: closed-loop results (copy into the report)

Nominal: 20 runs, randomised start position (+-1 m), rover phase and wind seed (run 0 = exact arena defaults). Headless PyBullet, `tiny` camera renderer.

| Metric | Requirement | Result |
|---|---|---|
| Touch-down vertical speed at scorer contact check [m/s] | >= -0.65 | mean -0.28, worst -0.60 |
| Peak descent speed during DESCEND [m/s] | (info) | worst 1.55 |
| Chassis tilt at touch-down [deg] | <= 30 | mean 5.8, worst 17.7 |
| Horizontal offset from pad centre at contact [cm] | minimise (scorer gate 50 cm) | mean 16.7, worst 77.6 |
| Time to acquire marker (leave SEARCH) [s] | n/a | mean 0.26, worst 0.31 |
| Time to land (scorer success) [s] | n/a | mean 5.3, worst 6.7 |
| Success rate, nominal | 100 % | 19/20 = 95 % (95 % Wilson CI 76-99 %) |
| Failure reasons | | GROUND COLLISION: Missed the rover entirely. |

### Robustness scenarios

| Scenario | Description | Success | Worst vz [m/s] | Worst offset [cm] | Mean time to land [s] | BLIND_DESCEND used |
|---|---|---|---|---|---|---|
| nominal | arena defaults + randomised start / rover phase / wind | 19/20 | -0.60 | 77.6 | 5.3 | 100 % |
| far_start | drone starts 6-8 m away (marker not visible: SEARCH spiral) | 5/6 | -0.37 | 420.4 | 23.9 | 83 % |
| dropout_track | camera blacked out for 1.0 s, 2 s after TRACK begins | 5/6 | -0.63 | 76.4 | 7.2 | 100 % |
| blackout_low | camera permanently blacked out below 1.0 m (BLIND_DESCEND) | 5/6 | -1.23 | 128.2 | 5.2 | 100 % |

## Dependencies line (Appendix A)

Python 3.14.7, NumPy 2.5.3, OpenCV 5.0.0 (cv2.aruco OK), PyBullet installed (pybullet.getAPIVersion=202010061); tested on macOS-26.6.2-arm64-arm-64bit-Mach-O.
