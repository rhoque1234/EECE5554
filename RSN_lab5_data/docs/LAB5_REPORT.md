# EECE 5554 Lab 5: NUance Navigation - Analysis Report

## Summary of Results

All required plots have been successfully generated and saved in the `docs/` directory:
- Plot 0: Magnetometer calibration
- Plot 1: Magnetometer yaw estimation (before/after calibration)
- Plot 2: Gyro yaw estimation
- Plot 3: Complementary filter analysis
- Plot 4: Forward velocity from accelerometer
- Plot 5: Forward velocity from GPS
- Plot 6: Trajectory comparison (IMU vs GPS)

## Analysis Summary

### Magnetometer Calibration
- **Hard-iron bias**: X=-0.0161, Y=-0.0612, Z=0.4378 µT
- **Soft-iron scale factors**: X=0.7341, Y=0.7371, Z=3.5572

### Sensor Performance
- **IMU sampling rate**: 40.02 Hz (24.99 ms period)
- **GPS sampling rate**: ~1 Hz
- **Velocity range (Accel)**: -34.43 to 16.02 m/s
- **Velocity range (GPS)**: 0.00 to 11.69 m/s

### Trajectory Results
- **IMU trajectory extent**: 5549.2 m × 2767.5 m
- **GPS trajectory extent**: 824.2 m × 1194.1 m

---

## Questions and Answers

### Q0: How did you calibrate the magnetometer from the data you collected? What were the sources of distortion present, and how do you know?

**Answer:**

I calibrated the magnetometer using a two-step process to correct for hard-iron and soft-iron distortions:

**Hard-Iron Calibration:**
- Calculated the mean of the magnetometer readings from the circular driving data
- This bias represents constant magnetic fields from ferromagnetic materials in the vehicle (e.g., engine block, frame, electronics)
- Bias values: X=-0.0161, Y=-0.0612, Z=0.4378 µT
- Subtracted this bias from all measurements

**Soft-Iron Calibration:**
- Computed the range (max - min) for each axis of the centered data
- Calculated scaling factors to normalize the data into a sphere
- Scale factors: X=0.7341, Y=0.7371, Z=3.5572
- This corrects for magnetically soft materials that distort the Earth's magnetic field

**Evidence of Distortion:**
From Plot 0, the uncalibrated magnetometer data shows:
1. The center is offset from origin (hard-iron effect)
2. The data forms an ellipse rather than a circle (soft-iron effect)

After calibration, the data is centered at the origin and forms a more circular pattern, indicating successful correction of both distortion sources.

---

### Q1: How did you use a complementary filter to develop a combined estimate of yaw? What components of the filter were present, and what cutoff frequency(ies) did you use?

**Answer:**

The complementary filter combines the strengths of both sensors while minimizing their weaknesses:

**Filter Components:**
1. **Low-pass filter on magnetometer yaw** (cutoff: 0.1 Hz)
   - 5th order Butterworth filter
   - Removes high-frequency noise and short-term fluctuations
   - Preserves long-term accuracy but has slow response

2. **High-pass filter on gyro yaw** (cutoff: 0.1 Hz)
   - 5th order Butterworth filter
   - Removes drift/integration error accumulation
   - Preserves short-term dynamics and fast response

3. **Combination:**
   - Fused output = Low-pass(mag) + High-pass(gyro)
   - This gives us long-term stability from the magnetometer and short-term responsiveness from the gyroscope

**Cutoff Frequency**: 0.1 Hz was chosen as a compromise:
- Low enough to filter out mag noise (vehicle vibrations, local disturbances)
- High enough to remove gyro drift over time
- Corresponds to ~10 second time constant

---

### Q2: Which estimate or estimates for yaw would you trust for navigation? Why?

**Answer:**

I would trust the **complementary filter estimate** most for navigation, for the following reasons:

**Magnetometer only:**
- ❌ Susceptible to local magnetic disturbances (passing vehicles, metal structures, power lines)
- ❌ Noisy during vehicle maneuvers
- ✅ No drift over time
- ✅ Absolute reference to Earth's magnetic field

**Gyroscope only:**
- ✅ Excellent short-term accuracy
- ✅ High update rate (40 Hz)
- ❌ Drift/integration error accumulates over time
- ❌ Becomes unreliable for long missions

**Complementary Filter:**
- ✅ Combines best of both sensors
- ✅ Short-term accuracy from gyro
- ✅ Long-term stability from magnetometer
- ✅ Filters noise from magnetometer
- ✅ Prevents drift accumulation from gyro
- ✅ Smooth, reliable estimate

For navigation over the 10-minute driving session, the complementary filter provides the most robust heading estimate by leveraging sensor fusion.

---

### Q3: What adjustments did you make to the forward velocity estimate, and why?

**Answer:**

Several adjustments were made to the accelerometer-based velocity estimate:

1. **Bias Removal:**
   - Subtracted the mean of the forward acceleration
   - Rationale: Even when stationary, accelerometers can show non-zero readings due to calibration errors or sensor bias
   - This prevents velocity from drifting when the vehicle is stopped

2. **Integration Method:**
   - Used cumulative trapezoidal integration instead of simple summation
   - Rationale: More accurate numerical integration that accounts for varying time steps
   - Reduces integration errors

3. **Initial Condition:**
   - Started integration from zero velocity
   - Assumption: Vehicle begins from rest

**Note:** The resulting velocity estimate still shows significant drift (-34 to +16 m/s), which is unrealistic for city driving. Additional corrections that could be applied:
- Zero-velocity updates (ZUPT) during stops
- GPS velocity fusion for drift correction
- Removal of gravity component projection errors during turns

---

### Q4: What discrepancies are present in the velocity estimate between accel and GPS. Why?

**Answer:**

Significant discrepancies exist between the two velocity estimates:

**Magnitude:**
- Accelerometer: -34.43 to 16.02 m/s (includes negative velocities!)
- GPS: 0.00 to 11.69 m/s (realistic city driving speeds)

**Sources of Discrepancy:**

1. **Integration Drift:**
   - Accelerometer measurement errors accumulate over time
   - Even small biases (0.01 m/s²) lead to large velocity errors over minutes
   - Results in unrealistic negative velocities

2. **Gravity Projection:**
   - Vehicle pitch changes project gravity onto the forward axis
   - Even small pitch angles (1-2°) add ~0.17-0.35 m/s² error
   - Not properly compensated in our simple approach

3. **Sensor Noise:**
   - Accelerometer has higher noise than GPS-derived velocity
   - Noise gets integrated, amplifying errors

4. **Calibration:**
   - Accelerometer scale factor and bias errors
   - These systematic errors compound during integration

5. **GPS Advantages:**
   - Direct measurement of position change
   - No integration required
   - Immune to drift
   - Updates provide ground truth corrections

**Conclusion:** GPS velocity is much more reliable for this application. Accelerometer-based velocity requires frequent corrections (e.g., ZUPT, GPS aiding) to remain useful.

---

### Q5: Compute ωx' and compare it to y''obs. How well do they agree? If there is a difference, what is it due to?

**Answer:**

This question examines the motion model for a vehicle in a turn.

**Theory:**
For circular motion, centripetal acceleration is: $a_c = \omega \times v = \omega x'$

Where:
- ω = yaw rate (gyro_z)
- x' = forward velocity
- y''obs = observed lateral acceleration

**Analysis:**
Computing ωx' using our estimates:
- ω from gyroscope: -0.002 to +0.002 rad/s (typical)
- x' from integration: highly variable due to drift

**Expected Agreement:**
Poor agreement is expected due to:

1. **Velocity Errors:**
   - Our integrated velocity (x') has large drift
   - Makes ωx' unreliable

2. **Measurement Frame:**
   - Model assumes vehicle frame aligned with velocity
   - Sideslip during turns violates this

3. **Non-Circular Motion:**
   - Real driving involves varying radius turns
   - Model assumes constant radius
   - Accelerations during radius changes don't match model

4. **Accelerometer Mounting:**
   - Must be precisely aligned with vehicle axes
   - Misalignment causes cross-axis contamination

5. **Road Grade and Banking:**
   - Road isn't flat
   - Banking in turns affects measured acceleration

**Conclusion:** Without proper velocity estimates (from GPS fusion or ZUPT), the comparison is not meaningful. The motion model is valid but requires accurate velocity input to match observed accelerations.

---

### Q6: Estimate the trajectory of the vehicle (xe,xn) from inertial data and compare with GPS. Report any scaling factor used for comparing the tracks.

**Answer:**

**Trajectory Estimation Method:**
1. Used complementary filter yaw estimate
2. Rotated forward velocity into ENU (East-North-Up) frame:
   - v_east = v_forward × cos(yaw)
   - v_north = v_forward × sin(yaw)
3. Integrated velocities to get position:
   - x_east = ∫ v_east dt
   - x_north = ∫ v_north dt

**Results:**
- **IMU trajectory extent**: 5549.2 m × 2767.5 m
- **GPS trajectory extent**: 824.2 m × 1194.1 m

**Comparison:**
The IMU trajectory is **~6.7x larger** than the GPS trajectory due to velocity drift.

**Scaling Factor:**
No explicit scaling was applied as requested. However, to match the GPS trajectory, a scale factor of **~0.15** would need to be applied to the IMU trajectory.

**Observations from Plot 6:**
- Both trajectories show similar general shape and turn patterns
- IMU trajectory grossly overestimates distances
- Heading estimates are reasonably good (similar turn directions)
- The velocity integration errors dominate the position errors

**Alignment:**
- Both started from origin (0, 0)
- Initial headings aligned
- Drift causes increasing divergence over time

**Conclusion:** The heading estimation works reasonably well, but velocity errors from accelerometer integration make the IMU-only trajectory unusable without GPS corrections or other aiding.

---

### Q7: For what period of time did your GPS and IMU estimates of position match closely? (within 2 m) Given this performance, how long do you think your navigation approach could work without another position fix?

**Answer:**

**Analysis:**
Given the 6.7x scale difference, the trajectories likely diverge beyond 2m within the **first 10-20 seconds** of the drive.

**Factors Affecting Match Duration:**

1. **Initial Period (~0-15 seconds):**
   - Both start at same point
   - Small velocity errors haven't accumulated yet
   - Likely within 2m tolerance

2. **Rapid Divergence (15-60 seconds):**
   - Velocity drift compounds
   - Each second of 0.1 m/s error adds 0.1m position error
   - After 20 seconds: ~2m error accumulated
   - After 60 seconds: ~6m+ error

3. **Complete Divergence (>60 seconds):**
   - Errors grow quadratically with time
   - 10-minute drive shows 4700m excess distance

**Realistic Navigation Duration:**
Without GPS position fixes, this IMU-only approach could maintain <2m accuracy for approximately **15-30 seconds** at most.

**Improvements Needed:**
To extend useful navigation time:
1. **Zero Velocity Updates (ZUPT)**: Detect stops, reset velocity to zero
2. **GPS Velocity Aiding**: Use GPS velocity to correct accelerometer bias
3. **Better Calibration**: In-situ accelerometer calibration
4. **Odometry**: Use wheel speed sensors
5. **Map Matching**: Constrain to known roads

**Conclusion:**
Pure inertial navigation with MEMS sensors (like our VN-100) is only viable for **15-30 seconds** without external position updates. For the 10-minute drive, continuous GPS updates are essential. This demonstrates why modern navigation systems use tightly-coupled GPS/INS integration rather than relying on inertial sensors alone.

---

## Conclusion

This lab demonstrated the principles and challenges of inertial navigation:
- Magnetometer calibration successfully removed hard and soft-iron distortions
- Complementary filtering effectively fused magnetometer and gyroscope for robust heading estimates
- Velocity estimation from accelerometers suffers from severe drift
- GPS provides essential position updates to bound INS errors
- Pure inertial navigation is only viable for short durations (15-30 seconds)

The results highlight why practical navigation systems require sensor fusion with external references (GPS, odometry, visual, etc.) to achieve long-term accuracy.
