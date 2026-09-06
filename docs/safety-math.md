# The math behind `src/robot/safety`

[Workaholic-Willy](../README.md) | sibling: [`grasping-math.md`](grasping-math.md) | package:
[`safety/`](../src/robot/safety/README.md)

The safety layer is small but math dense: closest-distance geometry, forward kinematics, and a
singular-value view of manipulability all live on the per-move hot path. This page explains that
math in one place, so the guard code reads as "apply the formula" rather than "re-derive it".
Everything here is closed form and deterministic, with no iterative solvers on the motion path
except one bounded scalar search in section 3.3.

The guard chain itself, its order and its reasons, is in
[`safety/`](../src/robot/safety/README.md). This page is only the geometry underneath it.

**Conventions used throughout.** Positions are millimetres, orientations are XYZW unit quaternions,
every pose is tagged with a frame, and rotations are right handed. Angles are radians internally and
formatted to degrees at the operator boundary, because pendants read degrees.

---

## 1. Workspace box and margin

The workspace guard is a Cartesian axis-aligned box test. A pose `p = (x, y, z)` is inside if and
only if

```
x_min <= x <= x_max  and  y_min <= y <= y_max  and  z_min <= z <= z_max
```

Before the guard runs, the box is shrunk by a safety margin `m` on every face:

```
x_min' = x_min + m,   x_max' = x_max - m      and likewise for y and z
```

A margin that would invert an axis, meaning `x_min' >= x_max'`, is a configuration error and raises.
This gives a keep-out shell around the true reachable box without re-authoring the box itself.

Code: [`workspace.py`](../src/robot/safety/workspace.py) and
[`preflight.py`](../src/robot/safety/preflight.py), `_shrink_workspace`

---

## 2. Quaternion geodesic distance

Both the continuity guard, on the orientation step, and the workspace diversity check need to know
how far apart two orientations are. For unit quaternions `q1` and `q2` the geodesic angle on the
unit sphere is

```
theta = 2 * arccos(abs(dot(q1, q2)))
```

The absolute value of the scalar product handles double cover, since `q` and `-q` are the same
rotation, so `theta` is the true minimal rotation angle in `[0, pi]`. This is what `Pose.angle_to`
returns; the guards compare `theta`, in degrees, against a configured cap.

---

## 3. Capsule collision geometry

The `capsule` self-collision backend approximates each arm link, the tool and the base column as a
capsule, meaning a swept sphere over a line segment: a segment `(p0, p1)` plus a radius `r`.
Fixtures are axis-aligned boxes. All distances are signed, and negative means penetration.

This backend is the fallback rather than the default. `self_collision.backend` ships as `fcl`, the
exact-mesh path of section 6; the capsule path is what runs when no collision engine or mesh bundle
resolves. It is materially coarser: its default link radius is 60 mm, which over-rejects legitimate
reach-down grasps, and it skips link pairs the exact backend checks.

### 3.1 Point to segment

Closest distance from a point `q` to a segment `(p0, p1)` projects `q` onto the segment and clamps
the parameter to `[0, 1]`:

```
d = p1 - p0
t = clamp(dot(q - p0, d) / dot(d, d), 0, 1)
closest = p0 + t * d
dist = norm(q - closest)
```

### 3.2 Segment to segment

The closest distance between two segments is the classic result from Ericson, Real-Time Collision
Detection, section 5.1.3. Writing `d1 = p1 - p0`, `d2 = q1 - q0`, `r = p0 - q0`, and

```
a = dot(d1, d1),  e = dot(d2, d2),  f = dot(d2, r),  b = dot(d1, d2),  c = dot(d1, r)
```

the interior solution for the two parameters `(s, t)` is

```
s = (b * f - c * e) / (a * e - b * b)      t = (b * s + f) / e
```

with the degenerate cases handled explicitly (either segment a point, and `a * e - b * b = 0` for
parallel segments), and both `s` and `t` clamped to `[0, 1]` with a re-solve so the closest points
stay on the segments. The distance is `norm((p0 + s * d1) - (q0 + t * d2))`.

### 3.3 Capsule against capsule, and capsule against box

A capsule is its segment inflated by `r`, so signed distances subtract the radii:

```
capsule vs capsule:  d = segment_segment(...) - (r_a + r_b)
capsule vs box:      d = segment_box(...) - r
```

The segment-to-box distance is a golden-section search rather than a sample, and the reason is
one-sided error. A minimum taken over samples can only overestimate the true minimum, so a sampled
implementation reports the capsule as further from the box than it is whenever it is wrong, which
is structural and not bad luck. A guard that overestimates distance passes a penetrating
configuration.

The search is exact because of convexity: distance to a convex set is a convex function of the
point, and a segment is affine in its parameter, so `t -> dist(P(t), box)` is convex on `[0, 1]`.
Golden section on a convex function converges to the global minimum with no Voronoi-region case
analysis and no local minima to settle in. Fifty iterations shrink the bracket by a factor of about
`0.618^50`, roughly `1e-11` of its width, which on a 500 mm link is on the order of `1e-8` mm, far
below input precision. Each evaluation clamps the offset to the box half extents axis by axis
(`nearest = center + clamp(q - center, +/- half)`).

One imprecision remains deliberately. With the segment inside the box the clamp yields zero, so the
result is `-radius_mm` rather than the true penetration depth. That is safe for a guard, since it is
already a rejection and it errs toward rejecting harder, but it is not a number to read as a depth.

Code: [`_capsule.py`](../src/robot/safety/_capsule.py)

---

## 4. Forward kinematics, Denavit-Hartenberg

To place the arm-link capsules, and the exact meshes, the guard needs each link frame in the base
frame. The bundled tables are the published Universal Robots DH parameters, one six-row table per
model for `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10`, `ur10e` and `ur16e`. Each joint contributes a
standard 4x4 DH transform for joint angle `theta` and row `(a, d, alpha)`:

```
         [ cos(th)  -sin(th)cos(al)   sin(th)sin(al)   a*cos(th) ]
T(th) =  [ sin(th)   cos(th)cos(al)  -cos(th)sin(al)   a*sin(th) ]
         [    0          sin(al)         cos(al)           d     ]
         [    0             0               0              1     ]
```

The DH theta offsets are zero for UR, because the controller's `q` is already the DH angle.
Chaining from the base gives every joint-frame origin, for the capsule end points, and every full
link frame, for mesh placement:

```
T_0 = I,   T_i = T_(i-1) * T(theta_i, row_i)
```

Lengths are metres internally and scaled to millimetres at the boundary, so a millimetre-space mesh
vertex maps straight into the millimetre base frame.

### Base-frame yaw reconcile

The bundled DH base frame can be rotated about `+Z` relative to the system base frame that poses and
fixtures live in. A single planar rotation about `+Z` reconciles them:

```
                                     [ cos(psi)  -sin(psi)   0 ]
R_z(psi) * origin,   with R_z(psi) = [ sin(psi)   cos(psi)   0 ]
                                     [    0          0       1 ]
```

Because this is a pure base-frame rotation it preserves every inter-link distance, so arm-against-arm
distances are invariant to it. Only the arm's placement relative to the base, tool and fixture
capsules moves. The key is `safety.self_collision.kinematics_base_yaw_deg` and its default `0.0`
leaves every existing cell unchanged; the Isaac cell ships `180.0` in
[`config/robot/robot.sim.yaml`](../config/robot/robot.sim.yaml), because the official UR DH base is
rotated half a turn about Z from that USD's `base_link`.

Code: [`_ur_kinematics.py`](../src/robot/safety/_ur_kinematics.py)

---

## 5. Singularity analysis, geometric Jacobian and SVD

A configuration is near singular when a small joint change produces almost no Cartesian motion in
some direction, so the arm loses a degree of freedom. The IK-quality guard measures this from the
geometric Jacobian `J`, a 6-by-n matrix estimated by central differences on the robot's own forward
kinematics, so no analytic Jacobian is required and the method stays vendor neutral:

```
J[:, i] = (twist(fk(q + delta * e_i)) - twist(fk(q - delta * e_i))) / (2 * delta)
```

where the 6-vector twist stacks the linear part, `delta_position / (2 * delta)` in metres, and the
angular part, the axis-angle of the relative rotation of the two probes divided by `2 * delta`. The
step `delta` defaults to `1e-4` radians. The singular values `sigma_1 >= ... >= sigma_k` from
`J = U * Sigma * transpose(V)` then classify risk on three independent tests:

| Quantity | Definition | Default | Fails when |
| --- | --- | --- | --- |
| rank | count of `sigma_i > rank_tol` | `rank_tol = 1e-6` | below `min(n, 6)`, a degree of freedom has collapsed |
| minimum singular value | `sigma_k` | `min_singular_value = 0.005` | below the threshold, a weak direction |
| condition number | `sigma_1 / sigma_k`, infinite if `sigma_k` is at or below `rank_tol` | `max_condition_number = 250.0` | above the threshold, ill conditioned |

Any failing test flags `is_near_singularity`, and the guard rejects with the offending numbers
attached. Intuitively, a large condition number means the manipulability ellipsoid is a thin sliver:
the arm moves freely along its long axis and is nearly stuck along the short one.

Code: [`singularity.py`](../src/robot/safety/singularity.py), consumed by
[`ik_quality.py`](../src/robot/safety/ik_quality.py)

---

## 6. Exact mesh distance, the default backend

`self_collision.backend` ships as `fcl`, which replaces the capsule proxy with closest-distance
queries on the real collision meshes through a BVH engine. Coal is preferred and python-fcl is the
fallback; the swap is behaviour identical, because both run the same meshes, pairs, thresholds and
distance query.

Each link mesh is pre-baked into its DH link frame once. At every evaluation only the collision
object's transform is updated, to `R_base(psi) * T_dh[frame](q)`, with no BVH rebuild. Pair
selection skips only links within one DH frame of each other, which are the adjacent links and the
rigid wrist cluster that touch by construction; every farther pair is checked exactly, which covers
the wrist pairs the capsule proxy has to skip. Each link is also checked against the fixtures.

**Broadphase cull.** For the continuous monitor a conservative bounding sphere, the mesh centroid
plus its maximum vertex radius, precedes each exact query. A pair is skipped when

```
norm(c_i - c_j) - r_i - r_j  >  min_distance_mm
```

Since the spheres bound the meshes, a skipped pair provably cannot violate the margin, so the cull
never changes a verdict; it only avoids exact queries that cannot matter.

**When the engine or the bundle is missing, this backend falls back rather than refusing.** It logs
one warning naming the reason and then runs the capsule path of section 3. The fallback is logged
with a reason token precisely because the capsule proxy is materially coarser and a silent swap
would be a safety-relevant regression. Ask a given box what resolved on it with
`python -m src.robot.safety.planning --check`.

Code: [`_fcl_self_collision.py`](../src/robot/safety/_fcl_self_collision.py) and
[`continuous_monitor.py`](../src/robot/safety/continuous_monitor.py)

---

## 7. Fail-closed authority, the rule the math serves

The formulas above only ever reject motion; they never grant it. The pipeline runs the guards in a
fixed order and returns the first rejection. A guard that cannot decide, because telemetry or an
asset is missing, returns `UNAVAILABLE`, and while its family is enforced that also refuses the
move. It fails closed and never passes silently. Safety outranks everything downstream:
deterministic geometry, recovery, and the optional learned layer may only reorder or filter
candidates the guards have already cleared, never override a rejection.

The one place where a missing asset does not produce `UNAVAILABLE` is the exact-mesh backend of
section 6, which falls back to a coarser but still conservative geometry and says so in the log.

Code: [`preflight.py`](../src/robot/safety/preflight.py) and
[`decision.py`](../src/robot/safety/decision.py)

---

## Honesty

All of this is simulation-grade collision and manipulability reasoning. It reduces risk in
simulation and it catches commanded configurations a controller would refuse. It is not a certified
functional-safety stop. A real cell still needs the vendor safety-rated stop, meaning an independent
emergency stop under ISO 10218, ISO/TS 15066 and ISO 13849, which runs independently of any of this
software. Nothing in this package can enable, disable, route or observe that stop.
