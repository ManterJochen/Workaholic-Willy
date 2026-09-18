# Combination evidence (`src/robot/safety/planning/robot/evidence`)

One committed file per measured combination of arm, hand, coupling, placement, planner margin and attach
slots. A planner starts only on a combination a file here measured: both drivers read it where a planner
starts, and the real cell checklist reads it at the desk. You do not call these files; you add one when
your cell's combination has none.

The file name is the combination, so nothing maps one to the other:

```
{arm}_{hand}_c{coupling}mm_{approach}{closing}_m{margin}mm_a{attach}.json
ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json
ur5e_robotiq_2f85_c0mm_+Z+X_m4mm_a0.json
```

## The two placements

`+Y+X` is the Isaac cell's frame, where the UR asset mounts its hand along flange +Y. `+Z+X` is a real UR
flange, the UR tool axis, which every real cell declares. Each placement has its own retract, because the
same joints hold the hand somewhere else, and so its own matrix: the ur5 with the EGU-50 has a file at
+Z and none at +Y. The ur16e has no file at either, because the schema does not admit a ur16e cell.

## Adding a combination

A placement or a hand that is not here is admitted by its own file and nothing else. Choose a retract,
then measure and write the file, spelling the tool rotation as four numbers in both commands:

```bash
python scripts/curobo/choose_ur_retract.py <arm> --tool-rotation-xyzw x y z w
python scripts/curobo/matrix_gate.py ... --tool-rotation-xyzw x y z w --write
```

A second run over an unchanged tree writes the same bytes, and no file holds a timestamp, so a diff here
is a real change in what was measured. A customer's hand gets its file the same way, as the evidence
step of [your_own_gripper.md](../../../../../../docs/runbooks/your_own_gripper.md).

## What a file binds

The hashes, not the names. `composed_sha256` covers the whole config the planner loaded, so the hand's
link, the coupling bodies and the margin are inside it. `guard_sha256` covers the placed arrays the
exact-mesh guard judges, so a turned hand, a plate and a coupling body all move it. A cell whose hashes
differ from the file's is refused by name rather than admitted on a matching file name, and a hash the
sidecar did not report is refused too: an unreported hash is not a hash that matched.

## Why a file, and not a list

Which UR arm can plan with which gripper is a measurement. A list of admitted arms written into each hand
bundle would answer for pairings nobody measured. The retract table knows one pose per pair, which is
enough to start a sidecar and says nothing about the rest of the space, so it is not read as an
admission.

## What the `b1` criterion asks

The planner accepts the pose its own sidecar starts from, it never calls a pose a collision without
naming the pair, and the gate's own control comes out at zero. It does not ask the sphere model and the
exact meshes to agree everywhere: the fitted spheres reach past the body by construction, so no fit can
meet that bar. The count of those disagreements is recorded anyway, as the price of the fit.

## Details

- [`../../evidence.py`](../../evidence.py) reads and checks these files
- Tests: `tests/test_combination_evidence.py`, `tests/test_committed_evidence.py`
