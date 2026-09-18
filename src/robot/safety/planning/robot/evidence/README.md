# Combination evidence

One file per measured combination of arm, hand, coupling, placement, planner margin and attach slots. The file name
is the combination, so nothing maps one to the other:

```
{arm}_{hand}_c{coupling}mm_{approach}{closing}_m{margin}mm_a{attach}.json
ur5e_robotiq_2f85_c0mm_+Y+X_m4mm_a0.json
ur5e_robotiq_2f85_c0mm_+Z+X_m4mm_a0.json
```

Two placements are measured. `+Y+X` is the Isaac cell's frame, where the UR asset mounts its hand along flange +Y.
`+Z+X` is a real UR flange, the UR tool axis, which every real cell declares. Each placement has its own retract,
because the same joints hold the hand somewhere else, and so its own matrix: 23 files at +Y and 24 at +Z, where the
ur5 with the EGU-50 has a retract it does not have at +Y. The four ur16e combinations are measured at neither, because
the schema does not admit a ur16e cell. A placement that is not here, a quarter turn of the closing for example, is
admitted by its own file and nothing else: `scripts/curobo/choose_ur_retract.py <arm> --tool-rotation-xyzw x y z w`,
then `scripts/curobo/matrix_gate.py ... --tool-rotation-xyzw x y z w --write`, both spelling the rotation as four
numbers. A customer's hand gets its file the same way, as the evidence step of
[`docs/runbooks/your_own_gripper.md`](../../../../../../docs/runbooks/your_own_gripper.md).

## Why there is a file at all

Which UR arm can plan with which gripper is a measurement. A list of admitted arms baked into each hand bundle when
the bundles were cut answers for pairings nobody measured, and four of the pairs such a list admitted are pairs cuRobo
refuses to plan from. The retract table knows one pose per pair, which is enough to start a sidecar and says nothing
about the rest of the space. The hand bundles carry no such list, and the table is not read as one.

## What a file binds

The hashes, not the names. `composed_sha256` covers the whole config the planner loaded, so the hand's link, the
coupling bodies and the margin are inside it; `guard_sha256` covers the placed arrays the exact-mesh guard judges, so a
turned hand, a plate and a coupling body all move it. A cell whose numbers differ from the file's is refused by name
rather than admitted on a matching file name.

A hash the sidecar did not report is refused too. An unreported hash is not a hash that matched, and treating it as
one is what an inherited admission list does.

## Writing them

`scripts/curobo/matrix_gate.py` measures a combination and writes its file; the files are committed. A second run over
an unchanged tree writes the same bytes, so a diff here is a real change in what was measured. Nothing in the files is
a timestamp.

## What the first rung asks for

`b1`: the planner accepts the pose its own sidecar starts from, it never calls a pose a collision without being able
to name the pair, and the gate's own control comes out at zero. It does not ask the sphere model and the exact meshes
to agree everywhere: the fitted spheres reach past the body by construction, so that rung is one nothing can reach.
The count of those disagreements is recorded anyway, because it is the price of the fit.
