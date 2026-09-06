# When Windows refuses to load a dependency

On Windows 11, Smart App Control and the WDAC family of policies refuse unsigned native code that
Microsoft's reputation service does not vouch for. This project's motion stack is exactly the kind of
software that trips it: Coal and cuRobo have no signed installers, they arrive as unsigned
conda-forge packages and locally compiled CUDA extensions, and a Python process loads them as DLLs.

The failure is silent and misattributed. It presents first as picks that do not work, then as
`ImportError: DLL load failed`, and neither says that an operating-system policy refused a file.

Run the doctor first. It names the file, the policy and the remedy, and exits `2` when a policy is
the cause. It exists so this page is rarely needed.

```bash
python -m src.robot.safety.planning --doctor   # 0 healthy | 1 degraded | 2 blocked by policy
```

## What the verdict depends on

The file's contents. Not its path, not its age on disk, not the environment it lives in. Two copies
of the same build are treated identically; two different builds of the same version are not, because
they are different bytes with different reputations. Re-downloading a refused file, copying it
somewhere else, or reinstalling changes nothing.

Two properties follow, and they are the whole point of this page.

**Reach, not recency, decides.** A build the world has downloaded is admitted; one published days ago
is refused until reputation accrues. conda-forge builds are far more exposed to this than PyPI
wheels, simply because fewer machines run them. Binaries signed by a reputable publisher, such as
Isaac Sim's own executables, are never evaluated on reputation at all.

**Verdicts change, in both directions.** A refusal can clear by itself within a day or two, and code
that has loaded for weeks can start being refused. Nothing about your install changed when it does.
The same is true within one package family: a newer wheel can be refused while the previous one
loads.

### The local cache

Admitted files carry an NTFS extended attribute named `$KERNEL.PURGE.NLLAV` holding the cached
verdict, and `fsutil file queryEA <path>` prints it. A file with no such attribute is judged live on
the next load. The cache is purgeable, which is why a previously working file can be re-judged. It is
a cache and not the authority, so do not try to manage it.

## What to do about it

**Pin the dependency to a build that loads.** That is the fix, and it is why `ext_deps/locks/*.lock`
are committed: they pin every conda package to an exact build by URL and hash.

```powershell
# the whole stack, from nothing, verified at the end
powershell -ExecutionPolicy Bypass -File scripts\ext_deps\install.ps1
```

When a new pin is refused, choose the previous build of that package and re-lock:

```powershell
ext_deps\micromamba\Library\bin\micromamba.exe install -p ext_deps\curobo_env `
    -c conda-forge "bzip2=1.0.8=h0ad9c76_9"
ext_deps\micromamba\Library\bin\micromamba.exe env export --explicit -p ext_deps\curobo_env `
    | Out-File ext_deps\locks\curobo_env.win-64.lock -Encoding utf8
python -m src.robot.safety.planning --doctor      # must exit 0
```

**Keep two independent backends where one exists.** cuRobo can run its CUDA kernels through a
just-in-time backend (`cuda.core`) or through compiled ones (`pybind`), and the install script builds
both on purpose. With a single backend installed, one policy verdict takes the planner down, and a
dead planner presents as picks that simply do not work. With both installed, a single verdict cannot
do that. The doctor reports `warn` when only one resolves, precisely so the gap is visible before it
matters, and it distinguishes a backend refused by policy from one that was never installed.

**Do not turn Smart App Control off.** It cannot be turned back on without reinstalling Windows, and
a customer's IT may enforce it regardless, so an install that only works with it disabled is not an
install. Nothing in this repository requires it to be off.

## Confirming it is really the cause

The doctor does this itself: `code_integrity_blocks` in `src/robot/safety/planning/doctor.py` queries
the event log and cross-references the refused filenames against the import error. To read the log by
hand:

```powershell
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-CodeIntegrity/Operational'; Id=3033,3077} -MaxEvents 50 |
  ForEach-Object { ([xml]$_.ToXml()).Event.EventData.Data |
    Where-Object { $_.Name -eq 'FileNameBuffer' } | ForEach-Object { $_.'#text' } } |
  Select-Object -Unique
```

Read the structured `FileNameBuffer` field rather than the rendered message: the message text is
localized and the field is not. Event `3033` means the file did not meet the required signing level
and `3077` means it was blocked by policy; both are emitted for the same refusal. The doctor matches
on this field for that reason, and treats its own list of localized message fragments as a fast path
only.

The current policy state lives in `HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy`, where
`VerifiedAndReputablePolicyState` is `0` for off, `1` for enforcing and `2` for evaluation.

## For a customer deployment

Say this plainly rather than discovering it on site.

- **Linux has none of this.** Production robot cells generally run Linux, where Coal and cuRobo
  install from conda-forge with no code-integrity layer in the way. That is the recommended target.
- **On Windows**, install from the committed lockfiles and let `--doctor` gate the result. An install
  that ends in exit `0` will keep working; one that ends in exit `2` names the package to re-pin.
- **The one durable fix this project does not have** is a signed distribution. Signing the shipped
  binaries with a reputable publisher certificate bypasses reputation entirely, which is why Isaac's
  own executables are never questioned. That is a commercial decision, an extended-validation
  certificate plus a signing step in the release pipeline, and not something an install script can
  do.

The mechanism is Microsoft's and undocumented in its specifics. Reputation is a cloud service whose
decisions can be observed but not predicted, so treat everything above as a description of observed
behaviour rather than as a specification.

## See also

- [`ext_deps/README.md`](../ext_deps/README.md), and its sections on
  [Coal](../ext_deps/README.md#coal-the-exact-mesh-self-collision-engine) and on
  [cuRobo](../ext_deps/README.md#curobo-the-collision-aware-motion-planner)
- [`src/robot/safety/planning/`](../src/robot/safety/planning/README.md)
- [make Isaac ready](isaac-ready.md)
