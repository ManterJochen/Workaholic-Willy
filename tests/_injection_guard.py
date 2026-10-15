"""A test may only inject what the arm already has (P0e, plan section 5.3).

About 45 tests write `arm._conn = MagicMock()`, others `arm._motion`, `arm._preflight`, `arm._curobo`, and some
override a verb (`arm.ik = ...`, `arm.wait_until_steady = ...`). Python accepts every one of those writes whether or not
the arm still reads that name: once P0e moves the planning half into `PlannedArm` and the RTDE calls behind a transport,
an assignment to a name nothing reads any more creates a fresh attribute and controls nothing, and the negative
assertions that follow it (`conn.moveJ.assert_not_called()`, the gating fences) pass on nothing at all.

So from the moment a UR arm or its motion controller is built, a write from outside the library must land on a name the
object already carries, on the instance or on its class. The library itself stays free to set what it likes, and an
object made with `__new__` (the hand-built arm in `test_ur_controller_identity.py`) is exempt, because nothing about it
was built. What this cannot see is a name that still exists but is no longer the one the code path reads; the tests in
`test_what_a_test_injects_reaches_the_arm.py` hold that half.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
#: Where the library lives in Willy: a write from here is the library's own, wherever it lands.
_LIBRARY = (_ROOT / "src", _ROOT / "api", _ROOT / "willy")
_BUILT = "_willy_injection_guard_built"


def _from_the_library(filename: str) -> bool:
    try:
        path = Path(filename).resolve()
        return any(path.is_relative_to(root) for root in _LIBRARY)
    except (OSError, ValueError):
        return False


def guard(cls: Any) -> None:
    """Wrap ``cls.__init__`` to mark a built object and ``cls.__setattr__`` to refuse a dead injection on it.

    Public for the fakes a test composes at run time (``tests/_fake_transport.py`` guards each class it builds), which
    :func:`install` cannot list because they do not exist yet when it runs.
    """
    if getattr(cls, "_willy_injection_guarded", False):
        return
    built_init = cls.__init__
    plain_setattr = cls.__setattr__

    @functools.wraps(built_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        built_init(self, *args, **kwargs)
        object.__setattr__(self, _BUILT, True)

    def __setattr__(self: Any, name: str, value: Any) -> None:
        state = self.__dict__
        if state.get(_BUILT) and name not in state and not hasattr(type(self), name):
            caller = sys._getframe(1).f_code.co_filename
            if not _from_the_library(caller):
                raise AttributeError(
                    f"{type(self).__name__} has no {name!r}, so this write would create it and control nothing: the "
                    f"code path does not read it. Inject a name the built {type(self).__name__} carries (see "
                    f"tests/_injection_guard.py); a renamed member means the test follows the rename."
                )
        plain_setattr(self, name, value)

    cls.__init__ = __init__
    cls.__setattr__ = __setattr__
    cls._willy_injection_guarded = True


def install() -> None:
    """Guard the planned arm, the UR arm and its motion controller for the whole run.

    ``PlannedArm`` since P0g slice 5: a test builds one over the fake transport with no vendor subclass, and a write
    to a name it does not carry controls nothing there either.
    """
    from src.robot.drivers.planned.arm import PlannedArm
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.drivers.ur.motion import MotionController

    for cls in (PlannedArm, URRobotArm, MotionController):
        guard(cls)
