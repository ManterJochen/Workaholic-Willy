"""Every registered arm vendor, built through the registry as a test may build it: no controller, no SDK (P0g slice 10).

The registry-driven suites iterate this table, and the table is pinned to ``available_vendors()``: a vendor that
registers a factory and has no row here fails ``test_the_table_is_the_registry`` until it says how a test builds it.
The kwargs are the test's, so the registry itself carries nothing for tests (prep decision, 2026-09-24).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["build", "links", "vendors"]

_PLAIN = (bool, int, float, str, bytes, list, tuple, dict, set, frozenset, type(None))


def _ur_config() -> Any:
    from src.config.schema.robot import RobotConfig

    return RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "192.168.1.10", "model": "ur5e"},
                                       "gripper": {"model": "robotiq_2f85"}})


def _sim_config() -> Any:
    from src.robot.drivers.sim.config import SimRobotConfig

    return SimRobotConfig(enabled=True, mock_mode=True)


#: Vendor -> the kwargs a test hands ``create_arm`` for it.
_KWARGS: dict[str, Callable[[], dict[str, Any]]] = {
    "dummy": lambda: {},
    "ur": lambda: {"config": _ur_config()},
    "sim": lambda: {"config": _sim_config()},
}


def vendors() -> tuple[str, ...]:
    """The vendors this table can build, sorted."""
    return tuple(sorted(_KWARGS))


def build(vendor: str) -> Any:
    """``vendor``'s arm, built by its registered factory with this table's kwargs; nothing is connected."""
    from src.robot.drivers import create_arm

    return create_arm(vendor, **_KWARGS[vendor]())


def links(arm: Any) -> list[Any]:
    """The objects that ARE this arm's controller link: its transport and what the transport holds, or the sim's
    session. A public attribute answering one of them hands a caller the controller past every guard."""
    found: list[Any] = []
    transport = vars(arm).get("_transport")
    if transport is not None:
        found.append(transport)
        # Objects only: a transport's plain state (a flag, a list of joints) is no link, and ``True is True`` would
        # make every boolean property look like one (the fake transport's ``connected``, measured in this suite).
        found.extend(value for value in vars(transport).values() if not isinstance(value, _PLAIN))
    session = vars(arm).get("_session")
    if session is not None:
        found.append(session)
    return found
