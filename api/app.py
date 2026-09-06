"""The FastAPI application.

Everything is mounted under ``/v1``, from the first route onward: a surface that starts versioned
never needs a deprecation shim to become versioned.

One error shape. Every failure, whether validation, refusal or unknown key, comes back as
``{code, message, detail}``. Routers raise ``HTTPException`` with that dict as the detail and the
handler below unwraps it, so a client has exactly one envelope to parse and ``code`` is always a
string it can branch on rather than prose it has to match.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from api import __version__
from api.cell import NoRobotConfigured, console
from api.constants import API_LOG_DIR, APP_LOG_FILE
from api.routers import cell as cell_router
from api.routers import config as config_router
from api.routers import diagnostics as diagnostics_router
from api.routers import history as history_router
from api.routers import media as media_router
from api.routers import pick as pick_router
from api.routers import preflight as preflight_router
from api.schemas import ErrorOut
from src.utility.log_cfg import create_logger

__all__ = ["create_app"]

logger = create_logger("ConsoleApp", APP_LOG_FILE, log_dir=API_LOG_DIR)

_DESCRIPTION = """\
The optional operator console for Willy the Workaholic.

Localhost only, no authentication. The console issues **tasks**, not parameters: it can start a pick and
give it a prompt, and it cannot change how a motion executes. Speed, acceleration, workspace limits and
safety toggles are read-only here and edited in YAML, next to the comments that justify them.

There is deliberately **no E-stop endpoint**. A stop that travels over a socket depends on latency, an
open tab and an awake laptop; offering one would invite someone to rely on it instead of the physical
mushroom button.
"""


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Nothing on the way in; put the cell down and close its camera on the way out.

    A build opens an ``rs.pipeline``, and nothing else in this server ever closes one. Without this,
    stopping the server leaves the device claimed until the OS reaps the process, which it does
    promptly on a clean exit and not promptly under ``--reload``, where a worker is replaced while
    the parent lives on, so the next worker cannot open the camera the last one is still holding.

    A lifespan rather than ``@app.on_event("shutdown")``: that decorator is deprecated in the FastAPI
    this repo pins, and it warns at import.

    Best-effort by construction: ``release()`` swallows what it cannot fix, because a shutdown that
    fails to shut down is worse than a device that fails to close.
    """
    yield
    logger.info("Shutting down: releasing the cell and closing its camera.")
    try:
        console().session.release()
    except Exception as exc:  # noqa: BLE001 (the process is going away; report, do not propagate)
        # A camera the shutdown failed to hand back is exactly what makes the next start fail, so
        # the reason is reported here rather than lost with the process.
        logger.warning("Releasing the cell during shutdown failed: %s: %s", type(exc).__name__, exc)


def create_app() -> FastAPI:
    """Build the app. A factory: every call returns a fresh instance, not shared module state."""
    app = FastAPI(
        lifespan=_lifespan,
        title="Willy the Workaholic: operator console",
        version=__version__,
        description=_DESCRIPTION,
        # Every failure leaves here as one `ErrorOut` envelope, produced by the handlers below and
        # not by any route. So `ErrorOut` appears in no route signature, and without this default
        # response it would appear in no OpenAPI document either: a generated client could type
        # every success and not one failure. Declaring it here documents it once, for every route,
        # without touching a single route.
        responses={"default": {"model": ErrorOut, "description": "The one failure envelope."}},
    )

    # Registered for Starlette's HTTPException, not FastAPI's. FastAPI's subclasses it, so this
    # still catches every route-raised error, and the router's own 404 for an unrouted path raises
    # the Starlette one directly: registered on the narrower class, that 404 escapes as
    # `{"detail": "Not Found"}`. The module docstring above promises one envelope for every failure,
    # and this registration is what makes that true of every response.
    @app.exception_handler(StarletteHTTPException)
    async def _typed_errors(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail: Any = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = {
                "code": detail["code"],
                "message": detail.get("message", ""),
                "detail": detail.get("detail", {}),
            }
        else:
            # FastAPI's own raises (404 on an unknown route, 405, ...) still arrive here. Giving them
            # the same envelope means the client never has to ask which kind of error it is holding.
            body = {"code": f"http_{exc.status_code}", "message": str(detail), "detail": {}}
        # One line per failure that leaves the server, at the level the status code deserves: 5xx is
        # a server fault, 4xx is the caller's. Logged at this single funnel, which every refusal
        # passes through, rather than at each raise site.
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("%s -> %d %s", request.url.path, exc.status_code, body["code"])
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(NoRobotConfigured)
    async def _no_robot(_request: Request, exc: NoRobotConfigured) -> JSONResponse:
        """A tree with no robot section is a misconfigured console, not a server fault.

        ``AppConfig.robot`` is genuinely optional, since a tree may configure cameras and models
        only, so this is reachable by pointing the console at the wrong directory or by omitting the
        profile that supplies the arm. Answering with the directory and the chain says which of the
        two happened.
        """
        logger.error(
            "No robot configured: tree %s, profile chain %s.", exc.root, exc.profile or "(none)"
        )
        return JSONResponse(
            status_code=409,
            content={
                "code": "no_robot_configured",
                "message": str(exc),
                "detail": {"root": str(exc.root), "profile": exc.profile},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _request_errors(request: Request, exc: RequestValidationError) -> JSONResponse:
        # The count and the field locations, never `exc.errors()` itself: that payload embeds the
        # rejected input, which on this server includes a connect token.
        logger.warning(
            "%s -> 422 bad_request (%d field error(s): %s)",
            request.url.path,
            len(exc.errors()),
            ", ".join(".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()[:5]),
        )
        return JSONResponse(
            status_code=422,
            content={
                "code": "bad_request",
                "message": "the request body or query does not match this endpoint.",
                "detail": {"errors": exc.errors()},
            },
        )

    @app.get("/v1/health", tags=["system"], summary="Is the server up?")
    def health() -> dict[str, str]:
        """Deliberately says nothing about the cell; that is what ``/v1/preflight`` is for.

        A health check that reports on hardware turns "the server is running" and "the robot is
        ready" into one answer, and then a green light means neither reliably.
        """
        return {"status": "ok", "version": __version__}

    app.include_router(preflight_router.router, prefix="/v1")
    app.include_router(config_router.router, prefix="/v1")
    app.include_router(cell_router.router, prefix="/v1")
    app.include_router(diagnostics_router.router, prefix="/v1")
    app.include_router(pick_router.router, prefix="/v1")
    app.include_router(media_router.router, prefix="/v1")
    app.include_router(history_router.router, prefix="/v1")
    _mount_console(app)

    logger.info("Console app built (version %s), 7 routers mounted under /v1.", __version__)
    return app


#: Where the ``frontend`` tree, built with ``npm run build``, writes the console bundle. Not checked
#: in: a built bundle in git is a second copy of the source that drifts from it silently.
_STATIC = Path(__file__).resolve().parent / "static"


def _mount_console(app: FastAPI) -> None:
    """Serve the built single-page console from this same server, if it has been built.

    Same origin is the point, not a convenience. With the SPA served here, the browser API base URL
    is the empty string: there is no host to configure, so no build and no bookmark can aim the
    console at a different cell than the one whose server it was opened from. During development
    ``npm run dev`` proxies ``/v1`` here and reproduces exactly that, so the client code is identical
    in both modes.

    Absent is not an error. A headless deployment and a CI run have no ``static/`` directory, and
    this must change nothing for them: the console is an optional face on a library, and the library
    does not need it. Mounted when built, skipped when not.

    The catch-all is scoped to paths that are not ``/v1``: an unknown API path must keep returning
    the JSON error envelope, never an HTML page. A client that receives ``index.html`` where it
    expected JSON reports a parse error, and the operator goes hunting for a serialisation bug that
    does not exist.
    """
    # Bound once, into the closures below, rather than read from the module at each request. The
    # difference is not stylistic: with a request-time read, the directory an app serves could
    # differ from the one it decided to mount for, so an app built for one bundle would quietly
    # serve another.
    root = _STATIC
    if not (root / "index.html").is_file():
        # Not a failure, and this line is the answer to "JSON arrives, but no page", which is
        # otherwise only discoverable by reading this function.
        logger.info("No built console bundle at %s; serving the API only.", root)
        return

    logger.info("Serving the console bundle from %s.", root)
    app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @app.get("/demo.html", include_in_schema=False)
    def _demo() -> FileResponse:
        return FileResponse(root / "demo.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str) -> FileResponse:
        """Any non-API path returns the app shell, so a deep link survives a reload."""
        if full_path.startswith("v1/"):
            raise StarletteHTTPException(status_code=404, detail="Not Found")
        candidate = (root / full_path).resolve()
        # Serve a real file when there is one (favicon, fonts), but only from inside the bundle: a
        # path that escapes it would turn the console into a file server for the whole box.
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            return FileResponse(candidate)
        return FileResponse(root / "index.html")


app = create_app()
