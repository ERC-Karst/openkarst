"""Server launch helpers for the browser-based openKARST viewer."""

import threading
import time
import webbrowser

from .app import create_openkarst_viewer_app
from .constants import DEFAULT_DEPTH_SCALE


def _in_google_colab():
    """Return True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _server_url(host, port):
    """Return a browser-friendly URL for a Dash server."""
    browser_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{browser_host}:{port}/"


def _run_dash_server(app, host, port, suppress_jupyter_display=False):
    """Run a Dash server with options that work in background threads."""
    run = app.run if hasattr(app, "run") else app.run_server
    kwargs = {
        "host": host,
        "port": port,
        "debug": False,
        "use_reloader": False,
    }

    if suppress_jupyter_display:
        try:
            run(**kwargs, jupyter_mode="external")
        except TypeError as exc:
            if "jupyter_mode" not in str(exc):
                raise
            run(**kwargs)
    else:
        run(**kwargs)


def _show_colab_iframe(port, iframe_height):
    """Display a running Dash app through Google Colab's port proxy."""
    from google.colab import output
    output.serve_kernel_port_as_iframe(
        port,
        path="/",
        height=iframe_height,
    )


def _print_colab_proxy_url(port):
    """Print a fallback proxied URL for Google Colab, when available."""
    from google.colab import output
    proxy_url = output.eval_js(f"google.colab.kernel.proxyPort({port})")
    print(f"Open the openKARST viewer through Colab here: {proxy_url}")


def launch_openkarst_viewer(
    results,
    geometry,
    obs_df=None,
    *,
    depth_scale=DEFAULT_DEPTH_SCALE,
    host="127.0.0.1",
    port=8050,
    open_browser=True,
    mode="auto",
    iframe_height=700,
):
    """Create, launch, and display the openKARST Dash viewer.

    Parameters
    ----------
    results, geometry, obs_df
        Simulation output, network geometry, and optional observation dataframe.
    depth_scale : float, default DEFAULT_DEPTH_SCALE
        Visual scaling factor for water-depth bars.
    host : str, default "127.0.0.1"
        Host used by the Dash server. In Colab, ``mode="auto"`` switches this
        to ``"0.0.0.0"`` so Colab's port proxy can reach the server.
    port : int, default 8050
        Port used by the Dash server.
    open_browser : bool, default True
        Backward-compatible local behavior. Ignored in Colab auto mode, where
        the viewer is shown in an iframe instead.
    mode : {"auto", "colab", "browser", "none"}, default "auto"
        Display mode. ``"auto"`` uses a Colab iframe inside Google Colab and a
        normal browser locally. ``"none"`` starts the server and only prints
        the URL, which is useful for advanced/custom embedding.
    iframe_height : int, default 700
        Height of the Colab iframe.

    Returns
    -------
    dash.Dash
        The Dash app instance. This keeps the old convenience API while still
        allowing advanced users to inspect or reuse the app.
    """
    valid_modes = {"auto", "colab", "browser", "none"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {mode!r}")

    in_colab = _in_google_colab()
    if mode == "auto":
        if in_colab:
            mode = "colab"
        elif open_browser:
            mode = "browser"
        else:
            mode = "none"

    if mode == "colab" and host == "127.0.0.1":
        # Colab's port proxy can reach the server reliably when it listens on
        # all interfaces inside the runtime container.
        host = "0.0.0.0"

    app = create_openkarst_viewer_app(
        results,
        geometry,
        obs_df,
        depth_scale=depth_scale,
    )

    thread = threading.Thread(
        target=_run_dash_server,
        args=(app, host, port, mode == "colab"),
        daemon=True,
    )
    thread.start()
    time.sleep(1.0)

    if mode == "colab":
        try:
            _show_colab_iframe(port, iframe_height)
        except Exception as exc:
            print("Could not show the openKARST viewer as a Colab iframe.")
            print(f"Reason: {exc}")
            try:
                _print_colab_proxy_url(port)
            except Exception:
                print(f"Viewer is running at {_server_url(host, port)}")
    elif mode == "browser":
        webbrowser.open(_server_url(host, port))
    else:
        print(f"Viewer is running at {_server_url(host, port)}")

    return app
