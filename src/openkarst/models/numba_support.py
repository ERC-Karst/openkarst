"""Shared helpers for optional Numba acceleration."""

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dependency
    NUMBA_AVAILABLE = False
    get_num_threads = None
    njit = None
    prange = range
    set_num_threads = None


def ensure_numba_available():
    """Raise a clear error when the optional Numba backend is requested."""
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "solver_settings.parallelization=True requires the optional "
            "'numba' package. Install numba in this environment or set "
            "parallelization=False."
        )


def configure_numba_threads(num_threads):
    """Set the active Numba thread count when requested and return the count."""
    ensure_numba_available()
    if num_threads is not None:
        set_num_threads(num_threads)
    return get_num_threads()
