"""Camera helpers for the Plotly 3D scene."""

from copy import deepcopy

from .constants import VIEW_CAMERAS


def _default_3d_camera(default_camera):
    return deepcopy(default_camera or dict(eye=dict(x=1.4, y=1.4, z=1.4)))


def _normalize_camera(camera, default_camera):
    if isinstance(camera, dict):
        if any(key in camera for key in ("eye", "up", "center", "projection")):
            return deepcopy(camera)
        if isinstance(camera.get("3d"), dict):
            return deepcopy(camera["3d"])
    return _default_3d_camera(default_camera)


def _camera_for_view(view_mode, default_camera):
    if view_mode in VIEW_CAMERAS:
        return deepcopy(VIEW_CAMERAS[view_mode])
    return _default_3d_camera(default_camera)


def _set_nested(target, path, value):
    cursor = target
    for part in path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[path[-1]] = value


def _camera_from_relayout(existing_camera, relayout_data):
    if not relayout_data:
        return None

    camera = deepcopy(existing_camera) if isinstance(existing_camera, dict) else {}
    if isinstance(relayout_data.get("scene.camera"), dict):
        for key, value in relayout_data["scene.camera"].items():
            if isinstance(value, dict) and isinstance(camera.get(key), dict):
                camera[key].update(deepcopy(value))
            else:
                camera[key] = deepcopy(value)
        return camera

    changed = False
    for key, value in relayout_data.items():
        if key.startswith("scene.camera."):
            _set_nested(camera, key.split(".")[2:], value)
            changed = True
    return camera if changed else None
