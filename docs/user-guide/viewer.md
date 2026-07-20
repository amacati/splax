# Viewer

`splax.viewer` serves splats to a web client using [viser](https://viser.studio). It requires the optional `viser` dependency:

```bash
pip install splax[viewer]
```

The module is imported lazily (`splax.viewer` or `from splax import viewer`), so the base package works without viser installed.

## Rigid objects

The viewer models a scene the same way `splax.render` does with dynamic transforms. It takes a set of named rigid objects, each with its own gaussians
and world pose. `add_splats` uploads an object's gaussians to the browser once, and `update_pose` moves it afterwards without re-uploading.

```python
import splax
from splax.viewer import Viewer

viewer = Viewer(port=8080, up_direction="+z")
viewer.add_splats("scene", *splax.io.load_ply("room.ply"))
viewer.add_splats("drone", *splax.io.load_ply("drone.ply"), position=(0.0, 0.0, 1.0))

for pos, wxyz in trajectory:  # e.g. from a simulator
    viewer.update_pose("drone", pos, wxyz)
```

Open `http://localhost:8080` in a browser to view the scene. Quaternions are wxyz, as everywhere in splax. `update_poses` sets several objects at once from
a `{name: (position, wxyz)}` dict, `remove` deletes an object, and `close` stops the server. Note that the server runs in a background thread. Keep the
process alive (e.g. block on `input()`) for as long as the viewer should stay reachable.

## Beyond splats

The wrapped `viser.ViserServer` is exposed as `Viewer.server` for anything the wrapper does not cover, such as GUI elements, meshes, or camera controls. See
the [viser documentation](https://viser.studio) for its full API.
