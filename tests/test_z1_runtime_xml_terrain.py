from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mujoco

TOOLS_Z1 = Path(__file__).resolve().parents[1] / "tools" / "z1"
if str(TOOLS_Z1) not in sys.path:
    sys.path.insert(0, str(TOOLS_Z1))

from deploy_onnx_mujoco import _make_runtime_xml  # noqa: E402


def _write_source_xml(root: Path) -> Path:
    meshes = root / "meshes"
    meshes.mkdir()
    mjcf = root / "mjcf"
    mjcf.mkdir()
    source = mjcf / "robot.xml"
    source.write_text(
        """
<mujoco model="terrain_test">
  <compiler angle="radian" meshdir="../meshes"/>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" friction="1.2 0.02 0.001"/>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return source


class Z1RuntimeXmlTerrainTest(unittest.TestCase):
    def test_plane_keeps_mjcf_floor_without_adding_runtime_plane(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_xml = _make_runtime_xml(_write_source_xml(root), root / "out", terrain="plane")
            text = runtime_xml.read_text(encoding="utf-8")

            self.assertEqual(text.count('name="floor"'), 1)
            self.assertNotIn("sim2sim_gravel_hfield", text)
            mujoco.MjModel.from_xml_path(str(runtime_xml))

    def test_gravel_removes_mjcf_floor_and_adds_hfield(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_xml = _make_runtime_xml(_write_source_xml(root), root / "out", terrain="gravel")
            text = runtime_xml.read_text(encoding="utf-8")

            self.assertNotIn('name="floor"', text)
            self.assertIn('name="sim2sim_gravel_hfield"', text)
            self.assertIn('name="sim2sim_gravel"', text)
            mujoco.MjModel.from_xml_path(str(runtime_xml))


if __name__ == "__main__":
    unittest.main()
