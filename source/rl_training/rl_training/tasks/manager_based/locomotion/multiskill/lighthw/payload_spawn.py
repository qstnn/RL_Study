"""R6专用URDF导入：强制使用已计算的惯量，不依赖导入器默认值。"""
from isaaclab.sim import clone
from isaaclab.sim.converters import UrdfConverter
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file


class PayloadUrdfConverter(UrdfConverter):
    def _get_urdf_import_config(self):
        config = super()._get_urdf_import_config()
        config.set_import_inertia_tensor(True)
        return config


@clone
def spawn_payload(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """按Isaac Lab 2.2的URDF流程生成资产，保留碰撞/传感器/驱动设置。"""
    converter = PayloadUrdfConverter(cfg)
    return _spawn_from_usd_file(prim_path, converter.usd_path, cfg, translation, orientation)
