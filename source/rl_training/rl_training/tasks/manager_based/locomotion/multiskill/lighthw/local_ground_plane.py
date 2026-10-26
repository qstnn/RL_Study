"""R6 离线平面：在当前 USD stage 创建平面，不引用 Nucleus/S3 资产。"""
from pxr import Gf, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter


def spawn_local_ground_plane(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """用途：创建与默认地面同类的无限碰撞平面；用于离线平地，返回根 prim。"""
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f'A prim already exists at path: {prim_path}')
    root = UsdGeom.Xform.Define(stage, prim_path)
    if translation is not None:
        root.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        root.AddOrientOp().Set(Gf.Quatf(orientation[0], Gf.Vec3f(*orientation[1:])))
    # PhysX 的 Plane 为无限碰撞面；height scanner 也按 Plane 创建无限射线地面。
    plane = UsdGeom.Plane.Define(stage, prim_path + '/CollisionPlane')
    plane.CreateAxisAttr('Z')
    plane.CreateWidthAttr(float(cfg.size[0]))
    plane.CreateLengthAttr(float(cfg.size[1]))
    plane.CreateDoubleSidedAttr(True)
    plane.CreateDisplayColorAttr([Gf.Vec3f(*(cfg.color or (.18, .18, .18)))])
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim()).CreateCollisionEnabledAttr(True)
    if cfg.physics_material is not None:
        material_path = prim_path + '/physicsMaterial'
        cfg.physics_material.func(material_path, cfg.physics_material)
        sim_utils.bind_physics_material(str(plane.GetPath()), material_path)
    return root.GetPrim()


class LocalTerrainImporter(TerrainImporter):
    """保留 TerrainImporter 的环境排列，只替换 plane 的资源创建。"""
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        prim_path = self.cfg.prim_path + '/' + name
        if prim_path in self.terrain_prim_paths:
            raise ValueError(f'Duplicate ground plane: {prim_path}')
        cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=(.18, .18, .18))
        spawn_local_ground_plane(prim_path, cfg)
        self.terrain_prim_paths.append(prim_path)
