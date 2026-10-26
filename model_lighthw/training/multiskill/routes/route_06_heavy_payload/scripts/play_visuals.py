"""R6 播放外观：复用本机 LightHW 配色、R1 V2 地面和随包分发的天空。"""
import json
from launch import ROOT, sha256
from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.local_ground_plane import LocalTerrainImporter

# 与 assets/deeprobotics.py::spawn_lighthw_ironman 完全相同的颜色、金属度和粗糙度。
PALETTE = {
    'ironman_dark_red': ((0.32, 0.015, 0.018), 0.58, 0.34),
    'ironman_thigh_gold': ((0.95, 0.32, 0.035), 0.72, 0.30),
    'ironman_shank_gold': ((1.0, 0.48, 0.06), 0.76, 0.28),
    'dark_graphite': ((0.025, 0.025, 0.03), 0.35, 0.45),
}
MATERIAL_BY_LINK = {'Base_link': 'ironman_dark_red'}
for _leg in ('FR', 'FL', 'RR', 'RL'):
    for _part in ('hip', 'thigh'):
        MATERIAL_BY_LINK[f'{_leg}_{_part}_link'] = 'ironman_thigh_gold'
    MATERIAL_BY_LINK[f'{_leg}_calf_link'] = 'ironman_shank_gold'
    MATERIAL_BY_LINK[f'{_leg}_wheel_link'] = 'dark_graphite'

# 当前本机 wheel_unified_v2/env_cfg.py 的 PreviewSurface 地面颜色。
GROUND_COLOR = (0.25, 0.28, 0.32)
VISUAL_GROUND_PATH = '/World/R6PlayGround'


def configure_play_view(cfg):
    """仅在 play 调用：校验离线 HDR 并设置天空/自由视角的初始机位，返回外观来源记录。"""
    import isaaclab.sim as sim_utils
    assets = ROOT/'assets/appearance'
    source = json.loads((assets/'sky_source.json').read_text())
    sky = assets/source['filename']
    if not sky.is_file() or sha256(sky) != source['sha256']:
        raise RuntimeError(f'R6 play sky missing or corrupt: {sky}; re-extract the complete USB appearance patch')
    cfg.scene.sky_light.spawn = sim_utils.DomeLightCfg(intensity=750.0, texture_file=str(sky.resolve()))
    # 在首次 sim.reset() 之前绑定材质。运行中修改实例材质会使 PhysX tensor view 失效。
    cfg.scene.robot.spawn.func = spawn_play_payload
    cfg.scene.terrain.class_type = PlayTerrainImporter
    # World 模式只设置初始机位，渲染回调不重设相机，鼠标可自由旋转/平移/缩放。
    # 保留原出生位置附近的构图；用户仍可在 Viewer Settings 中手动选择 Robot 跟随。
    cfg.viewer.origin_type = 'world'
    cfg.viewer.asset_name = 'robot'
    cfg.viewer.env_index = 0
    initial_pos = cfg.scene.robot.init_state.pos
    cfg.viewer.eye = tuple(p + offset for p, offset in zip(initial_pos, (2.0, -2.0, 0.8)))
    cfg.viewer.lookat = tuple(p + offset for p, offset in zip(initial_pos, (0.0, 0.0, 0.05)))
    return {'style': 'local_lighthw', 'sky': source, 'sky_local_path': str(sky.resolve()),
            'ground_color': GROUND_COLOR, 'palette': PALETTE,
            'camera': {'origin_type': cfg.viewer.origin_type, 'asset_name': 'robot', 'env_index': 0,
                       'eye': cfg.viewer.eye, 'lookat': cfg.viewer.lookat}}


def _bind(target, material):
    import isaaclab.sim as sim_utils
    sim_utils.bind_visual_material(target, material)
    _check_binding(target, material)


def _check_binding(target, material):
    import isaaclab.sim as sim_utils
    from pxr import UsdShade
    prim = sim_utils.get_current_stage().GetPrimAtPath(target)
    if not prim.IsValid():
        raise RuntimeError(f'R6 play prim missing: {target}')
    actual = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not actual or str(actual.GetPath()) != material:
        raise RuntimeError(f'R6 play material did not bind: {target} -> {material}')


def spawn_play_payload(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """仅播放：沿用原 R6 惯量导入，在物理初始化前补上 LightHW 材质。"""
    import isaaclab.sim as sim_utils
    from rl_training.tasks.manager_based.locomotion.multiskill.lighthw.payload_spawn import spawn_payload
    prim = spawn_payload(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    for root in sim_utils.find_matching_prim_paths(prim_path):
        for name, (color, metallic, roughness) in PALETTE.items():
            material = sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=metallic, roughness=roughness)
            material.func(f'{root}/Looks/{name}', material)
        for link, name in MATERIAL_BY_LINK.items():
            _bind(f'{root}/{link}', f'{root}/Looks/{name}')
    return prim


class PlayTerrainImporter(LocalTerrainImporter):
    """只复用当前本机地面的视觉材质，保留原 R6 无限碰撞平面。"""
    def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom
        super().import_ground_plane(name, size)
        # PhysX 的无限 Plane 用于碰撞/扫描；RTX 需要显式 Mesh 才能显示地面和阴影。
        # 单独放在扫描目标 /World/ground 外，且不应用任何碰撞或刚体 API。
        mesh = UsdGeom.Mesh.Define(sim_utils.get_current_stage(), VISUAL_GROUND_PATH)
        mesh.CreatePointsAttr([(-1000., -1000., 0.), (1000., -1000., 0.),
                               (1000., 1000., 0.), (-1000., 1000., 0.)])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateExtentAttr([(-1000., -1000., 0.), (1000., 1000., 0.)])
        material_path = self.cfg.prim_path + '/r6_play_ground'
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=GROUND_COLOR)
        material.func(material_path, material)
        _bind(VISUAL_GROUND_PATH, material_path)


def verify_play_materials(env):
    """仿真启动后只读核对实际绑定，避免运行中改材质使物理视图失效。"""
    import isaaclab.sim as sim_utils
    bindings = {}
    roots = sim_utils.find_matching_prim_paths(env.cfg.scene.robot.prim_path)
    if not roots:
        raise RuntimeError('R6 play robot prim not found')
    for root in roots:
        for link, name in MATERIAL_BY_LINK.items():
            target, material = f'{root}/{link}', f'{root}/Looks/{name}'
            _check_binding(target, material)
            bindings[target] = material
    ground = VISUAL_GROUND_PATH
    material = env.cfg.scene.terrain.prim_path + '/r6_play_ground'
    _check_binding(ground, material)
    bindings[ground] = material
    print(f'[R6 PLAY] LightHW colors on {len(roots)*len(MATERIAL_BY_LINK)} links; local ground/HDR; free camera (World); optional follow in Viewer Settings.', flush=True)
    return bindings
