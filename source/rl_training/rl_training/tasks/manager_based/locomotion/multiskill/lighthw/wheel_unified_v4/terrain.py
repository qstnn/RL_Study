"""固定11条真实坡面；全部坡度同时存在，没有隐式地形晋级。"""
import numpy as np
import torch
import trimesh
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from . import core, spec


class UnifiedTerrain(TerrainImporter):
    def __init__(self,cfg):
        cfg.validate()
        self.cfg=cfg
        self.device=sim_utils.SimulationContext.instance().device
        self.terrain_prim_paths=[]
        self._terrain_flat_patches={}
        self.terrain_origins=None
        self.env_origins=None
        meshes=[]
        for i,angle in enumerate(spec.ANGLES):
            vertices,faces=core.slope_mesh(angle,(i-5)*14.)
            meshes.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
        self.import_mesh("terrain",trimesh.util.concatenate(meshes))
        angles,_,_=core.assignments(cfg.num_envs,spec.SLOPE_MIX,spec.PAYLOADS,spec.TOWED_MASSES, cfg.fixed_slope_deg,cfg.fixed_payload_kg,cfg.fixed_towed_kg)
        # 训练只使用清单内离散坡度；固定评估也必须对应真实网格。
        types=torch.tensor([spec.ANGLES.index(float(a)) for a in angles],device=self.device)
        origins=torch.tensor([[0.,(i-5)*14.,0.] for i in range(len(spec.ANGLES))],device=self.device)
        self.terrain_origins=origins.unsqueeze(0)
        self.terrain_types=types
        self.terrain_levels=torch.zeros(cfg.num_envs,dtype=torch.long,device=self.device)
        self.max_terrain_level=1
        self.env_origins=origins[types].clone()
        self.set_debug_vis(cfg.debug_vis)


@configclass
class UnifiedTerrainCfg(TerrainImporterCfg):
    class_type: type=UnifiedTerrain
    fixed_slope_deg: float | None=None
    fixed_payload_kg: float | None=None
    fixed_towed_kg: float | None=None
