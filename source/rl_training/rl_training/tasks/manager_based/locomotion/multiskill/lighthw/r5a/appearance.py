"""离线 LightHW 原配色、深灰地面和本地 HDR；World 视角仅设置初始位置。"""
import hashlib
import json
from .spec import MULTISKILL

PALETTE = {'red':((.32,.015,.018),.58,.34), 'gold':((.95,.32,.035),.72,.30),
           'calf':((1.,.48,.06),.76,.28), 'wheel':((.025,.025,.03),.35,.45)}


def spawn_robot(prim_path,cfg,translation=None,orientation=None,**kwargs):
    import isaaclab.sim as sim
    from ..payload_spawn import spawn_payload
    prim=spawn_payload(prim_path,cfg,translation=translation,orientation=orientation,**kwargs)
    for root in sim.find_matching_prim_paths(prim_path):
        for name,(color,metal,rough) in PALETTE.items():
            material=sim.PreviewSurfaceCfg(diffuse_color=color,metallic=metal,roughness=rough)
            material.func(f'{root}/Looks/r5a_{name}',material)
        for leg in ('FR','FL','RR','RL'):
            for part,name in (('hip','gold'),('thigh','gold'),('calf','calf'),('wheel','wheel')):
                sim.bind_visual_material(f'{root}/{leg}_{part}_link',f'{root}/Looks/r5a_{name}')
        sim.bind_visual_material(f'{root}/Base_link',f'{root}/Looks/r5a_red')
    return prim


def sky_path():
    folder=MULTISKILL/'assets/appearance'
    info=json.loads((folder/'sky_source.json').read_text())
    path=folder/info['filename']
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=info['sha256']:
            raise ValueError('R5A local HDR checksum mismatch')
    return str(path)


def terrain_class():
    from ..local_ground_plane import LocalTerrainImporter
    class OfflineTerrain(LocalTerrainImporter):
        def import_ground_plane(self,name,size=(2.e6,2.e6)):
            import isaaclab.sim as sim
            from pxr import UsdGeom
            super().import_ground_plane(name,size)
            path='/World/R5AGroundVisual'
            mesh=UsdGeom.Mesh.Define(sim.get_current_stage(),path)
            mesh.CreatePointsAttr([(-1000.,-1000.,0.),(1000.,-1000.,0.),(1000.,1000.,0.),(-1000.,1000.,0.)])
            mesh.CreateFaceVertexCountsAttr([4]); mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
            mesh.CreateSubdivisionSchemeAttr('none')
            mat=sim.PreviewSurfaceCfg(diffuse_color=(.25,.28,.32))
            mat_path=self.cfg.prim_path+'/r5a_material'
            mat.func(mat_path,mat); sim.bind_visual_material(path,mat_path)
    return OfflineTerrain
