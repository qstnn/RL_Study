"""本地运行与检查点来源；没有下载、Git或外部服务。"""
from datetime import datetime
import hashlib,json,shutil,sys
from pathlib import Path
from zoneinfo import ZoneInfo
from .spec import ROOT,CONTRACT
CORE=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw'
sys.path.insert(0,str(ROOT/'model_lighthw/training/multiskill'))
from common import asset_manifest,module_from_file,URDF,snapshot

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def implementation():
 paths=list((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v5').glob('*.py'))+list((ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_auto_v5/bootstrap').glob('*'))+[ROOT/'model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/scripts/r5a_auto_v5.sh',CORE/'r5a/appearance.py',CORE/'payload_spawn.py',CORE/'local_ground_plane.py']
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def checkpoint(p):
 import torch
 x=torch.load(p,map_location='cpu',weights_only=False);info=x.get('infos') or {}
 if info.get('contract')!=CONTRACT:raise ValueError('不是当前自主空翻v5模型；v3只能经内置Actor显式迁移，不能直接续训')
 if info.get('implementation_sha256')!=implementation():raise ValueError('检查点与当前实现不同；需要显式版本迁移')
 if not isinstance(info.get('completed_updates'),int) or info['completed_updates']<0:raise ValueError('Missing update count')
 def finite(v):
  if isinstance(v,torch.Tensor):return bool(torch.isfinite(v).all())
  if isinstance(v,dict):return all(finite(a) for a in v.values())
  if isinstance(v,(tuple,list)):return all(finite(a) for a in v)
  return True
 if not finite(x['model_state_dict']) or not finite(x['optimizer_state_dict']):raise ValueError('Non-finite checkpoint')
 return x

def prepare_run(run,request):
 run.mkdir(parents=True,exist_ok=False)
 write(run/'request.json',request);write(run/'contract.json',CONTRACT)
 assets=asset_manifest();write(run/'assets.json',assets)
 model=module_from_file('_autoflip_payload_asset',CORE/'payload_asset.py').write_payload_urdf(URDF,run/'input/robot.urdf',0.)
 write(run/'model.json',model);snapshot(run)
 hashes=read(run/'source_manifest.json')
 # Archive the actual bundled Isaac Lab framework sources for later offline audit.
 for framework in ('isaaclab','isaaclab_rl','isaaclab_assets','isaaclab_tasks'):
  root=ROOT/'Isaaclab/source'/framework/framework
  for source in root.rglob('*.py'):
   rel=str(source.relative_to(ROOT));out=run/'source_snapshot'/rel
   out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out);hashes[rel]=sha(source)
 for rel,h in implementation().items():
  p=run/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,p);hashes[rel]=h
 write(run/'source_manifest.json',hashes)

def check_sources(run):
 hashes=read(run/'source_manifest.json');bad=[p for p,h in hashes.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
 write(run/'source_check.json',dict(files=len(hashes),changed=bad,passed=not bad))
 if bad:raise RuntimeError('运行期间源码改变: '+str(bad))

def seal(folder):
 # 保留原始缓存，但机器相关USD缓存不作必须回收的审查证据。
 paths=[p for p in folder.rglob('*') if p.is_file() and p.name!='SHA256.json' and 'asset_cache' not in p.relative_to(folder).parts and '__pycache__' not in p.parts]
 write(folder/'SHA256.json',{str(p.relative_to(folder)):sha(p) for p in sorted(paths)})
 return len(paths)

def pack_review(folder):
 """USB归档只包含SHA清单和证据，排除可重建的机器USD缓存；不删除原目录。"""
 import zipfile
 folder=Path(folder);hashes=read(folder/'SHA256.json')
 dest=folder.with_suffix('.zip');temp=dest.with_name(dest.name+'.packing')
 with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
  for name in sorted(hashes):
   p=(folder/name).resolve()
   if not p.is_relative_to(folder.resolve()) or not p.is_file() or sha(p)!=hashes[name]:raise ValueError('归档证据不匹配: '+name)
   z.write(p,folder.name+'/'+name)
  z.write(folder/'SHA256.json',folder.name+'/SHA256.json')
 temp.replace(dest)
 dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
 return dest
