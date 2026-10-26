"""离线重算冻结验收与物理采样；不把数据一致性等同任务通过。"""
import argparse,csv,json,sys,importlib.util,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from launch import PROTOCOL,write_json,sha256,SOURCE,module_from_file
TORQUE=module_from_file('_v4_audit_torque',SOURCE/'wheel_unified_v4/torque.py')
def compare(a,b,path=''):
 if isinstance(a,dict):
  assert set(a)<=set(b),(path,set(a)-set(b))
  for k,v in a.items():compare(v,b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
 elif isinstance(a,(float,int)) and not isinstance(a,bool):
  assert np.isclose(a,b,rtol=1e-6,atol=1e-7,equal_nan=False),(path,a,b)
 else:assert a==b,(path,a,b)
def audit(folder):
 report=json.loads((folder/'evaluation.json').read_text());samples=0
 for index,c in enumerate(report['cases']):
  # Evaluation index is local to each seed's 16 cases.
  prefix=f"seed{c['seed']}_case{index%16:02d}"
  rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((folder/(prefix+'.csv')).open())]
  recalculated=PROTOCOL.summarize(rows,c['case'],c['failure_reason']);compare(recalculated,c,prefix)
  z=np.load(folder/(prefix+'_motor_trace.npz'))
  assert len(z['applied'])==len(rows)*20
  for k in ('applied','requested','velocity'):assert np.isfinite(z[k]).all(),(prefix,k)
  assert np.all(abs(z['applied'])<=z['limits']+1e-4)
  trace={k:z[k] for k in ('applied','requested','velocity')}
  motor=TORQUE.summarize_trace(trace,z['joint_names'].tolist(),z['limits'],c['success'])
  compare(motor,json.loads((folder/(prefix+'_motor.json')).read_text()),prefix+'/motor')
  samples+=len(z['applied'])
 manifest=json.loads((folder/'source_manifest.json').read_text())
 for name,digest in manifest.items():assert sha256(folder/'source_snapshot'/name)==digest,name
 result=dict(status='passed',case_count=len(report['cases']),physics_sample_times=samples,measured_joints=16,
   source_snapshot_files=len(manifest),complete=report['complete_count'],task_pass=report['pass_count'],
   note='Data audit only; task acceptance remains evaluation.json. Physical assumptions are in runtime_check.json.')
 write_json(folder/'audit.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');a=p.parse_args()
 for path in a.folders:print(path,json.dumps(audit(path)),flush=True)
