"""实际学生状态上的影子教师查询；仅诊断，不生成成功示范。"""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_student import parent,read,write,sha
from teacher_sustain import SustainTeacher


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows=[];hashes={}
    for name in ['2026-10-22_bc_01_autonomous','2026-10-22_bc_02_autonomous']:
        folder=HERE/'runs'/name
        with np.load(folder/'trace_seed43.npz') as z:data={k:z[k] for k in z.files}
        hashes[name]=sha(folder/'trace_seed43.npz')
        teacher=SustainTeacher([parent.candidate(.8)]*2,read(folder/'geometry.json'),'cpu')
        for k in range(len(data['policy_time'])):
            j=max(0,k*20-1)
            metrics={key:torch.tensor(data[key][j]) for key in ['gravity','ang_vel','wheel_pos','com','com_vel','force_z']}
            base=SimpleNamespace(metrics=lambda:metrics,
                physics_steps=torch.round(torch.tensor(data['policy_physics_time'][k])/.001).long(),physics_dt=.001,
                state=SimpleNamespace(hold_origin=torch.tensor(data['policy_hold_origin'][k]),return_time=torch.tensor(data['policy_return_time'][k])))
            action=teacher(base,torch.tensor(data['policy_time'][k])).numpy()
            if k in [160,175,200]:
                rows.append(dict(run=name,time=k*.02,teacher_rear=action[:,14:].tolist(),
                    student_rear=data['policy_actor_action'][k,:,14:].tolist(),
                    leg_max_difference=abs(action[:,:12]-data['policy_actor_action'][k,:,:12]).max(-1).tolist()))
    write(HERE/'validation/actual_state_teacher_query.json',dict(rows=rows,trace_hashes=hashes,
        limitation='Shadow action comparison on actual student states; these unexecuted labels are diagnostic only and were not used as successful training data'))


if __name__=='__main__':main()
