"""独立复算器的完整成功与末段失稳反例；合成轨迹不是动力学证明。"""
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from test_v2 import contract
from verify_trace import verify


class TraceTests(unittest.TestCase):
    def test_positive_front_rear_and_terminal_instability(self):
        n=12000;t=(np.arange(1,n+1,dtype=np.float32)*np.float32(.001))
        ts=np.repeat(t[:,None],3,axis=1);modes=np.tile([0,1,0],(n,1))
        amp=np.minimum(np.clip((t-1)/3,0,1),np.clip((10-t)/3,0,1));amp=amp**2*(3-2*amp)
        pitches=np.array([.91,-.98,.91],np.float32)[None,:]*amp[:,None]
        target=np.stack((np.sin(pitches),np.zeros_like(pitches),-np.cos(pitches)),-1)
        hold=(t>=4)&(t<7);force=np.full((n,3,4),28.62,np.float32)
        wheels=np.zeros((n,3,4,3),np.float32);wheels[:,:,:,2]=.087501
        wheels[:,:,0::2,1]=-.2;wheels[:,:,1::2,1]=.2
        for i,mode in enumerate([0,1,0]):
            stance=[2*mode,2*mode+1];swing=[(j+2)%4 for j in stance]
            for j in swing:force[hold,i,j]=0.;wheels[hold,i,j,2]+=.1;wheels[:,i,j,0]=-.6
            for j in stance:force[hold,i,j]=57.24
        com=np.zeros((n,3,3),np.float32);com[:,:,2]=.35
        vel=np.zeros_like(com);vel[t>11.8,2,0]=.5
        zero=np.zeros((n,3),np.float32)
        data=dict(t=ts,task_t=ts,mode=modes,active=np.ones((n,3),bool),origin=np.zeros((n,3,2),np.float32),
                  wheel_pos=wheels,force_z=force,clearance=wheels[:,:,:,2]-.087501,com=com,com_vel=vel,
                  gravity=target,target_gravity=target,ang_vel=np.zeros_like(com),base_z=zero+.416,
                  line_error=zero,body_force=zero,torque_peak=zero)
        cases=[dict(trace='trace_seed43.npz',trace_env=i,entered=True,held=True,failed=False,success=i<2,
                    reason=0,best_hold=3.,best_recovery=2.001 if i<2 else 1.8,start_time=1.,best_pair=3.,timeout=True) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            np.savez_compressed(folder/'trace_seed43.npz',**data)
            for name,value in [('evaluation.json',dict(cases=cases)),('contract.json',contract()),
                               ('geometry.json',dict(candidates=[dict(pitch_rad=.91),dict(pitch_rad=-.98)]))]:
                (folder/name).write_text(json.dumps(value))
            proof=verify(folder)
            self.assertEqual(proof['cases'],3);self.assertEqual(proof['physics_samples'],36000)
            cases[-1]['success']=True
            (folder/'evaluation.json').write_text(json.dumps(dict(cases=cases)))
            with self.assertRaisesRegex(AssertionError,'success'):verify(folder)


if __name__=='__main__':unittest.main()
