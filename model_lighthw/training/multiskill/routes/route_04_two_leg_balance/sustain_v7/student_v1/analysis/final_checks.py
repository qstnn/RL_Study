"""本轮结束检查：来源、数据、两轮BC权重/Adam、已有行为报告及文档链接。"""
from pathlib import Path
import ast
import json
import re
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common_student import read,write,sha,require,parent,load_actor,now


def main():
    torch.set_num_threads(4)
    legacy=0
    for name in ('source_manifest.json','comparison_source_manifest.json'):
        for relative,digest in read(parent.CONTROL/name).items():
            assert sha(parent.REPO/relative)==sha(parent.CONTROL/'source_snapshot'/relative)==digest
            legacy+=1
    assert sha(parent.ACTOR)==parent.ACTOR_SHA
    parent.require(parent.HERE/'runs/2026-10-22_sustain_30s_blend800')
    dataset_checks=[]
    for name in ['2026-10-22_nominal_verified','2026-10-22_corrected_02']:
        folder=HERE/'datasets'/name;require(folder)
        meta=read(folder/'dataset.json')
        assert sha(folder/'dataset.npz')==meta['dataset_sha256']
        with np.load(folder/'dataset.npz') as z:
            for i,episode in enumerate(meta['episodes']):
                path=Path(episode['run'])/episode['trace'];assert sha(path)==episode['trace_sha256']
                with np.load(path) as src:
                    active=src['policy_active'][:,episode['slot']]
                    np.testing.assert_array_equal(z['observation'][z['episode']==i],src['policy_observation'][active,episode['slot']])
                    np.testing.assert_array_equal(z['action'][z['episode']==i],src['actions'][::20,episode['slot']][active])
            assert len(z['observation'])==meta['frames']
        dataset_checks.append(dict(name=name,frames=meta['frames'],episodes=len(meta['episodes']),actual_arrays_exact=True))
    models=[];first_state=None
    for name in ['2026-10-22_bc_01','2026-10-22_bc_02']:
        folder=HERE/'runs'/name;model,value=load_actor(folder/'actor.pt');result=read(folder/'result.json')
        with np.load(Path(value['dataset'])/'dataset.npz') as z,torch.no_grad():
            e=(model(torch.from_numpy(z['observation']))-torch.from_numpy(z['action'])).square().mean(-1).numpy()
            stage=[float(e[z['stage']==i].mean()) for i in range(4)]
        np.testing.assert_allclose(stage,result['after']['stage_mse'],atol=2e-10,rtol=1e-4)
        steps=sorted({int(s['step']) for s in value['optimizer_state_dict']['state'].values()});assert steps==[6000]
        if first_state is not None:
            initial=torch.load(folder/'initial_actor_state.pt',map_location='cpu',weights_only=True)
            assert all(torch.equal(initial[k],first_state[k]) for k in first_state)
        first_state=value['actor_state_dict']
        models.append(dict(name=name,actor_sha256=sha(folder/'actor.pt'),cpu_stage_mse=stage,optimizer_steps=steps))
    try:load_actor(parent.ACTOR)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Old 305-dimensional Actor incorrectly accepted')
    evaluations=[]
    for name in ['2026-10-22_bc_01_autonomous','2026-10-22_bc_01_takeover3200','2026-10-22_bc_02_autonomous']:
        folder=HERE/'runs'/name;require(folder)
        assert read(folder/'session.json')['status']=='completed'
        result=read(folder/'result.json');verification=read(folder/'verification.json')
        assert verification['independent_physics_verified']
        assert sum(c['observation_pass'] for c in verification['cases'])==result['observation_passes']
        for filename,digest in result['trace_hashes'].items():assert sha(folder/filename)==digest
        evaluations.append(dict(name=name,cases=len(result['cases']),passes=result['observation_passes'],
            autonomous=result['autonomous'],physics_samples=verification['physics_samples'],
            autonomous_gate=verification['autonomous_10s_gate'],verification_sha256=sha(folder/'verification.json')))
    paths=list(HERE.glob('*.py'))+list((HERE/'analysis').glob('*.py'))
    for p in paths:ast.parse(p.read_text())
    links=0
    for doc in [HERE/'README.md',HERE/'VALIDATION.md',parent.HERE/'README.md',parent.HERE/'NEXT_DIRECTION.md']:
        for dest in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' not in dest and not dest.startswith('#'):
                target=(doc.parent/dest.split('#')[0]).resolve()
                # 本检查生成的报告在所有检查结束后写入，不能要求它预先存在。
                if target != HERE/'validation/final_checks.json':
                    assert target.exists(),(doc,dest)
                links+=1
    report=dict(at=now(),legacy_execution_files_preserved=legacy,parent_sources_preserved=len(parent.sources()),
        legacy_actor_preserved=True,old_semantic_actor_rejected=True,python_files=len(paths),local_doc_links=links,
        datasets=dataset_checks,models=models,evaluations=evaluations,
        new_physics_samples=sum(r['physics_samples'] for r in evaluations),bc_optimizer_updates=12000,
        robot_ppo_updates=0,ppo_admission=False,analysis_sources={p.name:sha(p) for p in (HERE/'analysis').glob('*.py')})
    write(HERE/'validation/final_checks.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
