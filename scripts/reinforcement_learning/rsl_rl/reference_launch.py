"""Explicit formal-run pointer: startup runs cannot become the evaluation target."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def write_reference_run(log_dir,log_root,agent_cfg,env_cfg,*,task,status,runner=None):
    run=Path(log_dir).resolve()
    record=dict(task=task,run_dir=str(run),run_name=agent_cfg.run_name,
        role=agent_cfg.lighthw_reference_run_role,observation_version=agent_cfg.lighthw_observation_version,
        status=status,updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        planned_updates=agent_cfg.max_iterations,num_envs=env_cfg.scene.num_envs,seed=agent_cfg.seed,
        fresh_start=not agent_cfg.resume,save_interval=agent_cfg.save_interval)
    checkpoints=sorted(run.glob('model_[0-9]*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if checkpoints:record['latest_checkpoint']=str(checkpoints[-1])
    paths=[run/'reference_run.json']
    if record['role']=='continuation':
        from reference_continuation import SOURCE, validate_source, validate_completion
        import torch
        validate_source()
        if runner is None or not runner._continuation_restored or run==SOURCE.parent:
            raise ValueError('Continuation needs an independently recorded exact restoration')
        record['continuation']=runner._continuation
        record['completed_total_updates']=runner._curriculum_term.schedule.total_updates
        record['short_check']=bool(agent_cfg.lighthw_reference_short_check)
        if status=='completed':
            validate_completion(torch.load(record['latest_checkpoint'],map_location='cpu',weights_only=False),
                                runner._continuation)
        if not record['short_check']: paths.append(Path(log_root)/'latest_continuation_run.json')
    if record['role']=='formal':
        if not record['fresh_start']:raise ValueError('Formal reference pointer only accepts fresh initialization')
        paths.append(Path(log_root)/'latest_formal_run.json')
    for path in paths:
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(path)
    return record
