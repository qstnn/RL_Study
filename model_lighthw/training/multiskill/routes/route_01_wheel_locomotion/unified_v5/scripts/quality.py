"""冻结行为验收门槛；按原已通过工况保护能力，不用总reward抵消某一任务丢失。"""
from pathlib import Path
import json
import shutil


class QualityRegression(RuntimeError):
    """连续冻结评估丢失受保护工况；保留证据并停止本次更新预算。"""


def case_key(case):
    return '|'.join(str(float(case[k])) for k in ('slope_deg','payload_kg','towed_kg','speed'))+'|'+str(float(case.get('goal_distance_m',1.)))


def review(evaluation, checkpoint, directory, completed_updates, *, patience=2):
    """保存基准、最近结果与最佳合格模型；同一工况连续两次退化才触发停止。

    初始化建立保护集合；后续新的通过项也纳入保护。只有全部受保护项合格才能晋升best。
    相同检查点无需更新训练状态；早期失败与完整停车失败都按原协议success处理。
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'quality_state.json'
    cases=evaluation['cases']
    # 多seed时所有seed通过才算该工况通过，避免覆盖同名工况掩盖失败。
    grouped={}
    for result in cases:
        key=case_key(result['case']);grouped.setdefault(key,[]).append(result)
    passing={key for key,rows in grouped.items() if all(row['success'] for row in rows)}
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(protected=sorted(passing),streaks={},best_score=None,baseline_updates=completed_updates)
    missing=set(state['protected'])-set(grouped)
    if missing:raise ValueError('Evaluation omitted protected conditions: '+repr(sorted(missing)))
    regressions=sorted(set(state['protected'])-passing)
    for key in state['protected']:
        state['streaks'][key]=state['streaks'].get(key,0)+1 if key in regressions else 0
    eligible=not regressions
    score=[len(passing),int(evaluation['complete_count'])]
    promoted=False
    if eligible and (state['best_score'] is None or score>=state['best_score']):
        target=directory/'best_eligible.pt';shutil.copyfile(checkpoint,target)
        state.update(best_score=score,best_checkpoint=str(target.resolve()),best_updates=completed_updates)
        state['protected']=sorted(set(state['protected'])|passing)
        promoted=True
    stop=any(state['streaks'].get(key,0)>=patience for key in regressions)
    record=dict(completed_updates=completed_updates,pass_count=evaluation['pass_count'],complete_count=evaluation['complete_count'],
                protected_count=len(state['protected']),regressions=regressions,eligible=eligible,promoted=promoted,
                stop=stop,patience=patience,best_checkpoint=state.get('best_checkpoint'))
    state.update(last=record)
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    with (directory/'quality_history.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    return record
