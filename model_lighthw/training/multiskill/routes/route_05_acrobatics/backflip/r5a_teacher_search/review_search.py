"""Summarize measured search and independent full-episode evidence, never infer success."""
from pathlib import Path
import json,math,csv
from datetime import datetime
from zoneinfo import ZoneInfo

def main():
 root=Path(__file__).resolve().parent;results=root/'results';out=root/'review';out.mkdir(exist_ok=True)
 searches=[];full=[];curves={}
 for path in sorted(results.glob('cem_*/candidates.jsonl')):
  rows=[]
  for line in path.read_text().splitlines():
   try:rows.append(json.loads(line))
   except json.JSONDecodeError:continue  # A live run may still be writing its last line.
  safe=[r for r in rows if r['air_validated'] and not r['failed'] and r.get('physics_consistent') is True]
  searched_success=[r for r in rows if r['success'] and r.get('physics_consistent') is True]
  searches.append(dict(run=path.parent.name,candidates=len(rows),finished=(path.parent/'result.json').exists(),
   maximum_angle_deg=max((math.degrees(r['angle']) for r in rows),default=0),
   maximum_screened_nonfailed_angle_deg=max((math.degrees(r['angle']) for r in safe),default=0),
   qualified_first_landing=sum(r['qualified_landing'] for r in rows),
   screened_nonfailed=len(safe),short_horizon_success=len(searched_success)))
  grouped={}
  for r in rows:grouped.setdefault(r['generation'],[]).append(r)
  best=-float('inf');series=[]
  for gen,group in sorted(grouped.items()):
   r=max(group,key=lambda r:r['score'])
   if r['score']>best:best=r['score'];angle=math.degrees(r['angle'])
   series.append((gen+1,angle))
  curves[path.parent.name]=series
 for path in sorted(results.glob('*/result.json')):
  r=json.loads(path.read_text())
  if 'online_agrees' not in r:continue
  full.append(dict(run=path.parent.name,**r))
 now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
 summary=dict(updated=now,device='cpu',ppo_updates=0,searches=searches,full_probes=full,
  total_search_candidates=sum(x['candidates'] for x in searches),
  independently_verified_successes=[r['run'] for r in full if r['success'] and r['online_agrees'] and r['physics_consistent']],
  note='Search score and short-horizon success are not full teacher acceptance; fixed-condition CPU evidence only.')
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 with (out/'searches.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(searches[0]));writer.writeheader();writer.writerows(searches)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,6))
 for name,series in curves.items():
  if series:ax.plot(*zip(*series),label=name.replace('cem_','').replace('_01',''))
 ax.axhspan(350,385,alpha=.08,color='green');ax.axhline(350,color='green',linestyle='--',linewidth=.7)
 ax.set(xlabel='CEM generation',ylabel='First-flight angle of best-scoring candidate (deg)',title='Teacher search: angle alone does not establish a successful backflip')
 ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=3);fig.tight_layout();fig.savefig(out/'search_progress.png',dpi=130);plt.close(fig)
 text=['# R5A 教师搜索实时汇总','',f'更新时间：{now}。本机 CPU PhysX，新增 PPO 更新 0。','',
  f"累计候选 {summary['total_search_candidates']}；完整独立复核成功 {len(summary['independently_verified_successes'])}。",'',
  '搜索最长转角可以包含失败样本；仅用于探索，不能作为成功证明。短回合没有失败也不能代替完整6秒验收。','',
  '| 完整复核 | 首次腾空角度 | 腾空秒数 | 非轮接触等失败 | 合格首次落地 | 终末站稳秒数 | H筛查 | 完整成功 |',
  '|---|---:|---:|---|---|---:|---|---|']
 for r in full:text.append(f"| {r['run']} | {r['airborne_angle_deg']:.2f}° | {r['flight_s']:.3f} | {r['failed']} | {r['qualified_landing']} | {r['stable_s']:.3f} | {r['physics_consistent']} | {r['success']} |")
 ready_path=root.parent/'offline_sessions/route_05a/READY_GUIDED_V1.json'
 if ready_path.exists():
  ready=json.loads(ready_path.read_text());prepared=Path(ready['preparation']).parent/'preparation.json'
  if prepared.exists():
   prepared=json.loads(prepared.read_text());s=prepared.get('student_baseline',{})
   text+=['','## 交付与学生学习起点','',
    '已交付限幅教师355.225°、连续站稳4.527秒；独立命令重算最大误差3.73e-9，非轮接触峰值0。',
    f"完整准备流程学生基线：{s.get('airborne_angle_deg',0):.3f}°，成功={s.get('success')}。学生与教师分别判定，学生落地仍需PPO改善。",
    '离线安装和训练见 [r5a_guided_v1](../../model_lighthw/training/multiskill/routes/route_05_acrobatics/backflip/r5a_guided_v1/README.md)。正式机器人PPO更新0；目标GPU先复核教师，再训练。']
 text+=['','![搜索进度](search_progress.png)','','原始1kHz轨迹和源码快照位于每次 results 子目录。当前结论以本表完整复核列为准；CPU结果不构成GPU或实机验收。']
 (out/'REVIEW.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k not in ('searches','full_probes')},ensure_ascii=False))
if __name__=='__main__':main()
