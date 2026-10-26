"""把原始训练与冻结评估日志画成可直接查看的离线图。"""
import json
import numpy as np

def plot_training(folder):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 curve=[];updates=[]
 for run in [folder/'training']+sorted(folder.glob('train_*')):
  if (run/'training_curve.jsonl').is_file():curve+=rows(run/'training_curve.jsonl')
  if (run/'updates.jsonl').is_file():updates+=rows(run/'updates.jsonl')
 if not curve:return
 fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True)
 x=[r['completed_updates'] for r in curve]
 for axis,key,title in [(axes[0],'mean_reward','Training reward (last 100 ended)'),(axes[1],'mean_episode_length','Episode length (steps)')]:
  axis.plot(x,[r[key] if r[key] is not None else np.nan for r in curve]);axis.set_ylabel(title)
 axes[1].axhline(300,color='gray',ls='--',label='6s timeout');axes[1].legend()
 axes[2].plot([r['completed_updates'] for r in updates],[r['rollout_KL'] for r in updates]);axes[2].set_ylabel('Rollout KL')
 quality=folder/'quality_history.json'
 if quality.is_file():
  q=json.loads(quality.read_text())
  for key in ('launch_rate','valid_flight_rate','qualified_rate','success_rate'):axes[3].plot([r['updates'] for r in q],[r[key] for r in q],marker='o',label=key)
  axes[3].legend(fontsize=8)
 axes[3].set_ylabel('Frozen rate');axes[3].set_xlabel('PPO updates')
 for a in axes:a.grid(alpha=.3)
 fig.tight_layout();fig.savefig(folder/'training_progress.png',dpi=130);plt.close(fig)
