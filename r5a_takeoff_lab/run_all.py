"""离线复现：三个种子冻结探针和一段录像，统一保存 USB 回收目录。"""
import argparse
from datetime import datetime
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[43,44,45])
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    profiles=json.loads((HERE/'profiles.json').read_text())
    required=['r5a.sh','r6.sh','offline/routes.json','source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/r5a_highjump/env.py']
    for name in required:
        if not (ROOT/name).is_file():raise RuntimeError('需要已经安装 highjump v3 的 R5A 工程: '+name)
    if a.check:
        subprocess.run([sys.executable,str(HERE/'test_control.py')],check=True)
        print('离线探针静态入口通过；不代表仿真或空翻通过。');return
    folder=ROOT/'usb_review/route_05a'/('takeoff_review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'profiles.json').write_text(json.dumps(profiles,ensure_ascii=False,indent=2))
    status={'status':'running','ppo_updates':0,'cases':[]}
    try:
        for seed in a.seeds:
            dest=folder/f'seed{seed}'
            subprocess.run([sys.executable,str(HERE/'run.py'),'--profiles',str(folder/'profiles.json'),'--run-dir',str(dest),'--seed',str(seed),'--seconds','4'],check=True)
            status['cases']+=json.loads((dest/'analysis.json').read_text())['cases']
        if not a.no_video:
            selected=[x for x in profiles if x['name']=='vertical_deep']
            (folder/'video_profile.json').write_text(json.dumps(selected,indent=2))
            subprocess.run([sys.executable,str(HERE/'run.py'),'--profiles',str(folder/'video_profile.json'),'--run-dir',str(folder/'video'),'--seed',str(a.seeds[0]),'--seconds','4','--video'],check=True)
        status['status']='completed'
    except BaseException:
        status['status']='failed';status['error']=traceback.format_exc();raise
    finally:
        (folder/'summary.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
        checks={str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='SHA256.json'}
        (folder/'SHA256.json').write_text(json.dumps(checks,indent=2)+'\n')
        (folder.parent/'LATEST_TAKEOFF_REVIEW.txt').write_text(str(folder.relative_to(ROOT))+'\n')
        print('[USB REVIEW] '+str(folder),flush=True)

if __name__=='__main__':main()
