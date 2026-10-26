"""汇总已审计的冻结校准，不用短测替代完整行为门槛。"""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from probe import V3,REPO,SOURCES,sha256,write_json,now

PATH_FIELDS={"/scene/robot/spawn/usd_dir","/scene/trailer/spawn/asset_path","/scene/trailer/spawn/usd_dir"}
def difference(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        return [d for k in sorted(a.keys()|b.keys()) for d in difference(a.get(k),b.get(k),path+"/"+str(k))]
    return [] if a==b else [dict(path=path,old=a,new=b)]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeat",type=Path)
    parser.add_argument("--route38-repeat",type=Path)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    base=V3/"validation"
    experiments=[("original",base/"transfer_review_02"),("original_d1",base/"transfer_damping1_01"),("wheelcap_d3",base/"calibration_wheelcap_full_01"),("wheelcap_d1",base/"calibration_wheelcap_d1_full_01")]
    if args.repeat:experiments.append(("wheelcap_d3_independent",args.repeat.resolve()))
    if args.route38_repeat:experiments.append(("wheelcap_d1_independent",args.route38_repeat.resolve()))
    configs={};summary=[];all_rows=[]
    for label,root in experiments:
        state=json.loads((root/"summary.json").read_text());assert state["status"]=="completed"
        configs[label]=yaml.load((root/"params/env.yaml").read_text(),Loader=yaml.BaseLoader)
        audit=json.loads((root/"audit.json").read_text());assert audit["status"]=="passed"
        for item in state["candidates"]:
            name=item["candidate"];folder=root/name;e=json.loads((folder/"evaluation.json").read_text())
            assert e["sha256"]==SOURCES[name]["sha256"]==sha256(SOURCES[name]["checkpoint"])
            row=dict(variant=label,candidate=name,complete=e["complete_count"],passed=e["pass_count"],teacher_passed=item["teacher_pre_screen_passed"],training_ready=False)
            for idx,title in [(0,"flat_zero"),(1,"flat_fast"),(7,"up_fast"),(13,"down_fast"),(10,"tow40_up_zero")]:
                c=e["cases"][idx]
                row[title]=dict(complete=c["complete"],success=c["success"],duration_s=c["observed"]["duration_s"],termination=c["observed"]["termination_causes"],failed_checks=c["failed_checks"],metrics=c.get("metrics"))
            row["payload15_complete"]=sum(c["complete"] for c in e["cases"] if c["case"]["payload_kg"]==15)
            row["payload15_passed"]=sum(c["success"] for c in e["cases"] if c["case"]["payload_kg"]==15)
            summary.append(row)
            for i,c in enumerate(e["cases"]):
                z=np.load(folder/f"seed43_case{i:02d}_motor_trace.npz")
                with (folder/f"seed43_case{i:02d}.csv").open() as stream:series=list(csv.DictReader(stream))
                all_rows.append(dict(variant=label,candidate=name,index=i,**c["case"],duration_s=c["observed"]["duration_s"],complete=c["complete"],success=c["success"],termination="|".join(c["observed"]["termination_causes"]),failed_checks="|".join(c["failed_checks"]),**c.get("metrics",{}),last_vx=float(series[-1]["vx"]),last_tilt_deg=float(series[-1]["tilt_deg"]),last_hitch_clearance_m=float(series[-1]["hitch_clearance_m"]),last_command=float(series[-1]["command_vx"]),leg_peak_Nm=float(np.abs(z["applied"][:,:12]).max()),wheel_peak_Nm=float(np.abs(z["applied"][:,12:]).max()),leg_clipped_fraction=float((np.abs(z["requested"][:,:12])>z["limits"][:12]+1e-4).mean()),wheel_clipped_fraction=float((np.abs(z["requested"][:,12:])>z["limits"][12:]+1e-4).mean())))
    diffs={}
    pairs=[("original_d1","wheelcap_d1",{"/scene/robot/spawn/func"}),("original","wheelcap_d3",{"/scene/robot/spawn/func"}),("wheelcap_d3","wheelcap_d1",{"/scene/robot/actuators/wheel/damping"})]
    if args.repeat:pairs.append(("wheelcap_d3","wheelcap_d3_independent",set()))
    if args.route38_repeat:pairs.append(("wheelcap_d1","wheelcap_d1_independent",set()))
    for old,new,allowed in pairs:
        delta=difference(configs[old],configs[new]);effective=[d for d in delta if d["path"] not in PATH_FIELDS]
        assert {d["path"] for d in effective}==allowed,(old,new,effective)
        diffs[old+"_to_"+new]=delta
    write_json(out/"configuration_deltas.json",diffs)
    fields=list(dict.fromkeys(k for r in all_rows for k in r))
    with (out/"case_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    report=dict(created_at=now(),ppo_updates=0,source_checkpoints_unchanged=True,experiments=[dict(variant=l,path=str(r),audit=str(r/"audit.json")) for l,r in experiments],candidates=summary,numerical_fix=dict(attribute="physxRigidBody:maxAngularVelocity",unit="deg/s",wheel_before=1000,wheel_after=3000,other_bodies=1000,nominal_wheel_deg_s_at3mps=3/.0875*180/math.pi),training_ready=False,limitations=["Frozen teacher uses privileged body velocity; deployable adapter not implemented.","Fixed initial states and seed43 screen, not randomized robustness acceptance.","No final full-task or minimum motor-sizing acceptance."])
    write_json(out/"review.json",report)
    fig,axes=plt.subplots(3,3,figsize=(13.5,9),sharex="col")
    labels={"original":"Original cap1000, D3","original_d1":"Original cap1000, D1","wheelcap_d3":"Cap3000, D3","wheelcap_d1":"Cap3000, D1","wheelcap_d3_independent":"Cap3000, D3 repeat","wheelcap_d1_independent":"Cap3000, D1 independent"}
    colors=["#b84442","#bd7b36","#24779b","#318748","#8657b5","#7e7685"]
    for k,(variant,root) in enumerate(experiments):
        if not (root/"route53").exists():continue
        for col,(idx,title) in enumerate([(1,"Unloaded flat: accelerate, brake, hold"),(15,"15 kg, -25 deg: move and hold"),(10,"40 kg trailer, +25 deg: zero command")]):
            data=np.genfromtxt(root/"route53"/f"seed43_case{idx:02d}.csv",delimiter=",",names=True)
            for row,key in enumerate(["vx","tilt_deg","hitch_clearance_m" if idx==10 else "stand_drift_m"]):
                axes[row,col].plot(data["t"],data[key],color=colors[k],label=labels[variant],lw=1.3,alpha=.9)
                axes[row,col].grid(alpha=.2)
            axes[0,col].set_title(title,fontsize=10)
            if variant=="wheelcap_d1":axes[0,col].plot(data["t"],data["command_vx"],"k--",label="Command",lw=1)
            axes[2,col].set_xlabel("Time (s)")
    axes[0,0].set_ylabel("Route vx (m/s)");axes[1,0].set_ylabel("Surface tilt (deg)");axes[2,0].set_ylabel("Zero-command drift (m)")
    axes[2,2].set_ylabel("Hitch clearance (m)");axes[2,2].axhline(.005,c="black",ls=":",lw=1)
    for col in range(3):axes[1,col].axhline(45,c="black",ls=":",lw=1)
    handles,lab=axes[0,0].get_legend_handles_labels();fig.legend(handles,lab,loc="lower center",ncol=3,fontsize=9)
    fig.suptitle("Route53 frozen calibration | valid samples only | no PPO updates",fontsize=13)
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(out/"calibration_comparison.png",dpi=160);plt.close(fig)
    print(json.dumps(dict(report=str(out/"review.json"),cases=len(all_rows),configuration_deltas_verified=True)))
if __name__=="__main__":main()
