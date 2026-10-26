"""验收门槛回归：新增通过项不能抵消原任务退化，容忍一次但保留最佳合格模型。"""
import importlib.util
from pathlib import Path

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/quality.py'
 s=importlib.util.spec_from_file_location('v5_quality_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def evaluation(flags):
 return dict(pass_count=sum(flags),complete_count=len(flags),cases=[dict(case=dict(slope_deg=0,payload_kg=0,towed_kg=0,speed=i),success=ok) for i,ok in enumerate(flags)])

def test_same_count_different_failures_does_not_replace_good_model(tmp_path):
 m=module();checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'baseline')
 first=m.review(evaluation([True,False]),checkpoint,tmp_path/'quality',0)
 assert first['eligible'] and first['promoted']
 checkpoint.write_bytes(b'regressed')
 one=m.review(evaluation([False,True]),checkpoint,tmp_path/'quality',100)
 assert one['regressions'] and not one['promoted'] and not one['stop']
 two=m.review(evaluation([False,True]),checkpoint,tmp_path/'quality',200)
 assert two['stop']
 assert (tmp_path/'quality/best_eligible.pt').read_bytes()==b'baseline'

def test_recovery_clears_streak_and_new_capability_is_protected(tmp_path):
 m=module();p=tmp_path/'m.pt';p.write_bytes(b'm')
 m.review(evaluation([True,False]),p,tmp_path/'q',0)
 m.review(evaluation([False,False]),p,tmp_path/'q',100)
 recovered=m.review(evaluation([True,True]),p,tmp_path/'q',200)
 assert recovered['protected_count']==2 and not recovered['stop']
 once=m.review(evaluation([False,True]),p,tmp_path/'q',300)
 assert not once['stop']


def test_shorter_distance_cannot_replace_protected_one_meter_task(tmp_path):
 import pytest
 m=module();p=tmp_path/'m.pt';p.write_bytes(b'm')
 original=evaluation([True]);original['cases'][0]['case']['goal_distance_m']=1.
 m.review(original,p,tmp_path/'q',0)
 reduced=evaluation([True]);reduced['cases'][0]['case']['goal_distance_m']=.1
 with pytest.raises(ValueError,match='omitted protected'):
  m.review(reduced,p,tmp_path/'q',100)
