"""旧技能样本占比实验的纯张量规则；不生成动作、不消费随机数。"""
from copy import deepcopy
import hashlib
import json

VERSION = "lighthw_retention_fraction_v1"
COURSE_VERSION = "lighthw_retention_fraction_flights_v1"


def retention_rule(fraction):
    if fraction == 0.5:
        return "even_environment_ids"
    if fraction == 0.75:
        return "environment_id_mod4_not3"
    raise ValueError("Only matched 50% or 75% retention is supported")


def retention_mask(ids, fraction):
    """返回保留组mask；75%保留全部旧偶数ID，增加模4余1的ID。"""
    retention_rule(fraction)
    return ids.remainder(2) == 0 if fraction == 0.5 else ids.remainder(4) != 3


def command_contract(original, fraction):
    if original.get("retention_fraction") != 0.5 or original.get("retention_rule") != "even_environment_ids":
        raise ValueError("Expected the original pinned 50% command contract")
    result = deepcopy(original)
    result.update(version=VERSION, retention_rule=retention_rule(fraction), retention_fraction=fraction)
    return result


def migrate_state(source, target_mapping, original_commands, fraction):
    """仅允许命令分组契约迁移；观测、腿轮映射、课程计数逐字段保留。"""
    digest = hashlib.sha256(json.dumps(source['action_mapping'], sort_keys=True).encode()).hexdigest()
    if digest != source['signature'] or source['action_mapping'].get('command_distribution') != original_commands:
        raise ValueError('Corrupt or foreign original action/command signature')
    target = deepcopy(source['action_mapping'])
    target['command_distribution'] = command_contract(original_commands, fraction)
    if target != target_mapping:
        raise ValueError('Unexpected observation or physical action change outside retention allocation')
    result = deepcopy(source)
    result['action_mapping'] = target
    result['signature'] = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
    return result
