"""R1源码准入：其他路线的纯目录元数据不改变R1训练合同。"""
import ast
import hashlib
from pathlib import Path

CATALOG_RELATIVE = 'source/rl_training/rl_training/tasks/manager_based/locomotion/multiskill/lighthw/catalog.py'


def catalog_r1_sha256(source):
    """保留R1条目及其余全部代码，仅忽略可静态求值的其他路线目录值。"""
    tree = ast.parse(source)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ROUTES'
                           for target in node.targets)]
    if len(assignments) != 1 or len(assignments[0].targets) != 1:
        raise ValueError('Expected one plain ROUTES assignment')
    routes = assignments[0].value
    if not isinstance(routes, ast.Dict):
        raise ValueError('ROUTES must be a literal dictionary')
    selected = []; seen = set()
    for key, value in zip(routes.keys, routes.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in seen:
            raise ValueError('Expected unique literal route keys')
        seen.add(key.value)
        if key.value == '01':
            selected.append((key, value))
        else:
            # 不执行目录代码；调用、属性访问、动态解包等不享受忽略规则。
            ast.literal_eval(value)
    if len(selected) != 1:
        raise ValueError('Missing R1 route')
    routes.keys = [selected[0][0]]
    routes.values = [selected[0][1]]
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def validate_sources(repo, record, sha256):
    """整文件SHA优先；仅catalog可使用经过GPU快照核对的R1范围指纹。"""
    for relative, digest in record.get('validated_source_sha256', {}).items():
        path = Path(repo) / relative
        if path.is_file() and sha256(path) == digest:
            continue
        if relative == CATALOG_RELATIVE and path.is_file():
            expected = record.get('catalog_r1_sha256')
            if expected:
                try:
                    if catalog_r1_sha256(path.read_text()) == expected:
                        continue
                except (ValueError, SyntaxError, TypeError):
                    pass
        raise ValueError('Validated V8 training source changed: ' + relative)
