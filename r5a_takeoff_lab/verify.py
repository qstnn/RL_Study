"""只读核对一个 USB 回收目录中的 SHA 清单。"""
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
checks=json.loads((root/'SHA256.json').read_text());bad=[]
for rel,expected in checks.items():
    p=(root/rel).resolve()
    if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:bad.append(rel)
if bad:raise SystemExit('FAILED: '+', '.join(bad))
print(f'PASS: {len(checks)} files')
