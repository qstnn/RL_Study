"""Verify every shipped file before preparing a copied USB bundle."""
from pathlib import Path
import hashlib
import json
import sys
import subprocess

root=Path(__file__).resolve().parent.parent
manifest=root/'offline/bundle_manifest.json'
if not manifest.is_file():
    raise SystemExit('bundle_manifest.json missing; use the complete USB ZIP')
data=json.loads(manifest.read_text())
errors=[]
for relative, expected in data['sha256'].items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        errors.append(relative + ': missing or invalid path')
        continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    if digest.hexdigest()!=expected: errors.append(relative + ': checksum mismatch')
if errors:
    print('\n'.join(errors), file=sys.stderr)
    raise SystemExit(1)
subprocess.run([sys.executable, '-I', str(root/'offline/source_integrity.py')], check=True)
print(f"[R6 USB] Bundle verified: {len(data['sha256'])} files")
