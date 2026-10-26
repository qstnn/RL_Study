"""Verify a returned review folder with Python 3 only; no project or simulator required."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent).resolve()
    data=json.loads((root/'manifest.json').read_text())
    failures=[]
    for relative, expected in data['sha256'].items():
        p=root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file():
            failures.append(relative); continue
        with p.open('rb') as stream:
            value=hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''): value.update(chunk)
        if value.hexdigest()!=expected: failures.append(relative)
    if failures:
        print('FAILED: '+', '.join(failures)); return 1
    print(f"PASSED: {len(data['sha256'])} files verified"); return 0


if __name__=='__main__':
    raise SystemExit(main())
