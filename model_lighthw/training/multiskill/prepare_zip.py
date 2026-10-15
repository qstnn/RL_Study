"""为 GitHub ZIP 补齐 LightHW 网格；只用 Python 标准库，不调用 Git 或仿真。"""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MANIFEST = ROOT / "records/portable_assets.json"
# 固定到已发布资产版本，避免分支更新时混用网格；仍逐个校验清单中的 SHA256。
ASSET_COMMIT = "977cdead4e2ae7f6c475afedea014a3544eb83ea"
ASSET_BASE = f"https://media.githubusercontent.com/media/qstnn/RL_Study/{ASSET_COMMIT}/"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mesh(relative, destination, expected, *, ca_bundle=None):
    """流式写入临时文件，完整哈希一致后才替换目标；失败保留原文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(ASSET_BASE + quote(relative, safe="/"),
                      headers={"User-Agent": "LightHW-R6-ZIP/1"})
    # 默认仍遵循 Python/SSL_CERT_FILE；显式 CA 只作用于本次下载，不修改系统信任库。
    context = ssl.create_default_context(cafile=str(ca_bundle.expanduser()) if ca_bundle else None)
    temporary = None
    try:
        digest = hashlib.sha256()
        with urlopen(request, timeout=30, context=context) as response:
            if response.status != 200:
                raise ValueError(f"Expected a full mesh download, got HTTP {response.status}: {relative}")
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".mesh-", suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Downloaded mesh checksum mismatch: {relative}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验本地网格，不联网、不写文件")
    parser.add_argument("--ca-bundle", type=Path,
                        help="PEM格式可信CA文件，例如系统证书库或单位提供的CA；保持TLS验证")
    args = parser.parse_args(argv)
    try:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["meshes"]
        pending = []
        for relative, digest in expected.items():
            destination = (REPO / relative).resolve()
            if not destination.is_relative_to(REPO) or not relative.startswith("model_lighthw/lighthw_urdf/meshes/"):
                raise ValueError(f"Unexpected mesh path: {relative}")
            if not destination.is_file() or sha256(destination) != digest:
                pending.append((relative, destination, digest))
        print(f"Verified meshes: {len(expected) - len(pending)}/{len(expected)}", flush=True)
        if args.check and pending:
            for relative, _, _ in pending:
                print(f"Missing, corrupt or Git LFS pointer: {relative}", file=sys.stderr)
            print("Run this script without --check to download the required meshes over HTTPS.", file=sys.stderr)
            return 1
        for index, (relative, destination, digest) in enumerate(pending, 1):
            print(f"Downloading {index}/{len(pending)}: {destination.name}", flush=True)
            download_mesh(relative, destination, digest, ca_bundle=args.ca_bundle)
            print(f"SHA256 verified: {destination.name}", flush=True)
        print(f"All {len(expected)} meshes match portable_assets.json. Run R6 check/smoke next.", flush=True)
        return 0
    except (OSError, ValueError) as error:
        print(f"Asset preparation failed: {error}", file=sys.stderr)
        reason = error.reason if isinstance(error, URLError) else error
        if isinstance(reason, ssl.SSLCertVerificationError):
            print("TLS certificate verification failed; this can happen with organization HTTPS inspection.", file=sys.stderr)
            print("Ubuntu: retry with --ca-bundle /etc/ssl/certs/ca-certificates.crt", file=sys.stderr)
            print("If needed, use the CA bundle supplied by your organization: --ca-bundle /path/to/company-ca.pem", file=sys.stderr)
            print("Existing scripts also accept SSL_CERT_FILE=/path/to/trusted-ca.pem for this command.", file=sys.stderr)
        elif args.ca_bundle and not args.ca_bundle.expanduser().is_file():
            print("The --ca-bundle file does not exist; use an existing trusted PEM certificate bundle.", file=sys.stderr)
        else:
            print("Check HTTPS access to media.githubusercontent.com and rerun; verified meshes are skipped.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun to continue with the remaining meshes.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
