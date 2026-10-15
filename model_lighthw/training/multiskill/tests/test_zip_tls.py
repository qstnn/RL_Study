"""Exercise ZIP downloads against real TLS with a temporary organization-style CA."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import ssl
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


MESH = b"fixed-mesh-content-for-tls-test"
RELATIVE = "model_lighthw/lighthw_urdf/meshes/test.STL"


@pytest.fixture(scope="module")
def tls_server(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("TLS regression requires the openssl command")
    root = tmp_path_factory.mktemp("r6-tls-ca")

    def run(*args):
        subprocess.run([openssl, *args], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    for name in ("ca", "other-ca"):
        run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", f"{name}.key",
            "-out", f"{name}.pem", "-days", "2", "-subj", f"/CN=R6 temporary {name}",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign")
    run("req", "-newkey", "rsa:2048", "-nodes", "-keyout", "server.key",
        "-out", "server.csr", "-subj", "/CN=R6 test server")
    (root / "extensions.cnf").write_text(
        "subjectAltName=IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
        "-CAcreateserial", "-out", "server.pem", "-days", "2", "-extfile", "extensions.cnf")
    (root / "chain.pem").write_bytes((root / "server.pem").read_bytes() + (root / "ca.pem").read_bytes())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.requests += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.server.body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.body, server.requests = MESH, 0
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(root / "chain.pem", root / "server.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, root
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.fixture
def downloader(tmp_path, monkeypatch, tls_server):
    script = Path(__file__).resolve().parents[1] / "prepare_zip.py"
    spec = importlib.util.spec_from_file_location("_zip_tls_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"meshes": {RELATIVE: hashlib.sha256(MESH).hexdigest()}}))
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "ASSET_BASE", f"https://127.0.0.1:{tls_server[0].server_port}/")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    destination = tmp_path / RELATIVE
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-file")
    return module, destination


def test_untrusted_chain_reports_certificate_fix(downloader, capsys):
    module, destination = downloader
    assert module.main([]) == 1
    error = capsys.readouterr().err
    assert "self-signed certificate in certificate chain" in error
    assert "--ca-bundle" in error
    assert destination.read_bytes() == b"old-file"
    assert not list(destination.parent.glob(".mesh-*.part"))


def test_explicit_trusted_ca_downloads_mesh(downloader, tls_server):
    module, destination = downloader
    assert module.main(["--ca-bundle", str(tls_server[1] / "ca.pem")]) == 0
    assert destination.read_bytes() == MESH


def test_ssl_cert_file_remains_supported(downloader, tls_server, monkeypatch):
    module, destination = downloader
    monkeypatch.setenv("SSL_CERT_FILE", str(tls_server[1] / "ca.pem"))
    assert module.main([]) == 0
    assert destination.read_bytes() == MESH


def test_wrong_ca_is_rejected(downloader, tls_server, capsys):
    module, destination = downloader
    assert module.main(["--ca-bundle", str(tls_server[1] / "other-ca.pem")]) == 1
    assert "CERTIFICATE_VERIFY_FAILED" in capsys.readouterr().err
    assert destination.read_bytes() == b"old-file"


def test_hostname_is_still_verified(downloader, tls_server, monkeypatch, capsys):
    module, destination = downloader
    monkeypatch.setattr(module, "ASSET_BASE", f"https://localhost:{tls_server[0].server_port}/")
    assert module.main(["--ca-bundle", str(tls_server[1] / "ca.pem")]) == 1
    assert "Hostname mismatch" in capsys.readouterr().err
    assert destination.read_bytes() == b"old-file"


def test_corrupt_mesh_with_trusted_tls_does_not_replace_file(downloader, tls_server, monkeypatch, capsys):
    module, destination = downloader
    monkeypatch.setattr(tls_server[0], "body", b"wrong-mesh")
    assert module.main(["--ca-bundle", str(tls_server[1] / "ca.pem")]) == 1
    assert "checksum mismatch" in capsys.readouterr().err
    assert destination.read_bytes() == b"old-file"
    assert not list(destination.parent.glob(".mesh-*.part"))


def test_offline_check_needs_neither_ca_nor_network(downloader, tls_server):
    module, destination = downloader
    destination.write_bytes(MESH)
    before = tls_server[0].requests
    assert module.main(["--check", "--ca-bundle", "/missing-test-ca.pem"]) == 0
    assert tls_server[0].requests == before
