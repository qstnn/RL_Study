"""退出清理的真实子进程回归，不启动仿真或训练。"""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
import psutil
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("_v6_launch_test",ROOT/"scripts/launch.py")
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
def test_cleanup_reclaims_grandchild_and_leaves_unrelated_process(tmp_path):
    marker=tmp_path/"child.pid"
    child_code="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
    parent_code="import subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c',"+repr(child_code)+"]);Path("+repr(str(marker))+").write_text(str(p.pid));time.sleep(30)"
    unrelated=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])
    parent=subprocess.Popen([sys.executable,"-c",parent_code])
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
        assert marker.exists();child=psutil.Process(int(marker.read_text()))
        launch.stop_owned_processes(parent,grace_s=.2)
        assert parent.poll() is not None
        assert not child.is_running() or child.status()==psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        if parent.poll() is None:launch.stop_owned_processes(parent,grace_s=.2)
        unrelated.terminate();unrelated.wait(timeout=3)
