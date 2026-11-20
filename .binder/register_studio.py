"""Register the personal launcher in the Python environment running Jupyter."""

import json
import shutil
import sys
import sysconfig
from pathlib import Path

if __name__ == "__main__":
    module = Path(__file__).with_name("dendrinet_binder.py")
    shutil.copyfile(module, Path(sysconfig.get_path("purelib")) / module.name)
    config = Path(sys.prefix) / "etc/jupyter/jupyter_server_config.d"
    config.mkdir(parents=True, exist_ok=True)
    settings = {"ServerApp": {"jpserver_extensions": {"dendrinet_binder": True}}}
    (config / "dendrinet_binder.json").write_text(json.dumps(settings) + "\n")
