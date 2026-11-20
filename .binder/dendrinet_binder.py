"""Open the local Studio belonging to this personal Jupyter server."""

import threading
from pathlib import Path

from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import JupyterHandler
from jupyter_server.extension.application import ExtensionApp
from jupyter_server.utils import url_path_join
from tornado import web


class OpenStudio(JupyterHandler):
    def initialize(self, studio):
        self.studio = studio

    @web.authenticated
    @authorized(action="execute", resource="kernels")
    def get(self):
        server = self.studio.start_for(self.request.host_name)
        # The fragment reaches the editor. It is never sent to the proxy as a query.
        base = url_path_join(self.base_url, "proxy", str(server.server_port))
        self.set_header("Cache-Control", "no-store")
        self.set_header("Referrer-Policy", "no-referrer")
        self.redirect(f"{base}/#{server.studio_token}")


class StudioSession(ExtensionApp):
    name = "dendrinet_binder"
    server = None

    def initialize_handlers(self):
        self.handlers.append((r"/open-studio/?", OpenStudio, {"studio": self}))

    def start_for(self, host):
        if self.server is None:
            from dendrinet_studio.server import make_server

            root = Path(self.serverapp.root_dir) / "studio-runs"
            self.server = make_server(root, port=0, threads=1, allowed_hosts=(host,))
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server

    async def stop_extension(self):
        if self.server is not None:
            self.server.studio_runs.close()
            self.server.shutdown()
            self.server.server_close()


def _jupyter_server_extension_points():
    return [{"module": __name__, "app": StudioSession}]
