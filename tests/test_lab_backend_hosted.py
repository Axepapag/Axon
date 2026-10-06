"""Exercise hosted routing without touching the live shared launcher."""
import http.server
import json

from fastapi.testclient import TestClient

from lab.backend.hosted import create_hosted_app


class ExistingSite(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.reply({"host": self.headers["Host"], "path": self.path})

    def do_POST(self):
        self.reply({"host": self.headers["Host"], "body":
                    self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()})

    def reply(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Set-Cookie", "first=1")
        self.send_header("Set-Cookie", "second=2")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_axon_public_and_local_hosts_use_real_api_and_deployed_frontend(tmp_path):
    deployed = tmp_path / "deployed"
    deployed.mkdir()
    (deployed / "index.html").write_text("<h1>Selected deployed Axon frontend</h1>", encoding="utf-8")
    app = create_hosted_app(frontend_path=deployed, state_path=tmp_path / "db", legacy_handler=ExistingSite)
    with TestClient(app) as client:
        for host in ("axon.gliksbot.com", "127.0.0.1:8080", "localhost:8080"):
            assert client.get("/api/v1/health", headers={"Host": host}).json()["status"] == "ok"
            assert "Selected deployed" in client.get("/", headers={"Host": host}).text
        assert client.get("/api/v1/readiness", headers={"Host": "axon.gliksbot.com"}).json()["training_authorized"] is False
        response = client.post("/api/v1/runs", json={"command_id": "probe"}, headers={"Host": "axon.gliksbot.com"})
        assert response.status_code == 503 and response.json()["code"] == "not_integrated"


def test_other_hosts_retain_original_host_path_body_and_response_headers(tmp_path):
    app = create_hosted_app(state_path=tmp_path / "db", legacy_handler=ExistingSite)
    with TestClient(app) as client:
        response = client.get("/chess/?example=1", headers={"Host": "gliksbot.com"})
        assert response.json() == {"host": "gliksbot.com", "path": "/chess/?example=1"}
        assert response.headers.get_list("set-cookie") == ["first=1", "second=2"]
        response = client.post("/Downloads/api/write", content='{"sample":1}',
                               headers={"Host": "gliksbot.com", "Content-Type": "application/json"})
        assert response.json() == {"host": "gliksbot.com", "body": '{"sample":1}'}
        legacy = client.get("/api/v1/health", headers={"Host": "plex.gliksbot.com"}).json()
        assert legacy == {"host": "plex.gliksbot.com", "path": "/api/v1/health"}
