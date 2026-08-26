"""test_team_reconnect_identity.py — team client 重连保持身份不漂移。

ClientRegistry.register() 的三级判定：
  ① claimed_client_id 命中 → 续用
  ② (hostname, label) 指纹命中 → 续用
  ③ 都没命中 → 发新 uuid

新 register API 行为同时验证 CLI 带参 connect 重连时透传 client_id 的端
到端链路。
"""
from __future__ import annotations


from fastapi import FastAPI
from fastapi.testclient import TestClient

from xskill.team.server import api as server_api
from xskill.team.server.client_registry import ClientRegistry
from xskill.team.shared.protocol import RegisterRequest


# ─────────────────────── ClientRegistry 单元 ───────────────────────


def test_register_reuses_when_claimed_client_id_known(tmp_path):
    """优先级 ①：自报 client_id 存在 → 续用同一个 ID。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    cid = reg.register(label="alice-laptop", hostname="alice")
    # 二次注册，自报上次拿到的 ID → 应原样返回（即使 hostname 变了）
    same = reg.register(
        label="alice-laptop", hostname="changed-host",
        claimed_client_id=cid,
    )
    assert same == cid
    # 也只有一行 client（没新增）
    assert len(reg.list()) == 1


def test_register_reuses_by_fingerprint_when_state_lost(tmp_path):
    """优先级 ②：state 文件丢失（无 claimed_client_id），但 (hostname,
    label) 指纹能查到历史身份 → 续用。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    cid = reg.register(label="alice-laptop", hostname="alice")
    # 同 hostname/label 再次注册，模拟客户端删了 team_client.json 重连
    same = reg.register(label="alice-laptop", hostname="alice")
    assert same == cid
    assert len(reg.list()) == 1


def test_register_issues_new_uuid_on_hostname_mismatch(tmp_path):
    """优先级 ③：hostname 不同 → 不属于同台机器，发新 uuid。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    a = reg.register(label="alice-laptop", hostname="alice")
    b = reg.register(label="alice-laptop", hostname="bob")
    assert a != b
    assert len(reg.list()) == 2


def test_register_issues_new_uuid_when_all_empty(tmp_path):
    """无 hostname、无 label、无 claimed_client_id → 每次发新 uuid，
    避免匿名 client 互相匹配上历史空记录。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    a = reg.register()
    b = reg.register()
    assert a != b
    assert len(reg.list()) == 2


def test_register_unknown_claimed_id_falls_through_to_fingerprint(tmp_path):
    """优先级 ① 未命中（claimed_id server 不认）→ 进 ② 指纹回查。
    既能续用原 ID，又不会因为 stale claimed_id 让客户端永远卡住。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    cid = reg.register(label="lab", hostname="host1")
    # claimed 是某个不存在的 ID（比如 DB 重置过 client 误持有了旧 ID）
    same = reg.register(
        label="lab", hostname="host1",
        claimed_client_id="ghost-id-not-in-db",
    )
    assert same == cid


def test_register_unknown_claimed_id_no_fingerprint_match_makes_new(tmp_path):
    """① 未命中、② 也未命中（label/hostname 都没历史记录）→ ③ 发新。"""
    reg = ClientRegistry(tmp_path / "clients.db")
    new = reg.register(
        label="lab", hostname="host1",
        claimed_client_id="ghost-id-not-in-db",
    )
    assert reg.exists(new)
    assert new != "ghost-id-not-in-db"


# ─────────────────────── HTTP API 端到端 ───────────────────────


def _make_client(tmp_path) -> TestClient:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    traj_root = tmp_path / "team_traj"
    reg = ClientRegistry(tmp_path / "clients.db")
    server_api.init_team_context(
        join_token="secret-token",
        client_registry=reg,
        skill_dir=skill_dir,
        traj_root=traj_root,
        register_dir=lambda path, label: None,
    )
    app = FastAPI()
    app.include_router(server_api.router)
    return TestClient(app)


def test_api_register_preserves_id_when_client_self_reports(tmp_path):
    """端到端：HTTP POST /register 带 claimed_client_id → server 续用。
    覆盖 ``xskill connect <addr> --token`` 带参重连不漂移场景。"""
    client = _make_client(tmp_path)
    r = client.post("/api/v1/team/register", json={
        "token": "secret-token", "client_label": "alice", "hostname": "a",
    })
    assert r.status_code == 200
    cid = r.json()["client_id"]
    # 第二次带 claimed_client_id 重连
    r2 = client.post("/api/v1/team/register", json={
        "token": "secret-token", "client_label": "alice", "hostname": "a",
        "claimed_client_id": cid,
    })
    assert r2.status_code == 200
    assert r2.json()["client_id"] == cid


def test_api_register_preserves_id_via_fingerprint(tmp_path):
    """端到端：state 文件丢失（不带 claimed_client_id），server 按
    (hostname, label) 指纹续用。"""
    client = _make_client(tmp_path)
    r = client.post("/api/v1/team/register", json={
        "token": "secret-token", "client_label": "alice", "hostname": "a",
    })
    cid = r.json()["client_id"]
    r2 = client.post("/api/v1/team/register", json={
        "token": "secret-token", "client_label": "alice", "hostname": "a",
    })
    assert r2.json()["client_id"] == cid


def test_api_register_protocol_accepts_missing_field(tmp_path):
    """老 client 不带 claimed_client_id 也能注册——字段 optional。"""
    # 协议层验证：不传该字段也能 build
    req = RegisterRequest(token="t", client_label="l", hostname="h")
    assert req.claimed_client_id is None
    # API 层验证：老 body schema 仍然 work
    client = _make_client(tmp_path)
    r = client.post("/api/v1/team/register", json={
        "token": "secret-token", "client_label": "l", "hostname": "h",
    })
    assert r.status_code == 200


# ─────────────────────── client daemon ───────────────────────


class _FakeHttp:
    """记录 register POST body 的最小 fake。"""

    def __init__(self):
        self.last_body: dict | None = None

    def post(self, url, json):  # noqa: A002  (shadow built-in OK in test)
        self.last_body = json

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"client_id": "issued-cid"}
        return _Resp()


def test_register_with_server_sends_claimed_client_id():
    """daemon.register_with_server 把 existing_client_id 透传成
    body.claimed_client_id。"""
    from xskill.team.client.daemon import register_with_server
    fake = _FakeHttp()
    cid = register_with_server(
        fake, token="t", label="l", hostname="h",
        existing_client_id="existing-cid",
    )
    assert cid == "issued-cid"
    assert fake.last_body["claimed_client_id"] == "existing-cid"


def test_register_with_server_omits_claimed_id_by_default():
    """不传 existing_client_id 时 body.claimed_client_id 为 None
    （保持向后兼容老 server，server 老逻辑应 ignore None）。"""
    from xskill.team.client.daemon import register_with_server
    fake = _FakeHttp()
    register_with_server(fake, token="t", label="l", hostname="h")
    assert fake.last_body["claimed_client_id"] is None


# ─────────────────────── CLI 带参 connect ───────────────────────


def test_cli_connect_with_address_reads_existing_state(tmp_path, monkeypatch):
    """``xskill connect <addr> --token`` 时如果本地 state 文件存在，
    应读出 client_id 一起发给 server。验证身份不漂移端到端。"""
    from xskill.config import XSKILL_HOME  # noqa: F401  (确保 import 可用)
    from xskill.team.client.state import ClientState, save_client_state
    from xskill.cli import build_parser, cmd_connect

    # 先在 tmp 写一份 state 文件（模拟历史已建立连接）
    state_path = tmp_path / "team_client.json"
    save_client_state(
        ClientState(server_url="http://old:8000",
                    client_id="historic-cid",
                    join_token="old-token"),
        state_path,
    )
    monkeypatch.setattr("xskill.config.get_team_client_state_path",
                        lambda: state_path)

    # 拦 httpx.Client → 用 _FakeHttp 记录 register body
    fake = _FakeHttp()

    def _fake_client(*a, **kw):
        return fake
    monkeypatch.setattr("httpx.Client", _fake_client)
    monkeypatch.setattr(
        "xskill.ecosystems.bundled_guide.install_bundled_xskill_guide",
        lambda target_root=None: [],
    )

    # 跑到 run_forever 之前会构造 TeamClient——也得拦下，否则会去拉
    # collector 等真实组件。直接让 TeamClient.run_forever 立即返回。
    monkeypatch.setattr(
        "xskill.team.client.daemon.TeamClient.run_forever",
        lambda self: None,
    )
    # collector 启动也要 mock 掉
    monkeypatch.setattr(
        "xskill.team.client.daemon.TeamClient.__init__",
        lambda self, **kw: None,
    )

    parser = build_parser()
    # --foreground：直接跑（被 mock 的）守护循环，验证握手 body，不走后台任务托管
    args = parser.parse_args(["connect", "1.2.3.4:8000", "--token", "new-tok",
                              "--label", "alice", "--foreground"])
    rc = cmd_connect(args)
    assert rc == 0
    # body 必须把历史 client_id 当 claimed_client_id 带过去
    assert fake.last_body is not None
    assert fake.last_body["claimed_client_id"] == "historic-cid"
    assert fake.last_body["token"] == "new-tok"


def test_cli_connect_with_address_no_state_sends_null_claimed_id(
    tmp_path, monkeypatch,
):
    """``xskill connect <addr> --token`` 且本地 state 不存在 →
    body.claimed_client_id 应为 None（首次握手）。"""
    from xskill.team.client.state import ClientState  # noqa: F401
    from xskill.cli import build_parser, cmd_connect

    state_path = tmp_path / "absent_team_client.json"  # 不存在
    monkeypatch.setattr("xskill.config.get_team_client_state_path",
                        lambda: state_path)

    fake = _FakeHttp()
    monkeypatch.setattr("httpx.Client", lambda *a, **kw: fake)
    monkeypatch.setattr(
        "xskill.ecosystems.bundled_guide.install_bundled_xskill_guide",
        lambda target_root=None: [],
    )
    monkeypatch.setattr(
        "xskill.team.client.daemon.TeamClient.run_forever",
        lambda self: None,
    )
    monkeypatch.setattr(
        "xskill.team.client.daemon.TeamClient.__init__",
        lambda self, **kw: None,
    )

    parser = build_parser()
    args = parser.parse_args(["connect", "1.2.3.4:8000", "--token", "tok",
                              "--foreground"])
    rc = cmd_connect(args)
    assert rc == 0
    assert fake.last_body["claimed_client_id"] is None
