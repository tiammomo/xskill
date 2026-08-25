"""catalog_vector_dirty 增量同步、generation fence 与低频修复。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from xskill.pipeline import registry as registry_mod
from xskill.pipeline.registry import get_connection, pooled_connection
from xskill.recommend.heavy_worker import (
    VECTOR_RECONCILE_INTERVAL_SECONDS,
    run_vector_sync,
)
from xskill.recommend.skill_vector_store import (
    DEFAULT_DIM,
    MemorySkillVectorIndex,
    MilvusLiteSkillVectorIndex,
    content_sha_for_text,
    fake_embed,
)
from xskill.recommend.vector_dirty import (
    clear_catalog_vector_dirty,
    list_catalog_vector_dirty,
    mark_catalog_vector_dirty_on_connection,
)
from xskill.skill.catalog_store import (
    delete_native_skill,
    rename_native_skill,
    upsert_native_skill,
)


class CountingIndex(MemorySkillVectorIndex):
    def __init__(self) -> None:
        super().__init__(dim=DEFAULT_DIM)
        self.calls = {"upsert": 0, "delete": 0, "get": 0, "list_keys": 0}

    def reset_calls(self) -> None:
        for key in self.calls:
            self.calls[key] = 0

    def upsert(self, *args, **kwargs) -> None:
        self.calls["upsert"] += 1
        super().upsert(*args, **kwargs)

    def delete(self, *args, **kwargs) -> None:
        self.calls["delete"] += 1
        super().delete(*args, **kwargs)

    def get(self, *args, **kwargs):
        self.calls["get"] += 1
        return super().get(*args, **kwargs)

    def list_keys(self) -> set[str]:
        self.calls["list_keys"] += 1
        return super().list_keys()


@pytest.fixture()
def registry_db(tmp_path: Path) -> Path:
    db = tmp_path / "registry.db"
    get_connection(db).close()
    return db


def _catalog_row(key: str, description: str, *, name: str | None = None) -> dict:
    return {
        "catalog_key": key,
        "name": name or key.split(":", 1)[-1],
        "source": "native",
        "description": description,
        "content_sha": content_sha_for_text(description),
        "distributable": 1,
    }


def _store_catalog(db: Path, row: dict, *, mark: bool = True) -> None:
    with pooled_connection(db) as conn:
        conn.execute(
            """
            INSERT INTO skills_catalog(
                catalog_key, name, source, state, description, distributable,
                content_sha
            ) VALUES (?, ?, ?, 'main', ?, ?, ?)
            ON CONFLICT(catalog_key) DO UPDATE SET
                name=excluded.name,
                source=excluded.source,
                description=excluded.description,
                distributable=excluded.distributable,
                content_sha=excluded.content_sha
            """,
            (
                row["catalog_key"], row["name"], row["source"],
                row["description"], row["distributable"], row["content_sha"],
            ),
        )
        if mark:
            mark_catalog_vector_dirty_on_connection(
                conn,
                row["catalog_key"],
                operation="upsert",
                content_sha=row["content_sha"],
            )
        conn.commit()


def _sync(db: Path, index, *, model: str = "model-a", now: float = 100.0, embed=None):
    return run_vector_sync(
        db_path=db,
        index=index,
        embed=embed or (lambda text: fake_embed(text, DEFAULT_DIM)),
        model_fingerprint=f"test:{model}:{DEFAULT_DIM}",
        now=now,
    )


def _write_native_skill(root: Path, name: str, description: str) -> Path:
    skill = root / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n",
        encoding="utf-8",
    )
    ref = skill / ".git" / "refs" / "heads" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("a" * 40, encoding="utf-8")
    return skill


@pytest.mark.performance_contract
def test_idle_tick_does_not_scan_vector_index(registry_db):
    index = CountingIndex()
    _store_catalog(registry_db, _catalog_row("native:alpha", "alpha"))
    first = _sync(registry_db, index)
    assert first["mode"] == "full"
    assert index.get("native:alpha") is not None

    index.reset_calls()
    embeds = []
    second = _sync(
        registry_db,
        index,
        now=101,
        embed=lambda text: embeds.append(text) or fake_embed(text, DEFAULT_DIM),
    )
    assert second == {
        "upserted": 0, "deleted": 0, "skipped": 0, "deferred": 0,
        "mode": "incremental", "reason": "",
    }
    assert index.calls == {"upsert": 0, "delete": 0, "get": 0, "list_keys": 0}
    assert embeds == []


def test_upgrade_bootstrap_reuses_matching_legacy_vectors(registry_db):
    index = CountingIndex()
    row = _catalog_row("native:alpha", "alpha")
    _store_catalog(registry_db, row, mark=False)
    index.upsert(
        row["catalog_key"],
        fake_embed(row["description"]),
        content_sha=row["content_sha"],
        source=row["source"],
        name=row["name"],
    )
    index.reset_calls()
    embeds = []
    stats = _sync(
        registry_db,
        index,
        embed=lambda text: embeds.append(text) or fake_embed(text, DEFAULT_DIM),
    )
    assert stats["reason"] == "bootstrap"
    assert stats["skipped"] == 1
    assert index.calls["upsert"] == 0
    assert embeds == []


@pytest.mark.performance_contract
def test_one_dirty_skill_only_updates_that_key(registry_db):
    index = CountingIndex()
    _store_catalog(registry_db, _catalog_row("native:alpha", "alpha"))
    _store_catalog(registry_db, _catalog_row("native:beta", "beta"))
    _sync(registry_db, index)
    index.reset_calls()

    _store_catalog(registry_db, _catalog_row("native:beta", "beta v2"))
    stats = _sync(registry_db, index, now=101)
    assert stats["mode"] == "incremental"
    assert stats["upserted"] == 1
    assert index.calls == {"upsert": 1, "delete": 0, "get": 0, "list_keys": 0}
    assert index._rows["native:alpha"]["content_sha"] == content_sha_for_text("alpha")
    assert index._rows["native:beta"]["content_sha"] == content_sha_for_text("beta v2")


def test_generation_cas_keeps_late_update_and_prevents_stale_write(registry_db):
    index = CountingIndex()
    _store_catalog(registry_db, _catalog_row("native:alpha", "v1"))
    _sync(registry_db, index)
    _store_catalog(registry_db, _catalog_row("native:alpha", "v2"))

    def racing_embed(text: str):
        assert text == "v2"
        _store_catalog(registry_db, _catalog_row("native:alpha", "v3"))
        return fake_embed(text, DEFAULT_DIM)

    stats = _sync(registry_db, index, now=101, embed=racing_embed)
    assert stats["upserted"] == 0
    assert stats["deferred"] == 1
    assert index._rows["native:alpha"]["content_sha"] == content_sha_for_text("v1")
    assert list_catalog_vector_dirty(db_path=registry_db)[0]["generation"] == 3

    stats = _sync(registry_db, index, now=102)
    assert stats["upserted"] == 1
    assert index._rows["native:alpha"]["content_sha"] == content_sha_for_text("v3")
    assert list_catalog_vector_dirty(db_path=registry_db) == []


def test_generation_watermark_prevents_aba_clear(registry_db):
    row = _catalog_row("native:alpha", "v1")
    _store_catalog(registry_db, row)
    first = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert clear_catalog_vector_dirty(
        first["catalog_key"], first["generation"], db_path=registry_db,
    )

    _store_catalog(registry_db, _catalog_row("native:alpha", "v2"))
    second = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert second["generation"] > first["generation"]
    assert not clear_catalog_vector_dirty(
        first["catalog_key"], first["generation"], db_path=registry_db,
    )
    assert list_catalog_vector_dirty(db_path=registry_db)[0]["generation"] == second[
        "generation"
    ]


def test_model_switch_forces_reembedding_and_periodic_repairs_ghost(registry_db):
    index = CountingIndex()
    _store_catalog(registry_db, _catalog_row("native:alpha", "alpha"))
    _sync(registry_db, index, model="model-a")
    index.reset_calls()

    changed = _sync(registry_db, index, model="model-b", now=101)
    assert changed["mode"] == "full"
    assert changed["reason"] == "model_changed"
    assert changed["upserted"] == 1
    assert index.calls["list_keys"] == 1
    assert index.calls["upsert"] == 1

    index.upsert(
        "native:ghost", fake_embed("ghost"), content_sha="ghost",
        source="native", name="ghost",
    )
    repaired = _sync(
        registry_db,
        index,
        model="model-b",
        now=101 + VECTOR_RECONCILE_INTERVAL_SECONDS,
    )
    assert repaired["reason"] == "periodic"
    assert repaired["deleted"] == 1
    assert index.get("native:ghost") is None


def test_full_reconcile_updates_metadata_when_content_sha_is_unchanged(registry_db):
    index = CountingIndex()
    row = _catalog_row("native:alpha", "same text", name="old-name")
    _store_catalog(registry_db, row)
    _sync(registry_db, index)

    renamed = {**row, "name": "new-name", "source": "skillhub"}
    _store_catalog(registry_db, renamed)
    embeds = []
    stats = _sync(
        registry_db,
        index,
        now=100 + VECTOR_RECONCILE_INTERVAL_SECONDS,
        embed=lambda text: embeds.append(text) or fake_embed(text, DEFAULT_DIM),
    )
    assert stats["mode"] == "full"
    assert stats["upserted"] == 1
    assert index._rows["native:alpha"]["name"] == "new-name"
    assert index._rows["native:alpha"]["source"] == "skillhub"
    assert embeds == []


def test_catalog_writes_coalesce_and_emit_rename_tombstone(registry_db, tmp_path):
    root = tmp_path / "skills"
    old = _write_native_skill(root, "old-name", "first description")
    upsert_native_skill(old, db_path=registry_db)
    event = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert event["catalog_key"] == "native:old-name"
    assert event["operation"] == "upsert"

    upsert_native_skill(old, db_path=registry_db)
    unchanged = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert unchanged["generation"] == event["generation"]

    assert clear_catalog_vector_dirty(
        unchanged["catalog_key"], unchanged["generation"], db_path=registry_db,
    )
    new = _write_native_skill(root, "new-name", "renamed description")
    rename_native_skill("old-name", new, db_path=registry_db)
    events = {row["catalog_key"]: row for row in list_catalog_vector_dirty(
        db_path=registry_db,
    )}
    assert events["native:old-name"]["operation"] == "delete"
    assert events["native:new-name"]["operation"] == "upsert"

    delete_native_skill("new-name", db_path=registry_db)
    events = {row["catalog_key"]: row for row in list_catalog_vector_dirty(
        db_path=registry_db,
    )}
    assert events["native:new-name"]["operation"] == "delete"


def test_retire_and_unretire_emit_delete_and_upsert(registry_db):
    row = _catalog_row("native:alpha", "alpha")
    _store_catalog(registry_db, row, mark=False)
    registry_mod.retire_skill(skill_name="alpha", set_by="test", db_path=registry_db)
    event = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert event["operation"] == "delete"
    assert registry_mod.unretire_skill(skill_name="alpha", db_path=registry_db)
    event = list_catalog_vector_dirty(db_path=registry_db)[0]
    assert event["operation"] == "upsert"
    assert event["generation"] == 2


def test_heavy_tick_reuses_known_embedding_dimension(registry_db, tmp_path):
    from xskill.recommend.heavy_worker import run_recommend_heavy_once

    class EmbedClient:
        dim = DEFAULT_DIM
        model = "known-dim"

        def __init__(self):
            self.calls = []

        def encode(self, text):
            self.calls.append(text)
            return fake_embed(text, DEFAULT_DIM)

    class Engine:
        embed_client = EmbedClient()

    stats = run_recommend_heavy_once(
        engine=Engine(),
        db_path=registry_db,
        vector_db_path=tmp_path / "vectors.db",
        memory_index=MemorySkillVectorIndex(),
    )
    assert stats["vector"]["mode"] == "full"
    assert Engine.embed_client.calls == []


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ({"fields": [{"name": "vector", "params": {"dim": 1536}}]}, 1536),
        ({"fields": [{"name": "vector", "dim": "1024"}]}, 1024),
        ({"fields": [{"name": "catalog_key", "params": {}}]}, None),
    ],
)
def test_milvus_description_dimension_parsing(description, expected):
    assert MilvusLiteSkillVectorIndex._described_vector_dim(description) == expected


def test_milvus_dimension_change_recreates_projection(monkeypatch):
    class Schema:
        def __init__(self):
            self.fields = []

        def add_field(self, *args, **kwargs):
            self.fields.append((args, kwargs))

    class Client:
        def __init__(self):
            self.dropped = []
            self.schema = Schema()
            self.created = []

        def has_collection(self, _name):
            return True

        def describe_collection(self, _name):
            return {"fields": [{"name": "vector", "params": {"dim": 4}}]}

        def drop_collection(self, name):
            self.dropped.append(name)

        def create_schema(self, **_kwargs):
            return self.schema

        def prepare_index_params(self):
            return SimpleNamespace(add_index=lambda **_kwargs: None)

        def create_collection(self, **kwargs):
            self.created.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(
            DataType=SimpleNamespace(INT64=1, VARCHAR=2, FLOAT_VECTOR=3),
        ),
    )
    index = object.__new__(MilvusLiteSkillVectorIndex)
    index.dim = DEFAULT_DIM
    index._client = Client()
    index._ensure_collection()
    assert index._client.dropped == ["skill_vectors"]
    assert index._client.created
    vector_fields = [
        kwargs for _args, kwargs in index._client.schema.fields
        if _args and _args[0] == "vector"
    ]
    assert vector_fields[0]["dim"] == DEFAULT_DIM


# ─────────────────────────────────────────────────────────────────
# 全量对账分批（issue #328）：没有积压时播种，之后按 limit 分批消费，
# 大目录天然拆成多轮；持久索引（这里用可跨调用复用的内存索引模拟）
# 跨轮次真正累积覆盖，进度通过 remaining 字段体现。
# ─────────────────────────────────────────────────────────────────

def _store_many(db: Path, n: int, *, prefix: str = "native:k") -> None:
    for i in range(n):
        _store_catalog(
            db, _catalog_row(f"{prefix}{i}", f"desc-{i}"), mark=False,
        )


class TestFullSweepBatching:
    def test_large_catalog_spreads_across_multiple_rounds(self, registry_db):
        """25 条、每轮 limit=10：分 3 轮才追平，期间索引跨轮累积覆盖。"""
        _store_many(registry_db, 25)
        index = CountingIndex()

        r1 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=10,
        )
        assert r1["mode"] == "full"
        assert r1["reason"] == "bootstrap"
        assert r1["upserted"] == 10
        assert r1["remaining"] == 15
        assert len(index.list_keys()) == 10

        r2 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=101.0, limit=10,
        )
        assert r2["mode"] == "full"
        assert r2["remaining"] == 5
        assert len(index.list_keys()) == 20

        r3 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=102.0, limit=10,
        )
        assert r3["mode"] == "full"
        assert r3["remaining"] == 0
        assert len(index.list_keys()) == 25

        # 追平之后再来一轮：不再是「full」，纯增量且没有任何积压。
        r4 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=103.0, limit=10,
        )
        assert r4["mode"] == "incremental"
        assert r4["upserted"] == 0

    def test_mid_sweep_does_not_reseed_and_lose_progress(self, registry_db):
        """还在消化上一次播种的积压时，再次触发 full 原因不应该重新播种
        （否则已经清掉的 key 会被重新标脏，进度被抹掉，永远追不平）。"""
        _store_many(registry_db, 12)
        index = CountingIndex()

        run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=5,
        )
        from xskill.recommend.vector_dirty import count_catalog_vector_dirty
        assert count_catalog_vector_dirty(db_path=registry_db) == 7

        # 第二轮：queue 里还有 7 条积压，不应该被重新播种成 12 条。
        run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=101.0, limit=5,
        )
        assert count_catalog_vector_dirty(db_path=registry_db) == 2

        run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=102.0, limit=5,
        )
        assert count_catalog_vector_dirty(db_path=registry_db) == 0
        assert len(index.list_keys()) == 12

    def test_sweep_deletes_stale_keys_no_longer_indexable(self, registry_db):
        """索引里有、但 catalog 里已经不可索引（被删）的 key 会被播种成
        delete 并在消费时清掉——和旧的全量对账行为一致。"""
        index = CountingIndex()
        index.upsert(
            "native:gone", fake_embed("gone", DEFAULT_DIM),
            content_sha="gone", source="native", name="gone",
        )
        _store_catalog(registry_db, _catalog_row("native:alpha", "alpha"), mark=False)

        stats = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=256,
        )
        assert stats["deleted"] == 1
        assert stats["remaining"] == 0
        assert index.get("native:gone") is None
        assert index.get("native:alpha") is not None

    def test_ephemeral_reason_force_upserts_without_reuse(self, registry_db):
        """ephemeral（无持久索引）强制重新 embed，即便 content_sha 没变——
        旧向量可能是别的模型算的，复用会用错模型的向量。"""
        index = CountingIndex()
        row = _catalog_row("native:alpha", "alpha")
        _store_catalog(registry_db, row, mark=False)
        index.upsert(
            row["catalog_key"], fake_embed("alpha", DEFAULT_DIM),
            content_sha=row["content_sha"], source=row["source"], name=row["name"],
        )
        embeds = []
        stats = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: embeds.append(text) or fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=256,
            force_full=True,
        )
        assert stats["reason"] == "ephemeral"
        assert stats["upserted"] == 1
        assert embeds == ["alpha"]


class TestSweepSeededMarkerReviewFindings:
    """PR #338 code review（tiammomo）里复现的三个跨轮状态问题的回归测试。"""

    def test_organic_dirty_item_does_not_block_full_seeding(self, registry_db):
        """review 复现：3 条 catalog，bootstrap 前只有 k0 有机标脏。旧逻辑
        「脏表非空就不播种」会把这当成「已经播种过」，k1/k2 永远进不了
        索引，还会误把这轮标记成对账完成。"""
        _store_catalog(registry_db, _catalog_row("native:k0", "d0"))  # mark=True，有机脏项
        _store_catalog(registry_db, _catalog_row("native:k1", "d1"), mark=False)
        _store_catalog(registry_db, _catalog_row("native:k2", "d2"), mark=False)
        index = CountingIndex()

        stats = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=256,
        )
        assert stats["mode"] == "full"
        assert stats["remaining"] == 0
        # 三条全部进了索引，不是只有有机标脏的那一条。
        assert index.list_keys() == {"native:k0", "native:k1", "native:k2"}

    def test_organic_dirty_item_does_not_block_multi_round_seeding(self, registry_db):
        """同上，但目录更大、limit 更小，确认播种在多轮场景下同样不受
        有机脏项影响——只播种一次，后续轮次正常消化剩余积压。"""
        _store_catalog(registry_db, _catalog_row("native:k0", "d0"))  # 有机脏项
        for i in range(1, 5):
            _store_catalog(registry_db, _catalog_row(f"native:k{i}", f"d{i}"), mark=False)
        index = CountingIndex()

        r1 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=2,
        )
        assert r1["mode"] == "full"
        assert r1["remaining"] == 3  # 5 条播种，本轮处理 2 条

        r2 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=101.0, limit=2,
        )
        assert r2["remaining"] == 1
        # 第二轮没有重新播种（否则 remaining 会跳回接近 5，而不是继续下降）。

        r3 = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=102.0, limit=2,
        )
        assert r3["remaining"] == 0
        assert index.list_keys() == {f"native:k{i}" for i in range(5)}

    def test_model_change_mid_sweep_reseeds_and_reembeds_everything(self, registry_db):
        """review 复现：sweep 从模型 B 开始，处理 2 条后模型切到 C，剩余
        条目用 C embed。旧逻辑不会因为指纹变化而重新播种，导致 k0/k1
        停留在 B 向量、水位却记成 C，形成混合模型索引。"""
        for i in range(5):
            _store_catalog(registry_db, _catalog_row(f"native:k{i}", f"d{i}"), mark=False)
        index = CountingIndex()

        embedded_with = {}

        def embed_b(text):
            embedded_with[text] = "B"
            return fake_embed(f"B:{text}", DEFAULT_DIM)

        def embed_c(text):
            embedded_with[text] = "C"
            return fake_embed(f"C:{text}", DEFAULT_DIM)

        r1 = run_vector_sync(
            db_path=registry_db, index=index, embed=embed_b,
            model_fingerprint="test:model-b:8", now=100.0, limit=2,
        )
        assert r1["remaining"] == 3
        assert index.calls["upsert"] == 2

        # 模型切到 C，sweep 还没追平；应重新播种，之前在 B 下已处理的
        # 也要用 C 重新 embed，不能停留在 B 向量。
        r2 = run_vector_sync(
            db_path=registry_db, index=index, embed=embed_c,
            model_fingerprint="test:model-c:8", now=101.0, limit=10,
        )
        assert r2["reason"] == "model_changed"
        assert r2["remaining"] == 0
        # 全部 5 条这一轮都要用 C 重新处理（3 条剩余 + 2 条被重新播种）。
        assert r2["upserted"] == 5
        for i in range(5):
            assert embedded_with[f"d{i}"] == "C"

    def test_persistent_index_replacement_mid_sweep_reseeds_everything(
        self, registry_db, tmp_path, monkeypatch,
    ):
        """持久索引文件在多轮 sweep 中途被替换时，新索引必须重新播种。

        模型算法和维度没有变化，只有数据库 inode 变化；旧逻辑使用不含
        索引 identity 的稳定 sweep key，会让新索引只消费旧队列剩余项，
        随后错误提交为已追平。
        """
        from xskill.recommend.heavy_worker import run_recommend_heavy_once

        for i in range(5):
            _store_catalog(
                registry_db, _catalog_row(f"native:k{i}", f"d{i}"), mark=False,
            )

        class PersistentIndex:
            def __init__(self, db_path):
                self.db_path = db_path
                self.inner = CountingIndex()

            def __getattr__(self, name):
                return getattr(self.inner, name)

        vector_db = tmp_path / "vectors.db"
        vector_db.write_bytes(b"old-index")
        old_index = PersistentIndex(vector_db)
        new_index = PersistentIndex(vector_db)
        indexes = iter((old_index, new_index))
        monkeypatch.setattr(
            "xskill.recommend.skill_vector_store.open_skill_vector_index",
            lambda *_args, **_kwargs: next(indexes),
        )

        first = run_recommend_heavy_once(
            engine=SimpleNamespace(embed_client=None),
            db_path=registry_db,
            vector_db_path=vector_db,
            vector_sync_batch_limit=2,
            mark_catalog_dirty=False,
        )
        assert first["vector"]["remaining"] == 3
        assert old_index.list_keys() == {"native:k0", "native:k1"}

        replacement = tmp_path / "replacement.db"
        replacement.write_bytes(b"new-index")
        replacement.replace(vector_db)
        second = run_recommend_heavy_once(
            engine=SimpleNamespace(embed_client=None),
            db_path=registry_db,
            vector_db_path=vector_db,
            vector_sync_batch_limit=10,
            mark_catalog_dirty=False,
        )

        assert second["vector"]["remaining"] == 0
        assert second["vector"]["upserted"] == 5
        assert new_index.list_keys() == {f"native:k{i}" for i in range(5)}

    def test_sweep_marker_cleared_on_completion_allows_next_periodic_seed(
        self, registry_db,
    ):
        """追平后播种标记要清掉——不然下一次同指纹的周期性对账（内容真的
        变了）会被误判成「已经播种过」而跳过。"""
        _store_catalog(registry_db, _catalog_row("native:k0", "d0"), mark=False)
        index = CountingIndex()
        first = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8", now=100.0, limit=256,
        )
        assert first["remaining"] == 0

        from xskill.recommend.vector_dirty import get_sweep_seeded_fingerprint
        assert get_sweep_seeded_fingerprint(db_path=registry_db) == ""

        # 内容真的变了 + 到了周期性对账窗口：应该能重新播种并处理。
        _store_catalog(registry_db, _catalog_row("native:k1", "d1"), mark=False)
        second = run_vector_sync(
            db_path=registry_db, index=index,
            embed=lambda text: fake_embed(text, DEFAULT_DIM),
            model_fingerprint="test:model-a:8",
            now=100.0 + VECTOR_RECONCILE_INTERVAL_SECONDS, limit=256,
        )
        assert second["reason"] == "periodic"
        assert "native:k1" in index.list_keys()


class TestRecommendsSafeToRecompute:
    """review 复现：全量对账没追平/内存兜底只覆盖当前批次时，不能拿这
    份不完整的索引重算并覆盖已有推荐结果。"""

    def test_incremental_always_safe(self):
        from xskill.recommend.heavy_worker import _recommends_safe_to_recompute
        assert _recommends_safe_to_recompute(
            {"mode": "incremental", "remaining": 0}, ephemeral_index=True,
        )
        assert _recommends_safe_to_recompute(
            {"mode": "incremental", "remaining": 0}, ephemeral_index=False,
        )

    def test_full_sweep_in_progress_is_never_safe(self):
        from xskill.recommend.heavy_worker import _recommends_safe_to_recompute
        assert not _recommends_safe_to_recompute(
            {"mode": "full", "remaining": 3}, ephemeral_index=False,
        )
        assert not _recommends_safe_to_recompute(
            {"mode": "full", "remaining": 3}, ephemeral_index=True,
        )

    def test_full_sweep_done_persistent_index_is_safe(self):
        from xskill.recommend.heavy_worker import _recommends_safe_to_recompute
        assert _recommends_safe_to_recompute(
            {"mode": "full", "remaining": 0}, ephemeral_index=False,
        )

    def test_full_sweep_done_ephemeral_single_round_is_safe(self):
        """整份可索引目录一次就在这批里播种+处理完（total_indexable 有值）
        ——这一轮内存里的索引确实是完整的。"""
        from xskill.recommend.heavy_worker import _recommends_safe_to_recompute
        assert _recommends_safe_to_recompute(
            {"mode": "full", "remaining": 0, "total_indexable": 3},
            ephemeral_index=True,
        )

    def test_full_sweep_done_ephemeral_multi_round_tail_is_not_safe(self):
        """多轮播种的尾轮：remaining 归零了，但这轮没有播种（没有
        total_indexable），内存里的索引只有这一轮那一小批，不是全量。"""
        from xskill.recommend.heavy_worker import _recommends_safe_to_recompute
        assert not _recommends_safe_to_recompute(
            {"mode": "full", "remaining": 0}, ephemeral_index=True,
        )
