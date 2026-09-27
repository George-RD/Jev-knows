"""The memory-mapped static embedder, persisted claim vectors, and the hook's embedder."""

from __future__ import annotations

import json
import os
import struct
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from unittest import mock

from jev_wiki import embedding, hooks

try:
    import numpy as np
    from tokenizers import Tokenizer, models, pre_tokenizers
except ImportError:  # The embed extra is optional.
    np = None

VOCAB = ["[UNK]", "basil", "tomatoes", "garden", "report", "friday", "dinner"]
# Two concepts: the food words point one way, the office words another.
TABLE = [
    [0.0, 0.0, 0.0, 1.0],
    [1.0, 0.1, 0.0, 0.0],
    [1.0, 0.0, 0.2, 0.0],
    [0.9, 0.1, 0.1, 0.0],
    [0.0, 1.0, 0.0, 0.1],
    [0.1, 1.0, 0.1, 0.0],
    [1.0, 0.0, 0.0, 0.1],
]


def write_model(directory: Path, *, dtype: str = "F32", weights=None, mapping=None) -> Path:
    """A model2vec-layout model: a word-level tokenizer and a safetensors table."""
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    rows = len(VOCAB) if mapping is None else int(max(mapping)) + 1
    table = np.asarray(TABLE[:rows])
    tensors = {"embeddings": (table.astype("<f4" if dtype == "F32" else "<f2"), dtype)}
    if weights is not None:
        tensors["weights"] = (np.asarray(weights, dtype="<f4"), "F32")
    if mapping is not None:
        tensors["mapping"] = (np.asarray(mapping, dtype="<i8"), "I64")
    header, blobs, offset = {}, [], 0
    for name, (array, kind) in tensors.items():
        data = array.tobytes()
        header[name] = {
            "dtype": kind,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with (directory / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
    (directory / "config.json").write_text('{"normalize": true}')
    return directory


def expected(directory: Path, text: str, *, weights=None, mapping=None) -> np.ndarray:
    """The model2vec definition: mean of known tokens' rows, unit length."""
    header_size = struct.unpack("<Q", (directory / "model.safetensors").read_bytes()[:8])[0]
    blob = (directory / "model.safetensors").read_bytes()[8 + header_size :]
    info = json.loads((directory / "model.safetensors").read_bytes()[8 : 8 + header_size])
    begin, end = info["embeddings"]["data_offsets"]
    dtype = "<f4" if info["embeddings"]["dtype"] == "F32" else "<f2"
    table = np.frombuffer(blob[begin:end], dtype=dtype).reshape(info["embeddings"]["shape"])
    ids = [VOCAB.index(w) for w in text.split() if w in VOCAB[1:]]
    rows = table[[mapping[i] for i in ids] if mapping is not None else ids].astype("<f4")
    if weights is not None:
        rows = rows * np.asarray([weights[i] for i in ids], dtype="<f4")[:, None]
    mean = rows.mean(axis=0)
    return mean / np.linalg.norm(mean)


@unittest.skipIf(np is None, "the embed extra (numpy, tokenizers) is not installed")
class StaticEmbedderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-vectors-")
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.model = write_model(self.tmp / "model")
        self.cache = self.tmp / "root" / embedding.VECTOR_DIR

    def embedder(self, **options):
        return embedding.StaticEmbedder(str(self.model), cache_dir=self.cache, **options)

    def test_encoding_is_the_mean_of_known_token_rows(self):
        vectors = self.embedder().encode(["basil garden unknownword", "nothing known", ""])
        np.testing.assert_allclose(vectors[0], expected(self.model, "basil garden"), atol=1e-6)
        self.assertFalse(vectors[1].any())
        self.assertFalse(vectors[2].any())

    def test_half_precision_weighted_and_mapped_tables(self):
        weights = [0.0, 2.0, 0.5, 1.0, 1.0, 3.0, 1.0]
        mapping = [0, 2, 1, 3, 3, 4, 5]
        model = write_model(self.tmp / "quantized", dtype="F16", weights=weights, mapping=mapping)
        vector = embedding.StaticEmbedder(str(model)).encode(["basil tomatoes friday"])[0]
        np.testing.assert_allclose(
            vector,
            expected(model, "basil tomatoes friday", weights=weights, mapping=mapping),
            atol=1e-3,
        )

    def test_vectors_persist_and_the_next_instance_encodes_nothing(self):
        texts = ["basil tomatoes", "report friday", "basil tomatoes"]
        first, encoded = self.embedder().vectors(texts)
        self.assertEqual(encoded, 2)
        again = self.embedder()
        with mock.patch.object(again, "encode", wraps=again.encode) as encode:
            second, encoded = again.vectors(texts)
        self.assertEqual(encoded, 0)
        encode.assert_not_called()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            again.similarities("dinner basil", texts),
            self.embedder().similarities("dinner basil", texts),
        )

    def records(self) -> int:
        vector_file = self.cache / f"{self.embedder().fingerprint.hex()[:16]}.vec"
        record = 32 + 4 * 4
        size = vector_file.stat().st_size - embedding._HEADER.size
        self.assertEqual(size % record, 0)
        return size // record

    def test_new_claims_are_appended_and_other_models_dropped_by_the_worker(self):
        self.cache.mkdir(parents=True)
        other = self.cache / "0123456789abcdef.vec"
        other.write_bytes(b"another model")
        self.embedder(deadline=time.monotonic() + 60).vectors(["basil"])
        self.assertTrue(other.exists())  # The hook never removes another model's vectors.
        self.embedder(deadline=time.monotonic() + 60).vectors(["basil", "report"])
        self.assertEqual(self.records(), 2)
        (self.cache / f"{self.embedder().fingerprint.hex()[:16]}.vec").unlink()
        self.embedder().vectors(["basil", "report"])
        self.assertFalse(other.exists())
        self.embedder().vectors(["basil", "report", "garden"])
        self.assertEqual(self.records(), 3)

    def test_stale_records_are_compacted_without_a_deadline_only(self):
        self.embedder().vectors(["basil", "report"])
        with mock.patch.object(embedding, "COMPACT_SLACK", 0):
            hook = self.embedder(deadline=time.monotonic() + 60)
            hook.vectors(["garden"])  # "basil" and "report" were forgotten.
            self.assertEqual(self.records(), 3)
            _, encoded = self.embedder().vectors(["garden"])
        self.assertEqual(encoded, 0)
        self.assertEqual(self.records(), 1)

    def test_torn_append_keeps_earlier_records_and_only_the_worker_rewrites(self):
        self.embedder().vectors(["basil", "report"])
        (vector_file,) = self.cache.glob("*.vec")
        with vector_file.open("ab") as stream:
            stream.write(b"half a record")  # Or another process's append in progress.
        torn = vector_file.read_bytes()
        hook = self.embedder(deadline=time.monotonic() + 60)
        vectors, encoded = hook.vectors(["basil", "report", "garden"])
        self.assertEqual(encoded, 1)
        np.testing.assert_allclose(vectors[0], expected(self.model, "basil"), atol=1e-6)
        self.assertEqual(vector_file.read_bytes(), torn)
        self.assertEqual(self.embedder().vectors(["basil", "report", "garden"])[1], 1)
        self.assertEqual(self.records(), 3)

    def test_foreign_vector_file_is_replaced(self):
        self.embedder().vectors(["basil"])
        (vector_file,) = self.cache.glob("*.vec")
        vector_file.write_bytes(b"JEVVEC0\n" + vector_file.read_bytes()[8:])
        vectors, encoded = self.embedder().vectors(["basil"])
        self.assertEqual(encoded, 1)
        np.testing.assert_allclose(vectors[0], expected(self.model, "basil"), atol=1e-6)
        self.assertEqual(self.embedder().vectors(["basil"])[1], 0)

    def test_symlinked_vector_file_is_neither_read_nor_written(self):
        self.embedder().vectors(["basil"])
        (vector_file,) = self.cache.glob("*.vec")
        target = self.tmp / "elsewhere.vec"
        vector_file.rename(target)
        vector_file.symlink_to(target)
        before = target.read_bytes()
        _, encoded = self.embedder(deadline=time.monotonic() + 60).vectors(["basil", "garden"])
        self.assertEqual(encoded, 2)
        self.assertEqual(target.read_bytes(), before)

    def test_unwritable_cache_keeps_the_computed_vectors(self):
        self.cache.parent.mkdir(parents=True)
        self.cache.write_text("a file, not a directory")
        similarities = self.embedder().similarities("dinner", ["basil", "report"])
        self.assertGreater(similarities[0], similarities[1])

    def test_copies_of_one_model_share_vectors(self):
        self.embedder().vectors(["basil"])
        copy = write_model(self.tmp / "copy")
        shared = embedding.StaticEmbedder(str(copy), cache_dir=self.cache)
        self.assertEqual(shared.fingerprint, self.embedder().fingerprint)
        self.assertEqual(shared.vectors(["basil"])[1], 0)

    def test_encoding_cap_saves_part_of_a_backlog_per_call(self):
        texts = ["basil", "report", "garden", "friday", "tomatoes"]
        for done in (2, 4):
            with self.assertRaises(embedding.EmbeddingTimeout):
                self.embedder(max_new=2).vectors(texts)
            self.assertEqual(self.records(), done)
        _, encoded = self.embedder(max_new=2).vectors(texts)
        self.assertEqual(encoded, 1)

    def test_passed_deadline_stops_loading_and_encoding(self):
        with self.assertRaises(embedding.EmbeddingTimeout):
            self.embedder(deadline=time.monotonic() - 1)
        embedder = self.embedder()
        embedder.deadline = time.monotonic() - 1
        with self.assertRaises(embedding.EmbeddingTimeout):
            embedder.similarities("basil", ["basil"])

    def test_vectors_encoded_late_are_saved_but_recall_stays_lexical(self):
        self.embedder().vectors(["report"])
        embedder = self.embedder()
        check = embedder._check_deadline
        embedder._check_deadline = lambda: None  # Encoding finishes, then the deadline passes.
        embedder.deadline = time.monotonic() - 1
        vectors, encoded = embedder.vectors(["report", "basil"])
        self.assertEqual(encoded, 1)
        np.testing.assert_allclose(vectors[1], expected(self.model, "basil"), atol=1e-6)
        self.assertEqual(self.records(), 2)
        embedder._check_deadline = check
        with self.assertRaises(embedding.EmbeddingTimeout):
            embedder.similarities("dinner", ["report", "basil"])

    def test_hub_ids_resolve_from_the_cache_without_downloading(self):
        hub = self.tmp / "hub"
        repo = hub / "models--org--tiny"
        write_model(repo / "snapshots" / "abc123")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("abc123")
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}):
            self.assertEqual(
                embedding.resolve_model("org/tiny", local_only=True),
                repo / "snapshots" / "abc123",
            )
            with self.assertRaises(FileNotFoundError):
                embedding.resolve_model("org/missing", local_only=True)

    def test_hub_cache_follows_huggingface_hub_resolution(self):
        cases = [
            ({"HF_HUB_CACHE": "/a", "HF_HOME": "/b"}, Path("/a")),
            ({"HUGGINGFACE_HUB_CACHE": "/c"}, Path("/c")),
            ({"HF_HOME": "~/hf"}, Path("~/hf").expanduser() / "hub"),
            ({"XDG_CACHE_HOME": "/x"}, Path("/x/huggingface/hub")),
        ]
        names = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME")
        for env, path in cases:
            clean = {name: "" for name in names}
            with self.subTest(env=env), mock.patch.dict(os.environ, {**clean, **env}):
                self.assertEqual(embedding._hub_cache(), path)


@unittest.skipIf(np is None, "the embed extra (numpy, tokenizers) is not installed")
class HookEmbedderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jev-wiki-hook-embed-")
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        self.model = write_model(self.tmp / "model")
        self.root = self.tmp / "memory"
        self.root.mkdir()

    def build(self, model: str, started: float | None = None):
        with mock.patch.dict(os.environ, {embedding.ENV_VAR: model}):
            lazy = hooks._hook_embedder(self.root, time.monotonic() if started is None else started)
            if lazy is not None:
                with suppress(LookupError):
                    lazy.similarities("dinner", ["basil"])
            return lazy

    def test_unset_model_means_no_hook_embedder(self):
        self.assertIsNone(self.build(""))

    def test_local_model_loads_with_the_hook_limits_and_vector_dir(self):
        embedder = self.build(str(self.model)).embedder
        self.assertEqual(embedder._cache_file.parent, self.root / embedding.VECTOR_DIR)
        self.assertLessEqual(embedder.deadline, time.monotonic() + hooks.HOOK_EMBEDDING_SECONDS)
        self.assertEqual(embedder.max_new, hooks.HOOK_MAX_NEW_CLAIMS)

    def test_model_is_not_loaded_for_an_empty_memory(self):
        project = self.tmp / "project"
        project.mkdir()
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": "dinner ideas",
            "session_id": "s1",
            "cwd": str(project),
        }
        with (
            mock.patch.dict(os.environ, {embedding.ENV_VAR: str(self.model)}),
            mock.patch.object(embedding, "StaticEmbedder") as built,
        ):
            self.assertEqual(hooks.handle_hook(self.root, payload, project_root=project), {})
        built.assert_not_called()

    def test_hook_never_downloads_and_falls_back_to_lexical(self):
        with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(self.tmp / "empty-hub")}):
            self.assertIsNone(self.build("org/not-downloaded").embedder)

    def test_spent_budget_leaves_the_hook_lexical(self):
        lazy = self.build(str(self.model), started=time.monotonic() - 10)
        self.assertIsNone(lazy.embedder)

    def test_hook_recall_uses_embeddings_end_to_end(self):
        from jev_wiki.engine import Engine
        from test_lifecycle import LifecycleDecisionFixture

        engine = Engine(self.root, LifecycleDecisionFixture())
        text = "basil tomatoes garden.\n\nThe report is due friday."
        self.assertEqual(engine.ingest(text, source_key="notes")["status"], "complete")
        project = self.tmp / "project"
        project.mkdir()
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": "dinner ideas",
            "session_id": "s1",
            "cwd": str(project),
        }
        with mock.patch.dict(os.environ, {embedding.ENV_VAR: str(self.model)}):
            result = hooks.handle_hook(self.root, payload, project_root=project)
        self.assertIn("basil", result["hookSpecificOutput"]["additionalContext"])
        self.assertTrue(list((self.root / embedding.VECTOR_DIR).glob("*.vec")))
        with mock.patch.dict(os.environ, {embedding.ENV_VAR: ""}):
            lexical = hooks.handle_hook(self.root, payload, project_root=project)
        self.assertEqual(lexical, {})


class HookTimeoutTests(unittest.TestCase):
    def test_hook_budget_is_not_swallowed_by_the_optional_embedder(self):
        from jev_wiki.engine import Engine

        class Expired:
            def similarities(self, query, texts):
                raise hooks.HookTimeout("budget")

        with tempfile.TemporaryDirectory() as root:
            from test_lifecycle import LifecycleDecisionFixture

            engine = Engine(root, LifecycleDecisionFixture(), embedder=Expired())
            engine.ingest("The report is due friday.", source_key="notes")
            with self.assertRaises(hooks.HookTimeout):
                engine.recall("report", offline=True)

    def test_from_env_reraises_the_hook_budget(self):
        with (
            mock.patch.dict(os.environ, {embedding.ENV_VAR: "default"}),
            mock.patch.object(embedding, "StaticEmbedder", side_effect=hooks.HookTimeout("budget")),
            self.assertRaises(hooks.HookTimeout),
        ):
            embedding.from_env()


if __name__ == "__main__":
    unittest.main()
