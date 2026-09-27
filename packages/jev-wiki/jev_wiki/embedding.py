"""Optional local embeddings: a second recall candidate source beside BM25.

Lexical matching cannot connect "dinner with my homegrown ingredients" to evidence
that only says "basil" and "cherry tomatoes". A small static embedding model can, on
CPU and without a network call per query. It is off unless the caller passes an
embedder, and ``from_env`` builds one only when ``JEV_WIKI_EMBEDDING_MODEL`` is set
and the ``embed`` extra (model2vec, which brings numpy and tokenizers) is installed.

A static model is a token-vector table plus a tokenizer, and a text's vector is the
mean of its tokens' rows. ``StaticEmbedder`` reads that table memory-mapped instead of
through model2vec's loader, so only the rows a text uses are read from disk: loading
takes about 0.1 s instead of 0.8 s, which is what lets the prompt hook use it within
its budget (docs/hook-embeddings-2026-09-27.md). Claim vectors can be persisted in an
append-only vector file per model under the memory root, so each recall encodes only
new claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

from .hooks import HookTimeout

ENV_VAR = "JEV_WIKI_EMBEDDING_MODEL"
# Chosen on LongMemEval_S; see docs/embedding-candidates-2026-09-27.md.
DEFAULT_MODEL = "minishlab/potion-retrieval-32M"
# The directory, under a memory root, that holds persisted claim vectors.
VECTOR_DIR = "embeddings"
# model2vec's default: texts are cut to this many tokens.
MAX_TOKENS = 512
ENCODE_BATCH = 256
# Bump when encode() changes, so vectors persisted by older code are not reused.
ENCODING_VERSION = 1
# A vector file is rewritten without stale records once they outnumber live ones by
# this many; only callers without a deadline (worker, CLI) rewrite.
COMPACT_SLACK = 1_000

_MAGIC = b"JEVVEC2\n"
_HEADER = struct.Struct("<8s32sQ")  # magic, model fingerprint, dim; then records.
_DTYPES = {"F32": "<f4", "F16": "<f2"}
_INDEX_DTYPES = {"I64": "<i8", "I32": "<i4", "U32": "<u4", "U64": "<u8"}


class EmbeddingTimeout(Exception):
    """The embedder ran past its caller's deadline or encoding cap; recall stays lexical."""


def _hub_cache() -> Path:
    """The Hugging Face hub cache, resolved as huggingface_hub resolves it."""
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            return Path(os.environ[variable]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    cache = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser()
    return cache / "huggingface" / "hub"


def _is_model_dir(path: Path) -> bool:
    return (path / "model.safetensors").is_file() and (path / "tokenizer.json").is_file()


def resolve_model(model: str, *, local_only: bool = False) -> Path:
    """The local directory holding ``model``: a path, or a Hugging Face id.

    An id resolves through the Hugging Face cache without a network call. Only when
    it is not cached and ``local_only`` is false is it downloaded, once.
    """
    path = Path(model).expanduser()
    if path.is_dir():
        if not _is_model_dir(path):
            raise FileNotFoundError(f"Not a model2vec model directory: {model}")
        return path
    repo = _hub_cache() / f"models--{model.replace('/', '--')}"
    try:
        commit = (repo / "refs" / "main").read_text().strip()
    except OSError:
        commit = ""
    if commit and _is_model_dir(repo / "snapshots" / commit):
        return repo / "snapshots" / commit
    if local_only:
        raise FileNotFoundError(f"Embedding model is not downloaded: {model}")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model))
    if not _is_model_dir(path):
        raise FileNotFoundError(f"Not a model2vec model: {model}")
    return path


def _tensors(path: Path) -> tuple[dict, int]:
    """A safetensors file's tensor table and the offset its data starts at."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 100_000_000:
            raise ValueError("Unreasonable safetensors header")
        header = json.loads(stream.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


class StaticEmbedder:
    """Cosine similarity through a model2vec static embedding model.

    Encoding matches model2vec's ``StaticModel.encode``: no special tokens, unknown
    tokens dropped, at most MAX_TOKENS tokens, the mean of the (weighted) token rows.
    Vectors are unit length. ``deadline`` (a ``time.monotonic()`` value) makes loading
    and encoding raise EmbeddingTimeout once passed. With ``cache_dir``, claim vectors
    are read from and appended to one file per model there. ``max_new`` caps how many
    claims one call encodes: past it, the capped share is encoded and saved, and the
    call raises EmbeddingTimeout, so a backlog is worked off over several calls.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        local_only: bool = False,
        cache_dir: str | Path | None = None,
        deadline: float | None = None,
        max_new: int | None = None,
    ):
        self.deadline = deadline
        self.max_new = max_new
        import numpy as np
        from tokenizers import Tokenizer

        self._np = np
        self.name = model
        directory = resolve_model(model, local_only=local_only)
        self._check_deadline()
        tokenizer_path = directory / "tokenizer.json"
        weights_path = directory / "model.safetensors"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        unk = getattr(self._tokenizer.model, "unk_token", None)
        self._unk = None if unk is None else self._tokenizer.token_to_id(unk)
        self._median_token_length: int | None = None
        header, start = _tensors(weights_path)

        def tensor(name: str, dtypes: dict) -> Any:
            info = header.get(name)
            if info is None:
                return None
            if info.get("dtype") not in dtypes:
                raise ValueError(f"Unsupported {name} dtype: {info.get('dtype')}")
            begin, end = info["data_offsets"]
            shape = tuple(info["shape"])
            dtype = np.dtype(dtypes[info["dtype"]])
            if end - begin != dtype.itemsize * int(np.prod(shape)):
                raise ValueError(f"Inconsistent {name} tensor")
            return np.memmap(weights_path, dtype=dtype, mode="r", offset=start + begin, shape=shape)

        self._table = tensor("embeddings", _DTYPES)
        if self._table is None or self._table.ndim != 2:
            raise ValueError("The model has no 2-D embeddings tensor")
        self._weights = tensor("weights", _DTYPES)
        self._mapping = tensor("mapping", _INDEX_DTYPES)
        self.dim = int(self._table.shape[1])
        self.fingerprint = self._fingerprint(weights_path, tokenizer_path, start)
        self._record = np.dtype([("key", "u1", (32,)), ("vector", "<f4", (self.dim,))])
        self._memory: dict[bytes, Any] = {}
        self._disk: tuple[dict[bytes, int], Any] | None = None
        self._disk_ok = False
        self._cache_file = None
        if cache_dir is not None:
            self._cache_file = Path(cache_dir) / f"{self.fingerprint.hex()[:16]}.vec"
        self._check_deadline()

    @staticmethod
    def _fingerprint(weights_path: Path, tokenizer_path: Path, data_start: int) -> bytes:
        """Identify the model by content, wherever it is stored, plus the encoding code.

        Hashes the tokenizer, the tensor table header, and sixteen 4 KiB samples of the
        tensor data, so two copies of one model share persisted vectors.
        """
        digest = hashlib.sha256(f"jev-wiki-encoding-{ENCODING_VERSION}".encode())
        digest.update(tokenizer_path.read_bytes())
        size = weights_path.stat().st_size
        with weights_path.open("rb") as stream:
            digest.update(stream.read(data_start))
            span = max(0, size - data_start - 4_096)
            for step in range(16):
                stream.seek(data_start + span * step // 15)
                digest.update(stream.read(4_096))
        digest.update(str(size).encode())
        return digest.digest()

    def _check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise EmbeddingTimeout("Embedding exceeded its time budget")

    def _median(self) -> int:
        if self._median_token_length is None:
            lengths = sorted(len(token) for token in self._tokenizer.get_vocab())
            self._median_token_length = int(self._np.median(lengths))
        return self._median_token_length

    def encode(self, texts: list[str]) -> Any:
        """Unit vectors, one row per text; a text with no known token is all zeros."""
        np = self._np
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for begin in range(0, len(texts), ENCODE_BATCH):
            self._check_deadline()
            batch = texts[begin : begin + ENCODE_BATCH]
            # model2vec cuts characters first, only past MAX_TOKENS × median token length.
            if any(len(text) > MAX_TOKENS for text in batch):
                limit = MAX_TOKENS * self._median()
                batch = [text[:limit] for text in batch]
            encodings = self._tokenizer.encode_batch(batch, add_special_tokens=False)
            for row, encoding in enumerate(encodings, begin):
                ids = [i for i in encoding.ids if i != self._unk][:MAX_TOKENS]
                if not ids:
                    continue
                rows = ids if self._mapping is None else self._mapping[ids]
                vectors = np.asarray(self._table[rows], dtype=np.float32)
                if self._weights is not None:
                    vectors = vectors * np.asarray(self._weights[ids], dtype=np.float32)[:, None]
                mean = vectors.mean(axis=0)
                out[row] = mean / max(float(np.linalg.norm(mean)), 1e-12)
        return out

    def _load_disk(self) -> tuple[dict[bytes, int], Any]:
        """Persisted vectors for this model, last record per key winning, or none.

        ``self._disk_ok`` records whether the file can be appended to: false when it
        is missing, belongs to another format, or ends in a torn append.
        """
        if self._disk is not None:
            return self._disk
        np = self._np
        self._disk, self._disk_ok = ({}, np.zeros(0, dtype=self._record)), False
        if self._cache_file is None or self._cache_file.parent.is_symlink():
            return self._disk
        try:
            fd = os.open(self._cache_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            return self._disk
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size < _HEADER.size:
                return self._disk
            magic, fingerprint, dim = _HEADER.unpack(stream.read(_HEADER.size))
            if magic != _MAGIC or fingerprint != self.fingerprint or dim != self.dim:
                return self._disk
            count, torn = divmod(info.st_size - _HEADER.size, self._record.itemsize)
            self._disk_ok = not torn
            if not count:
                return self._disk
            # Mapped: a recall reads the keys, and vectors only as it indexes them.
            records = np.memmap(
                stream, dtype=self._record, mode="r", offset=_HEADER.size, shape=(count,)
            )
        keys = records["key"].tobytes()
        index = {keys[i * 32 : (i + 1) * 32]: i for i in range(count)}
        self._disk = (index, records)
        return self._disk

    def _records(self, known: dict[bytes, Any]) -> bytes:
        records = self._np.zeros(len(known), dtype=self._record)
        if known:
            keys = self._np.frombuffer(b"".join(known), dtype="u1")
            records["key"] = keys.reshape(-1, 32)
            records["vector"] = self._np.stack(list(known.values()))
        return records.tobytes()

    def _write_all(self, known: dict[bytes, Any]) -> None:
        """Replace this model's vector file with ``known``; drop other models' files."""
        directory = self._cache_file.parent
        fd, temporary = tempfile.mkstemp(prefix=".vectors-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_HEADER.pack(_MAGIC, self.fingerprint, self.dim))
                stream.write(self._records(known))
            os.replace(temporary, self._cache_file)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        for other in directory.glob("*.vec"):
            if other.name != self._cache_file.name and not other.is_symlink():
                other.unlink(missing_ok=True)

    def _append(self, new: dict[bytes, Any]) -> None:
        """Append whole records in one write, so concurrent appends do not interleave."""
        fd = os.open(self._cache_file, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "wb", buffering=0) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("The vector file must be a regular file")
            stream.write(self._records(new))

    def _save(self, known: dict[bytes, Any], new: dict[bytes, Any]) -> None:
        """Persist new vectors, rewriting the file when it is unusable or mostly stale.

        Best effort: vectors are only a cache, so a failed write leaves recall as is.
        A rewrite costs the whole file, so under a deadline (the hook) only a missing
        or unusable file is rewritten; stale records are compacted by the worker/CLI.
        """
        directory = self._cache_file.parent
        try:
            if directory.is_symlink():
                raise ValueError("The vector directory must not be a symlink")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _, stored = self._load_disk()
            stale = len(stored) + len(new) - len(known)
            if not self._disk_ok or (self.deadline is None and stale > len(known) + COMPACT_SLACK):
                self._write_all(known)
            elif new:
                self._append(new)
        except (OSError, ValueError):
            pass
        self._disk = None

    def vectors(self, texts: list[str]) -> tuple[Any, int]:
        """Unit vectors for ``texts`` and how many had to be encoded now.

        Vectors come from this instance, then the vector file, and are encoded
        otherwise; with a cache directory, newly encoded ones are saved. Past
        ``max_new`` new texts, only that many are encoded and saved, and
        EmbeddingTimeout is raised.
        """
        np = self._np
        digests = [hashlib.sha256(text.encode("utf-8")).digest() for text in texts]
        index, stored = self._load_disk()
        matrix = np.empty((len(texts), self.dim), dtype=np.float32)
        missing: dict[bytes, list[int]] = {}
        from_disk: list[tuple[int, int]] = []
        for row, digest in enumerate(digests):
            if digest in self._memory:
                matrix[row] = self._memory[digest]
            elif digest in index:
                from_disk.append((row, index[digest]))
            else:
                missing.setdefault(digest, []).append(row)
        if from_disk:
            rows, positions = zip(*from_disk)
            matrix[list(rows)] = stored["vector"][list(positions)]
        todo = list(missing.items())
        backlog = self.max_new is not None and len(todo) > self.max_new
        if backlog:
            todo = todo[: self.max_new]
        new = {}
        if todo:
            encoded = self.encode([texts[rows[0]] for _, rows in todo])
            for vector, (digest, rows) in zip(encoded, todo):
                self._memory[digest] = new[digest] = vector
                matrix[rows] = vector
        live = len(set(digests))
        due = self.deadline is None and len(stored) - live > live + COMPACT_SLACK
        if self._cache_file is not None and (new or due) and not self._late():
            known = {d: matrix[row] for row, d in enumerate(digests) if d not in missing}
            self._save({**known, **new}, new)
        if backlog:
            raise EmbeddingTimeout(f"{len(missing) - len(todo)} claims still need vectors")
        return matrix, len(todo)

    def _late(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def similarities(self, query: str, texts: list[str]) -> list[float]:
        """Cosine similarity of the query to each text, in input order."""
        if not texts:
            return []
        query_vector = self.encode([query])[0]
        matrix, _ = self.vectors(texts)
        return (matrix @ query_vector).tolist()


def from_env(
    *,
    local_only: bool = False,
    cache_dir: str | Path | None = None,
    deadline: float | None = None,
    max_new: int | None = None,
) -> StaticEmbedder | None:
    """The embedder named by JEV_WIKI_EMBEDDING_MODEL, or None when unset or unavailable.

    ``default`` selects DEFAULT_MODEL. A missing extra, a model that is not downloaded
    (with ``local_only``), a model that fails to load, or a load that runs past
    ``deadline`` returns None so recall stays lexical rather than failing.
    """
    name = os.environ.get(ENV_VAR, "").strip()
    if not name:
        return None
    try:
        return StaticEmbedder(
            DEFAULT_MODEL if name == "default" else name,
            local_only=local_only,
            cache_dir=cache_dir,
            deadline=deadline,
            max_new=max_new,
        )
    except HookTimeout:
        raise  # The hook's own budget ran out: stop, do not continue lexically.
    except Exception:  # noqa: BLE001 - optional accelerator; lexical recall still works.
        return None
