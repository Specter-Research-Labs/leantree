from dataclasses import dataclass
from pathlib import Path
from typing import Self

from leantree.core.proof_tree import ProofTree
from leantree.file_span import FileSpan

@dataclass
class StoredError:
    error: str

    def serialize(self) -> dict:
        return {
            "error": self.error
        }

    @classmethod
    def deserialize(cls, data: dict) -> Self:
        return cls(data["error"])

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(str(e))


@dataclass(eq=False)
class LeanTacticBlock:
    theorem: "LeanTheorem"
    tree: ProofTree | StoredError | None
    span: FileSpan

    # TODO: the decision what to serialize should be in the dataset generator, not here
    def serialize(self) -> dict:
        if isinstance(self.tree, StoredError):
            tree = {"nodes": [], "root_id": None, "error": self.tree.error}
        else:
            tree = {**self.tree.serialize(), "error": None}
        return {
            "tree": tree,
            "span": self.span.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict, theorem: "LeanTheorem") -> "LeanTacticBlock":
        tree_data = data["tree"]
        error = tree_data.get("error")
        if error is not None:
            tree = StoredError(error)
        else:
            tree = ProofTree.deserialize(tree_data)
        return LeanTacticBlock(
            theorem=theorem,
            tree=tree,
            span=FileSpan.deserialize(data["span"]),
        )

# TODO: start_proof(env) method
@dataclass(eq=False)
class LeanTheorem:
    span: FileSpan
    file: "LeanFile | None"

    # Single theorem can contain multiple `by` clauses.
    by_blocks: list[LeanTacticBlock]

    # Can contain clauses `open` (including `hiding`, `renaming`, etc.), `variable`, `universe`
    context: list[str]
    name: str | None = None

    @staticmethod
    def _serialize_block(b) -> dict:
        if isinstance(b, StoredError):
            return {
                "tree": {"nodes": [], "root_id": None, "error": b.error},
                "span": None,
            }
        return b.serialize()

    def serialize(self) -> dict:
        return {
            "span": self.span.serialize(),
            "by_blocks": [self._serialize_block(b) for b in self.by_blocks],
            "context": self.context,
            "name": self.name,
            "error": None,
        }

    @classmethod
    def deserialize(cls, data: dict, file: "LeanFile | None" = None) -> "LeanTheorem":
        by_blocks = []
        thm = LeanTheorem(
            span=FileSpan.deserialize(data["span"]),
            file=file,
            by_blocks=by_blocks,
            context=data["context"],
            name=data.get("name"),
        )
        for block_data in data["by_blocks"]:
            if "tree" not in block_data:
                # Legacy format: bare StoredError in by_blocks
                by_blocks.append(StoredError.deserialize(block_data))
            else:
                by_blocks.append(LeanTacticBlock.deserialize(block_data, thm))
        return thm

    def load_source(self) -> str:
        return self.span.read_from_file(self.file.path)

@dataclass(eq=False)
class LeanFile:
    path: Path
    imports: list[str]
    theorems: list[LeanTheorem | StoredError]
    relative_path: Path | None = None

    @staticmethod
    def _serialize_theorem(t) -> dict:
        if isinstance(t, StoredError):
            return {
                "span": None,
                "by_blocks": [],
                "context": [],
                "name": None,
                "error": t.error,
            }
        return t.serialize()

    def serialize(self) -> dict:
        return {
            "path": str(self.relative_path or self.path),
            "imports": self.imports,
            "theorems": [self._serialize_theorem(t) for t in self.theorems]
        }

    @classmethod
    def deserialize(cls, data: dict) -> "LeanFile":
        theorems = []
        file = LeanFile(
            path=Path(data["path"]),
            imports=data["imports"],
            theorems=theorems,
        )
        for thm_data in data["theorems"]:
            error = thm_data.get("error")
            if error is not None:
                theorems.append(StoredError(error))
            else:
                theorems.append(LeanTheorem.deserialize(thm_data, file))
        return file
