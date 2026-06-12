"""FactWorld — controlled instrument for parametric vs in-context, static vs stateful knowledge."""
from .config import WorldConfig
from .corpus import (
    Corpus,
    Document,
    aux_operator_documents,
    build_corpus,
    history_only_documents,
    recall_documents,
)
from .edits import Edit, append_pair
from .eval import Episode, EvalItem, easy_suite, hard_suite, recall_suite
from .oracle import Oracle
from .render import Renderer, classify
from .tokenizer import Tokenizer
from .world import Event, World

__all__ = [
    "WorldConfig", "World", "Event", "Oracle",
    "Renderer", "classify", "Tokenizer",
    "Episode", "EvalItem", "recall_suite", "easy_suite", "hard_suite",
    "Edit", "append_pair",
    "Document", "Corpus", "recall_documents", "history_only_documents",
    "aux_operator_documents", "build_corpus",
]
