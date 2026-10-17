"""Doctest the code blocks in the README, the docs and the docstrings."""

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

from sybil import Sybil
from sybil.parsers import markdown, rest

_flags = ELLIPSIS | NORMALIZE_WHITESPACE

pytest_collect_file = (
    Sybil(
        parsers=[
            markdown.PythonCodeBlockParser(doctest_optionflags=_flags),
            markdown.SkipParser(),
        ],
        patterns=["*.md"],
    )
    + Sybil(parsers=[rest.DocTestParser(optionflags=_flags)], patterns=["*.py"])
).pytest()
