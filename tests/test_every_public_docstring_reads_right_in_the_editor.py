"""Every docstring ``willy`` exports reads in the editor's preview as it reads in the source (the owner, 2026-10-10:
"die Vorschau ist dann nicht richtig").

VS Code's preview is the docstring as Pylance turns it into Markdown, with pyright's converter
(``docStringConversion.ts``), and two ways of writing broke it there:

- an example indented under its paragraph is plain text to the converter: its lines run together into one, so a
  ``print(...)`` stood behind the call before it. Written as a fenced ```` ```python ```` block it is code, coloured;
- an entry of an ``Args:`` or ``Attributes:`` block whose line ends with a colon, wrapped right after ``(default:``,
  is not an entry to the converter but a line of text, and ran into the entry above it.

Measured with pyright 1.1.414's language server over all 415 names and members: 7 hovers with a code block before, 47
after; 10 entries run into the one above before, none after. This test holds the source to both rules.
"""

from __future__ import annotations

import inspect
import re
import unittest
from collections.abc import Iterator

import willy
from tests.test_every_public_name_says_what_to_pass_and_what_comes_back import _targets

#: A Google-style section header on a line of its own.
_HEADER = re.compile(r"^(Args|Arguments|Attributes|Returns|Yields|Raises|Note|Notes|Example|Examples|Warning|"
                     r"See Also):\s*$")
_LIST_ITEM = re.compile(r"^(- |\* |\d+[.)] )")


def _docstrings() -> Iterator[tuple[str, str]]:
    seen: set[int] = set()
    yield "willy", willy.__doc__ or ""
    for label, value, kind in _targets():
        raw = value.fget.__doc__ if kind == "property" else value.__doc__
        if raw and id(raw) not in seen:
            seen.add(id(raw))
            yield label, raw


def unfenced_examples(doc: str) -> list[str]:
    """The first line of every block indented past the margin, outside a section and a fence, after a blank line."""
    found: list[str] = []
    in_section = in_fence = list_item = False
    previous_blank = True
    for line in inspect.cleandoc(doc).splitlines():
        text = line.strip()
        if text.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not text:
            previous_blank = True
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_section = bool(_HEADER.match(text))
            list_item = bool(_LIST_ITEM.match(text))
        elif not in_section and previous_blank and not (list_item and indent <= 4):
            found.append(text)
        previous_blank = False
    return found


def entries_ending_in_a_colon(doc: str) -> list[str]:
    """Every line at an entry's indentation in a section that ends with a colon."""
    found: list[str] = []
    in_section = in_fence = False
    entry_indent: int | None = None
    for line in inspect.cleandoc(doc).splitlines():
        text = line.strip()
        if text.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not text:
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_section, entry_indent = bool(_HEADER.match(text)), None
            continue
        if not in_section:
            continue
        if entry_indent is None:
            entry_indent = indent
        if indent == entry_indent and text.endswith(":"):
            found.append(text)
    return found


class EveryPublicDocstringReadsRightInTheEditorTests(unittest.TestCase):

    def test_every_example_is_a_fenced_python_block(self) -> None:
        problems = [f"{label}: {line}" for label, doc in _docstrings() for line in unfenced_examples(doc)]
        self.assertEqual([], problems, "an example indented under its paragraph runs into one line in the editor; "
                                       "write it as a ```python block:\n" + "\n".join(problems))

    def test_no_entry_of_a_section_ends_its_line_with_a_colon(self) -> None:
        problems = [f"{label}: {line}" for label, doc in _docstrings() for line in entries_ending_in_a_colon(doc)]
        self.assertEqual([], problems, "an entry whose line ends with a colon runs into the entry above it in the "
                                       "editor; move the colon's word to the next line:\n" + "\n".join(problems))

    def test_the_readers_find_what_broke_and_pass_what_did_not(self) -> None:
        broken = '''Summary.

            run = f(x)
            print(run)

        Args:
            a (int): One (default:
                1).
            b (int): Two.
        '''
        mended = '''Summary.

        ```python
        run = f(x)
            print(run)
        ```

        - an item that wraps
          onto its next line

        Args:
            a (int): One, with a description that wraps onto
                its next line: still the same entry (default: 1).
            b (int): Two.
        '''
        self.assertEqual(["run = f(x)"], unfenced_examples(broken))
        self.assertEqual(["a (int): One (default:"], entries_ending_in_a_colon(broken))
        self.assertEqual([], unfenced_examples(mended))
        self.assertEqual([], entries_ending_in_a_colon(mended))


if __name__ == "__main__":
    unittest.main()
