"""The knowledge an atom's nucleus draws from."""

from __future__ import annotations

import os
from pathlib import Path

from chatomon.errors import ChatomonConfigError

__all__ = ["Knowledge"]


class Knowledge:
    """Plain-text knowledge added to a prompt in a delimited section.

    The same object can be given to the atom and, explicitly, to an orbital such
    as a judge: chatomon never shares it implicitly.

    Args:
        text: The knowledge text.
        title: Optional title rendered in the section header.

    Raises:
        ChatomonConfigError: If `text` is empty.

    Examples:
        ```python
        from chatomon import Knowledge

        faq = Knowledge("Returns are accepted within 30 days.", title="FAQ")
        assert faq.render().startswith('<knowledge title="FAQ">')
        ```
    """

    def __init__(self, text: str, *, title: str | None = None) -> None:
        if not text.strip():
            raise ChatomonConfigError("Knowledge text must not be empty")
        self.text = text
        self.title = title

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        title: str | None = None,
        encoding: str = "utf-8",
    ) -> Knowledge:
        """Read knowledge from a text file.

        Args:
            path: Path of the file.
            title: Optional title rendered in the section header.
            encoding: File encoding.

        Returns:
            The knowledge.

        Raises:
            OSError: If the file cannot be read.
            ChatomonConfigError: If the file is empty.
        """  # noqa: DOC502 - raised by open() and the constructor
        return cls(Path(path).read_text(encoding=encoding), title=title)

    def render(self) -> str:
        """Render the knowledge as a delimited prompt section.

        Returns:
            The text wrapped in `<knowledge>` tags.
        """
        header = f'<knowledge title="{self.title}">' if self.title else "<knowledge>"
        return f"{header}\n{self.text}\n</knowledge>"

    def __repr__(self) -> str:
        return f"Knowledge(title={self.title!r}, chars={len(self.text)})"
