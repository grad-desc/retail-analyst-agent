"""Prompt registry: prompts live in files, are re-read when they change (no redeploy), and are
rendered in a single pass so user-supplied text can never inject template variables."""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

_VAR = re.compile(r"\{\{(\w+)\}\}")


class PromptError(RuntimeError):
    pass


class PromptRegistry:
    def __init__(self, directory) -> None:
        self.dir = Path(directory)
        self._cache: Dict[str, Tuple[int, str]] = {}

    def load(self, name: str) -> str:
        path = self.dir / f"{name}.md"
        try:
            mtime = path.stat().st_mtime_ns
        except FileNotFoundError as e:
            raise PromptError(f"prompt '{name}' not found in {self.dir}") from e
        cached = self._cache.get(name)
        if cached is None or cached[0] != mtime:  # hot reload on edit
            self._cache[name] = (mtime, path.read_text(encoding="utf-8"))
        return self._cache[name][1]

    def render(self, name: str, **variables) -> str:
        variables.setdefault("persona", self.load("persona"))
        variables.setdefault("today", datetime.now(timezone.utc).date().isoformat())
        text = self.load(name)
        return _VAR.sub(lambda m: str(variables.get(m.group(1), m.group(0))), text)

    def version(self, name: str) -> str:
        """Short content hash, logged with every trace so behaviour changes are attributable."""
        return hashlib.sha1(self.load(name).encode()).hexdigest()[:8]
