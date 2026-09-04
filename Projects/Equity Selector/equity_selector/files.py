"""Atomic text replacement coordinated with an open SQLite transaction."""

import os
import tempfile
from pathlib import Path


def atomic_write_text(path, text):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def commit_with_text(connection, path, text):
    """Rollback SQL and restore mapping on ordinary I/O/commit exceptions.

    A process/power failure between resources is not cross-resource atomic;
    callers should retain a backup before production schema migrations.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8") if path.exists() else None
    changed = False
    try:
        atomic_write_text(path, text)
        changed = True
        connection.commit()
    except Exception:
        connection.rollback()
        if changed:
            if original is None:
                path.unlink()
            else:
                atomic_write_text(path, original)
        raise
