from pathlib import Path
import pickle
import logging
from typing import Any, Iterable, Optional

import pandas as pd


def find_project_root(start: Optional[Path] = None, marker_dirs: Iterable[str] = ('data',)) -> Path:
    """Search upward from `start` (or cwd) for a directory that contains any of `marker_dirs`.

    Returns the first directory that contains a marker directory. Falls back to cwd if none found.
    """
    cur = Path(start) if start is not None else Path.cwd()
    cur = cur.resolve()
    root = cur
    for parent in [cur] + list(cur.parents):
        for m in marker_dirs:
            if (parent / m).exists():
                return parent
    return root


def resolve_path(path_like: Optional[str] = None, project_root: Optional[Path] = None) -> Path:
    """Resolve `path_like` relative to `project_root` if provided, else return Path(path_like).

    If `path_like` is None, returns a sensible default inside the project (e.g., project_root/data).
    """
    if project_root is None:
        project_root = find_project_root()
    if path_like is None:
        return project_root
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def setup_logger(name: str = 'car_price', level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
