"""Versioned universe-keyed feature maps, with explicit legacy line compatibility."""

import ast
import copy
import json
from pathlib import Path

from .files import atomic_write_text

LEGACY_UNIVERSES = (
    "High Liquidity 30",
    "Medium Liquidity 30",
    "Lower Liquidity 30",
    "Sector Spread 30",
    "Intraday High Liquidity 30",
    "Intraday Medium Liquidity 30",
    "Liquidity Barbell 30",
    "Institutional Liquidity 60",
    "Medium Small Liquidity 60",
    "Medium Large Liquidity 60",
    "All Liquidity 90",
)
LEGACY_INDICES = {name: index for index, name in enumerate(LEGACY_UNIVERSES)}


def canonical_universe(name):
    return "Intraday High Liquidity 30" if name == "Intraday Higher Liquidity 30" else name


def _validate(mapping):
    if not isinstance(mapping, dict):
        raise ValueError("Feature mapping must be a Target -> Features dictionary")
    for target, features in mapping.items():
        if (
            not isinstance(target, str)
            or not isinstance(features, list)
            or not all(isinstance(feature, str) for feature in features)
        ):
            raise ValueError("Feature mapping requires string targets and lists of feature names")
    return copy.deepcopy(mapping)


def parse_feature_mappings(text, legacy_indices=None):
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and "schema_version" in document:
        if document["schema_version"] != 1 or not isinstance(document.get("universes"), dict):
            raise ValueError("Unsupported feature mapping schema")
        result = {}
        for universe, mapping in document["universes"].items():
            key = canonical_universe(universe)
            if key in result:
                raise ValueError(f"Duplicate universe identity: {key}")
            result[key] = _validate(mapping)
        return result
    indices = LEGACY_INDICES if legacy_indices is None else legacy_indices
    by_index = {index: canonical_universe(name) for name, index in indices.items()}
    lines = text.splitlines()
    result = {}
    for index, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"Blank legacy feature line {index}; universe identity is ambiguous")
        if index not in by_index:
            raise ValueError(f"No universe identity for legacy feature line {index}")
        try:
            mapping = ast.literal_eval(line)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Invalid legacy feature line {index}") from error
        result[by_index[index]] = _validate(mapping)
    return result


def load_feature_mapping(path, universe, legacy_indices=None):
    maps = parse_feature_mappings(Path(path).read_text(), legacy_indices)
    key = canonical_universe(universe)
    if key not in maps:
        raise ValueError(f"No selected features stored for universe: {universe}")
    return maps[key]


def serialize_feature_mappings(mappings):
    checked = {canonical_universe(name): _validate(value) for name, value in mappings.items()}
    if len(checked) != len(mappings):
        raise ValueError("Duplicate universe aliases")
    return json.dumps({"schema_version": 1, "universes": checked}, indent=2) + "\n"


def updated_feature_mapping_text(path, universe, mapping, legacy_indices=None):
    path = Path(path)
    maps = parse_feature_mappings(path.read_text(), legacy_indices) if path.exists() else {}
    maps[canonical_universe(universe)] = _validate(mapping)
    return serialize_feature_mappings(maps)


def save_feature_mapping(path, universe, mapping):
    atomic_write_text(path, updated_feature_mapping_text(path, universe, mapping))
