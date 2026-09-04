import json

import pytest

from equity_selector.feature_mapping import (
    LEGACY_UNIVERSES,
    load_feature_mapping,
    parse_feature_mappings,
    save_feature_mapping,
    serialize_feature_mappings,
)


def test_legacy_mapping_roundtrip_preserves_all_seven_universes(tmp_path):
    legacy = [{f"Target {i}": [f"Feature {i}", "Common"]} for i in range(7)]
    path = tmp_path / "Selected_Features.txt"
    path.write_text("\n".join(map(str, legacy)))
    for universe, mapping in zip(LEGACY_UNIVERSES, legacy):
        assert load_feature_mapping(path, universe) == mapping
    save_feature_mapping(path, "Liquidity Barbell 30", {"Replacement": []})
    document = json.loads(path.read_text())
    assert document["schema_version"] == 1
    assert len(document["universes"]) == 7
    for universe, mapping in zip(LEGACY_UNIVERSES[:6], legacy[:6]):
        assert load_feature_mapping(path, universe) == mapping
    assert load_feature_mapping(path, "Liquidity Barbell 30") == {"Replacement": []}
    assert load_feature_mapping(path, "Intraday Higher Liquidity 30") == legacy[4]


def test_new_file_can_start_with_any_universe_without_positional_padding(tmp_path):
    path = tmp_path / "Selected_Features.txt"
    save_feature_mapping(path, "Liquidity Barbell 30", {"Target": ["Feature"]})
    assert load_feature_mapping(path, "Liquidity Barbell 30") == {"Target": ["Feature"]}
    with pytest.raises(ValueError, match="No selected features"):
        load_feature_mapping(path, "High Liquidity 30")
    save_feature_mapping(path, "Liquidity Barbell 30", {"Target": ["Updated"]})
    assert len(json.loads(path.read_text())["universes"]) == 1


@pytest.mark.parametrize(
    "text",
    [
        "\n{}",
        "{}\n\n{}",
        "\n".join(["{}"] * 12),
        '{"schema_version":2,"universes":{}}',
        "{'Target': 'not a list'}",
    ],
)
def test_ambiguous_or_invalid_mapping_is_not_reinterpreted(text):
    with pytest.raises(ValueError):
        parse_feature_mappings(text)


def test_keyed_order_does_not_change_universe_identity():
    original = {"Sector Spread 30": {"One": ["x"]}, "High Liquidity 30": {"Two": ["y"]}}
    assert parse_feature_mappings(serialize_feature_mappings(original)) == original


def test_new_mapping_removed_if_database_commit_fails(tmp_path):
    from equity_selector.files import commit_with_text

    class FailedCommit:
        rolled_back = False

        def commit(self):
            raise RuntimeError("Injected commit failure")

        def rollback(self):
            self.rolled_back = True

    connection = FailedCommit()
    path = tmp_path / "Selected_Features.txt"
    with pytest.raises(RuntimeError, match="Injected"):
        commit_with_text(connection, path, serialize_feature_mappings({"Example": {}}))
    assert connection.rolled_back
    assert not path.exists()
