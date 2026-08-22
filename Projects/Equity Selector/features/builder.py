import logging

from .registry import FEATURE_GROUPS

logger = logging.getLogger(__name__)


def build_features(df, groups, options=None):
    if isinstance(groups, str):
        groups = [groups]

    if options is None:
        options = {}

    for group in groups:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group}")

        logger.info("Adding feature group: %s", group)
        group_options = options.get(group, {})
        df = FEATURE_GROUPS[group](df, **group_options)

    return df
