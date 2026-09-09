import logging

from .registry import TARGET_GROUPS

logger = logging.getLogger(__name__)


def build_targets(df, groups, options=None):
    if isinstance(groups, str):
        groups = [groups]

    if options is None:
        options = {}

    for group in groups:
        if group not in TARGET_GROUPS:
            raise ValueError(f"Unknown target group: {group}")

        logger.info("Adding target group: %s", group)
        group_options = options.get(group, {})
        df = TARGET_GROUPS[group](df, **group_options)

    return df
