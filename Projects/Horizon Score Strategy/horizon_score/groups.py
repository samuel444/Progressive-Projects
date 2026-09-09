"""Canonical reconstruction groups; simulation search space is unchanged.
Legacy names remain available for already-frozen runs. Validators accept every
configuration that the simulation launcher can select.
"""

from copy import deepcopy

SIMULATION_GROUPS = [
    {
        "Name": "Balanced",
        "Ranking": 0.3,
        "Direction": 0.25,
        "Risk": 0.25,
        "Opportunity": 0.15,
        "Special": 0.05,
    },
    {
        "Name": "Equal Weight",
        "Ranking": 0.2,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.2,
        "Special": 0.2,
    },
    {
        "Name": "Core Balanced",
        "Ranking": 0.35,
        "Direction": 0.3,
        "Risk": 0.25,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking Heavy",
        "Ranking": 0.5,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Risk",
        "Ranking": 0.45,
        "Direction": 0.15,
        "Risk": 0.3,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Direction",
        "Ranking": 0.45,
        "Direction": 0.3,
        "Risk": 0.15,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Direction Heavy",
        "Ranking": 0.25,
        "Direction": 0.45,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Direction And Risk",
        "Ranking": 0.25,
        "Direction": 0.4,
        "Risk": 0.3,
        "Opportunity": 0.05,
        "Special": 0.0,
    },
    {
        "Name": "Risk Heavy",
        "Ranking": 0.25,
        "Direction": 0.2,
        "Risk": 0.45,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Conservative",
        "Ranking": 0.3,
        "Direction": 0.15,
        "Risk": 0.45,
        "Opportunity": 0.05,
        "Special": 0.05,
    },
    {
        "Name": "Opportunity Heavy",
        "Ranking": 0.25,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.35,
        "Special": 0.0,
    },
    {
        "Name": "Ranking And Special",
        "Ranking": 0.4,
        "Direction": 0.15,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.15,
    },
]

LEGACY_GROUPS = [
    {
        "Name": "Equal Weight Baseline",
        "Ranking": 0.2,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.2,
        "Special": 0.2,
    },
    {
        "Name": "Ranking Focused",
        "Ranking": 0.5,
        "Direction": 0.2,
        "Risk": 0.2,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
    {
        "Name": "Risk Aware",
        "Ranking": 0.35,
        "Direction": 0.2,
        "Risk": 0.35,
        "Opportunity": 0.1,
        "Special": 0.0,
    },
]


def validation_groups():
    return deepcopy(SIMULATION_GROUPS + LEGACY_GROUPS)
