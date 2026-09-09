"""Original research orchestration, launched explicitly through the CLI.

Large stages retain their original sequential state and researcher settings.
Shared calculations and validation live in the parent package and are testable
independently. Importing a stage does not execute its research workload.
"""
