"""Read-only database checks. Exit: 0 checks pass; 1 failure; 2 incomplete evidence."""

from equity_selector.database_audit import main

if __name__ == "__main__":
    raise SystemExit(main())
