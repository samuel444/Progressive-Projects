
# Database design

SQLite is appropriate for this local portfolio project because the outputs are
tabular, queryable and small enough to fit comfortably in one file.

Store in SQL:
- run metadata and valuation dates;
- adjusted equity prices;
- option-chain snapshots;
- model-comparison metrics;
- priced option universe;
- portfolio positions and current risk;
- scenario definitions and aggregated/position results;
- Greek attribution results.

Do not store in SQL:
- full Monte Carlo path matrices;
- large intermediate NumPy arrays;
- figures.

Those are better as `.npz`, Parquet or image files under `data/processed` and
`outputs/figures`.
