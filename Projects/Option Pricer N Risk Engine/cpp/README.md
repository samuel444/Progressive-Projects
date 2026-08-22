
# Optional C++ acceleration

The Python/NumPy implementation remains the reference implementation. NumPy,
SciPy and scikit-learn already execute most numerical kernels in compiled code,
so rewriting every formula in C++ would add complexity without a meaningful
benefit.

The extension targets the two genuinely Python-loop-heavy operations:

1. **Batch implied-volatility inversion** across thousands of contracts.
2. **Batch option revaluation** across positions multiplied by scenarios.

A C++ Monte Carlo terminal-price accumulator would also be worthwhile only when
moving to millions of simulations or when avoiding the memory cost of storing
all paths. The current 10,000-path NumPy implementation is already appropriate
for this CV project.

Build with `make build-cpp`, then copy/symlink the resulting `fast_options`
extension onto the Python path. Always benchmark against the vectorised Python
reference before claiming a speed-up.
