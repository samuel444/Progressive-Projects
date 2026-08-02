
# C++ acceleration decisions

## Worth accelerating

- **Implied volatility for a large chain:** the current Python code calls a root
  solver once per quote. A batch C++ loop avoids thousands of Python calls.
- **Scenario repricing at production scale:** positions × scenarios can reach
  millions of Black-Scholes evaluations. A batch kernel can materially reduce
  loop overhead.
- **Very large Monte Carlo simulations:** a streaming C++ implementation can
  avoid storing every path and can parallelise payoff accumulation.

## Keep in Python

- pandas transformations and SQL persistence;
- plotting and notebook presentation;
- scikit-learn models and PCA, which already call compiled libraries;
- GARCH from `arch`, which already performs its numerical work below Python;
- the 10,000-path NumPy Monte Carlo used in this project unless benchmarking
  proves a bottleneck.

The CV project should lead with correctness, validation and clear findings. C++
is presented as an optional engineering extension, not as complexity for its own
sake.
