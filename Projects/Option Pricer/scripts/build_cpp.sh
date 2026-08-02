
#!/usr/bin/env bash
set -euo pipefail
python -m pip install pybind11
cmake -S cpp -B cpp/build -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --config Release -j
printf '
Built extension files:
'
find cpp/build -name 'fast_options*.so' -o -name 'fast_options*.dylib' -o -name 'fast_options*.pyd'
