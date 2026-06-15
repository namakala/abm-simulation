# Calibration Package

Adjusts PSS-10 item parameters so the population mean falls within literature norm [13, 15].

## Contents

- `calibrate_pss10_population.py` — batch tuning script
- `__init__.py` — public API

## Usage

```python
from src.python.calibration import run_calibration
result = run_calibration(N=200, D=100, seeds=10)
```

## Files keep under 300 lines
