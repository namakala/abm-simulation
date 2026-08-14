# Calibration Package

Adjusts PSS-10 item parameters so the population mean falls within the
literature norm [13, 15] and SD [6, 8] (Cohen, Kamarck & Mermelstein 1983).

## Contents

- `calibrate_pss10_population.py` — gradient-descent calibration with persistence
- `run_full_calibration.py` — CLI entry point
- `__init__.py` — public API

## Usage

```python
from src.python.calibration import run_calibration
result = run_calibration(N=200, max_days=100, seeds=5, max_iterations=50)
```

## How It Works

1. **Measure** — runs simulations across multiple seeds, computes PSS-10
   population mean and SD.
2. **Adjust** — nudges all 10 item means proportionally toward the target
   range using gradient descent.
3. **Persist** — on convergence, writes calibrated item means to:
   - `src/python/config.py` (default array)
   - `.env` and `.env.example` (PSS10_ITEM_MEAN line)
   - `src/python/tests/test_pss10_comprehensive.py` (expected_means)
4. **Verify** — runs `test_pss10_comprehensive.py` and
   `TestPSS10Distribution` tests. If they fail, calibration continues.

## No Timeout

The calibration script has no timeout. Let it run until convergence and
all dependent tests pass.

## Files keep under 300 lines
