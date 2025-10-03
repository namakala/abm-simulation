import numpy as np
import pandas as pd
from src.python.model import StressModel

model = StressModel(N=10, max_days=100, seed=42)

while True:
    model.step()
    if not model.running:
        break

tbl = model.get_time_series_data()

print(tbl)
