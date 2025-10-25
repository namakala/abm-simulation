## IMPORT MODULES

import pandas as pd


## EDA

# Read data
tbl_model = pd.read_csv("data/raw/model.csv")
tbl_agent = pd.read_csv("data/raw/agent.csv")

tbl_model.info()
tbl_agent.info()

# Basic overview
tbl_model.describe().T
tbl_agent.describe().T

# Table summary
agent_summary = (
    tbl_agent.groupby('AgentID')
    .agg({'pss10':'mean', 'resilience':'mean', 'current_stress':'mean'})
    .rename(columns={'pss10':'avg_pss10', 'resilience':'avg_resilience', 'current_stress':'avg_stress'})
)

step_summary = (
    tbl_agent.groupby('Step')
    .agg({'pss10':'mean', 'resilience':'mean', 'current_stress':'mean'})
    .rename(columns={'pss10':'avg_pss10', 'resilience':'avg_resilience', 'current_stress':'avg_stress'})
)
