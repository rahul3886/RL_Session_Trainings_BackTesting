import pandas as pd
import numpy as np

# Load trade log
df = pd.read_csv("c:/RAHUL BOGI/RL_Session Trainings/xauusd_v3/output/backtest/trade_log.csv")

# Ensure PnL is numeric
df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
df['win'] = df['pnl'] > 0

# Scenario analysis
scenarios = df.groupby('opening_scenario_label').agg(
    total_trades=('pnl', 'count'),
    wins=('win', 'sum'),
    total_pnl=('pnl', 'sum'),
    ny_open_trades=('opened_in_ny_open', 'sum'),
    late_trades=('opened_in_ny_open', lambda x: (~x).sum())
).reset_index()

scenarios['win_rate'] = (scenarios['wins'] / scenarios['total_trades'] * 100).round(1)

print("\n--- SCENARIO BREAKDOWN (BACKTEST) ---")
print(scenarios[['opening_scenario_label', 'total_trades', 'win_rate', 'total_pnl', 'ny_open_trades', 'late_trades']])

# Overall stats
print(f"\nTotal Trades: {len(df)}")
print(f"Overall Win Rate: {(df['win'].mean()*100):.1f}%")
print(f"Overall PnL: {df['pnl'].sum():.2f}")
print(f"NY Open Trades: {df['opened_in_ny_open'].sum()}")
print(f"Late Trades: {(~df['opened_in_ny_open']).sum()}")
