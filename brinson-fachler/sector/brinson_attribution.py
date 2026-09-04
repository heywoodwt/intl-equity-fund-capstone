#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_carino_linking(r_p_series, r_b_series):
    """
    Calculates the Carino/Menchero multi-period linking coefficients.
    Returns a series of coefficients c_t such that sum(E_t * c_t) = total_effect
    """
    R_p_total = np.prod(1 + r_p_series) - 1
    R_b_total = np.prod(1 + r_b_series) - 1
    
    if np.isclose(R_p_total, R_b_total):
        K = 1 / (1 + R_p_total)
    else:
        K = (R_p_total - R_b_total) / (np.log(1 + R_p_total) - np.log(1 + R_b_total))
        
    k_t = np.zeros(len(r_p_series))
    for i, (rp, rb) in enumerate(zip(r_p_series, r_b_series)):
        if np.isclose(rp, rb):
            k_t[i] = 1 / (1 + rp)
        else:
            k_t[i] = (rp - rb) / (np.log(1 + rp) - np.log(1 + rb))
            
    return K / k_t

def main():
    # Define paths relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    panel_path = os.path.join(script_dir, "combined_panel.csv")
    parquet_path = os.path.join(script_dir, "benchmark_constituents_monthly.parquet")
    monthly_path = os.path.join(script_dir, "combined_monthly.csv")

    print("Loading datasets...")
    panel = pd.read_csv(panel_path)
    parquet = pd.read_parquet(parquet_path)
    monthly = pd.read_csv(monthly_path)

    # Standardize sector names to lowercase and alphanumeric only
    print("Standardizing sector names...")
    parquet['sector_clean'] = parquet['sector'].astype(str).str.lower().replace(r'[^a-z0-9]', '_', regex=True).replace(r'_+', '_', regex=True)
    panel['sector_clean'] = panel['sector'].astype(str).str.lower().replace(r'[^a-z0-9]', '_', regex=True).replace(r'_+', '_', regex=True)

    print("Extracting ticker-to-region mapping dynamically from benchmark constituents...")
    # Map from fsym_id in panel to ticker_region in parquet
    fsym_to_region = parquet[['fsym_id', 'ticker_region']].dropna().drop_duplicates().set_index('fsym_id')['ticker_region'].to_dict()
    panel['ticker_region'] = panel['fsym_id'].map(fsym_to_region)

    # Clean dates to Period format for accurate monthly merging
    print("Standardizing dates...")
    panel['month_dt'] = pd.to_datetime(panel['month']).dt.to_period('M')
    parquet['month_dt'] = pd.to_datetime(parquet['month']).dt.to_period('M')
    monthly['month_dt'] = pd.to_datetime(monthly['month']).dt.to_period('M')

    # 1. Extract Portfolio and Benchmark Weights from combined_monthly.csv
    # We use actual target allocations (bench_sect_wt) rather than a 1/3 ETF blend
    sect_wt_cols = [c for c in monthly.columns if c.startswith('sect_wt_')]
    bench_sect_wt_cols = [c for c in monthly.columns if c.startswith('bench_sect_wt_')]

    w_p = monthly[['month_dt'] + sect_wt_cols].melt(id_vars='month_dt', var_name='sector', value_name='w_p')
    w_p['sector'] = w_p['sector'].str.replace('sect_wt_', '')

    w_b = monthly[['month_dt'] + bench_sect_wt_cols].melt(id_vars='month_dt', var_name='sector', value_name='w_b')
    w_b['sector'] = w_b['sector'].str.replace('bench_sect_wt_', '')

    weights_df = pd.merge(w_p, w_b, on=['month_dt', 'sector'], how='outer').fillna(0)

    # 2. Calculate Portfolio (Fund) Sector Returns (R_p)
    print("Calculating portfolio sector returns...")
    r_p = panel.groupby(['month_dt', 'sector_clean'])['ret'].mean().reset_index(name='r_p')

    # 3. Impute Benchmark Sector Returns using weight-change dynamics from actual target allocations
    print("Imputing benchmark sector returns from target weight dynamics...")
    weight_dynamics = pd.merge(w_b, monthly[['month_dt', 'bench_avg_ret']], on='month_dt', how='inner')
    weight_dynamics = weight_dynamics.sort_values(['sector', 'month_dt'])
    weight_dynamics['prev_w_b'] = weight_dynamics.groupby('sector')['w_b'].shift(1)
    
    # Calculate imputed return, falling back to overall benchmark return when there is no previous month
    weight_dynamics['r_b_imputed'] = (weight_dynamics['w_b'] / weight_dynamics['prev_w_b']) * (1.0 + weight_dynamics['bench_avg_ret']) - 1.0
    weight_dynamics['r_b_imputed'] = weight_dynamics['r_b_imputed'].fillna(weight_dynamics['bench_avg_ret'])
    
    # Cap extreme returns to avoid anomalies but allow a reasonable range since we now use multi-period smoothing
    weight_dynamics['r_b_imputed'] = weight_dynamics['r_b_imputed'].clip(lower=-0.99, upper=2.0)

    # 4. Merge returns and weights to calculate Brinson-Fachler effects
    brinson = pd.merge(weights_df, r_p, left_on=['month_dt', 'sector'], right_on=['month_dt', 'sector_clean'], how='left').fillna(0)
    brinson = pd.merge(brinson, weight_dynamics[['month_dt', 'sector', 'r_b_imputed']], on=['month_dt', 'sector'], how='inner')
    brinson.rename(columns={'r_b_imputed': 'r_b'}, inplace=True)
    brinson = pd.merge(brinson, monthly[['month_dt', 'bench_avg_ret', 'fund_ret']].rename(columns={'bench_avg_ret': 'R_B_total', 'fund_ret': 'R_P_total'}), on='month_dt', how='left')

    # Calculate single-period unlinked Brinson effects
    brinson['allocation_unlinked'] = (brinson['w_p'] - brinson['w_b']) * (brinson['r_b'] - brinson['R_B_total'])
    brinson['selection_unlinked'] = brinson['w_b'] * (brinson['r_p'] - brinson['r_b'])
    brinson['interaction_unlinked'] = (brinson['w_p'] - brinson['w_b']) * (brinson['r_p'] - brinson['r_b'])
    brinson['total_active_unlinked'] = brinson['allocation_unlinked'] + brinson['selection_unlinked'] + brinson['interaction_unlinked']

    # 5. Apply Multi-Period Linking (Carino/Menchero algorithm)
    print("Applying Carino/Menchero multi-period smoothing...")
    # Calculate linking coefficient for each month
    monthly_totals = brinson.groupby('month_dt')[['R_P_total', 'R_B_total']].first().reset_index()
    monthly_totals['linking_coef'] = calculate_carino_linking(monthly_totals['R_P_total'], monthly_totals['R_B_total'])
    
    brinson = pd.merge(brinson, monthly_totals[['month_dt', 'linking_coef']], on='month_dt', how='left')
    
    # Apply coefficient to smooth effects
    brinson['allocation'] = brinson['allocation_unlinked'] * brinson['linking_coef']
    brinson['selection'] = brinson['selection_unlinked'] * brinson['linking_coef']
    brinson['interaction'] = brinson['interaction_unlinked'] * brinson['linking_coef']
    brinson['total_active_return'] = brinson['allocation'] + brinson['selection'] + brinson['interaction']

    # 6. Aggregate results
    print("Aggregating smoothed results...")
    summary = brinson.groupby('sector')[['allocation', 'selection', 'interaction', 'total_active_return']].sum().reset_index()

    # Save outputs to CSV
    output_path = os.path.join(script_dir, "brinson_attribution_results_parquet.csv")
    summary.to_csv(output_path, index=False)

    print(f"Results saved to:\n  - {output_path}")

    # Print Brinson table
    print("\n--- Brinson-Fachler Attribution (Multi-Period Smoothed) ---")
    summary_sorted = summary.sort_values('total_active_return', ascending=False)
    formatted = summary_sorted.copy()
    for col in ['allocation', 'selection', 'interaction', 'total_active_return']:
        formatted[col] = formatted[col].map(lambda x: f'{x*100:.4f}%')
    print(formatted.to_string(index=False))

    # 7. Generate Chart
    print("\nGenerating charts...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    summary_sorted_plot = summary.sort_values('total_active_return')
    x = np.arange(len(summary_sorted_plot['sector']))
    width = 0.25

    ax.barh(x - width, summary_sorted_plot['allocation'], width, label='Allocation', color='#4c72b0')
    ax.barh(x, summary_sorted_plot['selection'], width, label='Selection', color='#55a868')
    ax.barh(x + width, summary_sorted_plot['interaction'], width, label='Interaction', color='#c44e52')
    
    ax.set_yticks(x)
    ax.set_yticklabels(summary_sorted_plot['sector'])
    ax.set_title("Brinson-Fachler Attribution (Carino Multi-Period Smoothed)", fontsize=14)
    ax.set_xlabel('Cumulative Return Contribution', fontsize=12)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.legend(fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(script_dir, "brinson_attribution_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved to: {chart_path}")
    print("Done!")

if __name__ == "__main__":
    main()
