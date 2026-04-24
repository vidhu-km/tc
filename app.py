import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta

# Constants
BBL_PER_M3 = 6.2898
B = 0.95
DI_ANNUAL = 2.4749435
DI_DAILY = DI_ANNUAL / 365.0  # daily nominal decline

# 1) Read Excel and expand monthly volumes to daily rates (bbl/d)
def read_and_expand(file):
    df = pd.read_excel(file)
    required = [
        'uwi','year',
        'jan_volume','feb_volume','mar_volume','apr_volume',
        'may_volume','jun_volume','jul_volume','aug_volume',
        'sep_volume','oct_volume','nov_volume','dec_volume'
    ]
    for c in required:
        if c not in df.columns:
            st.error(f"Missing required column: {c}")
            return None

    months_map = {
        'jan_volume': 1, 'feb_volume': 2, 'mar_volume': 3, 'apr_volume': 4,
        'may_volume': 5, 'jun_volume': 6, 'jul_volume': 7, 'aug_volume': 8,
        'sep_volume': 9, 'oct_volume': 10, 'nov_volume': 11, 'dec_volume': 12
    }

    # Melt monthly volumes to long format
    df_long = df.melt(id_vars=['uwi','year'],
                      value_vars=list(months_map.keys()),
                      var_name='month_col',
                      value_name='volume_m3')
    df_long['month'] = df_long['month_col'].map(months_map)
    df_long = df_long[['uwi','year','month','volume_m3']]
    df_long = df_long.dropna(subset=['volume_m3'])

    daily_rows = []
    for _, row in df_long.iterrows():
        y = int(row['year'])
        m = int(row['month'])
        uwi = row['uwi']
        vol_m3 = row['volume_m3']

        # days in month
        days = calendar.monthrange(y, m)[1]
        if vol_m3 <= 0 or np.isnan(vol_m3):
            continue

        vol_bbl = vol_m3 * BBL_PER_M3
        daily_rate = vol_bbl / days if days > 0 else 0.0  # bbl/d

        for day in range(1, days + 1):
            date = datetime(year=y, month=m, day=day)
            daily_rows.append({
                'uwi': uwi,
                'date': date,
                'rate_bbl_d': daily_rate
            })

    df_daily = pd.DataFrame(daily_rows)
    if df_daily.empty:
        return df_daily
    df_daily = df_daily.sort_values(['uwi','date']).reset_index(drop=True)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    return df_daily

# 2) For each well, compute peak, normalize, and forecast beyond last data point
def compute_curves(df_daily):
    if df_daily is None or df_daily.empty:
        return None

    # Peak rate per well
    peak_by_well = df_daily.groupby('uwi')['rate_bbl_d'].max().reset_index().rename(columns={'rate_bbl_d':'peak_rate'})

    # Peak date per well (date of max rate)
    idx_peak = df_daily.groupby('uwi')['rate_bbl_d'].idxmax()
    peak_dates = df_daily.loc[idx_peak, ['uwi','date']].rename(columns={'date':'peak_date'})

    # Merge peak info for normalization
    df_with_peak = df_daily.merge(peak_by_well, on='uwi')
    df_with_peak = df_with_peak.merge(peak_dates, on='uwi')

    # Normalized rate and time since peak
    df_with_peak['norm_rate'] = df_with_peak['rate_bbl_d'] / df_with_peak['peak_rate']
    df_with_peak['t_since_peak'] = (df_with_peak['date'] - df_with_peak['peak_date']).dt.days
    df_with_peak['t_since_peak'] = df_with_peak['t_since_peak'].astype(int)

    # Forecast beyond last data point for each well
    forecast_rows = []
    wells = df_with_peak['uwi'].unique()
    for uwi in wells:
        # last observed date for this well
        last_date = df_daily[df_daily['uwi'] == uwi]['date'].max()

        # peak info for this well
        peak_row = df_with_peak[df_with_peak['uwi'] == uwi].iloc[0]
        peak_date = peak_row['peak_date']
        peak_rate = peak_row['peak_rate']

        # qi: last normalized rate
        last_norm = df_with_peak[(df_with_peak['uwi'] == uwi)].sort_values('date').tail(1)
        if last_norm.empty:
            continue
        qi = float(last_norm['norm_rate'].values[0])
        if qi <= 0:
            continue

        t_model = 0
        while True:
            t_model += 1
            q_norm = qi / ((1 + B * DI_DAILY * t_model) ** (1.0 / B))  # normalized
            r_actual = q_norm * peak_rate  # de-normalized (bbl/d)

            if r_actual <= 2.0:
                break

            date_next = last_date + timedelta(days=t_model)
            t_since_peak = (date_next - peak_date).days
            forecast_rows.append({
                'uwi': uwi,
                'date': date_next,
                't_since_peak': int(t_since_peak),
                'norm_rate': float(q_norm),
                'denorm_rate': float(r_actual)
            })

    df_forecast = pd.DataFrame(forecast_rows)

    # Observed data for plotting/alignment
    df_obs = df_with_peak[['uwi','date','t_since_peak','norm_rate']].rename(columns={'norm_rate':'norm_rate'})

    # Combine observed and forecast into a single time series per well
    df_all = pd.concat([df_obs, df_forecast], ignore_index=True, sort=False)

    # Compute peak-rate percentiles across wells for de-normalization later
    peak_rates = df_daily.groupby('uwi')['rate_bbl_d'].max().values
    P10_peak = np.percentile(peak_rates, 10)
    P50_peak = np.percentile(peak_rates, 50)
    P90_peak = np.percentile(peak_rates, 90)

    # Percentiles by time since peak (normalized)
    by_t = df_all.groupby('t_since_peak')['norm_rate'].apply(list).to_dict()
    t_values = sorted(by_t.keys())

    p10 = {}
    p50 = {}
    p90 = {}

    for t in t_values:
        arr = np.array(by_t[t])
        if arr.size == 0:
            continue
        p10[t] = np.percentile(arr, 10)
        p50[t] = np.percentile(arr, 50)
        p90[t] = np.percentile(arr, 90)

    # Enforce flat segment: 0..44 days after peak set to 1.0
    for t in range(0, 45):
        p10[t] = 1.0
        p50[t] = 1.0
        p90[t] = 1.0

    # Unified time axis
    t_axis = sorted(set(list(t_values) + list(range(0, 45))))
    P10_norm = []
    P50_norm = []
    P90_norm = []

    # Fill normalized percentile curves, with simple interpolation if needed
    for t in t_axis:
        v10 = p10.get(t, None)
        v50 = p50.get(t, None)
        v90 = p90.get(t, None)

        if (v10 is None) or (v50 is None) or (v90 is None):
            # Simple interpolation using available points
            known_ts = np.array(sorted([k for k in p10.keys() if k is not None]))
            if known_ts.size == 0:
                P10_norm.append(np.nan)
                P50_norm.append(np.nan)
                P90_norm.append(np.nan)
                continue
            P10_norm.append(float(np.interp(t, known_ts, [p10[k] for k in known_ts])))
            P50_norm.append(float(np.interp(t, known_ts, [p50[k] for k in known_ts])))
            P90_norm.append(float(np.interp(t, known_ts, [p90[k] for k in known_ts])))
        else:
            P10_norm.append(v10)
            P50_norm.append(v50)
            P90_norm.append(v90)

    P10_norm = np.array(P10_norm, dtype=float)
    P50_norm = np.array(P50_norm, dtype=float)
    P90_norm = np.array(P90_norm, dtype=float)

    # De-normalize percentile curves using peak percentiles
    P10_actual = P10_norm * P10_peak
    P50_actual = P50_norm * P50_peak
    P90_actual = P90_norm * P90_peak

    cum_P10 = np.cumsum(np.nan_to_num(P10_actual))
    cum_P50 = np.cumsum(np.nan_to_num(P50_actual))
    cum_P90 = np.cumsum(np.nan_to_num(P90_actual))

    return {
        't_axis': np.array(t_axis),
        'P10_actual': P10_actual,
        'P50_actual': P50_actual,
        'P90_actual': P90_actual,
        'cum_P10': cum_P10,
        'cum_P50': cum_P50,
        'cum_P90': cum_P90,
        'overlay_df': df_all
    }

# 3) Run the app
def main():
    st.set_page_config(layout="wide")
    st.title("Production type curves from well data")
    st.write(
        "Upload `wellprod.xlsx` with columns: uwi, year, jan_volume, feb_volume, mar_volume, apr_volume, may_volume, jun_volume, jul_volume, aug_volume, sep_volume, oct_volume, nov_volume, dec_volume (volumes in m³)."
    )

    uploaded = st.file_uploader("Upload wellprod.xlsx", type=["xlsx"])
    if uploaded is None:
        st.info("Awaiting file upload.")
        return

    # 1) Expand to daily series
    with st.spinner("Expanding monthly data to daily rates..."):
        df_daily = read_and_expand(uploaded)
    if df_daily is None or df_daily.empty:
        st.error("No data found in the uploaded file.")
        return

    # 2) Compute curves (normalize, forecast, and percentile curves)
    with st.spinner("Computing peak-normalized histories and forecasts..."):
        results = compute_curves(df_daily)

    if results is None:
        st.error("Failed to compute curves.")
        return

    t_axis = results['t_axis']
    P10_actual = results['P10_actual']
    P50_actual = results['P50_actual']
    P90_actual = results['P90_actual']
    cum_P10 = results['cum_P10']
    cum_P50 = results['cum_P50']
    cum_P90 = results['cum_P90']
    overlay_df = results['overlay_df']

    # 3b) Simple statistics for display (peak rates)
    peak_rates = df_daily.groupby('uwi')['rate_bbl_d'].max()
    P10_peak = np.percentile(peak_rates.values, 10)
    P50_peak = np.percentile(peak_rates.values, 50)
    P90_peak = np.percentile(peak_rates.values, 90)

    st.subheader("Peak rate percentiles (historical, bbl/d)")
    perf_table = pd.DataFrame({
        'Percentile': ['P10', 'P50', 'P90'],
        'Peak_rate_bbl_per_day': [P10_peak, P50_peak, P90_peak]
    })
    st.table(perf_table)

    # 4) Plot daily rates and cumulative production
    overlay = st.checkbox("Overlay individual well curves", value=False)

    import matplotlib.pyplot as plt

    # Plot 1: Daily rate vs time since peak (P10/P50/P90)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t_axis, P10_actual, label='P10', color='blue')
    ax1.plot(t_axis, P50_actual, label='P50', color='green')
    ax1.plot(t_axis, P90_actual, label='P90', color='red')
    if overlay:
        # Overlay each well's normalized historical curves
        for uwi, g in overlay_df.groupby('uwi'):
            ax1.plot(g['t_since_peak'], g['norm_rate'], color='gray', alpha=0.15, linewidth=0.8)
    ax1.set_xlabel("Days since peak (t)")
    ax1.set_ylabel("Daily production rate (bbl/d)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # Plot 2: Cumulative production for P10/P50/P90
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_axis, cum_P10, label='P10 cumulative', color='blue')
    ax2.plot(t_axis, cum_P50, label='P50 cumulative', color='green')
    ax2.plot(t_axis, cum_P90, label='P90 cumulative', color='red')
    ax2.set_xlabel("Days since peak (t)")
    ax2.set_ylabel("Cumulative production (bbl)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()
