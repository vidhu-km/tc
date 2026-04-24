import streamlit as st
import pandas as pd
import numpy as np
import calendar
import os
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BBL_PER_M3 = 6.2898
Q_LIMIT = 2.0  # economic limit (bbl/d)
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))  # ≈44 days

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def hyp_rate(t, qi, di, b):
    """Hyperbolic decline: q(t) = qi / (1 + b*di*t)^(1/b), t in days, di nominal daily."""
    return qi / ((1.0 + b * di * t) ** (1.0 / b))


def hyp_cum(t, qi, di, b):
    """Cumulative production from hyperbolic decline (bbl), t in days."""
    if b == 0 or di == 0:
        return qi * t
    return (qi ** b / ((1.0 - b) * di)) * (qi ** (1.0 - b) - hyp_rate(t, qi, di, b) ** (1.0 - b))


def hyp_time_to_rate(qi, di, b, q_limit):
    """Days from t=0 of decline to reach q_limit under hyperbolic decline."""
    if qi <= q_limit:
        return 0.0
    if di <= 0 or b <= 0:
        return 1e6
    return ((qi / q_limit) ** b - 1.0) / (b * di)


def fit_hyperbolic(t_days, q_vals):
    """
    Fit qi, Di (daily nominal), b to observed (t, q) using least-squares.
    Returns (qi, di_daily, b) or None on failure.
    """
    if len(t_days) < 4:
        return None
    try:
        # initial guesses
        qi0 = float(q_vals[0]) if q_vals[0] > 0 else 100.0
        popt, _ = curve_fit(
            hyp_rate,
            t_days.astype(float),
            q_vals.astype(float),
            p0=[qi0, 0.005, 1.0],
            bounds=([0.1, 1e-7, 0.01], [qi0 * 5, 0.1, 2.0]),
            maxfev=20000,
        )
        return tuple(popt)  # qi, di, b
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1) Read Excel → monthly rates (bbl/d) per well
# ──────────────────────────────────────────────────────────────────────────────
def read_production(file=None, path=None):
    if path:
        df = pd.read_excel(path)
    elif file:
        df = pd.read_excel(file)
    else:
        return None

    required = ['uwi', 'year'] + [
        f'{m}_volume' for m in
        ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    ]
    for c in required:
        if c not in df.columns:
            st.error(f"Missing column: {c}")
            return None

    months_map = {f'{m}_volume': i+1 for i, m in enumerate(
        ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])}

    df_long = df.melt(id_vars=['uwi','year'],
                      value_vars=list(months_map.keys()),
                      var_name='mcol', value_name='vol_m3')
    df_long['month'] = df_long['mcol'].map(months_map)
    df_long = df_long.dropna(subset=['vol_m3'])
    df_long = df_long[df_long['vol_m3'] > 0]

    rows = []
    for _, r in df_long.iterrows():
        y, m = int(r['year']), int(r['month'])
        days = calendar.monthrange(y, m)[1]
        rate = r['vol_m3'] * BBL_PER_M3 / days
        date = datetime(y, m, 15)  # mid‑month representative date
        rows.append({'uwi': r['uwi'], 'date': date, 'rate': rate, 'days': days, 'vol_bbl': r['vol_m3'] * BBL_PER_M3})

    out = pd.DataFrame(rows).sort_values(['uwi','date']).reset_index(drop=True)
    out['date'] = pd.to_datetime(out['date'])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2) Per‑well analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyse_well(df_well):
    """
    Given a single well's monthly‑rate dataframe, find peak, build decline
    from peak to end, fit hyperbolic, forecast to Q_LIMIT.
    Returns dict with everything needed for display.
    """
    df_well = df_well.sort_values('date').reset_index(drop=True)
    peak_idx = df_well['rate'].idxmax()
    peak_rate = df_well.loc[peak_idx, 'rate']
    peak_date = df_well.loc[peak_idx, 'date']

    # Production from peak onward
    df_from_peak = df_well[df_well['date'] >= peak_date].copy()
    df_from_peak['t_days'] = (df_from_peak['date'] - peak_date).dt.days.astype(float)

    # Decline portion (after peak month)
    df_decline = df_from_peak[df_from_peak['t_days'] > 0].copy()

    # Fit hyperbolic to decline portion
    fit = None
    if len(df_decline) >= 4:
        fit = fit_hyperbolic(df_decline['t_days'].values, df_decline['rate'].values)

    # Defaults if fit fails
    if fit is None:
        qi = float(peak_rate)
        di = 0.005
        b = 1.2
    else:
        qi, di, b = fit

    # Historical cumulative (all months, not just from peak)
    hist_cum = df_well['vol_bbl'].sum()

    # Last data point
    last_date = df_well['date'].max()
    last_rate = df_well.loc[df_well['date'].idxmax(), 'rate']

    return {
        'peak_rate': peak_rate,
        'peak_date': peak_date,
        'qi_fit': qi,
        'di_fit': di,
        'b_fit': b,
        'df_from_peak': df_from_peak,
        'df_all': df_well,
        'hist_cum': hist_cum,
        'last_date': last_date,
        'last_rate': last_rate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3) Build forecast array from peak using given parameters
# ──────────────────────────────────────────────────────────────────────────────
def build_forecast(peak_rate, qi, di, b, t_start_days, max_days=30000):
    """
    Generate daily forecast from t_start_days (days after peak) until rate
    drops to Q_LIMIT.  Returns arrays (t_days, rates).
    qi is the rate at t=0 of the hyperbolic (= peak_rate for a fresh fit starting at peak).
    """
    ts = []
    qs = []
    for t in range(int(t_start_days), int(t_start_days) + max_days):
        q = hyp_rate(t, qi, di, b)
        if q < Q_LIMIT:
            break
        ts.append(t)
        qs.append(q)
    return np.array(ts, dtype=float), np.array(qs, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# 4) EUR calculation
# ──────────────────────────────────────────────────────────────────────────────
def calc_eur(peak_rate, qi, di, b, hist_cum_before_peak, df_from_peak):
    """
    EUR = historical cum (before peak) + historical cum (peak→end) + forecast cum (end→limit).
    """
    if len(df_from_peak) == 0:
        return hist_cum_before_peak

    last_t = df_from_peak['t_days'].max()
    # forecast from last observed t to economic limit
    t_end = hyp_time_to_rate(qi, di, b, Q_LIMIT)
    if t_end <= last_t:
        forecast_add = 0.0
    else:
        # cumulative under hyperbolic from last_t to t_end
        cum_to_end = hyp_cum(t_end, qi, di, b) if t_end > 0 else 0.0
        cum_to_last = hyp_cum(last_t, qi, di, b) if last_t > 0 else 0.0
        forecast_add = cum_to_end - cum_to_last

    hist_from_peak = df_from_peak['vol_bbl'].sum() if 'vol_bbl' in df_from_peak.columns else 0.0
    # historical before peak
    eur = hist_cum_before_peak + hist_from_peak + forecast_add
    return eur


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Decline Curve Analysis", layout="wide")
    st.title("🛢️ Production Decline Curve Analysis & Forecasting")
    st.markdown(
        "Upload **wellprod.xlsx** with columns: `uwi`, `year`, "
        "`jan_volume` … `dec_volume` (monthly volumes in **m³**)."
    )

    # ── File input ──
    uploaded = st.file_uploader("Upload wellprod.xlsx", type=["xlsx"])
    local_path = "wellprod.xlsx"
    source = None
    if uploaded:
        source = ('file', uploaded)
    elif os.path.exists(local_path):
        source = ('path', local_path)
        st.info(f"Auto‑loaded local `{local_path}`.")
    else:
        st.info("Awaiting file upload or place `wellprod.xlsx` alongside this script.")
        return

    # ── Read data ──
    with st.spinner("Reading production data…"):
        if source[0] == 'file':
            df_prod = read_production(file=source[1])
        else:
            df_prod = read_production(path=source[1])
    if df_prod is None or df_prod.empty:
        st.error("No valid production data found."); return

    wells = sorted(df_prod['uwi'].unique(), key=str)
    st.success(f"Loaded **{len(wells)}** wells.")

    # ──────────────────────────────────────────────────────────────────────────
    # PER‑WELL SECTION
    # ──────────────────────────────────────────────────────────────────────────
    # Store results for aggregate curves
    well_results = {}

    for uwi in wells:
        st.markdown("---")
        st.subheader(f"Well: `{uwi}`")

        df_w = df_prod[df_prod['uwi'] == uwi].copy()
        info = analyse_well(df_w)

        peak_rate = info['peak_rate']
        peak_date = info['peak_date']
        df_from_peak = info['df_from_peak']

        # Editable decline parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            qi_user = st.number_input(
                "qi (bbl/d)", value=round(float(info['qi_fit']), 2),
                min_value=1.0, step=10.0, key=f"qi_{uwi}")
        with col2:
            di_user = st.number_input(
                "Di (daily nominal)", value=round(float(info['di_fit']), 6),
                min_value=0.000001, step=0.0005, format="%.6f", key=f"di_{uwi}")
        with col3:
            b_user = st.number_input(
                "b exponent", value=round(float(info['b_fit']), 4),
                min_value=0.01, max_value=2.5, step=0.05, format="%.4f", key=f"b_{uwi}")

        # Convert di to annual for display
        di_annual_user = di_user * 365.0

        # Effective annual decline at t=0:  De = 1 - (1 + b*Di_nom)^(-1/b)  (first‑year)
        de_annual = 1.0 - (1.0 + b_user * di_annual_user) ** (-1.0 / b_user) if b_user > 0 else 1.0 - np.exp(-di_annual_user)

        # Historical cum before peak
        hist_before_peak = df_w[df_w['date'] < peak_date]['vol_bbl'].sum()

        # Re‑add vol_bbl to df_from_peak for EUR calc
        df_from_peak_full = df_from_peak.merge(
            df_w[['date','vol_bbl']], on='date', how='left', suffixes=('','_y'))
        if 'vol_bbl_y' in df_from_peak_full.columns:
            df_from_peak_full['vol_bbl'] = df_from_peak_full['vol_bbl_y']

        eur = calc_eur(peak_rate, qi_user, di_user, b_user,
                       hist_before_peak, df_from_peak_full)

        # Time to economic limit
        t_econ = hyp_time_to_rate(qi_user, di_user, b_user, Q_LIMIT)

        # Display parameters table
        params_df = pd.DataFrame({
            'Parameter': [
                'Peak Rate (bbl/d)',
                'qi – fit/user (bbl/d)',
                'Di – daily nominal (1/d)',
                'Di – annual nominal (1/yr)',
                'De – effective annual (%)',
                'b exponent',
                f'Time to {Q_LIMIT} bbl/d (years)',
                'EUR (bbl)',
                'EUR (Mbbl)',
            ],
            'Value': [
                f"{peak_rate:,.1f}",
                f"{qi_user:,.1f}",
                f"{di_user:.6f}",
                f"{di_annual_user:,.4f}",
                f"{de_annual*100:,.2f}",
                f"{b_user:.4f}",
                f"{t_econ/365.25:,.1f}",
                f"{eur:,.0f}",
                f"{eur/1000:,.1f}",
            ]
        })
        st.table(params_df)

        # ── Build plot data ──
        # Historical from peak
        hist_t = df_from_peak['t_days'].values
        hist_q = df_from_peak['rate'].values

        # Forecast from last historical month onward
        last_t = float(df_from_peak['t_days'].max())
        fc_t, fc_q = build_forecast(peak_rate, qi_user, di_user, b_user,
                                    t_start_days=last_t + 1)

        # Full model curve from t=0 for visual reference
        model_t = np.arange(0, max(t_econ, last_t + 1) + 1)
        model_q = hyp_rate(model_t, qi_user, di_user, b_user)
        model_q = np.where(model_q < Q_LIMIT, np.nan, model_q)

        # Plot
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.scatter(hist_t, hist_q, s=18, color='black', zorder=5, label='Historical (from peak)')
        ax.plot(model_t, model_q, color='steelblue', linewidth=1.2, alpha=0.5, label='Full fit curve')
        if len(fc_t) > 0:
            ax.plot(fc_t, fc_q, color='red', linewidth=1.8, label='Forecast')
        ax.axhline(Q_LIMIT, color='grey', linestyle='--', linewidth=0.8, label=f'Econ. limit ({Q_LIMIT} bbl/d)')
        ax.set_xlabel("Days since peak")
        ax.set_ylabel("Rate (bbl/d)")
        ax.set_title(f"{uwi}  —  Peak {peak_rate:,.0f} bbl/d on {peak_date.strftime('%Y‑%m')}")
        ax.legend(fontsize=8)
        ax.grid(True, ls='--', alpha=0.4)
        ax.set_xlim(left=-10)
        ax.set_ylim(bottom=0)
        st.pyplot(fig)
        plt.close(fig)

        # Store for aggregate
        well_results[uwi] = {
            'peak_rate': peak_rate,
            'qi': qi_user,
            'di': di_user,
            'b': b_user,
            'eur': eur,
            'hist_t': hist_t,
            'hist_q': hist_q,
            'last_t': last_t,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # AGGREGATE P10 / P50 / P90 TYPE CURVES
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📊 Aggregate Type Curves — P10 / P50 / P90")
    st.markdown(
        f"Each percentile curve uses a **flat peak rate for {FLAT_MONTHS} months "
        f"({FLAT_DAYS} days)**, then **hyperbolic decline to {Q_LIMIT} bbl/d**."
    )

    if len(well_results) < 2:
        st.warning("Need at least 2 wells for meaningful percentile curves.")
        return

    # Collect arrays of fitted parameters
    peak_rates_arr = np.array([v['peak_rate'] for v in well_results.values()])
    qi_arr = np.array([v['qi'] for v in well_results.values()])
    di_arr = np.array([v['di'] for v in well_results.values()])
    b_arr = np.array([v['b'] for v in well_results.values()])
    eur_arr = np.array([v['eur'] for v in well_results.values()])

    # Percentiles of peak rate (used as the flat‑period rate AND qi for the type curve)
    p10_peak = float(np.percentile(peak_rates_arr, 90))   # P10 = high case
    p50_peak = float(np.percentile(peak_rates_arr, 50))
    p90_peak = float(np.percentile(peak_rates_arr, 10))   # P90 = low case

    # Percentiles of decline parameters
    # For P10 (optimistic): lower Di, higher b → slower decline
    # For P90 (pessimistic): higher Di, lower b → faster decline
    p10_di = float(np.percentile(di_arr, 10))
    p50_di = float(np.percentile(di_arr, 50))
    p90_di = float(np.percentile(di_arr, 90))

    p10_b = float(np.percentile(b_arr, 90))
    p50_b = float(np.percentile(b_arr, 50))
    p90_b = float(np.percentile(b_arr, 10))

    # EUR percentiles
    p10_eur = float(np.percentile(eur_arr, 90))
    p50_eur = float(np.percentile(eur_arr, 50))
    p90_eur = float(np.percentile(eur_arr, 10))

    # Summary table
    summ = pd.DataFrame({
        'Statistic': ['P10 (High)', 'P50 (Mid)', 'P90 (Low)'],
        'Peak Rate (bbl/d)': [f"{p10_peak:,.1f}", f"{p50_peak:,.1f}", f"{p90_peak:,.1f}"],
        'Di daily (1/d)': [f"{p10_di:.6f}", f"{p50_di:.6f}", f"{p90_di:.6f}"],
        'b exponent': [f"{p10_b:.3f}", f"{p50_b:.3f}", f"{p90_b:.3f}"],
        'EUR (Mbbl)': [f"{p10_eur/1000:,.1f}", f"{p50_eur/1000:,.1f}", f"{p90_eur/1000:,.1f}"],
    })
    st.table(summ)

    # Build percentile type curves
    def build_type_curve(qi_peak, di, b, flat_days, q_limit):
        """Return (t_days_array, rate_array, cum_array) for a type curve."""
        t_list = []
        q_list = []

        # Flat segment
        for d in range(flat_days):
            t_list.append(d)
            q_list.append(qi_peak)

        # Hyperbolic segment after flat period
        # At t=flat_days the decline starts with qi = qi_peak
        t_decline = 0
        while True:
            t_decline += 1
            q = hyp_rate(t_decline, qi_peak, di, b)
            if q < q_limit:
                break
            t_list.append(flat_days + t_decline)
            q_list.append(q)

        t_arr = np.array(t_list, dtype=float)
        q_arr = np.array(q_list, dtype=float)
        cum_arr = np.cumsum(q_arr)  # daily cum (bbl)
        return t_arr, q_arr, cum_arr

    t10, q10, c10 = build_type_curve(p10_peak, p10_di, p10_b, FLAT_DAYS, Q_LIMIT)
    t50, q50, c50 = build_type_curve(p50_peak, p50_di, p50_b, FLAT_DAYS, Q_LIMIT)
    t90, q90, c90 = build_type_curve(p90_peak, p90_di, p90_b, FLAT_DAYS, Q_LIMIT)

    # ── Overlay individual well observed curves (from peak) ──
    overlay_agg = st.checkbox("Overlay individual well histories on type curves", value=True)

    # ── Rate plot ──
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    if overlay_agg:
        for uwi, wr in well_results.items():
            ax3.plot(wr['hist_t'], wr['hist_q'], color='grey', alpha=0.25, linewidth=0.7)

    ax3.plot(t10, q10, color='green', linewidth=2.2, label=f'P10  (qi={p10_peak:,.0f}, Di={p10_di:.5f}, b={p10_b:.2f})')
    ax3.plot(t50, q50, color='blue', linewidth=2.2, label=f'P50  (qi={p50_peak:,.0f}, Di={p50_di:.5f}, b={p50_b:.2f})')
    ax3.plot(t90, q90, color='red', linewidth=2.2, label=f'P90  (qi={p90_peak:,.0f}, Di={p90_di:.5f}, b={p90_b:.2f})')

    ax3.axhline(Q_LIMIT, color='grey', ls='--', lw=0.8)
    ax3.axvline(FLAT_DAYS, color='orange', ls=':', lw=0.9, label=f'End flat ({FLAT_MONTHS} mo)')
    ax3.set_xlabel("Days since peak")
    ax3.set_ylabel("Rate (bbl/d)")
    ax3.set_title("Aggregate Type Curves — Daily Rate")
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, ls='--', alpha=0.4)
    ax3.set_xlim(left=-10)
    ax3.set_ylim(bottom=0)
    st.pyplot(fig3)
    plt.close(fig3)

    # ── Cumulative plot ──
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(t10, c10, color='green', linewidth=2.2, label=f'P10 EUR ≈ {c10[-1]/1000:,.0f} Mbbl')
    ax4.plot(t50, c50, color='blue', linewidth=2.2, label=f'P50 EUR ≈ {c50[-1]/1000:,.0f} Mbbl')
    ax4.plot(t90, c90, color='red', linewidth=2.2, label=f'P90 EUR ≈ {c90[-1]/1000:,.0f} Mbbl')
    ax4.axvline(FLAT_DAYS, color='orange', ls=':', lw=0.9, label=f'End flat ({FLAT_MONTHS} mo)')
    ax4.set_xlabel("Days since peak")
    ax4.set_ylabel("Cumulative Production (bbl)")
    ax4.set_title("Aggregate Type Curves — Cumulative Production")
    ax4.legend(fontsize=8)
    ax4.grid(True, ls='--', alpha=0.4)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:,.0f}k'))
    ax4.set_xlim(left=-10)
    st.pyplot(fig4)
    plt.close(fig4)

    # ── Final EUR table for all wells ──
    st.markdown("---")
    st.subheader("Individual Well EUR Summary")
    eur_rows = []
    for uwi in wells:
        wr = well_results.get(uwi)
        if wr is None:
            continue
        eur_rows.append({
            'UWI': uwi,
            'Peak Rate (bbl/d)': round(wr['peak_rate'], 1),
            'qi (bbl/d)': round(wr['qi'], 1),
            'Di daily': round(wr['di'], 6),
            'b': round(wr['b'], 4),
            'EUR (Mbbl)': round(wr['eur'] / 1000, 1),
        })
    st.dataframe(pd.DataFrame(eur_rows), use_container_width=True)


if __name__ == "__main__":
    main()