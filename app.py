import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════════════════════
Q_LIMIT = 2.0          # economic limit bbl/d
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))  # ≈44 days
DATA_FILE = "tcgenprod.xlsx"

COLORS = {
    'p10': '#2ecc71',
    'p50': '#3498db',
    'p90': '#e74c3c',
    'hist': '#2c3e50',
    'forecast': '#e67e22',
    'grid': '#ecf0f1',
    'bg': '#fafbfc',
    'accent': '#8e44ad',
}

# ══════════════════════════════════════════════════════════════════════════════
# DECLINE MATH
# ══════════════════════════════════════════════════════════════════════════════

def hyp_rate(t, qi, di, b):
    """Hyperbolic: q(t) = qi / (1 + b·Di·t)^(1/b)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return qi / np.power(1.0 + b * di * t, 1.0 / b)


def hyp_cum_analytical(t, qi, di, b):
    """Analytical cumulative for hyperbolic decline (bbl), t in days."""
    if abs(b - 1.0) < 1e-6:
        return (qi / di) * np.log(1.0 + di * t)
    return (qi / ((1.0 - b) * di)) * (1.0 - np.power(1.0 + b * di * t, (b - 1.0) / b))


def exp_rate(t, qi, di):
    """Exponential: q(t) = qi·exp(-Di·t)"""
    return qi * np.exp(-di * t)


def hyp_time_to_rate(qi, di, b, q_target):
    """Days from t=0 to reach q_target."""
    if qi <= q_target or di <= 0:
        return 0.0
    return ((qi / q_target) ** b - 1.0) / (b * di)


def fit_hyperbolic(t_days, q_vals):
    """Least-squares fit of (qi, di, b). Returns tuple or None."""
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = (t > 0) & (q > 0) & np.isfinite(t) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if len(t) < 4:
        return None
    qi0 = float(q[0])
    try:
        popt, _ = curve_fit(
            hyp_rate, t, q,
            p0=[qi0, 0.003, 1.2],
            bounds=([0.1, 1e-8, 0.01], [qi0 * 3, 0.05, 2.5]),
            maxfev=30000,
        )
        return tuple(popt)
    except Exception:
        pass
    # fallback: fix b=1.0 harmonic
    try:
        popt, _ = curve_fit(
            lambda t, qi, di: hyp_rate(t, qi, di, 1.0), t, q,
            p0=[qi0, 0.003],
            bounds=([0.1, 1e-8], [qi0 * 3, 0.05]),
            maxfev=20000,
        )
        return (popt[0], popt[1], 1.0)
    except Exception:
        return None


def fit_exponential(t_days, q_vals):
    """Least-squares exponential fit. Returns (qi, di) or None."""
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = (t > 0) & (q > 0)
    t, q = t[mask], q[mask]
    if len(t) < 3:
        return None
    try:
        popt, _ = curve_fit(
            exp_rate, t, q,
            p0=[float(q[0]), 0.003],
            bounds=([0.1, 1e-8], [q[0] * 3, 0.05]),
            maxfev=20000,
        )
        return tuple(popt)
    except Exception:
        return None


def nominal_to_effective(di_daily, b):
    """Convert nominal daily Di to effective annual De (fraction)."""
    di_ann = di_daily * 365.25
    if b > 0:
        return 1.0 - (1.0 + b * di_ann) ** (-1.0 / b)
    return 1.0 - np.exp(-di_ann)


def calc_eur_from_curve(t_days, rates):
    """Trapezoidal EUR from arrays of (days, bbl/d)."""
    if len(t_days) < 2:
        return 0.0
    return float(np.trapezoid(rates, t_days))


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Read tcgenprod.xlsx → cleaned DataFrame with date, t_days from peak, etc."""
    if not os.path.exists(DATA_FILE):
        return None, "File not found"

    df = pd.read_excel(DATA_FILE)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        if 'uwi' in c:
            rename_map[c] = 'uwi'
        elif 'month' in c or 'date' in c:
            rename_map[c] = 'month'
        elif 'bbl' in c or 'rate' in c:
            rename_map[c] = 'rate'
    df = df.rename(columns=rename_map)

    for req in ['uwi', 'month', 'rate']:
        if req not in df.columns:
            return None, f"Missing column: {req}"

    df['date'] = pd.to_datetime(df['month'], errors='coerce')
    df = df.dropna(subset=['date', 'rate'])
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df = df.dropna(subset=['rate'])
    df = df[df['rate'] >= 0].copy()
    df = df.sort_values(['uwi', 'date']).reset_index(drop=True)

    # Approximate days-in-month for volume calc
    df['days_in_month'] = df['date'].dt.days_in_month
    df['monthly_vol'] = df['rate'] * df['days_in_month']

    return df, None


# ══════════════════════════════════════════════════════════════════════════════
# PER-WELL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_well(df_w):
    """Full analysis of one well. Returns rich dict."""
    df_w = df_w.sort_values('date').reset_index(drop=True)

    # Peak
    peak_idx = df_w['rate'].idxmax()
    peak_rate = df_w.loc[peak_idx, 'rate']
    peak_date = df_w.loc[peak_idx, 'date']

    # Time axis from peak
    df_w['t_days'] = (df_w['date'] - peak_date).dt.days.astype(float)

    # From peak onward
    df_post = df_w[df_w['t_days'] >= 0].copy()
    df_pre = df_w[df_w['t_days'] < 0].copy()

    # Decline data (after peak month)
    df_decline = df_post[df_post['t_days'] > 0].copy()

    # Fit hyperbolic
    hyp_fit = fit_hyperbolic(df_decline['t_days'].values, df_decline['rate'].values)
    # Fit exponential for comparison
    exp_fit = fit_exponential(df_decline['t_days'].values, df_decline['rate'].values)

    if hyp_fit:
        qi_h, di_h, b_h = hyp_fit
        pred_h = hyp_rate(df_decline['t_days'].values, qi_h, di_h, b_h)
        ss_res = np.sum((df_decline['rate'].values - pred_h) ** 2)
        ss_tot = np.sum((df_decline['rate'].values - np.mean(df_decline['rate'].values)) ** 2)
        r2_hyp = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        qi_h, di_h, b_h, r2_hyp = peak_rate, 0.003, 1.2, 0.0

    if exp_fit:
        qi_e, di_e = exp_fit
        pred_e = exp_rate(df_decline['t_days'].values, qi_e, di_e)
        ss_res = np.sum((df_decline['rate'].values - pred_e) ** 2)
        ss_tot = np.sum((df_decline['rate'].values - np.mean(df_decline['rate'].values)) ** 2)
        r2_exp = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        qi_e, di_e, r2_exp = peak_rate, 0.003, 0.0

    # EUR via trapezoidal on full provided data (pre + post peak)
    eur_trap = calc_eur_from_curve(
        (df_w['date'] - df_w['date'].min()).dt.days.values,
        df_w['rate'].values
    )

    # Cumulative volumes
    cum_vol = df_w['monthly_vol'].cumsum().values

    # Normalised rate (q / q_peak)
    df_w['norm_rate'] = df_w['rate'] / peak_rate

    # Producing life
    first_date = df_w['date'].min()
    last_date = df_w['date'].max()
    life_days = (last_date - first_date).days
    life_years = life_days / 365.25

    # Last rate
    last_rate = df_w['rate'].iloc[-1]

    # Months on production
    n_months = len(df_w)

    # Months from peak to end
    months_post_peak = len(df_post)

    return {
        'df': df_w,
        'df_post': df_post,
        'df_decline': df_decline,
        'peak_rate': peak_rate,
        'peak_date': peak_date,
        'last_rate': last_rate,
        'last_date': last_date,
        'first_date': first_date,
        'life_years': life_years,
        'n_months': n_months,
        'months_post_peak': months_post_peak,
        'qi_h': qi_h, 'di_h': di_h, 'b_h': b_h, 'r2_hyp': r2_hyp,
        'qi_e': qi_e, 'di_e': di_e, 'r2_exp': r2_exp,
        'eur_trap': eur_trap,
        'cum_vol': cum_vol,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_well_production(info, show_fit=True, log_scale=False):
    """Two-panel plot: rate + cumulative for one well."""
    df_w = info['df']
    df_post = info['df_post']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS['bg'])

    # ── Left: Rate vs Date ──
    ax1.bar(df_w['date'], df_w['rate'], width=25, color=COLORS['hist'],
            alpha=0.55, label='Monthly rate', zorder=2)
    ax1.scatter(df_w['date'], df_w['rate'], s=12, color=COLORS['hist'], zorder=3)

    # Peak marker
    ax1.axhline(info['peak_rate'], color=COLORS['accent'], ls=':', lw=0.8, alpha=0.6)
    ax1.scatter([info['peak_date']], [info['peak_rate']], s=80, color=COLORS['accent'],
                marker='*', zorder=5, label=f"Peak {info['peak_rate']:,.0f} bbl/d")

    # Hyperbolic fit overlay
    if show_fit and len(info['df_decline']) >= 4:
        t_fit = np.linspace(0, df_post['t_days'].max(), 500)
        q_fit = hyp_rate(t_fit, info['qi_h'], info['di_h'], info['b_h'])
        dates_fit = [info['peak_date'] + pd.Timedelta(days=float(d)) for d in t_fit]
        ax1.plot(dates_fit, q_fit, color=COLORS['p50'], lw=1.8, alpha=0.8,
                 label=f"Hyp fit (R²={info['r2_hyp']:.3f})")

    if info['last_rate'] <= Q_LIMIT * 5:
        ax1.axhline(Q_LIMIT, color=COLORS['p90'], ls='--', lw=0.7, alpha=0.5,
                     label=f'{Q_LIMIT} bbl/d limit')

    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Rate (bbl/d)", fontsize=10)
    ax1.set_title("Production Rate", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, ls='--', alpha=0.3)
    if log_scale:
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=max(0.5, df_w['rate'].min() * 0.5))
    else:
        ax1.set_ylim(bottom=0)
    fig.autofmt_xdate(rotation=30)

    # ── Right: Cumulative ──
    cum = info['cum_vol']
    ax2.fill_between(df_w['date'], 0, cum / 1000, color=COLORS['p10'], alpha=0.25)
    ax2.plot(df_w['date'], cum / 1000, color=COLORS['p10'], lw=2)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Cumulative (Mbbl)", fontsize=10)
    ax2.set_title("Cumulative Production", fontsize=11, fontweight='bold')
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def plot_well_normalised(info, log_scale=False):
    """Normalised rate (q/q_peak) vs days-since-peak."""
    df_post = info['df_post']

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(COLORS['bg'])

    ax.scatter(df_post['t_days'], df_post['rate'] / info['peak_rate'],
               s=20, color=COLORS['hist'], zorder=3, label='Observed')

    if len(info['df_decline']) >= 4:
        t_fit = np.linspace(0, df_post['t_days'].max(), 500)
        q_fit_norm = hyp_rate(t_fit, info['qi_h'], info['di_h'], info['b_h']) / info['peak_rate']
        ax.plot(t_fit, q_fit_norm, color=COLORS['p50'], lw=1.8, alpha=0.8,
                label=f"Hyp fit  b={info['b_h']:.2f}")

    ax.set_xlabel("Days since peak", fontsize=10)
    ax.set_ylabel("Normalised rate (q / q_peak)", fontsize=10)
    ax.set_title("Normalised Decline", fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, ls='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.001)
    else:
        ax.set_ylim(0, 1.15)
    ax.set_xlim(left=-10)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE TYPE CURVES
# ══════════════════════════════════════════════════════════════════════════════

def build_type_curve_from_percentiles(df_all_norm, percentile, flat_days=FLAT_DAYS):
    """
    Given all wells' normalised rates aligned by t_days (from peak),
    compute a percentile curve day-by-day, enforce flat period,
    return (t_array, norm_rate_array).
    """
    grouped = df_all_norm.groupby('t_days')['norm_rate']
    t_vals = sorted(df_all_norm['t_days'].unique())
    t_out, q_out = [], []

    for t in t_vals:
        if t < 0:
            continue
        vals = grouped.get_group(t)
        if len(vals) < 1:
            continue
        t_out.append(t)
        if t <= flat_days:
            q_out.append(1.0)
        else:
            q_out.append(np.percentile(vals, percentile))

    return np.array(t_out), np.array(q_out)


def build_parametric_type_curve(qi, di, b, flat_days, q_limit):
    """Build type curve from parameters: flat then hyperbolic."""
    t_list, q_list = [], []
    for d in range(flat_days + 1):
        t_list.append(d)
        q_list.append(qi)
    t_dec = 0
    while True:
        t_dec += 1
        q = hyp_rate(t_dec, qi, di, b)
        if q < q_limit:
            # add the limit point
            t_list.append(flat_days + t_dec)
            q_list.append(max(q, q_limit))
            break
        t_list.append(flat_days + t_dec)
        q_list.append(q)
        if t_dec > 50000:
            break
    return np.array(t_list), np.array(q_list)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Type Curve Generator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──
    st.sidebar.title("🛢️ Type Curve Generator")
    st.sidebar.markdown("---")

    # Load data
    df_raw, err = load_data()
    if df_raw is None:
        st.error(f"Cannot load `{DATA_FILE}`: {err}")
        st.info(f"Place `{DATA_FILE}` in the same folder as this script.\n\n"
                f"Expected columns: `uwi`, `month` (YYYY-MM), `bbl/d`")
        return

    wells = sorted(df_raw['uwi'].unique(), key=str)
    n_wells = len(wells)

    st.sidebar.metric("Wells loaded", n_wells)
    st.sidebar.metric("Total data points", f"{len(df_raw):,}")
    date_range = f"{df_raw['date'].min().strftime('%Y-%m')}  →  {df_raw['date'].max().strftime('%Y-%m')}"
    st.sidebar.caption(f"Date range: {date_range}")
    st.sidebar.markdown("---")

    # Navigation
    section = st.sidebar.radio(
        "Navigate",
        ["📈 Individual Well Analysis", "📊 Aggregate Type Curves", "📋 Summary Table"],
        index=0,
    )

    # Global settings in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Settings")
    log_scale = st.sidebar.checkbox("Log scale on rate plots", value=False)
    show_fit = st.sidebar.checkbox("Show hyperbolic fit on well plots", value=True)

    # ── Pre-compute all wells ──
    @st.cache_data(show_spinner=False)
    def compute_all_wells(_df_raw, _wells):
        results = {}
        for uwi in _wells:
            df_w = _df_raw[_df_raw['uwi'] == uwi].copy()
            if len(df_w) < 2:
                continue
            results[uwi] = analyse_well(df_w)
        return results

    with st.spinner("Analysing all wells…"):
        all_results = compute_all_wells(df_raw, wells)

    valid_wells = list(all_results.keys())
    if not valid_wells:
        st.error("No wells with sufficient data.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: INDIVIDUAL WELL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    if section == "📈 Individual Well Analysis":
        st.title("📈 Individual Well Analysis")

        # Well selector
        col_sel1, col_sel2 = st.columns([3, 1])
        with col_sel1:
            selected_uwi = st.selectbox(
                "Select well",
                valid_wells,
                format_func=lambda x: f"{x}  ({all_results[x]['n_months']} months)",
            )

        info = all_results[selected_uwi]

        # ── KPI cards ──
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Peak Rate", f"{info['peak_rate']:,.0f} bbl/d")
        k2.metric("Peak Date", info['peak_date'].strftime('%Y-%m'))
        k3.metric("Last Rate", f"{info['last_rate']:,.1f} bbl/d")
        k4.metric("Producing Life", f"{info['life_years']:.1f} yrs")
        k5.metric("Months on Prod", info['n_months'])
        k6.metric("EUR (trap.)", f"{info['eur_trap']/1000:,.1f} Mbbl")

        # ── Decline parameters ──
        st.markdown("#### Decline Curve Parameters")
        pc1, pc2 = st.columns(2)

        with pc1:
            st.markdown("**Hyperbolic Fit**")
            di_ann = info['di_h'] * 365.25
            de_eff = nominal_to_effective(info['di_h'], info['b_h'])
            hyp_params = pd.DataFrame({
                'Parameter': [
                    'qi (bbl/d)', 'b exponent',
                    'Di nominal daily (1/d)', 'Di nominal annual (1/yr)',
                    'De effective annual (%)', 'R²',
                    f"Time to {Q_LIMIT} bbl/d (yrs)",
                ],
                'Value': [
                    f"{info['qi_h']:,.1f}", f"{info['b_h']:.3f}",
                    f"{info['di_h']:.6f}", f"{di_ann:.4f}",
                    f"{de_eff*100:.1f}", f"{info['r2_hyp']:.4f}",
                    f"{hyp_time_to_rate(info['qi_h'], info['di_h'], info['b_h'], Q_LIMIT)/365.25:.1f}",
                ]
            })
            st.dataframe(hyp_params, hide_index=True, use_container_width=True)

        with pc2:
            st.markdown("**Exponential Fit** (comparison)")
            de_exp = 1.0 - np.exp(-info['di_e'] * 365.25)
            exp_params = pd.DataFrame({
                'Parameter': [
                    'qi (bbl/d)', 'Di nominal daily (1/d)',
                    'Di nominal annual (1/yr)', 'De effective annual (%)', 'R²',
                ],
                'Value': [
                    f"{info['qi_e']:,.1f}", f"{info['di_e']:.6f}",
                    f"{info['di_e']*365.25:.4f}", f"{de_exp*100:.1f}",
                    f"{info['r2_exp']:.4f}",
                ]
            })
            st.dataframe(exp_params, hide_index=True, use_container_width=True)

        # ── Production plot ──
        st.markdown("#### Production History")
        fig_prod = plot_well_production(info, show_fit=show_fit, log_scale=log_scale)
        st.pyplot(fig_prod)
        plt.close(fig_prod)

        # ── Data table ──
        with st.expander("📄 View raw data for this well"):
            display_df = info['df'][['date', 'rate', 't_days', 'monthly_vol']].copy()
            display_df.columns = ['Date', 'Rate (bbl/d)', 'Days from Peak', 'Monthly Vol (bbl)']
            display_df['Cum Vol (Mbbl)'] = info['cum_vol'] / 1000
            st.dataframe(display_df.style.format({
                'Rate (bbl/d)': '{:,.1f}',
                'Days from Peak': '{:,.0f}',
                'Monthly Vol (bbl)': '{:,.0f}',
                'Cum Vol (Mbbl)': '{:,.1f}',
            }), use_container_width=True, height=400)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: AGGREGATE TYPE CURVES
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "📊 Aggregate Type Curves":
        st.title("📊 Aggregate Type Curves — P10 / P50 / P90")
        st.markdown(
            f"Each percentile curve: **flat peak for {FLAT_MONTHS} months ({FLAT_DAYS} days)** → "
            f"**hyperbolic decline to {Q_LIMIT} bbl/d**."
        )

        if len(all_results) < 3:
            st.warning("Need ≥ 3 wells for meaningful percentiles.")
            return

        # ── Collect parameter distributions ──
        peak_rates = np.array([v['peak_rate'] for v in all_results.values()])
        qi_arr = np.array([v['qi_h'] for v in all_results.values()])
        di_arr = np.array([v['di_h'] for v in all_results.values()])
        b_arr = np.array([v['b_h'] for v in all_results.values()])
        eur_arr = np.array([v['eur_trap'] for v in all_results.values()])

        # P10=optimistic (high), P90=pessimistic (low)
        p10_peak = float(np.percentile(peak_rates, 90))
        p50_peak = float(np.percentile(peak_rates, 50))
        p90_peak = float(np.percentile(peak_rates, 10))

        # For decline: P10 optimistic → slow decline (low di, high b)
        p10_di = float(np.percentile(di_arr, 10))
        p50_di = float(np.percentile(di_arr, 50))
        p90_di = float(np.percentile(di_arr, 90))

        p10_b = float(np.percentile(b_arr, 90))
        p50_b = float(np.percentile(b_arr, 50))
        p90_b = float(np.percentile(b_arr, 10))

        p10_eur = float(np.percentile(eur_arr, 90))
        p50_eur = float(np.percentile(eur_arr, 50))
        p90_eur = float(np.percentile(eur_arr, 10))

        # ── Parameter distributions ──
        st.markdown("### Parameter Distributions")
        dist_df = pd.DataFrame({
            'Statistic': ['P10 (Optimistic)', 'P50 (Mid)', 'P90 (Conservative)',
                          'Mean', 'Std Dev', 'Min', 'Max'],
            'Peak Rate (bbl/d)': [
                f"{p10_peak:,.0f}", f"{p50_peak:,.0f}", f"{p90_peak:,.0f}",
                f"{peak_rates.mean():,.0f}", f"{peak_rates.std():,.0f}",
                f"{peak_rates.min():,.0f}", f"{peak_rates.max():,.0f}",
            ],
            'Di daily (1/d)': [
                f"{p10_di:.6f}", f"{p50_di:.6f}", f"{p90_di:.6f}",
                f"{di_arr.mean():.6f}", f"{di_arr.std():.6f}",
                f"{di_arr.min():.6f}", f"{di_arr.max():.6f}",
            ],
            'b exponent': [
                f"{p10_b:.3f}", f"{p50_b:.3f}", f"{p90_b:.3f}",
                f"{b_arr.mean():.3f}", f"{b_arr.std():.3f}",
                f"{b_arr.min():.3f}", f"{b_arr.max():.3f}",
            ],
            'EUR (Mbbl)': [
                f"{p10_eur/1000:,.1f}", f"{p50_eur/1000:,.1f}", f"{p90_eur/1000:,.1f}",
                f"{eur_arr.mean()/1000:,.1f}", f"{eur_arr.std()/1000:,.1f}",
                f"{eur_arr.min()/1000:,.1f}", f"{eur_arr.max()/1000:,.1f}",
            ],
        })
        st.dataframe(dist_df, hide_index=True, use_container_width=True)

        # ── Histograms of key parameters ──
        st.markdown("### Parameter Histograms")
        fig_hist, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        fig_hist.patch.set_facecolor(COLORS['bg'])

        for ax, data, title, fmt in zip(
            axes,
            [peak_rates, di_arr * 365.25, b_arr, eur_arr / 1000],
            ['Peak Rate (bbl/d)', 'Di annual (1/yr)', 'b exponent', 'EUR (Mbbl)'],
            ['{:,.0f}', '{:.2f}', '{:.2f}', '{:,.0f}'],
        ):
            ax.hist(data, bins=min(20, max(5, len(data) // 2)),
                    color=COLORS['p50'], alpha=0.7, edgecolor='white')
            ax.axvline(np.median(data), color=COLORS['p90'], ls='--', lw=1.5, label='Median')
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, ls='--', alpha=0.2)
        fig_hist.tight_layout()
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # ── Build type curves (parametric) ──
        t10, q10 = build_parametric_type_curve(p10_peak, p10_di, p10_b, FLAT_DAYS, Q_LIMIT)
        t50, q50 = build_parametric_type_curve(p50_peak, p50_di, p50_b, FLAT_DAYS, Q_LIMIT)
        t90, q90 = build_parametric_type_curve(p90_peak, p90_di, p90_b, FLAT_DAYS, Q_LIMIT)

        # Cumulative (trapezoidal)
        c10 = np.cumsum(q10)  # approx daily (dt=1 day steps)
        c50 = np.cumsum(q50)
        c90 = np.cumsum(q90)

        overlay_wells = st.checkbox("Overlay individual well histories", value=True)

        # ── RATE TYPE CURVE ──
        st.markdown("### Rate Type Curve")
        fig_tc, ax_tc = plt.subplots(figsize=(14, 6))
        fig_tc.patch.set_facecolor(COLORS['bg'])

        if overlay_wells:
            for uwi, info in all_results.items():
                df_post = info['df_post']
                ax_tc.plot(df_post['t_days'], df_post['rate'],
                           color='grey', alpha=0.15, lw=0.6)

        ax_tc.plot(t10, q10, color=COLORS['p10'], lw=2.5,
                   label=f"P10: qi={p10_peak:,.0f}, Di={p10_di:.5f}, b={p10_b:.2f}")
        ax_tc.plot(t50, q50, color=COLORS['p50'], lw=2.5,
                   label=f"P50: qi={p50_peak:,.0f}, Di={p50_di:.5f}, b={p50_b:.2f}")
        ax_tc.plot(t90, q90, color=COLORS['p90'], lw=2.5,
                   label=f"P90: qi={p90_peak:,.0f}, Di={p90_di:.5f}, b={p90_b:.2f}")

        ax_tc.axhline(Q_LIMIT, color='grey', ls='--', lw=0.7, alpha=0.5)
        ax_tc.axvline(FLAT_DAYS, color=COLORS['forecast'], ls=':', lw=1,
                      label=f"End flat ({FLAT_MONTHS} mo)")

        ax_tc.set_xlabel("Days since peak", fontsize=11)
        ax_tc.set_ylabel("Rate (bbl/d)", fontsize=11)
        ax_tc.set_title("P10 / P50 / P90 Rate Type Curves", fontsize=13, fontweight='bold')
        ax_tc.legend(fontsize=8, loc='upper right')
        ax_tc.grid(True, ls='--', alpha=0.3)
        ax_tc.set_xlim(left=-20)
        if log_scale:
            ax_tc.set_yscale('log')
            ax_tc.set_ylim(bottom=1)
        else:
            ax_tc.set_ylim(bottom=0)
        fig_tc.tight_layout()
        st.pyplot(fig_tc)
        plt.close(fig_tc)

        # ── CUMULATIVE TYPE CURVE ──
        st.markdown("### Cumulative Type Curve")
        fig_cum, ax_cum = plt.subplots(figsize=(14, 5))
        fig_cum.patch.set_facecolor(COLORS['bg'])

        ax_cum.fill_between(t10, 0, c10 / 1000, color=COLORS['p10'], alpha=0.08)
        ax_cum.fill_between(t90, 0, c90 / 1000, color=COLORS['p90'], alpha=0.08)
        ax_cum.plot(t10, c10 / 1000, color=COLORS['p10'], lw=2.2,
                    label=f"P10 EUR ≈ {c10[-1]/1000:,.0f} Mbbl")
        ax_cum.plot(t50, c50 / 1000, color=COLORS['p50'], lw=2.2,
                    label=f"P50 EUR ≈ {c50[-1]/1000:,.0f} Mbbl")
        ax_cum.plot(t90, c90 / 1000, color=COLORS['p90'], lw=2.2,
                    label=f"P90 EUR ≈ {c90[-1]/1000:,.0f} Mbbl")

        ax_cum.axvline(FLAT_DAYS, color=COLORS['forecast'], ls=':', lw=1)
        ax_cum.set_xlabel("Days since peak", fontsize=11)
        ax_cum.set_ylabel("Cumulative Production (Mbbl)", fontsize=11)
        ax_cum.set_title("P10 / P50 / P90 Cumulative Type Curves", fontsize=13, fontweight='bold')
        ax_cum.legend(fontsize=9)
        ax_cum.grid(True, ls='--', alpha=0.3)
        ax_cum.set_xlim(left=-20)
        ax_cum.set_ylim(bottom=0)
        fig_cum.tight_layout()
        st.pyplot(fig_cum)
        plt.close(fig_cum)

        # ── Normalised overlay ──
        st.markdown("### Normalised Rate Overlay (q / q_peak)")
        fig_no, ax_no = plt.subplots(figsize=(14, 5))
        fig_no.patch.set_facecolor(COLORS['bg'])

        for uwi, info in all_results.items():
            df_post = info['df_post']
            ax_no.plot(df_post['t_days'], df_post['rate'] / info['peak_rate'],
                       color='grey', alpha=0.2, lw=0.6)

        # Normalised type curves
        ax_no.plot(t10, q10 / p10_peak, color=COLORS['p10'], lw=2.2, label='P10 norm')
        ax_no.plot(t50, q50 / p50_peak, color=COLORS['p50'], lw=2.2, label='P50 norm')
        ax_no.plot(t90, q90 / p90_peak, color=COLORS['p90'], lw=2.2, label='P90 norm')

        ax_no.axvline(FLAT_DAYS, color=COLORS['forecast'], ls=':', lw=1)
        ax_no.set_xlabel("Days since peak", fontsize=11)
        ax_no.set_ylabel("q / q_peak", fontsize=11)
        ax_no.set_title("Normalised Decline — All Wells + Type Curves", fontsize=13, fontweight='bold')
        ax_no.legend(fontsize=8)
        ax_no.grid(True, ls='--', alpha=0.3)
        ax_no.set_xlim(left=-20)
        if log_scale:
            ax_no.set_yscale('log')
            ax_no.set_ylim(bottom=0.001)
        else:
            ax_no.set_ylim(0, 1.15)
        fig_no.tight_layout()
        st.pyplot(fig_no)
        plt.close(fig_no)

        # ── EUR distribution ──
        st.markdown("### EUR Distribution")
        fig_eur, ax_eur = plt.subplots(figsize=(10, 4))
        fig_eur.patch.set_facecolor(COLORS['bg'])

        eur_mbbl = eur_arr / 1000
        ax_eur.hist(eur_mbbl, bins=min(20, max(5, len(eur_mbbl) // 2)),
                     color=COLORS['p50'], alpha=0.6, edgecolor='white')
        for pval, label, col in [
            (p10_eur / 1000, 'P10', COLORS['p10']),
            (p50_eur / 1000, 'P50', COLORS['p50']),
            (p90_eur / 1000, 'P90', COLORS['p90']),
        ]:
            ax_eur.axvline(pval, color=col, ls='--', lw=2, label=f"{label}: {pval:,.0f} Mbbl")
        ax_eur.set_xlabel("EUR (Mbbl)", fontsize=11)
        ax_eur.set_ylabel("Count", fontsize=11)
        ax_eur.set_title("EUR Distribution", fontsize=13, fontweight='bold')
        ax_eur.legend(fontsize=8)
        ax_eur.grid(True, ls='--', alpha=0.2)
        fig_eur.tight_layout()
        st.pyplot(fig_eur)
        plt.close(fig_eur)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "📋 Summary Table":
        st.title("📋 All Wells — Summary Table")

        rows = []
        for uwi in valid_wells:
            info = all_results[uwi]
            de_eff = nominal_to_effective(info['di_h'], info['b_h'])
            rows.append({
                'UWI': uwi,
                'Peak Rate (bbl/d)': round(info['peak_rate'], 1),
                'Peak Date': info['peak_date'].strftime('%Y-%m'),
                'Last Rate (bbl/d)': round(info['last_rate'], 1),
                'Months': info['n_months'],
                'Life (yrs)': round(info['life_years'], 1),
                'qi fit (bbl/d)': round(info['qi_h'], 1),
                'Di daily': round(info['di_h'], 6),
                'Di annual': round(info['di_h'] * 365.25, 4),
                'De annual (%)': round(de_eff * 100, 1),
                'b': round(info['b_h'], 3),
                'R² hyp': round(info['r2_hyp'], 3),
                'R² exp': round(info['r2_exp'], 3),
                'EUR (Mbbl)': round(info['eur_trap'] / 1000, 1),
            })

        summary_df = pd.DataFrame(rows)

        # Stats row
        numeric_cols = ['Peak Rate (bbl/d)', 'Last Rate (bbl/d)', 'Months',
                        'Life (yrs)', 'qi fit (bbl/d)', 'Di daily', 'Di annual',
                        'De annual (%)', 'b', 'R² hyp', 'R² exp', 'EUR (Mbbl)']

        st.dataframe(
            summary_df.style.format({
                'Peak Rate (bbl/d)': '{:,.1f}',
                'Last Rate (bbl/d)': '{:,.1f}',
                'qi fit (bbl/d)': '{:,.1f}',
                'Di daily': '{:.6f}',
                'Di annual': '{:.4f}',
                'De annual (%)': '{:.1f}',
                'b': '{:.3f}',
                'R² hyp': '{:.3f}',
                'R² exp': '{:.3f}',
                'EUR (Mbbl)': '{:,.1f}',
            }).background_gradient(subset=['EUR (Mbbl)'], cmap='YlGn'),
            use_container_width=True,
            height=min(800, 40 + 35 * len(rows)),
        )

        # Aggregate stats
        st.markdown("### Aggregate Statistics")
        agg_stats = summary_df[numeric_cols].describe().T
        agg_stats.columns = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
        st.dataframe(agg_stats.style.format('{:,.3f}'), use_container_width=True)

        # Download
        csv = summary_df.to_csv(index=False)
        st.download_button(
            "⬇ Download summary CSV",
            csv,
            file_name="well_summary.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()