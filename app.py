import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
Q_LIMIT = 5.0
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))  # ≈44 days
DATA_FILE = "tcgenprod.xlsx"
N_BOOTSTRAP = 500

# ══════════════════════════════════════════════════════════════════════════════
# DECLINE MATH
# ══════════════════════════════════════════════════════════════════════════════

def hyp_rate(t, qi, di, b):
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return qi / np.power(1.0 + b * di * t, 1.0 / b)


def hyp_time_to_rate(qi, di, b, q_target):
    if qi <= q_target or di <= 0 or b <= 0:
        return 0.0
    return ((qi / q_target) ** b - 1.0) / (b * di)


def fit_hyperbolic(t_days, q_vals):
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
            bounds=([0.1, 1e-8, 0.01], [qi0 * 5, 0.1, 2.5]),
            maxfev=30000,
        )
        return tuple(popt)
    except Exception:
        pass
    try:
        popt, _ = curve_fit(
            lambda t, qi, di: hyp_rate(t, qi, di, 1.0), t, q,
            p0=[qi0, 0.003],
            bounds=([0.1, 1e-8], [qi0 * 5, 0.1]),
            maxfev=20000,
        )
        return (popt[0], popt[1], 1.0)
    except Exception:
        return None


def build_type_curve(qi, di, b, flat_days, q_limit):
    t_list, q_list = [], []
    for d in range(flat_days + 1):
        t_list.append(d)
        q_list.append(qi)
    t_dec = 0
    while True:
        t_dec += 1
        q = hyp_rate(t_dec, qi, di, b)
        if q < q_limit or t_dec > 60000:
            t_list.append(flat_days + t_dec)
            q_list.append(max(q, q_limit))
            break
        t_list.append(flat_days + t_dec)
        q_list.append(q)
    return np.array(t_list, dtype=float), np.array(q_list, dtype=float)


def calc_eur_from_arrays(t_arr, q_arr):
    if len(t_arr) < 2:
        return 0.0
    return float(np.trapezoid(q_arr, t_arr))


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists(DATA_FILE):
        return None, f"`{DATA_FILE}` not found in script directory."
    df = pd.read_excel(DATA_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if 'uwi' in cl:
            rename[c] = 'uwi'
        elif 'month' in cl or 'date' in cl:
            rename[c] = 'month'
        elif 'bbl' in cl or 'rate' in cl:
            rename[c] = 'rate'
    df = df.rename(columns=rename)
    for req in ['uwi', 'month', 'rate']:
        if req not in df.columns:
            return None, f"Missing column: `{req}`"
    df['date'] = pd.to_datetime(df['month'], errors='coerce')
    df = df.dropna(subset=['date', 'rate'])
    df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
    df = df.dropna(subset=['rate'])
    df = df[df['rate'] >= 0].copy()
    df = df.sort_values(['uwi', 'date']).reset_index(drop=True)
    df['days_in_month'] = df['date'].dt.days_in_month
    df['monthly_vol'] = df['rate'] * df['days_in_month']
    return df, None


# ══════════════════════════════════════════════════════════════════════════════
# PER-WELL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_well(df_w):
    df_w = df_w.sort_values('date').reset_index(drop=True)
    peak_idx = df_w['rate'].idxmax()
    peak_rate = df_w.loc[peak_idx, 'rate']
    peak_date = df_w.loc[peak_idx, 'date']
    df_w['t_days'] = (df_w['date'] - peak_date).dt.days.astype(float)
    df_post = df_w[df_w['t_days'] >= 0].copy()
    df_decline = df_post[df_post['t_days'] > 0].copy()

    hyp_fit = fit_hyperbolic(df_decline['t_days'].values, df_decline['rate'].values)

    if hyp_fit:
        qi_h, di_h, b_h = hyp_fit
        pred = hyp_rate(df_decline['t_days'].values, qi_h, di_h, b_h)
        ss_res = np.sum((df_decline['rate'].values - pred) ** 2)
        ss_tot = np.sum((df_decline['rate'].values - df_decline['rate'].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        qi_h, di_h, b_h, r2 = peak_rate, 0.003, 1.2, 0.0

    eur_trap = calc_eur_from_arrays(
        (df_w['date'] - df_w['date'].min()).dt.days.values.astype(float),
        df_w['rate'].values,
    )
    cum_vol = df_w['monthly_vol'].cumsum().values
    last_rate = df_w['rate'].iloc[-1]
    last_date = df_w['date'].max()
    first_date = df_w['date'].min()
    life_years = (last_date - first_date).days / 365.25

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
        'n_months': len(df_w),
        'qi': qi_h, 'di': di_h, 'b': b_h, 'r2': r2,
        'eur_trap': eur_trap,
        'cum_vol': cum_vol,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BOOTSTRAP MONTE CARLO
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def run_bootstrap(param_array, n_iter, flat_days, q_limit):
    """
    param_array: Nx3 array of (qi, di, b) from fitted wells.
    Resample rows with replacement, build type curve for each, compute EUR.
    Return dict with curves + EUR array.
    """
    n_wells = len(param_array)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_wells, size=n_iter)

    eurs = np.zeros(n_iter)
    curves = []

    for i in range(n_iter):
        qi, di, b = param_array[indices[i]]
        t_arr, q_arr = build_type_curve(qi, di, b, flat_days, q_limit)
        eur = calc_eur_from_arrays(t_arr, q_arr)
        eurs[i] = eur
        curves.append((t_arr, q_arr, qi, di, b))

    return eurs, curves


def select_percentile_curves(eurs, curves, percentiles):
    """For each target percentile, find the simulation whose EUR is closest."""
    result = {}
    for p in percentiles:
        target_eur = np.percentile(eurs, p)
        idx = int(np.argmin(np.abs(eurs - target_eur)))
        t_arr, q_arr, qi, di, b = curves[idx]
        eur_actual = eurs[idx]
        result[p] = {
            't': t_arr, 'q': q_arr,
            'qi': qi, 'di': di, 'b': b,
            'eur': eur_actual,
        }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

PCOLORS = {
    10: '#2ecc71',
    25: '#27ae60',
    50: '#3498db',
    75: '#e67e22',
    90: '#e74c3c',
}

PLOTLY_LAYOUT = dict(
    template='plotly_white',
    font=dict(family='Inter, Arial, sans-serif', size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode='x unified',
)


def well_rate_figure(info, show_fit):
    df_w = info['df']
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Production Rate', 'Cumulative Production'),
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Bar(
        x=df_w['date'], y=df_w['rate'],
        marker_color='#2c3e50', opacity=0.5, name='Monthly rate',
        hovertemplate='%{x|%Y-%m}<br>%{y:,.1f} bbl/d',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[info['peak_date']], y=[info['peak_rate']],
        mode='markers', marker=dict(size=12, color='#8e44ad', symbol='star'),
        name=f"Peak {info['peak_rate']:,.0f} bbl/d",
        hovertemplate='Peak: %{y:,.0f} bbl/d',
    ), row=1, col=1)

    if show_fit and len(info['df_decline']) >= 4:
        df_post = info['df_post']
        t_fit = np.linspace(0, df_post['t_days'].max(), 300)
        q_fit = hyp_rate(t_fit, info['qi'], info['di'], info['b'])
        dates_fit = pd.to_datetime(info['peak_date']) + pd.to_timedelta(t_fit, unit='D')
        fig.add_trace(go.Scatter(
            x=dates_fit, y=q_fit,
            mode='lines', line=dict(color='#3498db', width=2, dash='dash'),
            name=f"Hyp fit (R²={info['r2']:.3f})",
            hovertemplate='%{y:,.1f} bbl/d',
        ), row=1, col=1)

    cum = info['cum_vol']
    fig.add_trace(go.Scatter(
        x=df_w['date'], y=cum / 1000,
        mode='lines', fill='tozeroy',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46,204,113,0.15)',
        name='Cumulative',
        hovertemplate='%{y:,.1f} Mbbl',
    ), row=1, col=2)

    fig.update_yaxes(title_text='Rate (bbl/d)', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative (Mbbl)', row=1, col=2)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=2)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=450,
        showlegend=True,
        legend=dict(orientation='h', y=-0.2),
    )
    return fig


def well_normalised_figure(info, show_fit):
    df_post = info['df_post']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_post['t_days'], y=df_post['rate'] / info['peak_rate'],
        mode='markers', marker=dict(size=5, color='#2c3e50'),
        name='Observed', hovertemplate='Day %{x:,.0f}<br>q/qp = %{y:.3f}',
    ))
    if show_fit and len(info['df_decline']) >= 4:
        t_fit = np.linspace(0, df_post['t_days'].max(), 300)
        q_norm = hyp_rate(t_fit, info['qi'], info['di'], info['b']) / info['peak_rate']
        fig.add_trace(go.Scatter(
            x=t_fit, y=q_norm,
            mode='lines', line=dict(color='#3498db', width=2, dash='dash'),
            name=f"Fit  b={info['b']:.2f}",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis_title='Days since peak',
        yaxis_title='q / q_peak',
        title='Normalised Decline',
    )
    fig.update_yaxes(range=[0, 1.15])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Type Curve Generator", page_icon="🛢️", layout="wide")

    st.sidebar.title("🛢️ Type Curve Generator")
    st.sidebar.markdown("---")

    df_raw, err = load_data()
    if df_raw is None:
        st.error(err)
        st.stop()

    wells = sorted(df_raw['uwi'].unique(), key=str)
    n_wells = len(wells)
    st.sidebar.metric("Wells", n_wells)
    st.sidebar.metric("Data points", f"{len(df_raw):,}")
    st.sidebar.caption(
        f"{df_raw['date'].min().strftime('%Y-%m')} → {df_raw['date'].max().strftime('%Y-%m')}"
    )
    st.sidebar.markdown("---")

    section = st.sidebar.radio(
        "Navigate",
        ["📈 Individual Wells", "📊 Type Curves (Bootstrap)", "📋 Summary"],
    )

    # ── Compute all wells ──
    @st.cache_data(show_spinner=False)
    def compute_all(_df, _wells):
        out = {}
        for uwi in _wells:
            dfw = _df[_df['uwi'] == uwi].copy()
            if len(dfw) < 3:
                continue
            out[uwi] = analyse_well(dfw)
        return out

    with st.spinner("Analysing wells…"):
        all_results = compute_all(df_raw, wells)

    valid_wells = list(all_results.keys())
    if not valid_wells:
        st.error("No wells with enough data."); st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL WELLS
    # ══════════════════════════════════════════════════════════════════════════
    if section == "📈 Individual Wells":
        st.title("📈 Individual Well Analysis")

        selected = st.selectbox(
            "Select well",
            valid_wells,
            format_func=lambda x: f"{x}  ({all_results[x]['n_months']} mo)",
        )
        info = all_results[selected]

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Peak Rate", f"{info['peak_rate']:,.0f} bbl/d")
        c2.metric("Peak Date", info['peak_date'].strftime('%Y-%m'))
        c3.metric("Last Rate", f"{info['last_rate']:,.1f} bbl/d")
        c4.metric("Producing Life", f"{info['life_years']:.1f} yrs")
        c5.metric("EUR (data)", f"{info['eur_trap']/1000:,.1f} Mbbl")

        # Arps parameters
        st.markdown("#### Arps Decline Parameters")
        di_ann = info['di'] * 365.25
        params = pd.DataFrame({
            'Parameter': ['qi (bbl/d)', 'Di (1/day)', 'Di (1/year)', 'b', 'R²'],
            'Value': [
                f"{info['qi']:,.1f}",
                f"{info['di']:.6f}",
                f"{di_ann:.4f}",
                f"{info['b']:.3f}",
                f"{info['r2']:.4f}",
            ],
        })
        st.dataframe(params, hide_index=True, use_container_width=True)

        # Plots
        st.plotly_chart(well_rate_figure(info, show_fit=True), use_container_width=True)
        st.plotly_chart(well_normalised_figure(info, show_fit=True), use_container_width=True)

        with st.expander("📄 Raw data"):
            disp = info['df'][['date', 'rate', 't_days', 'monthly_vol']].copy()
            disp.columns = ['Date', 'Rate (bbl/d)', 'Days from Peak', 'Monthly Vol (bbl)']
            disp['Cum (Mbbl)'] = info['cum_vol'] / 1000
            st.dataframe(disp, use_container_width=True, height=400)

    # ══════════════════════════════════════════════════════════════════════════
    # BOOTSTRAP TYPE CURVES
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "📊 Type Curves (Bootstrap)":
        st.title("📊 Type Curves — Bootstrap Monte Carlo")
        st.markdown(
            f"**Method:** {N_BOOTSTRAP:,} iterations resampling full *(qi, Di, b)* parameter "
            f"sets from the {len(valid_wells)} fitted wells (with replacement). "
            f"Each sample generates a type curve: **flat for {FLAT_MONTHS} months**, "
            f"then **hyperbolic to {Q_LIMIT} bbl/d**. "
            f"Percentile curves selected by matching EUR rank — preserving physical "
            f"parameter correlations."
        )

        # Build parameter array (qi, di, b) — one row per well
        param_rows = []
        for uwi in valid_wells:
            info = all_results[uwi]
            param_rows.append([info['qi'], info['di'], info['b']])
        param_array = np.array(param_rows)

        with st.spinner(f"Running {N_BOOTSTRAP:,} bootstrap simulations…"):
            eurs, curves = run_bootstrap(param_array, N_BOOTSTRAP, FLAT_DAYS, Q_LIMIT)

        percentiles = [10, 25, 50, 75, 90]
        pctile_curves = select_percentile_curves(eurs, curves, percentiles)

        # ── Summary table ──
        st.markdown("### Selected Type Curve Parameters")
        tc_rows = []
        for p in percentiles:
            c = pctile_curves[p]
            tc_rows.append({
                'Percentile': f'P{p}',
                'qi (bbl/d)': f"{c['qi']:,.1f}",
                'Di (1/day)': f"{c['di']:.6f}",
                'Di (1/year)': f"{c['di']*365.25:.4f}",
                'b': f"{c['b']:.3f}",
                'EUR (Mbbl)': f"{c['eur']/1000:,.1f}",
            })
        st.dataframe(pd.DataFrame(tc_rows), hide_index=True, use_container_width=True)

        # ── Rate type curve ──
        st.markdown("### Rate Type Curves")
        fig_rate = go.Figure()

        # Individual wells as background
        for uwi in valid_wells:
            info = all_results[uwi]
            df_post = info['df_post']
            fig_rate.add_trace(go.Scatter(
                x=df_post['t_days'], y=df_post['rate'],
                mode='lines', line=dict(color='grey', width=0.5),
                opacity=0.2, showlegend=False, hoverinfo='skip',
            ))

        for p in percentiles:
            c = pctile_curves[p]
            fig_rate.add_trace(go.Scatter(
                x=c['t'], y=c['q'],
                mode='lines',
                line=dict(color=PCOLORS[p], width=3),
                name=f"P{p}  qi={c['qi']:,.0f}  Di={c['di']:.5f}  b={c['b']:.2f}",
                hovertemplate=f"P{p}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} bbl/d",
            ))

        fig_rate.add_hline(y=Q_LIMIT, line_dash='dot', line_color='grey', opacity=0.5,
                           annotation_text=f'{Q_LIMIT} bbl/d limit')
        fig_rate.add_vline(x=FLAT_DAYS, line_dash='dot', line_color='orange', opacity=0.6,
                           annotation_text=f'{FLAT_MONTHS} mo flat')

        fig_rate.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
            xaxis_title='Days since peak',
            yaxis_title='Rate (bbl/d)',
            title='P10 / P25 / P50 / P75 / P90 — Rate Type Curves',
            legend=dict(font=dict(size=10)),
        )
        fig_rate.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig_rate, use_container_width=True)

        # ── Cumulative type curve ──
        st.markdown("### Cumulative Type Curves")
        fig_cum = go.Figure()

        for p in percentiles:
            c = pctile_curves[p]
            cum = np.cumsum(c['q'])  # daily steps → bbl
            fig_cum.add_trace(go.Scatter(
                x=c['t'], y=cum / 1000,
                mode='lines',
                line=dict(color=PCOLORS[p], width=3),
                name=f"P{p}  EUR={c['eur']/1000:,.0f} Mbbl",
                hovertemplate=f"P{p}<br>Day %{{x:,.0f}}<br>%{{y:,.0f}} Mbbl",
            ))

        fig_cum.add_vline(x=FLAT_DAYS, line_dash='dot', line_color='orange', opacity=0.6)
        fig_cum.update_layout(
            **PLOTLY_LAYOUT,
            height=480,
            xaxis_title='Days since peak',
            yaxis_title='Cumulative (Mbbl)',
            title='P10 / P25 / P50 / P75 / P90 — Cumulative Type Curves',
        )
        fig_cum.update_yaxes(rangemode='tozero')
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── EUR distribution ──
        st.markdown("### EUR Distribution (Bootstrap)")
        fig_eur = go.Figure()
        eur_mbbl = eurs / 1000
        fig_eur.add_trace(go.Histogram(
            x=eur_mbbl, nbinsx=60,
            marker_color='#3498db', opacity=0.6,
            name='Simulated EUR',
        ))
        for p in percentiles:
            val = pctile_curves[p]['eur'] / 1000
            fig_eur.add_vline(
                x=val, line_dash='dash', line_color=PCOLORS[p], line_width=2,
                annotation_text=f"P{p}: {val:,.0f}",
                annotation_font_color=PCOLORS[p],
            )
        fig_eur.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            xaxis_title='EUR (Mbbl)',
            yaxis_title='Count',
            title=f'EUR Distribution — {N_BOOTSTRAP:,} Bootstrap Samples',
            bargap=0.05,
        )
        st.plotly_chart(fig_eur, use_container_width=True)

        # ── Parameter scatter ──
        st.markdown("### Parameter Space (Fitted Wells)")
        fig_scat = make_subplots(rows=1, cols=3,
                                  subplot_titles=('qi vs Di', 'qi vs b', 'Di vs b'))
        fig_scat.add_trace(go.Scatter(
            x=param_array[:, 1] * 365.25, y=param_array[:, 0],
            mode='markers', marker=dict(size=7, color='#2c3e50', opacity=0.7),
            name='Wells', showlegend=False,
        ), row=1, col=1)
        fig_scat.add_trace(go.Scatter(
            x=param_array[:, 2], y=param_array[:, 0],
            mode='markers', marker=dict(size=7, color='#2c3e50', opacity=0.7),
            showlegend=False,
        ), row=1, col=2)
        fig_scat.add_trace(go.Scatter(
            x=param_array[:, 2], y=param_array[:, 1] * 365.25,
            mode='markers', marker=dict(size=7, color='#2c3e50', opacity=0.7),
            showlegend=False,
        ), row=1, col=3)

        # Mark selected percentile wells
        for p in percentiles:
            c = pctile_curves[p]
            fig_scat.add_trace(go.Scatter(
                x=[c['di'] * 365.25], y=[c['qi']],
                mode='markers', marker=dict(size=12, color=PCOLORS[p], symbol='diamond'),
                name=f'P{p}', showlegend=True,
            ), row=1, col=1)
            fig_scat.add_trace(go.Scatter(
                x=[c['b']], y=[c['qi']],
                mode='markers', marker=dict(size=12, color=PCOLORS[p], symbol='diamond'),
                showlegend=False,
            ), row=1, col=2)
            fig_scat.add_trace(go.Scatter(
                x=[c['b']], y=[c['di'] * 365.25],
                mode='markers', marker=dict(size=12, color=PCOLORS[p], symbol='diamond'),
                showlegend=False,
            ), row=1, col=3)

        fig_scat.update_xaxes(title_text='Di (1/yr)', row=1, col=1)
        fig_scat.update_yaxes(title_text='qi (bbl/d)', row=1, col=1)
        fig_scat.update_xaxes(title_text='b', row=1, col=2)
        fig_scat.update_yaxes(title_text='qi (bbl/d)', row=1, col=2)
        fig_scat.update_xaxes(title_text='b', row=1, col=3)
        fig_scat.update_yaxes(title_text='Di (1/yr)', row=1, col=3)
        fig_scat.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            title='Fitted Parameter Correlations (diamonds = selected percentile wells)',
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "📋 Summary":
        st.title("📋 All Wells — Summary")
        rows = []
        for uwi in valid_wells:
            info = all_results[uwi]
            rows.append({
                'UWI': uwi,
                'Peak (bbl/d)': round(info['peak_rate'], 1),
                'Peak Date': info['peak_date'].strftime('%Y-%m'),
                'Last (bbl/d)': round(info['last_rate'], 1),
                'Months': info['n_months'],
                'qi (bbl/d)': round(info['qi'], 1),
                'Di (1/d)': round(info['di'], 6),
                'Di (1/yr)': round(info['di'] * 365.25, 4),
                'b': round(info['b'], 3),
                'R²': round(info['r2'], 3),
                'EUR (Mbbl)': round(info['eur_trap'] / 1000, 1),
            })
        sdf = pd.DataFrame(rows)
        st.dataframe(
            sdf.style.format({
                'Peak (bbl/d)': '{:,.1f}',
                'Last (bbl/d)': '{:,.1f}',
                'qi (bbl/d)': '{:,.1f}',
                'Di (1/d)': '{:.6f}',
                'Di (1/yr)': '{:.4f}',
                'b': '{:.3f}',
                'R²': '{:.3f}',
                'EUR (Mbbl)': '{:,.1f}',
            }).background_gradient(subset=['EUR (Mbbl)'], cmap='YlGn'),
            use_container_width=True,
            height=min(800, 45 + 35 * len(rows)),
        )

        st.markdown("### Descriptive Statistics")
        num_cols = ['Peak (bbl/d)', 'qi (bbl/d)', 'Di (1/d)', 'Di (1/yr)', 'b', 'R²', 'EUR (Mbbl)']
        st.dataframe(sdf[num_cols].describe().T, use_container_width=True)

        csv = sdf.to_csv(index=False)
        st.download_button("⬇ Download CSV", csv, "well_summary.csv", "text/csv")


if __name__ == "__main__":
    main()