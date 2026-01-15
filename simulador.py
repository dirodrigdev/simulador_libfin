import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- FUNCIÓN PRINCIPAL QUE LLAMA EL DASHBOARD ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0):
    
    # Inicializar memoria
    if 'current_results' not in st.session_state: st.session_state.current_results = None

    # --- CONFIGURACIÓN TÉCNICA ---
    @dataclass
    class AssetBucket:
        name: str; weight: float = 0.0; mu_nominal: float = 0.0; sigma_nominal: float = 0.0

    @dataclass
    class WithdrawalTramo:
        from_year: int; to_year: int; amount_nominal_monthly_start: float

    @dataclass
    class SimulationConfig:
        horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1_000_000; n_sims: int = 2000
        inflation_mean: float = 0.035; inflation_vol: float = 0.01; prob_crisis: float = 0.05
        crisis_drift: float = 0.75; crisis_vol: float = 1.25

    # --- ESCENARIOS PREDEFINIDOS ---
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 4.0, "rv": 7.0, "inf": 4.5, "vol": 22.0, "crisis": 15, "desc": "Alta inflación, retornos bajos."},
        "Estable (Base) ☁️": {"rf": 6.0, "rv": 10.0, "inf": 3.0, "vol": 16.0, "crisis": 5, "desc": "Promedios históricos normales."},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2, "desc": "Mercados alcistas."},
        "Personalizado 🛠️": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5, "desc": "Tus datos reales."}
    }

    # --- MOTOR MATEMÁTICO (NOMINAL) ---
    def run_simulation(cfg, assets, withdrawals):
        dt = 1/cfg.steps_per_year
        n_steps = int(cfg.horizon_years * cfg.steps_per_year)
        
        # Arrays
        capital = np.zeros((cfg.n_sims, n_steps + 1)); capital[:, 0] = cfg.initial_capital
        cpi = np.ones((cfg.n_sims, n_steps + 1))
        
        # Parámetros ponderados del portafolio
        mu_port = sum(a.weight * a.mu_nominal for a in assets)
        sig_port = np.sqrt(sum((a.weight * a.sigma_nominal)**2 for a in assets)) # Simplificado
        
        # Crisis
        p_crisis = 1 - (1 - cfg.prob_crisis)**dt
        in_crisis = np.zeros(cfg.n_sims, dtype=bool)

        for t in range(1, n_steps + 1):
            # Inflación
            inf_shock = np.random.normal(cfg.inflation_mean * dt, cfg.inflation_vol * np.sqrt(dt), cfg.n_sims)
            cpi[:, t] = cpi[:, t-1] * (1 + inf_shock)
            
            # Crisis Switch
            new_c = np.random.rand(cfg.n_sims) < p_crisis
            in_crisis = np.logical_or(in_crisis, new_c)
            in_crisis[np.random.rand(cfg.n_sims) < 0.15] = False # Salida aleatoria
            
            # Ajuste Retornos
            curr_mu = np.where(in_crisis, mu_port * dt * cfg.crisis_drift, mu_port * dt)
            curr_sig = np.where(in_crisis, sig_port * np.sqrt(dt) * cfg.crisis_vol, sig_port * np.sqrt(dt))
            
            # Crecimiento (Browniano Geométrico)
            shock = np.random.normal(0, 1, cfg.n_sims)
            ret = np.exp(curr_mu - 0.5*curr_sig**2 + curr_sig*shock)
            
            # Retiros (Ajustados por Inflación Acumulada)
            year = t / 12
            wd_base = next((w.amount_nominal_monthly_start for w in withdrawals if w.from_year <= year < w.to_year), 0)
            wd_actual = wd_base * cpi[:, t]
            
            # Flujo
            prev = capital[:, t-1]
            alive = prev > 0
            capital[alive, t] = (prev[alive] * ret[alive]) - wd_actual[alive]
            capital[~alive, t] = 0
            
        return capital, cpi

    # --- INTERFAZ DEL SIMULADOR ---
    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # SIDEBAR
    with st.sidebar:
        st.header("1. Escenario")
        s_key = st.selectbox("Modo:", list(SCENARIOS.keys()), index=1)
        scen = SCENARIOS[s_key]
        st.caption(scen['desc'])
        
        is_cust = (s_key == "Personalizado 🛠️")
        with st.expander("Variables", expanded=is_cust):
            inf_in = st.number_input("Inflación (%)", value=scen["inf"], disabled=not is_cust)
            rf_in = st.number_input("Retorno RF (%)", value=scen["rf"], disabled=not is_cust)
            rv_in = st.number_input("Retorno RV (%)", value=scen["rv"], disabled=not is_cust)
            vol_in = st.number_input("Volatilidad RV", value=scen["vol"], disabled=not is_cust)
        
        st.divider()
        n_sims = st.slider("Simulaciones", 500, 5000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 60, 40)
        btn_run = st.button("🚀 EJECUTAR", type="primary")

    # MAIN INPUTS
    st.markdown("### 💰 Capital Inicial")
    # Calculamos total inicial
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: cap = clean("Capital Total ($)", ini_def, "cap")
    with c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60)
    with c3: st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")

    st.markdown("### 💸 Plan de Retiro (Valor Hoy)")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7)
    with g2: r2 = clean("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13)
    with g3: r3 = clean("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")

    # LOGICA EJECUCION
    if btn_run:
        # Configurar
        assets = [
            AssetBucket("RV", pct_rv/100, rv_in/100, vol_in/100),
            AssetBucket("RF", (100-pct_rv)/100, rf_in/100, 0.05)
        ]
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horiz, r3)
        ]
        cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap, n_sims=n_sims, inflation_mean=inf_in/100, prob_crisis=scen["crisis"]/100)
        
        with st.spinner("Procesando..."):
            paths, cpi = run_simulation(cfg, assets, wds)
            
            # Resultados
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            # Herencia ajustada a valor real de hoy (deflactada)
            legacy_real = np.median(final_nom / cpi[:, -1])
            
            st.session_state.current_results = {"succ": success, "leg": legacy_real, "paths": paths, "in": (cap, r1)}

    # RESULTADOS
    if st.session_state.current_results:
        res = st.session_state.current_results
        succ = res["succ"]
        
        # Color Semáforo
        clr = "green" if succ > 90 else "orange" if succ > 75 else "red"
        st.markdown(f"""
        <div style="text-align:center; padding:15px; border:2px solid {clr}; border-radius:10px; margin-top:20px;">
            <h2 style="color:{clr}; margin:0;">Probabilidad de Éxito: {succ:.1f}%</h2>
            <p style="margin:0;">Escenario: <b>{s_key}</b> | Herencia Real Est.: <b>${fmt(res['leg'])}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # Gráfico
        st.subheader("🔭 Proyección")
        y = np.arange(res["paths"].shape[1])/12
        p10 = np.percentile(res["paths"], 10, axis=0)
        p50 = np.percentile(res["paths"], 50, axis=0)
        p90 = np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='blue', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(0,0,255,0.1)', line=dict(width=0), name='Rango 80%'))
        
        fig.update_layout(height=400, yaxis_title="Capital Nominal ($)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
