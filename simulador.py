# NOMBRE DEL ARCHIVO: simulador.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0):
    
    # --- 1. GESTIÓN DE ESTADO (Para permitir edición sobre presets) ---
    # Inicializamos claves de inputs si no existen
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = {
            "inf": 3.0, "rf": default_ret_rf, "rv": default_ret_rv, "vol": 16.0, "crisis": 5
        }
    if 'last_scenario' not in st.session_state:
        st.session_state.last_scenario = "Estable (Base) ☁️"
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None

    # --- 2. CONFIGURACIÓN TÉCNICA ---
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

    # --- 3. DEFINICIÓN DE ESCENARIOS (Suavizados) ---
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 5.0, "rv": 8.0, "inf": 4.0, "vol": 20.0, "crisis": 10}, # Ajustado para no ser tan extremo (25%)
        "Estable (Base) ☁️": {"rf": 6.5, "rv": 10.5, "inf": 3.0, "vol": 16.0, "crisis": 5},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2},
        "Mis Datos (Home) 🏠": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5}
    }

    # --- 4. CALLBACKS (Lógica de Actualización) ---
    def update_params():
        # Cuando cambia el selectbox, actualizamos los valores de los inputs
        sel = st.session_state.scenario_selector
        if sel in SCENARIOS:
            vals = SCENARIOS[sel]
            st.session_state.sim_params["inf"] = vals["inf"]
            st.session_state.sim_params["rf"] = vals["rf"]
            st.session_state.sim_params["rv"] = vals["rv"]
            st.session_state.sim_params["vol"] = vals["vol"]
            st.session_state.sim_params["crisis"] = vals["crisis"]

    # --- 5. MOTOR MATEMÁTICO (NOMINAL) ---
    def run_simulation(cfg, assets, withdrawals):
        dt = 1/cfg.steps_per_year
        n_steps = int(cfg.horizon_years * cfg.steps_per_year)
        
        capital = np.zeros((cfg.n_sims, n_steps + 1)); capital[:, 0] = cfg.initial_capital
        cpi = np.ones((cfg.n_sims, n_steps + 1))
        
        mu_port = sum(a.weight * a.mu_nominal for a in assets)
        sig_port = np.sqrt(sum((a.weight * a.sigma_nominal)**2 for a in assets)) 
        
        p_crisis = 1 - (1 - cfg.prob_crisis)**dt
        in_crisis = np.zeros(cfg.n_sims, dtype=bool)
        
        ruin_idx = np.full(cfg.n_sims, -1) # -1 = No quebró

        for t in range(1, n_steps + 1):
            inf_shock = np.random.normal(cfg.inflation_mean * dt, cfg.inflation_vol * np.sqrt(dt), cfg.n_sims)
            cpi[:, t] = cpi[:, t-1] * (1 + inf_shock)
            
            new_c = np.random.rand(cfg.n_sims) < p_crisis
            in_crisis = np.logical_or(in_crisis, new_c)
            in_crisis[np.random.rand(cfg.n_sims) < 0.15] = False
            
            curr_mu = np.where(in_crisis, mu_port * dt * cfg.crisis_drift, mu_port * dt)
            curr_sig = np.where(in_crisis, sig_port * np.sqrt(dt) * cfg.crisis_vol, sig_port * np.sqrt(dt))
            
            shock = np.random.normal(0, 1, cfg.n_sims)
            ret = np.exp(curr_mu - 0.5*curr_sig**2 + curr_sig*shock)
            
            year = t / 12
            wd_base = next((w.amount_nominal_monthly_start for w in withdrawals if w.from_year <= year < w.to_year), 0)
            wd_actual = wd_base * cpi[:, t]
            
            prev = capital[:, t-1]
            alive = prev > 0
            
            # Cálculo capital
            capital[alive, t] = (prev[alive] * ret[alive]) - wd_actual[alive]
            
            # Detectar Ruina Exacta
            just_died = (capital[:, t] <= 0) & (prev > 0)
            ruin_idx[just_died] = t
            capital[~alive, t] = 0
            capital[capital[:, t] < 0, t] = 0
            
        return capital, cpi, ruin_idx

    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- 6. INTERFAZ (SIDEBAR) ---
    with st.sidebar:
        st.header("1. Escenario Base")
        
        # SELECTOR PRINCIPAL
        st.selectbox(
            "Cargar Preset:", 
            list(SCENARIOS.keys()), 
            key="scenario_selector",
            on_change=update_params, # Al cambiar, actualiza los inputs de abajo
            index=1
        )
        
        st.markdown("---")
        st.markdown("### 🔧 Ajuste Fino (Editable)")
        # Inputs conectados a Session State. Si los cambias, se usan estos valores.
        
        p_inf = st.number_input("Inflación Anual (%)", value=st.session_state.sim_params["inf"], step=0.1, format="%.1f", key="in_inf")
        p_rf = st.number_input("Retorno Nom. RF (%)", value=st.session_state.sim_params["rf"], step=0.1, format="%.1f", key="in_rf")
        p_rv = st.number_input("Retorno Nom. RV (%)", value=st.session_state.sim_params["rv"], step=0.1, format="%.1f", key="in_rv")
        p_vol = st.slider("Volatilidad RV", 10.0, 30.0, st.session_state.sim_params["vol"], key="in_vol")
        p_cris = st.slider("Prob. Crisis (%)", 0, 20, st.session_state.sim_params["crisis"], key="in_cris")

        # Actualizamos el diccionario manual con lo que el usuario haya tocado
        st.session_state.sim_params.update({"inf": p_inf, "rf": p_rf, "rv": p_rv, "vol": p_vol, "crisis": p_cris})

        st.markdown("---")
        st.header("2. Simulación")
        n_sims = st.slider("Simulaciones", 500, 5000, 2000)
        horiz = st.slider("Horizonte (Años)", 10, 60, 40)
        
        # CONFIGURACIÓN MÉTRICA RUINA
        st.markdown("### 💀 Análisis de Ruina")
        ruin_percentile = st.slider("Sensibilidad de Riesgo (%)", 70, 99, 90, help="Si eliges 90%, calculamos a partir de qué año ocurren el 90% de las quiebras (ignorando los casos extremos tempranos).")

        btn_run = st.button("🚀 EJECUTAR ANÁLISIS", type="primary")

    # --- 7. INTERFAZ (MAIN) ---
    # CAPITAL
    st.markdown("### 💰 Estructura de Capital")
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: cap = clean("Capital Total ($)", ini_def, "cap")
    with c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60)
    
    monto_rv = cap * (pct_rv/100); monto_rf = cap * ((100-pct_rv)/100)
    with c3:
        st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"Usando retornos: RF {p_rf}% | RV {p_rv}%")

    # GASTOS
    st.markdown("### 💸 Flujo de Retiros")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7)
    with g2: r2 = clean("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13)
    with g3: r3 = clean("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")

    # --- 8. EJECUCIÓN ---
    if btn_run:
        assets = [
            AssetBucket("RV", pct_rv/100, p_rv/100, p_vol/100),
            AssetBucket("RF", (100-pct_rv)/100, p_rf/100, 0.05)
        ]
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horiz, r3)
        ]
        cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap, n_sims=n_sims, inflation_mean=p_inf/100, prob_crisis=p_cris/100)
        
        with st.spinner("Procesando futuros..."):
            paths, cpi, ruin_idx = run_simulation(cfg, assets, wds)
            
            # --- CÁLCULO DE MÉTRICAS AVANZADAS ---
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            prob_ruin = 100 - success
            legacy_real = np.median(final_nom / cpi[:, -1])
            
            # Análisis de Ruina
            fails = ruin_idx[ruin_idx > -1] # Solo índices de quiebra
            if len(fails) > 0:
                fail_years = fails / 12
                median_ruin_year = np.median(fail_years)
                # Percentil: ¿A partir de qué año ocurre el X% de las ruinas?
                # Si threshold es 90%, buscamos el percentil 10 (el 10% más temprano se ignora, el 90% restante ocurre post-fecha)
                # O como lo pediste: "a partir de qué año se ubica el 85-90%"
                # Usaremos Percentil (100 - X). Ej: 90% -> Percentil 10.
                pct_risk_year = np.percentile(fail_years, 100 - ruin_percentile)
            else:
                median_ruin_year = 0
                pct_risk_year = 0

            st.session_state.current_results = {
                "succ": success, "ruin": prob_ruin, "leg": legacy_real, 
                "paths": paths, "med_ruin": median_ruin_year, "risk_start": pct_risk_year,
                "n_fails": len(fails)
            }

    # --- 9. RESULTADOS ---
    if st.session_state.current_results:
        res = st.session_state.current_results
        
        # Color Semáforo
        clr = "green" if res["succ"] > 90 else "orange" if res["succ"] > 75 else "red"
        
        # TARJETA PRINCIPAL
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border:2px solid {clr}; border-radius:15px; background-color: rgba(255,255,255,0.05); margin-top:20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="color:{clr}; margin:0; font-size: 3rem;">{res['succ']:.1f}%</h1>
            <p style="margin:0; text-transform:uppercase; letter-spacing:1px; color:gray;">Probabilidad de Éxito</p>
        </div>
        """, unsafe_allow_html=True)

        # MÉTRICAS DETALLADAS (LO QUE PEDISTE)
        st.markdown("### 💀 Análisis de Supervivencia")
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric("Probabilidad de Ruina", f"{res['ruin']:.1f}%", help="Porcentaje de escenarios donde el dinero se agota antes del fin.")
        
        with c2:
            val = f"Año {res['med_ruin']:.1f}" if res['n_fails'] > 0 else "Nunca"
            st.metric("Mediana de Ruina", val, help="En los casos que fallan, este es el año central donde ocurre el desastre.")
            
        with c3:
            val = f"Año {res['risk_start']:.1f}" if res['n_fails'] > 0 else "N/A"
            st.metric(f"Zona de Riesgo ({ruin_percentile}%)", val, help=f"El {ruin_percentile}% de las quiebras ocurren DESPUÉS de este año. (Filtra las desgracias muy tempranas).")
        
        with c4:
            st.metric("Herencia Real (Hoy)", f"${fmt(res['leg'])}")

        # GRÁFICO
        st.markdown("---")
        y = np.arange(res["paths"].shape[1])/12
        p10 = np.percentile(res["paths"], 10, axis=0)
        p50 = np.percentile(res["paths"], 50, axis=0)
        p90 = np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#2563eb', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', line=dict(width=0), name='Rango 80%'))
        
        # Línea de Ruina
        if res['risk_start'] > 0:
            fig.add_vline(x=res['risk_start'], line_dash="dot", line_color="red", annotation_text=f"Inicio Riesgo {ruin_percentile}%")
            
        fig.update_layout(height=450, yaxis_title="Capital Nominal ($)", hovermode="x unified", title="Evolución del Patrimonio")
        st.plotly_chart(fig, use_container_width=True)
