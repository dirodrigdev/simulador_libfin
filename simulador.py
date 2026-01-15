# NOMBRE DEL ARCHIVO: simulador.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, replace
from typing import List, Dict
import re

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0):
    
    # 1. GESTIÓN DE ESTADO
    if 'current_results' not in st.session_state: st.session_state.current_results = None
    if 'sim_params' not in st.session_state:
        st.session_state.sim_params = {"inf": 3.0, "rf": default_ret_rf, "rv": default_ret_rv, "vol": 16.0, "crisis": 5}

    # --- 2. MOTOR MATEMÁTICO AVANZADO (Recuperado de tu archivo) ---
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

    class AdvancedSimulator:
        def __init__(self, config, assets, withdrawals):
            self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
            self.dt = 1/config.steps_per_year
            self.total_steps = int(config.horizon_years * config.steps_per_year)
            # Matriz de Correlación (Simplificada para 2 activos base + USD implícito si aplica)
            self.corr_matrix = np.eye(len(assets))

        def run(self):
            n_sims, n_steps, n_assets = self.cfg.n_sims, self.total_steps, len(self.assets)
            
            # Inicializar
            capital_paths = np.zeros((n_sims, n_steps + 1))
            capital_paths[:, 0] = self.cfg.initial_capital
            
            # Asset Values individuales (Para simular correlación)
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight
            
            cpi_paths = np.ones((n_sims, n_steps + 1))
            
            # Cholesky para correlación
            try: L = np.linalg.cholesky(self.corr_matrix)
            except: L = np.eye(n_assets)
            
            # Crisis
            p_crisis = 1 - (1 - self.cfg.prob_crisis)**self.dt
            in_crisis = np.zeros(n_sims, dtype=bool)

            for t in range(1, n_steps + 1):
                # Inflación
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # Crisis Switch
                new_c = np.random.rand(n_sims) < p_crisis
                in_crisis = np.logical_or(in_crisis, new_c)
                in_crisis[np.random.rand(n_sims) < 0.15] = False 
                
                # Shocks Correlacionados
                z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
                z_corr = np.dot(z_uncorr, L.T)
                
                # Evolución de cada activo
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    if np.any(in_crisis):
                        mu *= self.cfg.crisis_drift
                        sig *= self.cfg.crisis_vol
                    
                    # Retorno Browniano: (mu - 0.5*sig^2)*dt + sig*sqrt(dt)*Z
                    step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]
                
                asset_values *= np.exp(step_rets)
                
                # Retiros y Rebalanceo
                total_cap = np.sum(asset_values, axis=1)
                
                year = t / 12
                wd_base = 0
                for w in self.withdrawals:
                    if w.from_year <= year < w.to_year:
                        wd_base = w.amount_nominal_monthly_start
                        break
                wd_nom = wd_base * cpi_paths[:, t]
                
                # Ratio de retiro
                ratio = np.divide(wd_nom, total_cap, out=np.zeros_like(total_cap), where=total_cap!=0)
                ratio = np.clip(ratio, 0, 1)
                asset_values *= (1 - ratio[:, np.newaxis])
                
                # Rebalanceo Anual
                if t % 12 == 0:
                    tot = np.sum(asset_values, axis=1)
                    alive = tot > 0
                    for i, asset in enumerate(self.assets):
                        asset_values[alive, i] = tot[alive] * asset.weight
                
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
            return capital_paths, cpi_paths

    # --- 3. HERRAMIENTAS AVANZADAS (GOAL SEEK & OPTIMIZER) ---
    def goal_seek(target_prob, base_sim, var_type, r_min, r_max):
        # Búsqueda binaria para encontrar el valor que da X% de éxito
        low, high = r_min, r_max
        best_val = low
        
        cfg_fast = replace(base_sim.cfg, n_sims=300) # Más rápido
        
        for _ in range(10):
            mid = (low + high) / 2
            
            # Clonar configuración
            new_wds = []
            new_horizon = base_sim.cfg.horizon_years
            
            if var_type == 'duration':
                new_horizon = int(mid)
                cfg_fast = replace(cfg_fast, horizon_years=new_horizon)
                # Ajustar tramos al nuevo horizonte
                for w in base_sim.withdrawals:
                    to_y = min(w.to_year, new_horizon)
                    if w == base_sim.withdrawals[-1]: to_y = new_horizon
                    if w.from_year < new_horizon:
                        new_wds.append(WithdrawalTramo(w.from_year, to_y, w.amount_nominal_monthly_start))
            else:
                # Ajustar montos
                factor = 1.0 # Si var_type es monto, ajustamos todos proporcionalmente o solo el primero
                for w in base_sim.withdrawals:
                    # Asumimos que Goal Seek ajusta el "Nivel de Vida" general (todos los tramos)
                    new_amt = w.amount_nominal_monthly_start * (mid / base_sim.withdrawals[0].amount_nominal_monthly_start)
                    new_wds.append(WithdrawalTramo(w.from_year, w.to_year, new_amt))
            
            sim = AdvancedSimulator(cfg_fast, base_sim.assets, new_wds)
            sim.corr_matrix = base_sim.corr_matrix
            paths, _ = sim.run()
            success = np.mean(paths[:, -1] > 0)
            
            if abs(success - target_prob) < 0.01: return mid
            
            if var_type == 'duration':
                if success < target_prob: high = mid # Menos años = más éxito
                else: low = mid
            else:
                if success < target_prob: high = mid # Menos plata = más éxito
                else: low = mid
            best_val = mid
            
        return best_val

    def optimize_mix(base_sim, base_prob):
        # Fuerza bruta inteligente: Probar mix de RV de 0 a 100
        results = []
        cfg_fast = replace(base_sim.cfg, n_sims=200)
        
        # Recuperar parametros base de los activos
        # Asumimos assets[0]=RV, assets[1]=RF por construcción abajo
        rv_asset = base_sim.assets[0]
        rf_asset = base_sim.assets[1]
        
        for pct_rv in range(0, 101, 10):
            pct_rf = 100 - pct_rv
            
            new_assets = [
                AssetBucket("RV", pct_rv/100, rv_asset.mu_nominal, rv_asset.sigma_nominal),
                AssetBucket("RF", pct_rf/100, rf_asset.mu_nominal, rf_asset.sigma_nominal)
            ]
            
            sim = AdvancedSimulator(cfg_fast, new_assets, base_sim.withdrawals)
            # Matriz correlación simple 2x2
            sim.corr_matrix = np.array([[1.0, 0.2], [0.2, 1.0]]) 
            
            paths, _ = sim.run()
            succ = np.mean(paths[:, -1] > 0)
            results.append({"RV": pct_rv, "RF": pct_rf, "Prob": succ})
            
        return pd.DataFrame(results)

    # --- INTERFAZ ---
    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # CSS Botón Flotante
    st.markdown("""
    <style>
        div.stButton > button[kind="primary"] {
            position: fixed; bottom: 20px; left: 20px; width: 300px; z-index: 999999;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.3); border-radius: 8px; padding: 12px;
            background-color: #ff4b4b; color: white; border: none; font-weight: bold;
        }
        @media (max-width: 640px) { div.stButton > button[kind="primary"] { left: auto; right: 20px; width: auto; } }
    </style>
    """, unsafe_allow_html=True)

    # SIDEBAR
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 5.0, "rv": 8.0, "inf": 4.0, "vol": 20.0, "crisis": 10},
        "Estable (Base) ☁️": {"rf": 6.5, "rv": 10.5, "inf": 3.0, "vol": 16.0, "crisis": 5},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2},
        "Mis Datos 🏠": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5}
    }

    def update_params():
        s = st.session_state.scenario_selector
        if s in SCENARIOS: st.session_state.sim_params.update(SCENARIOS[s])

    with st.sidebar:
        st.header("1. Escenario")
        st.selectbox("Preset:", list(SCENARIOS.keys()), key="scenario_selector", index=1, on_change=update_params)
        
        with st.expander("Variables", expanded=True):
            p_inf = st.number_input("Inflación (%)", value=st.session_state.sim_params["inf"], step=0.1, key="in_inf")
            p_rf = st.number_input("Retorno RF (%)", value=st.session_state.sim_params["rf"], step=0.1, key="in_rf")
            p_rv = st.number_input("Retorno RV (%)", value=st.session_state.sim_params["rv"], step=0.1, key="in_rv")
            p_vol = st.slider("Volatilidad RV", 10.0, 30.0, st.session_state.sim_params["vol"], key="in_vol")
            p_cris = st.slider("Prob. Crisis (%)", 0, 20, st.session_state.sim_params["crisis"], key="in_cris")
            st.session_state.sim_params.update({"inf": p_inf, "rf": p_rf, "rv": p_rv, "vol": p_vol, "crisis": p_cris})

        st.divider()
        n_sims = st.slider("Simulaciones", 500, 5000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 60, 40)
        
        # GOAL SEEK INTEGRADO
        if st.session_state.current_results:
            st.markdown("---")
            st.markdown("### 🎯 Goal Seek")
            goal_type = st.selectbox("Objetivo:", ["Monto Retiro Mensual", "Duración (Años)"])
            target_prob = st.slider("Prob. Éxito Deseada", 50, 95, 90) / 100
            if st.button("Calcular Objetivo"):
                with st.spinner("Buscando..."):
                    res = st.session_state.current_results
                    val = goal_seek(target_prob, res["sim_obj"], "duration" if "Duración" in goal_type else "amount", 1, 100 if "Duración" in goal_type else res["inputs"][0]*0.01)
                    if "Duración" in goal_type: st.success(f"El dinero dura: **{val:.0f} Años**")
                    else: st.success(f"Gasto Máximo: **${fmt(val)}/mes**")

    # MAIN
    st.markdown("### 💰 Capital Inicial")
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: cap = clean("Capital Total ($)", ini_def, "cap")
    with c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60)
    with c3: 
        st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"RF: {p_rf}% | RV: {p_rv}%")

    st.markdown("### 💸 Plan de Retiro (Nominal)")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7)
    with g2: r2 = clean("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13)
    with g3: r3 = clean("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")

    btn_run = st.button("🚀 EJECUTAR ANÁLISIS", type="primary")

    if btn_run:
        # Construir Activos con volatilidad diferenciada
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
        
        sim = AdvancedSimulator(cfg, assets, wds)
        # Correlación RV-RF (0.2)
        sim.corr_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
        
        with st.spinner("Simulando Escenarios..."):
            paths, cpi = sim.run()
            
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            median_legacy = np.median(final_nom / cpi[:, -1])
            
            st.session_state.current_results = {
                "succ": success, "leg": median_legacy, "paths": paths, 
                "sim_obj": sim, "inputs": (cap, r1)
            }

    if st.session_state.current_results:
        res = st.session_state.current_results
        
        clr = "#10b981" if res["succ"] > 90 else "#f59e0b" if res["succ"] > 75 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center; padding:15px; border:2px solid {clr}; border-radius:10px; margin-top:20px;">
            <h2 style="color:{clr}; margin:0;">Probabilidad de Éxito: {res['succ']:.1f}%</h2>
            <p style="margin:0;">Herencia Real Estimada: <b>${fmt(res['leg'])}</b></p>
        </div>
        """, unsafe_allow_html=True)

        y = np.arange(res["paths"].shape[1])/12
        p10 = np.percentile(res["paths"], 10, axis=0)
        p50 = np.percentile(res["paths"], 50, axis=0)
        p90 = np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#3b82f6', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(width=0), name='Rango 80%'))
        st.plotly_chart(fig, use_container_width=True)
        
        # OPTIMIZADOR AUTOMÁTICO (Recuperado)
        if res["succ"] < 99:
            with st.expander("💡 Optimización de Portafolio", expanded=True):
                st.write("Analizando qué combinación RV/RF maximiza tu éxito...")
                df_opt = optimize_mix(res["sim_obj"], res["succ"])
                best = df_opt.loc[df_opt["Prob"].idxmax()]
                
                c_opt1, c_opt2 = st.columns([2,1])
                with c_opt1:
                    fig_opt = go.Figure()
                    fig_opt.add_trace(go.Scatter(x=df_opt["RV"], y=df_opt["Prob"]*100, mode='lines+markers'))
                    fig_opt.update_layout(title="Curva de Eficiencia", xaxis_title="% Renta Variable", yaxis_title="Prob. Éxito %", height=250)
                    st.plotly_chart(fig_opt, use_container_width=True)
                with c_opt2:
                    st.success(f"Mejor Mix: **{int(best['RV'])}% RV**")
                    st.metric("Probabilidad Máxima", f"{best['Prob']*100:.1f}%", delta=f"{(best['Prob']*100 - res['succ']):.1f}%")
