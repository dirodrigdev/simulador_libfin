# NOMBRE DEL ARCHIVO: simulador.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, replace
import re

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0):
    
    # --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
    if 'current_results' not in st.session_state: st.session_state.current_results = None
    
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 5.0, "rv": 8.0, "inf": 4.5, "vol": 20.0, "crisis": 10},
        "Estable (Base) ☁️": {"rf": 6.5, "rv": 10.5, "inf": 3.0, "vol": 16.0, "crisis": 5},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2},
        "Mis Datos 🏠": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5}
    }

    # Inicialización de estado para inputs
    if "in_inf" not in st.session_state: st.session_state.in_inf = 3.0
    if "in_rf" not in st.session_state: st.session_state.in_rf = default_ret_rf
    if "in_rv" not in st.session_state: st.session_state.in_rv = default_ret_rv
    if "in_vol" not in st.session_state: st.session_state.in_vol = 16.0
    if "in_cris" not in st.session_state: st.session_state.in_cris = 5

    def update_params_callback():
        sel = st.session_state.scenario_selector
        if sel in SCENARIOS:
            vals = SCENARIOS[sel]
            st.session_state.in_inf = vals["inf"]
            st.session_state.in_rf = vals["rf"]
            st.session_state.in_rv = vals["rv"]
            st.session_state.in_vol = vals["vol"]
            st.session_state.in_cris = vals["crisis"]

    # --- 2. MOTOR MATEMÁTICO ---
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
        simple_mode: bool = False # Nuevo flag para modo optimista

    class AdvancedSimulator:
        def __init__(self, config, assets, withdrawals):
            self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
            self.dt = 1/config.steps_per_year
            self.total_steps = int(config.horizon_years * config.steps_per_year)
            # Matriz de Correlación Base
            self.corr_matrix = np.eye(len(assets))

        def run(self):
            n_sims, n_steps, n_assets = self.cfg.n_sims, self.total_steps, len(self.assets)
            
            capital_paths = np.zeros((n_sims, n_steps + 1)); capital_paths[:, 0] = self.cfg.initial_capital
            cpi_paths = np.ones((n_sims, n_steps + 1))
            
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight

            # Lógica Modo Simple vs Avanzado
            if self.cfg.simple_mode:
                L = np.eye(n_assets) # Sin correlación
                p_crisis = 0 # Sin crisis
            else:
                try: L = np.linalg.cholesky(self.corr_matrix)
                except: L = np.eye(n_assets)
                p_crisis = 1 - (1 - self.cfg.prob_crisis)**self.dt

            in_crisis = np.zeros(n_sims, dtype=bool)

            for t in range(1, n_steps + 1):
                # Inflación
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # Crisis
                if not self.cfg.simple_mode:
                    new_c = np.random.rand(n_sims) < p_crisis
                    in_crisis = np.logical_or(in_crisis, new_c)
                    in_crisis[np.random.rand(n_sims) < 0.15] = False 
                
                # Retornos
                z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
                z_corr = np.dot(z_uncorr, L.T)
                
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    if np.any(in_crisis):
                        mu *= self.cfg.crisis_drift
                        sig *= self.cfg.crisis_vol
                    
                    # FÓRMULA CLAVE: (mu - 0.5*sig^2) es el crecimiento geométrico real
                    step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]
                
                asset_values *= np.exp(step_rets)
                
                # Retiros
                total_cap = np.sum(asset_values, axis=1)
                year = t / 12
                wd_base = 0
                for w in self.withdrawals:
                    if w.from_year <= year < w.to_year:
                        wd_base = w.amount_nominal_monthly_start
                        break
                wd_nom = wd_base * cpi_paths[:, t]
                
                ratio = np.divide(wd_nom, total_cap, out=np.zeros_like(total_cap), where=total_cap!=0)
                ratio = np.clip(ratio, 0, 1)
                asset_values *= (1 - ratio[:, np.newaxis])
                
                # Rebalanceo
                if t % 12 == 0:
                    tot = np.sum(asset_values, axis=1)
                    alive = tot > 0
                    if np.any(alive):
                        for i, asset in enumerate(self.assets):
                            asset_values[alive, i] = tot[alive] * asset.weight
                
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
            return capital_paths, cpi_paths

    # --- 3. TOOLS ---
    def goal_seek(target_prob, base_sim, var_type, r_min, r_max):
        low, high = r_min, r_max
        best_val = low
        cfg_fast = replace(base_sim.cfg, n_sims=300)
        
        for _ in range(12):
            mid = (low + high) / 2
            new_wds = []
            new_horizon = base_sim.cfg.horizon_years
            
            if var_type == 'duration':
                new_horizon = int(mid)
                cfg_fast = replace(cfg_fast, horizon_years=new_horizon)
                for w in base_sim.withdrawals:
                    to_y = min(w.to_year, new_horizon)
                    if w == base_sim.withdrawals[-1]: to_y = new_horizon
                    if w.from_year < new_horizon:
                        new_wds.append(WithdrawalTramo(w.from_year, to_y, w.amount_nominal_monthly_start))
            else: 
                factor = mid / base_sim.withdrawals[0].amount_nominal_monthly_start if base_sim.withdrawals[0].amount_nominal_monthly_start > 0 else 1
                for w in base_sim.withdrawals:
                    new_wds.append(WithdrawalTramo(w.from_year, w.to_year, w.amount_nominal_monthly_start * factor))
            
            sim = AdvancedSimulator(cfg_fast, base_sim.assets, new_wds)
            sim.corr_matrix = base_sim.corr_matrix
            paths, _ = sim.run()
            success = np.mean(paths[:, -1] > 0)
            
            if abs(success - target_prob) < 0.01: return mid
            if var_type == 'duration':
                if success < target_prob: high = mid 
                else: low = mid
            else:
                if success < target_prob: high = mid 
                else: low = mid
            best_val = mid
        return best_val

    def optimize_mix(base_sim):
        results = []
        cfg_fast = replace(base_sim.cfg, n_sims=200)
        rv_mu, rv_sig = base_sim.assets[0].mu_nominal, base_sim.assets[0].sigma_nominal
        rf_mu, rf_sig = base_sim.assets[1].mu_nominal, base_sim.assets[1].sigma_nominal
        
        for pct_rv in range(0, 101, 10):
            pct_rf = 100 - pct_rv
            new_assets = [
                AssetBucket("RV", pct_rv/100, rv_mu, rv_sig),
                AssetBucket("RF", pct_rf/100, rf_mu, rf_sig)
            ]
            sim = AdvancedSimulator(cfg_fast, new_assets, base_sim.withdrawals)
            sim.corr_matrix = base_sim.corr_matrix
            paths, _ = sim.run()
            succ = np.mean(paths[:, -1] > 0)
            results.append({"RV": pct_rv, "RF": pct_rf, "Prob": succ})
        return pd.DataFrame(results)

    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- 4. UI ---
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

    with st.sidebar:
        st.header("1. Escenario")
        st.selectbox("Preset:", list(SCENARIOS.keys()), key="scenario_selector", index=1, on_change=update_params_callback)
        
        with st.expander("Variables (Editables)", expanded=True):
            p_inf = st.number_input("Inflación (%)", key="in_inf", step=0.1)
            p_rf = st.number_input("Retorno RF (%)", key="in_rf", step=0.1)
            p_rv = st.number_input("Retorno RV (%)", key="in_rv", step=0.1)
            p_vol = st.slider("Volatilidad RV", 10.0, 30.0, key="in_vol")
            p_cris = st.slider("Prob. Crisis (%)", 0, 20, key="in_cris")

        st.markdown("---")
        # SWITCH PARA MODO SIMPLE VS REALISTA
        use_simple = st.checkbox("Modo Optimista (Sin Correlación)", value=False, help="Ignora crisis y asume correlación perfecta. Da resultados más altos pero menos realistas.")
        
        st.divider()
        n_sims = st.slider("Simulaciones", 500, 5000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 60, 40)
        
        st.markdown("---")
        st.markdown("### 🎯 Goal Seek")
        if st.session_state.current_results:
            gs_type = st.selectbox("Objetivo:", ["Monto Retiro Mensual", "Duración (Años)"])
            target_prob = st.slider("Prob. Éxito Deseada", 50, 95, 90) / 100
            if st.button("🔍 Calcular Objetivo"):
                res = st.session_state.current_results
                val = goal_seek(target_prob, res["sim_obj"], "duration" if "Duración" in gs_type else "amount", 1, 100 if "Duración" in gs_type else res["inputs"][0]*0.01)
                st.markdown("#### Resultado:")
                if "Duración" in gs_type: st.success(f"Duración: **{val:.0f} Años**")
                else: st.success(f"Gasto Máx: **${fmt(val)}/mes**")
        else:
            st.info("Ejecuta simulación primero.")

    st.markdown("### 💰 Capital Inicial")
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: cap = clean("Capital Total ($)", ini_def, "cap")
    with c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60)
    with c3: 
        st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"Nominales: RF {p_rf}% | RV {p_rv}%")

    st.markdown("### 💸 Plan de Retiro (Nominal)")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7)
    with g2: r2 = clean("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13)
    with g3: r3 = clean("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")

    if st.button("🚀 EJECUTAR ANÁLISIS", type="primary"):
        assets = [
            AssetBucket("RV", pct_rv/100, p_rv/100, p_vol/100),
            AssetBucket("RF", (100-pct_rv)/100, p_rf/100, 0.05)
        ]
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horiz, r3)
        ]
        cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap, n_sims=n_sims, inflation_mean=p_inf/100, prob_crisis=p_cris/100, simple_mode=use_simple)
        
        sim = AdvancedSimulator(cfg, assets, wds)
        sim.corr_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
        
        with st.spinner("Calculando..."):
            paths, cpi = sim.run()
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            median_legacy = np.median(final_nom / cpi[:, -1])
            
            st.session_state.current_results = {"succ": success, "leg": median_legacy, "paths": paths, "sim_obj": sim, "inputs": (cap, r1)}

    if st.session_state.current_results:
        res = st.session_state.current_results
        
        # AUDITORÍA MATEMÁTICA
        with st.expander("🧮 Auditoría del Cálculo (¿Por qué me da esto?)", expanded=True):
            avg_nom = (p_rv * pct_rv/100) + (p_rf * (100-pct_rv)/100)
            real_rate = avg_nom - p_inf
            # Estimación de arrastre por volatilidad (aprox)
            vol_drag = 0.5 * ((p_vol/100)**2 * pct_rv/100) * 100
            geo_real = real_rate - vol_drag
            
            ka, kb, kc = st.columns(3)
            ka.metric("Retorno Nominal Promedio", f"{avg_nom:.1f}%")
            kb.metric("Retorno Real (Sin Inflación)", f"{real_rate:.1f}%", help="Nominal - Inflación")
            kc.metric("Crecimiento Geométrico Real", f"~{geo_real:.1f}%", help="Lo que realmente crece tu dinero después de volatilidad.", delta="-Volatility Drag" if not use_simple else "Optimista")
            
            if geo_real < 4.0 and not use_simple:
                st.warning(f"⚠️ **Alerta Matemática:** Tu portafolio crece realmente al ~{geo_real:.1f}%, pero estás intentando retirar un 4-5%. Por eso la probabilidad baja al 60%.")

        clr = "#10b981" if res["succ"] > 90 else "#f59e0b" if res["succ"] > 75 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center; padding:15px; border:2px solid {clr}; border-radius:10px; margin-top:10px;">
            <h2 style="color:{clr}; margin:0;">Probabilidad de Éxito: {res['succ']:.1f}%</h2>
            <p style="margin:0;">Herencia Real Estimada: <b>${fmt(res['leg'])}</b></p>
        </div>
        """, unsafe_allow_html=True)

        y = np.arange(res["paths"].shape[1])/12
        p10, p50, p90 = np.percentile(res["paths"], 10, axis=0), np.percentile(res["paths"], 50, axis=0), np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#3b82f6', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(width=0), name='Rango 80%'))
        st.plotly_chart(fig, use_container_width=True)

        if res["succ"] < 99:
            st.markdown("### 💡 Optimización")
            with st.expander("Ver Frontera Eficiente"):
                df_opt = optimize_mix(res["sim_obj"])
                best = df_opt.loc[df_opt["Prob"].idxmax()]
                fig_opt = go.Figure()
                fig_opt.add_trace(go.Scatter(x=df_opt["RV"], y=df_opt["Prob"]*100, mode='lines+markers'))
                st.plotly_chart(fig_opt, use_container_width=True)
                st.success(f"Mejor Mix: **{int(best['RV'])}% RV** -> Prob: **{best['Prob']*100:.1f}%**")
