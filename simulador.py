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
        name: str
        weight: float = 0.0
        mu_nominal: float = 0.0
        sigma_nominal: float = 0.0
        is_bond: bool = False

    @dataclass
    class WithdrawalTramo:
        from_year: int
        to_year: int
        amount_nominal_monthly_start: float

    @dataclass
    class SimulationConfig:
        horizon_years: int = 40
        steps_per_year: int = 12
        initial_capital: float = 1_000_000
        n_sims: int = 2000
        inflation_mean: float = 0.035
        inflation_vol: float = 0.01
        prob_crisis: float = 0.05
        crisis_drift: float = 0.75
        crisis_vol: float = 1.25
        
        # PARAMETROS AVANZADOS
        use_fat_tails: bool = True
        use_mean_reversion: bool = True
        use_guardrails: bool = True
        
        # PARAMETRIZACIÓN DE GUARDRAILS (NUEVO)
        guardrail_trigger: float = 0.15  # Caída del 15% activa el modo ahorro
        guardrail_cut: float = 0.10      # Recorte del 10% del gasto

    class InstitutionalSimulator:
        def __init__(self, config, assets, withdrawals):
            self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
            self.dt = 1/config.steps_per_year
            self.total_steps = int(config.horizon_years * config.steps_per_year)
            self.corr_matrix = np.eye(len(assets))

        def run(self):
            n_sims, n_steps, n_assets = self.cfg.n_sims, self.total_steps, len(self.assets)
            
            capital_paths = np.zeros((n_sims, n_steps + 1)); capital_paths[:, 0] = self.cfg.initial_capital
            cpi_paths = np.ones((n_sims, n_steps + 1))
            
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight

            try: L = np.linalg.cholesky(self.corr_matrix)
            except: L = np.eye(n_assets)
            
            p_crisis = 1 - (1 - self.cfg.prob_crisis)**self.dt
            in_crisis = np.zeros(n_sims, dtype=bool)

            # Para Guardrails: Rastrear el "High Water Mark" (Máximo valor real histórico)
            max_real_wealth = np.full(n_sims, self.cfg.initial_capital)

            for t in range(1, n_steps + 1):
                # 1. Inflación
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # 2. Crisis Switch
                new_c = np.random.rand(n_sims) < p_crisis
                in_crisis = np.logical_or(in_crisis, new_c)
                in_crisis[np.random.rand(n_sims) < 0.15] = False 

                # 3. Shocks
                if self.cfg.use_fat_tails:
                    df = 5; std_adj = np.sqrt((df-2)/df)
                    z_uncorr = np.random.standard_t(df, (n_sims, n_assets)) * std_adj
                else:
                    z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
                z_corr = np.dot(z_uncorr, L.T)
                
                # 4. Evolución Activos
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    if np.any(in_crisis): mu *= self.cfg.crisis_drift; sig *= self.cfg.crisis_vol
                    
                    # Mean Reversion para Bonos
                    if self.cfg.use_mean_reversion and asset.is_bond:
                        step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]
                    else:
                        step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]
                
                asset_values *= np.exp(step_rets)
                
                # 5. RETIROS & GUARDRAILS
                total_cap = np.sum(asset_values, axis=1)
                
                # Actualizar Máximo Histórico Real (High Water Mark)
                current_real_wealth = total_cap / cpi_paths[:, t]
                max_real_wealth = np.maximum(max_real_wealth, current_real_wealth)
                
                # Calcular Monto Base (Nominal sin ajuste aún)
                year = t / 12
                wd_base_start = 0
                for w in self.withdrawals:
                    if w.from_year <= year < w.to_year:
                        wd_base_start = w.amount_nominal_monthly_start
                        break
                
                if self.cfg.use_guardrails:
                    # Calcular Drawdown actual (Caída desde el máximo)
                    drawdown = (max_real_wealth - current_real_wealth) / max_real_wealth
                    
                    # Si la caída es mayor al TRIGGER, aplicamos el CUT
                    in_trouble = drawdown > self.cfg.guardrail_trigger
                    
                    wd_nom = np.zeros(n_sims)
                    # Caso Normal: Inflación completa
                    wd_nom[~in_trouble] = wd_base_start * cpi_paths[~in_trouble, t]
                    # Caso Guardrail Activado: Recorte % sobre el valor inflado
                    # (Significa: "Debería retirar 100, pero retiro 90 por la crisis")
                    wd_nom[in_trouble] = (wd_base_start * cpi_paths[in_trouble, t]) * (1.0 - self.cfg.guardrail_cut)
                else:
                    wd_nom = np.full(n_sims, wd_base_start) * cpi_paths[:, t]

                # Ejecutar retiro
                ratio = np.divide(wd_nom, total_cap, out=np.zeros_like(total_cap), where=total_cap!=0)
                ratio = np.clip(ratio, 0, 1)
                asset_values *= (1 - ratio[:, np.newaxis])
                
                # 6. Rebalanceo
                if t % 12 == 0:
                    tot = np.sum(asset_values, axis=1)
                    alive = tot > 0
                    if np.any(alive):
                        for i, asset in enumerate(self.assets):
                            asset_values[alive, i] = tot[alive] * asset.weight
                
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
            return capital_paths, cpi_paths

    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- 4. INTERFAZ ---
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
        
        with st.expander("Variables", expanded=True):
            p_inf = st.number_input("Inflación (%)", key="in_inf", step=0.1)
            p_rf = st.number_input("Retorno RF (%)", key="in_rf", step=0.1)
            p_rv = st.number_input("Retorno RV (%)", key="in_rv", step=0.1)
            p_vol = st.slider("Volatilidad RV", 10.0, 30.0, key="in_vol")
            p_cris = st.slider("Prob. Crisis (%)", 0, 20, key="in_cris")

        st.divider()
        st.markdown("### 🧠 Configuración Institucional")
        
        # --- AQUÍ ESTÁ LA NUEVA PARAMETRIZACIÓN DE GUARDRAILS ---
        use_guard = st.checkbox("🛡️ Guardrails (Gasto Dinámico)", value=True)
        if use_guard:
            # Inputs condicionales que aparecen solo si activas Guardrails
            c_g1, c_g2 = st.columns(2)
            gr_trigger = c_g1.number_input("Activar si cae (%)", value=15, step=5, help="Si el portafolio cae X% desde su máximo, se activa el recorte.")
            gr_cut = c_g2.number_input("Recorte Gasto (%)", value=10, step=5, help="Porcentaje que reduces tu retiro mientras dure la crisis.")
        else:
            gr_trigger, gr_cut = 15, 10 # Defaults invisibles

        use_fat = st.checkbox("📉 Fat Tails (Eventos Extremos)", value=True)
        use_bond = st.checkbox("🔄 Bonos Reales (Mean Rev)", value=True)
        
        st.divider()
        n_sims = st.slider("Simulaciones", 500, 5000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 60, 40)

    # MAIN
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

    if st.button("🚀 EJECUTAR ANÁLISIS PRO", type="primary"):
        assets = [
            AssetBucket("RV", pct_rv/100, p_rv/100, p_vol/100, is_bond=False),
            AssetBucket("RF", (100-pct_rv)/100, p_rf/100, 0.05, is_bond=True)
        ]
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horiz, r3)
        ]
        cfg = SimulationConfig(
            horizon_years=horiz, initial_capital=cap, n_sims=n_sims, 
            inflation_mean=p_inf/100, prob_crisis=p_cris/100,
            use_guardrails=use_guard, use_fat_tails=use_fat, use_mean_reversion=use_bond,
            # Pasamos los parámetros de usuario a la config
            guardrail_trigger=gr_trigger/100.0, guardrail_cut=gr_cut/100.0
        )
        
        sim = InstitutionalSimulator(cfg, assets, wds)
        sim.corr_matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
        
        with st.spinner("Corriendo Montecarlo Institucional..."):
            paths, cpi = sim.run()
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            median_legacy = np.median(final_nom / cpi[:, -1])
            
            st.session_state.current_results = {"succ": success, "leg": median_legacy, "paths": paths}

    if st.session_state.current_results:
        res = st.session_state.current_results
        
        clr = "#10b981" if res["succ"] > 90 else "#f59e0b" if res["succ"] > 75 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border:2px solid {clr}; border-radius:10px; margin-top:10px; background-color: rgba(0,0,0,0.02);">
            <h2 style="color:{clr}; margin:0; font-size: 2.5rem;">Probabilidad de Éxito: {res['succ']:.1f}%</h2>
            <p style="margin:0; font-size: 1.1rem; color: gray;">Herencia Real Mediana: <b>${fmt(res['leg'])}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🔎 Configuración de Seguridad"):
            c_a, c_b, c_c = st.columns(3)
            if use_guard:
                c_a.success(f"🛡️ **Guardrails:** Activo.\n(Si cae **{gr_trigger}%**, recortas **{gr_cut}%**).")
            else:
                c_a.warning("🛡️ **Guardrails:** Inactivo.")
            c_b.info(f"📉 **Fat Tails:** {'Activo' if use_fat else 'Inactivo'}.")
            c_c.info(f"🔄 **Bonos:** {'Mean Rev' if use_bond else 'Random Walk'}.")

        y = np.arange(res["paths"].shape[1])/12
        p10, p50, p90 = np.percentile(res["paths"], 10, axis=0), np.percentile(res["paths"], 50, axis=0), np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#3b82f6', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(width=0), name='Rango 80%'))
        fig.update_layout(title="Proyección Patrimonial (Institucional)", yaxis_title="Capital Nominal ($)", height=450)
        st.plotly_chart(fig, use_container_width=True)
