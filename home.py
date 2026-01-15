import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, replace
from typing import List, Dict
import re
import datetime

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930,
        default_ret_rf=6.0, default_ret_mx=8.0, default_ret_rv=10.0, default_ret_usd=4.5):
    
    # Gestión de Estado
    if 'current_results' not in st.session_state: st.session_state.current_results = None
    if 'prev_results' not in st.session_state: st.session_state.prev_results = None

    # --- CLASES DE LÓGICA FINANCIERA (MONTECARLO) ---
    @dataclass
    class AssetBucket:
        name: str; weight: float = 0.0; mu_nominal: float = 0.0; sigma_nominal: float = 0.0; fee_annual: float = 0.0 

    @dataclass
    class WithdrawalTramo:
        from_year: int; to_year: int; amount_real_monthly: float

    @dataclass
    class SimulationConfig:
        horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1_000_000
        n_sims: int = 2000; random_seed: int = 42
        inflation_mean_annual: float = 0.035; inflation_sd_annual: float = 0.01; model_inflation_stochastic: bool = True
        use_regime_switching: bool = True; p_enter_crisis_annual: float = 0.05
        crisis_drift_multiplier: float = 0.75; crisis_vol_multiplier: float = 1.25

    class FinancialSimulator:
        def __init__(self, config, target_weights, asset_names):
            self.cfg = config; self.dt = 1/config.steps_per_year
            self.total_steps = int(config.horizon_years * config.steps_per_year)
            self.assets = []; self.withdrawals = []
            self.target_weights = target_weights; self.asset_names = asset_names
            self.corr_matrix_normal = None; self.corr_matrix_crisis = None

        def run(self):
            np.random.seed(self.cfg.random_seed)
            n_sims, n_steps, n_assets = self.cfg.n_sims, self.total_steps, len(self.assets)
            
            # Inicialización
            capital_paths = np.zeros((n_sims, n_steps + 1))
            capital_paths[:, 0] = self.cfg.initial_capital
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight
            
            cpi_paths = np.ones((n_sims, n_steps + 1))
            
            # Pre-cálculo Crisis
            crisis_state = np.zeros(n_sims, dtype=bool)
            p_crisis = 1 - (1 - self.cfg.p_enter_crisis_annual)**(self.dt)

            # Matriz Cholesky
            try: L = np.linalg.cholesky(self.corr_matrix_normal)
            except: L = np.eye(n_assets)

            # Bucle Temporal (Vectorizado por pasos)
            for t in range(1, n_steps + 1):
                # Inflación
                inf_step = np.random.normal(self.cfg.inflation_mean_annual * self.dt, self.cfg.inflation_sd_annual * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_step)
                
                # Crisis Switch
                new_crisis = np.random.rand(n_sims) < p_crisis
                crisis_state = np.logical_or(crisis_state, new_crisis) # Simplificación: una vez crisis, puede mantenerse o salir (aquí simple)
                # Reset crisis aleatorio para no ser eterno
                crisis_state[np.random.rand(n_sims) < 0.1] = False 

                # Retornos Activos
                z = np.random.normal(0, 1, (n_sims, n_assets))
                shocks = np.dot(z, L.T)
                
                # Aplicar retornos
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    if np.any(crisis_state): # Ajuste Crisis
                        mu *= self.cfg.crisis_drift_multiplier
                        sig *= self.cfg.crisis_vol_multiplier
                    step_rets[:, i] = (mu * self.dt) + (sig * np.sqrt(self.dt) * shocks[:, i])
                
                asset_values *= np.exp(step_rets)
                
                # Retiros y Rebalanceo
                total_cap = np.sum(asset_values, axis=1)
                
                # Determinar retiro del mes (basado en tramos)
                step_year = t / 12
                wd_amount = 0
                for tramo in self.withdrawals:
                    if tramo.from_year <= step_year < tramo.to_year:
                        wd_amount = tramo.amount_real_monthly
                        break
                
                wd_nominal = wd_amount * cpi_paths[:, t]
                ratio = np.divide(wd_nominal, total_cap, out=np.zeros_like(total_cap), where=total_cap!=0)
                ratio = np.clip(ratio, 0, 1)
                
                asset_values *= (1 - ratio[:, np.newaxis])
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
                # Rebalanceo Anual
                if t % 12 == 0:
                    tot = np.sum(asset_values, axis=1)
                    for i, name in enumerate(self.asset_names):
                        asset_values[:, i] = tot * self.target_weights.get(name, 0)

            return capital_paths, cpi_paths

    # --- INTERFAZ UI ---
    st.title("💸 Simulador de Libertad Financiera")

    def clean_input(label, default, key):
        val = st.text_input(label, value=f"{int(default):,}".replace(",", "."), key=key)
        return int(re.sub(r'\D', '', val)) if val else 0

    # --- SIDEBAR CONFIGURACIÓN ---
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # 1. FUENTE DE DATOS (Por defecto MANUAL como pediste)
        source = st.radio("Fuente de Datos", ["Manual", "Pegar JSON (Gems)"], index=0) # <--- MANUAL POR DEFECTO
        
        # Lógica de carga
        cap_inicial = 1800000000 # Default pedido: 1.800MM
        pct_rv_user = 60 # Default pedido: 60%
        
        if source == "Pegar JSON (Gems)":
            txt = st.text_area("JSON", height=100)
            if txt: st.success("JSON Cargado (Simulado)") # Aquí iría lógica real si se pega
        else:
            # Input Manual pero pre-llenado con lo que viene de Home.py si existe, o el default 1800MM
            val_show = default_rf + default_rv + default_usd_nominal * default_tc
            if val_show == 0: val_show = 1800000000
            cap_inicial = clean_input("Capital Inicial ($)", val_show, "cap_ini_sidebar")
            
            # Slider RV
            pct_rv_user = st.slider("% Renta Variable", 0, 100, 60, key="slider_rv") # <--- DEFAULT 60

        st.divider()
        n_sims = st.slider("Simulaciones", 100, 5000, 2000)
        horizonte = st.slider("Horizonte (Años)", 10, 60, 40) # <--- DEFAULT 40
        
        st.markdown("---")
        st.caption("Parámetros Avanzados")
        with st.expander("Ver Detalles"):
            inf = st.number_input("Inflación (%)", 3.5)/100
            vol_rv = st.number_input("Volatilidad RV", 0.18)
            vol_rf = st.number_input("Volatilidad RF", 0.05)

        ejecutar = st.button("🚀 EJECUTAR ESCENARIOS", type="primary")

    # --- MAIN UI ---
    st.header("1. Posición Inicial")
    
    # Calcular montos según el % del slider
    monto_rv = cap_inicial * (pct_rv_user / 100)
    monto_rf = cap_inicial * ((100 - pct_rv_user) / 100)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Capital Total", f"$ {cap_inicial:,.0f}".replace(",", "."))
    c2.metric("Renta Variable (RV)", f"{pct_rv_user}%", f"$ {monto_rv:,.0f}".replace(",", "."))
    c3.metric("Renta Fija (RF)", f"{100-pct_rv_user}%", f"$ {monto_rf:,.0f}".replace(",", "."))

    st.header("2. Estrategia de Retiro")
    c_r1, c_r2, c_r3 = st.columns(3)
    with c_r1:
        r1 = clean_input("Retiro Fase 1 ($)", 6000000, "r1")
        d1 = st.number_input("Años Fase 1", value=7)
    with c_r2:
        r2 = clean_input("Retiro Fase 2 ($)", 5500000, "r2")
        d2 = st.number_input("Años Fase 2", value=13)
    with c_r3:
        r3 = clean_input("Retiro Fase 3 ($)", 5000000, "r3")
        st.caption(f"Resto ({horizonte - d1 - d2} años)")

    # --- EJECUCIÓN ---
    if ejecutar:
        cfg = SimulationConfig(horizon_years=horizonte, initial_capital=cap_inicial, n_sims=n_sims, inflation_mean_annual=inf)
        
        # Activos construidos dinámicamente según el slider
        assets = [
            AssetBucket("RV", pct_rv_user/100, default_ret_rv/100, vol_rv),
            AssetBucket("RF", (100-pct_rv_user)/100, default_ret_rf/100, vol_rf)
        ]
        target_w = {"RV": pct_rv_user/100, "RF": (100-pct_rv_user)/100}
        
        # Tramos de Retiro
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horizonte, r3)
        ]
        
        sim = FinancialSimulator(cfg, target_w, ["RV", "RF"])
        sim.assets = assets; sim.withdrawals = wds
        # Correlación simple
        sim.corr_matrix_normal = np.array([[1.0, 0.2], [0.2, 1.0]]) 
        
        with st.spinner("Simulando futuros..."):
            paths, cpi = sim.run()
            
            # Análisis Rápido
            final_cap = paths[:, -1]
            success = np.mean(final_cap > 0) * 100
            median_end = np.median(final_cap)
            
            # Gráfico
            st.divider()
            st.subheader("Resultados")
            
            k1, k2 = st.columns(2)
            k1.metric("Probabilidad de Éxito", f"{success:.1f}%", delta="Objetivo > 90%")
            k2.metric("Herencia Mediana (Nominal)", f"$ {median_end:,.0f}".replace(",", "."))
            
            # Plot
            p10 = np.percentile(paths, 10, axis=0)
            p50 = np.percentile(paths, 50, axis=0)
            p90 = np.percentile(paths, 90, axis=0)
            x_axis = np.arange(paths.shape[1]) / 12
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_axis, y=p50, line=dict(color='blue', width=3), name='Mediana'))
            fig.add_trace(go.Scatter(x=x_axis, y=p10, line=dict(color='rgba(0,0,255,0.2)'), fill=None, name='P10'))
            fig.add_trace(go.Scatter(x=x_axis, y=p90, line=dict(color='rgba(0,0,255,0.2)'), fill='tonexty', name='Rango 80%'))
            fig.update_layout(title="Proyección de Capital", xaxis_title="Años", yaxis_title="Monto ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)
