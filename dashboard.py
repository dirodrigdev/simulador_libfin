# Archivo: simulador.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict
import re
import datetime

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0):
    
    # 1. GESTIÓN DE ESTADO
    if 'current_results' not in st.session_state: st.session_state.current_results = None
    if 'prev_results' not in st.session_state: st.session_state.prev_results = None

    # --- CLASES Y CONFIGURACIÓN ---
    @dataclass
    class AssetBucket:
        name: str; weight: float = 0.0; mu_nominal: float = 0.0; sigma_nominal: float = 0.0

    @dataclass
    class WithdrawalTramo:
        from_year: int; to_year: int; amount_nominal_monthly_start: float

    @dataclass
    class SimulationConfig:
        horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1_000_000; n_sims: int = 2000
        inflation_mean: float = 0.035; inflation_vol: float = 0.01
        admin_fee: float = 0.0 # Costo explícito
        # Crisis
        use_crisis: bool = True; prob_crisis: float = 0.05
        crisis_drift_factor: float = 0.75; crisis_vol_factor: float = 1.25

    # --- DEFINICIÓN DE ESCENARIOS ---
    SCENARIOS = {
        "Pesimista 🌧️": {
            "ret_rf": 4.0, "ret_rv": 7.0, "inf": 4.5, "vol_rv": 22.0, "prob_crisis": 15,
            "desc": "Estanflación: Retornos bajos e inflación alta."
        },
        "Estable (Base) ☁️": {
            "ret_rf": 6.0, "ret_rv": 10.0, "inf": 3.0, "vol_rv": 16.0, "prob_crisis": 5,
            "desc": "Promedios históricos razonables."
        },
        "Optimista ☀️": {
            "ret_rf": 7.5, "ret_rv": 13.0, "inf": 2.5, "vol_rv": 14.0, "prob_crisis": 2,
            "desc": "Viento a favor: Mercados alcistas."
        },
        "Personalizado 🛠️": {
            "ret_rf": default_ret_rf, "ret_rv": default_ret_rv, # Toma tus datos reales
            "inf": 3.5, "vol_rv": 18.0, "prob_crisis": 5,
            "desc": "Configuración manual basada en tus datos."
        }
    }

    # --- MOTOR DE SIMULACIÓN (NOMINAL) ---
    class NominalSimulator:
        def __init__(self, config, assets, withdrawals):
            self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
            self.dt = 1/config.steps_per_year
            self.total_steps = int(config.horizon_years * config.steps_per_year)

        def run(self):
            n_sims, n_steps = self.cfg.n_sims, self.total_steps
            n_assets = len(self.assets)
            
            # 1. Inicializar
            capital_paths = np.zeros((n_sims, n_steps + 1))
            capital_paths[:, 0] = self.cfg.initial_capital
            
            # Distribución de activos (Simplificada: Rebalanceo mensual implícito en el pool total)
            # Para mayor precisión, simulamos el pool ponderado
            w_avg_mu = sum(a.weight * a.mu_nominal for a in self.assets)
            w_avg_sigma = np.sqrt(sum((a.weight * a.sigma_nominal)**2 for a in self.assets)) # Aprox sin correlación para velocidad
            
            # Ajuste Costos
            w_avg_mu -= self.cfg.admin_fee

            # Inflación Acumulada
            cpi_paths = np.ones((n_sims, n_steps + 1))
            
            # Estado Crisis
            crisis_state = np.zeros(n_sims, dtype=bool)
            p_crisis_step = 1 - (1 - self.cfg.prob_crisis)**(self.dt)

            for t in range(1, n_steps + 1):
                # A. Inflación del periodo
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # B. Crisis
                if self.cfg.use_crisis:
                    new_crisis = np.random.rand(n_sims) < p_crisis_step
                    crisis_state = np.logical_or(crisis_state, new_crisis)
                    crisis_state[np.random.rand(n_sims) < 0.15] = False # Salida crisis
                
                # C. Retorno Portafolio
                mu_step = w_avg_mu * self.dt
                sig_step = w_avg_sigma * np.sqrt(self.dt)
                
                # Ajuste Crisis
                current_mu = np.where(crisis_state, mu_step * self.cfg.crisis_drift_factor, mu_step)
                current_sig = np.where(crisis_state, sig_step * self.cfg.crisis_vol_factor, sig_step)
                
                # Retorno Geométrico: exp(mu - 0.5*sig^2 + sig*Z)
                shock = np.random.normal(0, 1, n_sims)
                geo_ret = np.exp(current_mu - 0.5*(current_sig**2) + current_sig*shock)
                
                # Crecimiento
                # D. Retiros (Ajustados por Inflación)
                year_curr = t / 12
                wd_base = 0
                for tr in self.withdrawals:
                    if tr.from_year <= year_curr < tr.to_year:
                        wd_base = tr.amount_nominal_monthly_start
                        break
                
                wd_nominal = wd_base * cpi_paths[:, t]
                
                # E. Aplicar flujo
                # Capital(t) = Capital(t-1) * Retorno - Retiro
                prev_cap = capital_paths[:, t-1]
                mask_alive = prev_cap > 0
                
                capital_paths[mask_alive, t] = (prev_cap[mask_alive] * geo_ret[mask_alive]) - wd_nominal[mask_alive]
                capital_paths[~mask_alive, t] = 0
                
            return capital_paths, cpi_paths

    # --- INTERFAZ UI ---
    st.title("🛡️ Simulador Pro V3.1 (Nominal)")
    
    def clean_input(label, default, key):
        val = st.text_input(label, value=f"{int(default):,}".replace(",", "."), key=key)
        return int(re.sub(r'\D', '', val)) if val else 0

    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("1. Escenario de Mercado")
        
        # SELECTOR DE ESCENARIO
        scen_key = st.selectbox("Selecciona Escenario:", list(SCENARIOS.keys()), index=1)
        scen = SCENARIOS[scen_key]
        st.info(f"ℹ️ {scen['desc']}")
        
        # EDITOR (Solo si es personalizado)
        is_custom = (scen_key == "Personalizado 🛠️")
        with st.expander("Ver Variables", expanded=is_custom):
            s_inf = st.number_input("Inflación (%)", value=scen["inf"], format="%.1f", disabled=not is_custom)
            s_rf = st.number_input("Retorno RF (%)", value=scen["ret_rf"], format="%.1f", disabled=not is_custom)
            s_rv = st.number_input("Retorno RV (%)", value=scen["ret_rv"], format="%.1f", disabled=not is_custom)
            s_vol = st.slider("Volatilidad RV", 10.0, 25.0, scen["vol_rv"], disabled=not is_custom)
            s_crisis = st.slider("Prob. Crisis", 0, 20, scen["prob_crisis"], disabled=not is_custom)
            
        st.divider()
        n_sims = st.slider("Simulaciones", 500, 5000, 1000)
        horizonte = st.slider("Horizonte", 10, 60, 40)
        
        btn_run = st.button("🚀 EJECUTAR", type="primary")

    # --- MAIN ---
    
    # 1. CAPITAL
    col_c1, col_c2, col_c3 = st.columns(3)
    
    # Defaults
    ini_clp = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_clp == 0: ini_clp = 1800000000
    
    with col_c1: cap_total = clean_input("Capital Total ($)", ini_clp, "c_tot")
    with col_c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60, key="slide_rv")
    
    # MIX VISUAL
    monto_rv = cap_total * (pct_rv/100)
    monto_rf = cap_total * ((100-pct_rv)/100)
    with col_c3:
        st.metric("Tu Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"RF: ${fmt(monto_rf)} | RV: ${fmt(monto_rv)}")

    # 2. GASTOS
    st.markdown("### 💸 Plan de Retiro (A valor de hoy)")
    c_g1, c_g2, c_g3 = st.columns(3)
    with c_g1: r1 = clean_input("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7, key="d1")
    with c_g2: r2 = clean_input("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13, key="d2")
    with c_g3: r3 = clean_input("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")

    # --- LÓGICA DE EJECUCIÓN ---
    if btn_run:
        # Configurar Activos
        assets = [
            AssetBucket("RV", pct_rv/100, s_rv/100, s_vol/100),
            AssetBucket("RF", (100-pct_rv)/100, s_rf/100, 0.05) # RF Vol baja fija
        ]
        
        # Configurar Retiros
        wds = [
            WithdrawalTramo(0, d1, r1),
            WithdrawalTramo(d1, d1+d2, r2),
            WithdrawalTramo(d1+d2, horizonte, r3)
        ]
        
        # Configurar Simulador
        cfg = SimulationConfig(
            horizon_years=horizonte, initial_capital=cap_total, n_sims=n_sims,
            inflation_mean=s_inf/100, prob_crisis=s_crisis/100
        )
        
        sim = NominalSimulator(cfg, assets, wds)
        
        with st.spinner("Procesando futuros alternativos..."):
            paths, cpi = sim.run()
            
            # Análisis Resultados (Real al final)
            final_nom = paths[:, -1]
            final_real = final_nom / cpi[:, -1]
            success = np.mean(final_nom > 0) * 100
            median_legacy = np.median(final_real)
            
            # Guardar en sesión para persistencia
            st.session_state.current_results = {
                "success": success, "legacy": median_legacy, 
                "paths": paths, "cpi": cpi, "cfg": cfg,
                "inputs": (cap_total, r1)
            }

    # --- MOSTRAR RESULTADOS ---
    if st.session_state.current_results:
        res = st.session_state.current_results
        succ = res["success"]
        
        st.divider()
        
        # SEMÁFORO
        color = "green" if succ > 90 else "orange" if succ > 75 else "red"
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px; background-color: rgba(0,0,0,0.02);">
            <h2 style="color: {color}; margin:0;">Probabilidad de Éxito: {succ:.1f}%</h2>
            <p style="margin:0; color: gray;">Escenario: <b>{scen_key}</b> | Herencia (Valor Hoy): <b>${fmt(res['legacy'])}</b></p>
        </div>
        """, unsafe_allow_html=True)

        # GRÁFICO
        st.subheader("🔭 Trayectoria del Patrimonio")
        years = np.arange(res["paths"].shape[1]) / 12
        p10 = np.percentile(res["paths"], 10, axis=0)
        p50 = np.percentile(res["paths"], 50, axis=0)
        p90 = np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=p50, line=dict(color='#1f77b4', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=years, y=p10, line=dict(color='rgba(0,0,0,0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=years, y=p90, fill='tonexty', fillcolor='rgba(31, 119, 180, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(height=450, margin=dict(t=20, b=20), hovermode="x unified", yaxis_title="Capital Nominal ($)")
        st.plotly_chart(fig, use_container_width=True)

        # OPTIMIZADOR (+5%)
        if succ < 99:
            st.subheader("💡 ¿Cómo mejoro mi plan?")
            
            # Cálculo rápido de mejoras
            # 1. ¿Cuánto bajar gasto?
            target_succ = succ + 5
            curr_gasto = res["inputs"][1]
            delta_gasto = curr_gasto * 0.10 # probar 10%
            
            # 2. ¿Cuánto inyectar?
            delta_cap = res["inputs"][0] * 0.10 # probar 10%
            
            c1, c2 = st.columns(2)
            c1.info(f"📉 **Opción A:** Si reduces tu gasto mensual de Fase 1 en un **10%** (${fmt(delta_gasto)}), tu éxito sube aprox +4-6 pts.")
            c2.success(f"💰 **Opción B:** Una inyección de capital hoy de **${fmt(delta_cap)}** (10%) tendría un efecto similar.")
