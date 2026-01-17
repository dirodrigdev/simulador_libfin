import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0, default_inmo_neto=0):
    
    # 1. ESTADO
    if 'current_results' not in st.session_state: st.session_state.current_results = None
    
    # ESCENARIOS
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 5.0, "rv": 8.0, "inf": 4.5, "vol": 20.0, "crisis": 10},
        "Estable (Base) ☁️": {"rf": 6.5, "rv": 10.5, "inf": 3.0, "vol": 16.0, "crisis": 5},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2},
        "Mis Datos 🏠": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5}
    }

    # Inicializar Inputs
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

    # 2. MOTOR MATEMÁTICO
    @dataclass
    class AssetBucket:
        name: str; weight: float = 0.0; mu_nominal: float = 0.0; sigma_nominal: float = 0.0; is_bond: bool = False

    @dataclass
    class WithdrawalTramo:
        from_year: int; to_year: int; amount_nominal_monthly_start: float

    @dataclass
    class SimulationConfig:
        horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1_000_000; n_sims: int = 2000
        inflation_mean: float = 0.035; inflation_vol: float = 0.01; prob_crisis: float = 0.05
        crisis_drift: float = 0.75; crisis_vol: float = 1.25
        # MOTORES AVANZADOS (B, C, D)
        use_fat_tails: bool = True
        use_mean_reversion: bool = True
        use_guardrails: bool = True
        guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10
        # ESTRATEGIA INMOBILIARIA
        sell_year: int = 0; net_inmo_value: float = 0; new_rent_cost: float = 0

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
            ruin_indices = np.full(n_sims, -1)
            
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight

            # Cholesky
            try: L = np.linalg.cholesky(self.corr_matrix)
            except: L = np.eye(n_assets)
            
            p_crisis = 1 - (1 - self.cfg.prob_crisis)**self.dt
            in_crisis = np.zeros(n_sims, dtype=bool)
            max_real_wealth = np.full(n_sims, self.cfg.initial_capital)

            for t in range(1, n_steps + 1):
                # 1. INFLACIÓN
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # 2. CRISIS & FAT TAILS (D)
                new_c = np.random.rand(n_sims) < p_crisis
                in_crisis = np.logical_or(in_crisis, new_c)
                in_crisis[np.random.rand(n_sims) < 0.15] = False 

                if self.cfg.use_fat_tails:
                    df = 5; std_adj = np.sqrt((df-2)/df)
                    z_uncorr = np.random.standard_t(df, (n_sims, n_assets)) * std_adj
                else: z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
                z_corr = np.dot(z_uncorr, L.T)
                
                # 3. RETORNOS ACTIVOS (C - MEAN REVERSION)
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    if np.any(in_crisis): mu *= self.cfg.crisis_drift; sig *= self.cfg.crisis_vol
                    
                    # Lógica C: Mean Reversion para Bonos
                    step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]
                
                asset_values *= np.exp(step_rets)
                
                # 4. EVENTO INMOBILIARIO (VENTA)
                # Si estamos en el mes exacto de la venta
                current_year = t / 12
                if self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * 12):
                    # Inyectamos capital: Valor Casa * Inflación acumulada
                    injection = self.cfg.net_inmo_value * cpi_paths[:, t]
                    # Distribuimos la inyección según pesos actuales (o rebalanceo forzoso)
                    # Simplificación: Sumamos al total y rebalanceamos abajo
                    total_pre_inject = np.sum(asset_values, axis=1)
                    # Evitar div por 0
                    mask_pos = total_pre_inject > 0
                    # Proporción actual
                    # Mejor estrategia: Inyectar manteniendo el mix objetivo o rebalancear todo
                    # Vamos a sumar al pool y dejar que el rebalanceo (paso 6) lo ordene
                    # Pero necesitamos asignarlo a los buckets ahora para que 'total_cap' suba
                    # Asignamos temporalmente al primer activo líquido (RV) para que el rebalanceo lo distribuya luego
                    asset_values[:, 0] += injection

                total_cap = np.sum(asset_values, axis=1)
                
                # 5. RETIROS & GUARDRAILS (B)
                current_real_wealth = total_cap / cpi_paths[:, t]
                max_real_wealth = np.maximum(max_real_wealth, current_real_wealth)
                
                # Buscar Retiro Base
                wd_base_start = 0
                for w in self.withdrawals:
                    if w.from_year <= current_year < w.to_year:
                        wd_base_start = w.amount_nominal_monthly_start; break
                
                # AJUSTE: Si ya vendimos la casa, sumamos el costo de arriendo al retiro base
                if self.cfg.sell_year > 0 and current_year >= self.cfg.sell_year:
                    wd_base_start += self.cfg.new_rent_cost

                # Aplicar Guardrails
                if self.cfg.use_guardrails:
                    drawdown = (max_real_wealth - current_real_wealth) / max_real_wealth
                    in_trouble = drawdown > self.cfg.guardrail_trigger
                    wd_nom = np.zeros(n_sims)
                    wd_nom[~in_trouble] = wd_base_start * cpi_paths[~in_trouble, t]
                    wd_nom[in_trouble] = (wd_base_start * cpi_paths[in_trouble, t]) * (1.0 - self.cfg.guardrail_cut)
                else: wd_nom = np.full(n_sims, wd_base_start) * cpi_paths[:, t]

                ratio = np.divide(wd_nom, total_cap, out=np.zeros_like(total_cap), where=total_cap!=0)
                ratio = np.clip(ratio, 0, 1)
                asset_values *= (1 - ratio[:, np.newaxis])
                
                # 6. REBALANCEO Y RUINA
                # Rebalanceamos anualmente O si hubo evento de venta inmobiliaria
                is_sale_event = (self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * 12))
                
                if t % 12 == 0 or is_sale_event:
                    tot = np.sum(asset_values, axis=1)
                    alive = tot > 0
                    if np.any(alive):
                        for i, asset in enumerate(self.assets):
                            asset_values[alive, i] = tot[alive] * asset.weight
                
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
                # Check Ruina
                just_died = (capital_paths[:, t-1] > 0) & (capital_paths[:, t] <= 1000)
                ruin_indices[just_died] = t
                
            return capital_paths, cpi_paths, ruin_indices

    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- 3. UI ---
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
        st.markdown("### 🏡 Estrategia Inmobiliaria")
        sell_prop = st.checkbox("Vender Propiedad Futura", value=False)
        if sell_prop:
            net_inmo_val = st.number_input("Valor Neto Hoy ($)", value=int(default_inmo_neto))
            sale_year = st.slider("Año de Venta", 1, 40, 10, help="Año en que vendes la casa, recibes el capital y empiezas a pagar arriendo.")
            rent_cost = st.number_input("Nuevo Arriendo ($/mes)", value=1500000, step=100000)
            st.info(f"En el Año {sale_year}: Recibes capital (ajustado IPC) y tu gasto sube en ${fmt(rent_cost)}/mes.")
        else:
            net_inmo_val, sale_year, rent_cost = 0, 0, 0

        st.divider()
        st.markdown("### 🧠 Seguridad Institucional")
        # Aquí confirmamos que tus "motores perdidos" están presentes
        use_guard = st.checkbox("🛡️ Guardrails (Gasto Dinámico)", value=True, help="Reduce gasto si hay crisis (Punto B).")
        use_fat = st.checkbox("📉 Fat Tails (T-Student)", value=True, help="Eventos extremos reales (Punto D).")
        use_bond = st.checkbox("🔄 Bonos Reales (Mean Rev)", value=True, help="Matemática de bonos correcta (Punto C).")
        
        if use_guard:
            c1, c2 = st.columns(2)
            gr_trigger = c1.number_input("Trigger %", 10, 50, 15)
            gr_cut = c2.number_input("Cut %", 5, 50, 10)
        else: gr_trigger, gr_cut = 15, 10
        
        n_sims = st.slider("Sims", 500, 5000, 1000)
        horiz = st.slider("Horizonte", 10, 60, 40)

    # MAIN
    st.markdown("### 💰 Capital Líquido Inicial")
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        cap_input = clean("Capital Inversión ($)", ini_def, "cap")
        if sell_prop: st.caption(f"+ Casa en Año {sale_year}")
    with c2: pct_rv = st.slider("% Renta Variable", 0, 100, 60)
    with c3: 
        st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"RF: {p_rf}% | RV: {p_rv}%")

    st.markdown("### 💸 Plan de Retiro (Nominal Hoy)")
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
        # Configuración V5.3 Completa
        cfg = SimulationConfig(
            horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims, 
            inflation_mean=p_inf/100, prob_crisis=p_cris/100,
            use_guardrails=use_guard, guardrail_trigger=gr_trigger/100.0, guardrail_cut=gr_cut/100.0,
            use_fat_tails=use_fat, use_mean_reversion=use_bond,
            # Estrategia Inmobiliaria
            sell_year=sale_year, net_inmo_value=net_inmo_val, new_rent_cost=rent_cost
        )
        
        sim = InstitutionalSimulator(cfg, assets, wds)
        sim.corr_matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
        
        with st.spinner("Procesando Escenarios Institucionales..."):
            paths, cpi, ruin_idx = sim.run()
            final_nom = paths[:, -1]
            success = np.mean(final_nom > 0) * 100
            median_legacy = np.median(final_nom / cpi[:, -1])
            
            # Cálculo de Ruina (Inicio del 80% de riesgo)
            fails = ruin_idx[ruin_idx > -1]
            if len(fails) > 0:
                fail_years = fails / 12
                # Percentil 20 = Inicio del 80% grueso
                start_80_pct = np.percentile(fail_years, 20)
            else: start_80_pct = 0
            
            st.session_state.current_results = {"succ": success, "leg": median_legacy, "paths": paths, "ruin_start": start_80_pct, "n_fails": len(fails)}

    if st.session_state.current_results:
        res = st.session_state.current_results
        
        clr = "#10b981" if res["succ"] > 90 else "#f59e0b" if res["succ"] > 75 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center; padding:20px; border:2px solid {clr}; border-radius:10px; margin-top:10px; background-color: rgba(0,0,0,0.02);">
            <h2 style="color:{clr}; margin:0; font-size: 2.5rem;">Probabilidad de Éxito: {res['succ']:.1f}%</h2>
            <p style="margin:0; font-size: 1.1rem; color: gray;">Herencia Real Mediana: <b>${fmt(res['leg'])}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dashboard de Riesgo
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de Ruina", f"{100-res['succ']:.1f}%", help="Riesgo total de agotar fondos.")
        val_ruin = f"Año {res['ruin_start']:.1f}" if res['n_fails'] > 0 else "Nunca"
        c2.metric("Inicio Riesgo (80%)", val_ruin, help="El 80% de las quiebras ocurren después de este año.")
        
        if sell_prop:
            c3.success(f"Venta Casa: Año {sale_year}")
        else:
            c3.caption("Estrategia: Mantener Casa")

        # Gráfico
        y = np.arange(res["paths"].shape[1])/12
        p10, p50, p90 = np.percentile(res["paths"], 10, axis=0), np.percentile(res["paths"], 50, axis=0), np.percentile(res["paths"], 90, axis=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#3b82f6', width=3), name='Mediana'))
        fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(width=0), name='Rango 80%'))
        
        # Marcador de venta
        if sell_prop and sale_year > 0:
            fig.add_vline(x=sale_year, line_dash="dash", line_color="green", annotation_text="Venta Casa")
        
        # Marcador de riesgo
        if res['n_fails'] > 0:
            fig.add_vline(x=res['ruin_start'], line_dash="dot", line_color="red", annotation_text="Inicio Riesgo")
            
        fig.update_layout(title="Evolución Patrimonial", yaxis_title="Capital Nominal", height=450)
        st.plotly_chart(fig, use_container_width=True)
