# NOMBRE DEL ARCHIVO: simulador.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- FUNCIÓN PRINCIPAL ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0, default_inmo_neto=0):
    
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
        crisis_drift: float = 0.85; crisis_vol: float = 1.25
        # MOTORES
        use_fat_tails: bool = True; use_mean_reversion: bool = True; use_guardrails: bool = True
        guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10
        use_smart_buckets: bool = True 
        sell_year: int = 0; net_inmo_value: float = 0; new_rent_cost: float = 0
        inmo_strategy: str = "portfolio"; annuity_rate: float = 0.05 

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
            debug_net_flow = np.zeros(n_steps + 1)
            
            asset_values = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight

            try: L = np.linalg.cholesky(self.corr_matrix)
            except: L = np.eye(n_assets)

            # --- FIX 1: ÍNDICES ROBUSTOS ---
            # Identificamos dinámicamente pero con validación
            try: rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
            except StopIteration: rv_idx = 0
            try: rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
            except StopIteration: rf_idx = 1
            
            # --- FIX 2: EXCLUSIÓN MUTUA FAT TAILS VS CRISIS ---
            # Si Fat Tails está ON, el slider de crisis solo afecta DRIFT (media), no Volatilidad
            # para no duplicar el castigo de varianza.
            p_crisis_manual = 1 - (1 - self.cfg.prob_crisis)**self.dt
            use_crisis_vol = not self.cfg.use_fat_tails # Solo inflar vol si no usamos colas gordas
            
            in_crisis = np.zeros(n_sims, dtype=bool)
            
            # SHADOW BENCHMARK
            shadow_rv_index = np.ones(n_sims) 
            shadow_rv_peak = np.ones(n_sims)  
            
            # ANUALIDAD
            annuity_monthly_payout_real = 0
            if self.cfg.sell_year > 0 and self.cfg.inmo_strategy == 'annuity':
                years_remaining = self.cfg.horizon_years - self.cfg.sell_year
                if years_remaining > 0:
                    months = years_remaining * 12
                    r_monthly = self.cfg.annuity_rate / 12
                    if r_monthly > 0:
                        factor = (1 + r_monthly)**months
                        annuity_monthly_payout_real = self.cfg.net_inmo_value * (r_monthly * factor) / (factor - 1)
                    else: annuity_monthly_payout_real = self.cfg.net_inmo_value / months

            # Estado para Mean Reversion (Ornstein-Uhlenbeck)
            # Guardamos el retorno acumulado "tendencial" de RF para hacerlo revertir
            rf_drift_state = np.zeros(n_sims) 

            max_real_wealth = np.full(n_sims, self.cfg.initial_capital)

            for t in range(1, n_steps + 1):
                # 1. Inflación (FIX: CLAMP para evitar inflación negativa absurda)
                inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
                inf_shock = np.maximum(inf_shock, -0.99) 
                cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + inf_shock)
                
                # 2. Crisis Script
                if p_crisis_manual > 0:
                    new_c = np.random.rand(n_sims) < p_crisis_manual
                    in_crisis = np.logical_or(in_crisis, new_c)
                    in_crisis[np.random.rand(n_sims) < 0.15] = False # Salida aleatoria de crisis

                # 3. Retornos
                if self.cfg.use_fat_tails:
                    df = 5; std_adj = np.sqrt((df-2)/df)
                    z_uncorr = np.random.standard_t(df, (n_sims, n_assets)) * std_adj
                else: z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
                z_corr = np.dot(z_uncorr, L.T)
                
                step_rets = np.zeros((n_sims, n_assets))
                for i, asset in enumerate(self.assets):
                    mu, sig = asset.mu_nominal, asset.sigma_nominal
                    
                    # Aplicar Crisis (Drift siempre, Vol solo si Fat Tails OFF)
                    if p_crisis_manual > 0 and np.any(in_crisis): 
                        mu *= self.cfg.crisis_drift 
                        if use_crisis_vol: sig *= self.cfg.crisis_vol
                    
                    # --- FIX 3: MEAN REVERSION REAL (NO PLACEBO) ---
                    if self.cfg.use_mean_reversion and asset.is_bond:
                        # Velocidad de reversión
                        kappa = 1.0 
                        # Ajustamos el drift basado en la desviación acumulada (simplificado)
                        # Si rf_drift_state es alto, empujamos abajo, y viceversa.
                        mr_adj = -kappa * rf_drift_state * self.dt
                        step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i] + mr_adj
                        # Actualizamos estado (simple random walk tracking)
                        rf_drift_state += z_corr[:, i] * np.sqrt(self.dt) 
                    else:
                        step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]

                asset_values *= np.exp(step_rets)
                
                # SHADOW BENCHMARK
                shadow_rv_index *= np.exp(step_rets[:, rv_idx])
                shadow_rv_peak = np.maximum(shadow_rv_peak, shadow_rv_index)
                market_drawdown = (shadow_rv_peak - shadow_rv_index) / shadow_rv_peak
                # FIX: Umbral de crisis más robusto (15%) para evitar ruido
                is_market_crash = market_drawdown > 0.15 
                sim_in_trouble = np.logical_or(in_crisis, is_market_crash)

                # --- FIX 4: USO DE steps_per_year EN LUGAR DE HARDCODE 12 ---
                spy = self.cfg.steps_per_year
                current_year = t / spy

                # 4. EVENTO VENTA
                if self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * spy):
                    if self.cfg.inmo_strategy == 'portfolio':
                        injection = self.cfg.net_inmo_value * cpi_paths[:, t]
                        asset_values[:, 0] += injection 

                total_cap = np.sum(asset_values, axis=1)
                current_real_wealth = total_cap / cpi_paths[:, t]
                max_real_wealth = np.maximum(max_real_wealth, current_real_wealth)

                # 5. GESTIÓN FLUJO
                living_base = 0
                for w in self.withdrawals:
                    if w.from_year <= current_year < w.to_year:
                        living_base = w.amount_nominal_monthly_start; break
                
                living_nom = np.zeros(n_sims)
                if self.cfg.use_guardrails:
                    drawdown_port = (max_real_wealth - current_real_wealth) / max_real_wealth
                    in_trouble_guard = drawdown_port > self.cfg.guardrail_trigger
                    living_nom[~in_trouble_guard] = living_base * cpi_paths[~in_trouble_guard, t]
                    living_nom[in_trouble_guard] = (living_base * cpi_paths[in_trouble_guard, t]) * (1.0 - self.cfg.guardrail_cut)
                else: living_nom = np.full(n_sims, living_base) * cpi_paths[:, t]

                rent_nom = np.zeros(n_sims)
                annuity_nom = np.zeros(n_sims)
                if self.cfg.sell_year > 0 and current_year >= self.cfg.sell_year:
                    rent_nom = np.full(n_sims, self.cfg.new_rent_cost) * cpi_paths[:, t]
                    if self.cfg.inmo_strategy == 'annuity':
                        annuity_nom = np.full(n_sims, annuity_monthly_payout_real) * cpi_paths[:, t]

                net_cashflow = annuity_nom - (living_nom + rent_nom)
                debug_net_flow[t] = np.median(net_cashflow)
                portfolio_adjustment = -net_cashflow 
                
                # --- FIX 5: REINVERSIÓN PROPORCIONAL (NO 100% RV) ---
                mask_surplus = portfolio_adjustment <= 0
                if np.any(mask_surplus):
                    # Reinvertimos según los pesos TARGET definidos en assets
                    # Esto mantiene el balance y reduce riesgo
                    surplus = -portfolio_adjustment[mask_surplus]
                    for i, asset in enumerate(self.assets):
                         asset_values[mask_surplus, i] += surplus * asset.weight

                # Caso 2: Déficit (Retiro)
                mask_deficit = portfolio_adjustment > 0
                if np.any(mask_deficit):
                    wd_req = portfolio_adjustment[mask_deficit]
                    
                    if self.cfg.use_smart_buckets:
                        # Intentar sacar de RF (protegido con max 0 para evitar underflow)
                        rf_bal = np.maximum(asset_values[mask_deficit, rf_idx], 0)
                        
                        take_rf = np.minimum(wd_req, rf_bal)
                        take_rv = wd_req - take_rf # El resto se saca de RV (o de los otros buckets si hubiera)
                        
                        asset_values[mask_deficit, rf_idx] -= take_rf
                        asset_values[mask_deficit, rv_idx] -= take_rv # Simplificación: todo el resto a RV principal
                    else:
                        tot_d = total_cap[mask_deficit]
                        ratio_d = np.divide(wd_req, tot_d, out=np.zeros_like(tot_d), where=tot_d!=0)
                        asset_values[mask_deficit] *= (1 - ratio_d[:, np.newaxis])

                # 6. REBALANCEO
                is_sale_event = (self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * spy))
                
                if (t % spy == 0) or is_sale_event:
                    if self.cfg.use_smart_buckets:
                        rebalance_mask = ~sim_in_trouble 
                        if np.any(rebalance_mask):
                            vals_ok = asset_values[rebalance_mask]
                            tot_ok = np.sum(vals_ok, axis=1)
                            alive_ok = tot_ok > 0
                            if np.any(alive_ok):
                                # Reestablecer a pesos originales
                                for i, asset in enumerate(self.assets):
                                    asset_values[rebalance_mask, i] = tot_ok * asset.weight
                    else:
                        tot = np.sum(asset_values, axis=1)
                        alive = tot > 0
                        if np.any(alive):
                            for i, asset in enumerate(self.assets):
                                asset_values[alive, i] = tot[alive] * asset.weight
                
                # CLAMP FINAL y Ruina
                asset_values = np.maximum(asset_values, 0)
                capital_paths[:, t] = np.sum(asset_values, axis=1)
                
                # Ruina lógica: Si capital < 0.1% del inicial (técnicamente 0)
                just_died = (capital_paths[:, t-1] > 0) & (capital_paths[:, t] <= 1000)
                ruin_indices[just_died] = t
                
            return capital_paths, cpi_paths, ruin_indices, annuity_monthly_payout_real, debug_net_flow

    def clean(lbl, d, k): 
        v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
        return int(re.sub(r'\D', '', v)) if v else 0
    def fmt(v): return f"{int(v):,}".replace(",", ".")

    # --- UI ---
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
            sale_year = st.slider("Año de Venta", 1, 40, 10)
            rent_cost = st.number_input("Nuevo Arriendo ($/mes)", value=1500000, step=100000)
            st.markdown("#### ¿Qué hacer con el dinero?")
            strat_mode = st.radio("Destino Capital:", ["Invertir en Portafolio", "Consumir (Anualidad)"], index=1)
            inmo_strat_code = 'portfolio' if "Invertir" in strat_mode else 'annuity'
            if inmo_strat_code == 'annuity':
                annuity_rate_ui = st.number_input("Tasa Rentabilidad Casa (%)", value=5.0, step=0.1)
            else: annuity_rate_ui = 0.0
        else:
            net_inmo_val, sale_year, rent_cost, inmo_strat_code, annuity_rate_ui = 0, 0, 0, 'portfolio', 0

        st.divider()
        st.markdown("### 🧠 Seguridad")
        use_smart = st.checkbox("🥛 Smart Buckets", value=True, help="Prioriza sacar de Renta Fija. Si se agota, saca de Variable.")
        use_guard = st.checkbox("🛡️ Guardrails", value=True)
        use_fat = st.checkbox("📉 Fat Tails", value=True)
        use_bond = st.checkbox("🔄 Bonos Reales", value=True, help="Activa reversión a la media en Renta Fija.")
        
        if use_guard:
            c1, c2 = st.columns(2)
            gr_trigger = c1.number_input("Trigger %", 10, 50, 15)
            gr_cut = c2.number_input("Cut %", 5, 50, 10)
        else: gr_trigger, gr_cut = 15, 10
        n_sims = st.slider("Sims", 500, 5000, 1000)
        horiz = st.slider("Horizonte", 10, 60, 40)

    # MAIN
    st.markdown("### 💰 Capital Inversión")
    # Calcular mix inicial
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1800000000
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        cap_input = clean("Capital Líquido ($)", ini_def, "cap")
        if sell_prop and inmo_strat_code == 'portfolio': st.success(f"Año {sale_year}: +${fmt(net_inmo_val)}")
    
    # MIX SLIDERS
    # Si tenemos dato real de MX, lo mostramos, si no, simplificamos a RF/RV
    has_mx = default_mx > 0
    
    with c2: 
        if has_mx:
            st.caption("Mix Detectado (Aprox):")
            # Un slider simple de 3 vias es dificil, simplificamos a RV vs (RF+MX)
            pct_rv = st.slider("% Renta Variable", 0, 100, int((default_rv/ini_def)*100) if ini_def>0 else 60)
        else:
            pct_rv = st.slider("% Renta Variable", 0, 100, 60)

    with c3: 
        st.metric("Mix", f"{100-pct_rv}% RF / {pct_rv}% RV")
        st.caption(f"Nominales: RF {p_rf}% | RV {p_rv}%")

    st.markdown("### 💸 Gastos de Vida (Sin Arriendo)")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1 ($)", 6000000, "r1"); d1 = st.number_input("Años", 7)
    with g2: r2 = clean("Fase 2 ($)", 5500000, "r2"); d2 = st.number_input("Años", 13)
    with g3: r3 = clean("Fase 3 ($)", 5000000, "r3"); st.caption("Resto vida")
    
    if sell_prop: 
        st.info(f"ℹ️ El arriendo (${fmt(rent_cost)}) se sumará SOLO desde el año {sale_year}.")

    if st.button("🚀 EJECUTAR ANÁLISIS PRO", type="primary"):
        # Construcción de Assets (Ahora soportamos 3 assets si quisiéramos, pero mantenemos 2 buckets lógicos)
        # Bucket 0: RV
        # Bucket 1: RF (
