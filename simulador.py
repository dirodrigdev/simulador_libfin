import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- CLASES DEL MOTOR ---
@dataclass
class AssetBucket:
    name: str; weight: float = 0.0; mu_nominal: float = 0.0; sigma_nominal: float = 0.0; is_bond: bool = False

@dataclass
class WithdrawalTramo:
    from_year: int; to_year: int; amount_nominal_monthly_start: float

@dataclass
class SimulationConfig:
    horizon_years: int = 40
    steps_per_year: int = 12
    initial_capital: float = 1_000_000
    n_sims: int = 2000
    inflation_mean: float = 0.035
    inflation_vol: float = 0.01
    prob_crisis: float = 0.05
    crisis_drift: float = 0.85
    crisis_vol: float = 1.25
    use_fat_tails: bool = True
    use_guardrails: bool = True
    guardrail_trigger: float = 0.15
    guardrail_cut: float = 0.10
    use_smart_buckets: bool = True 
    sell_year: int = 0
    net_inmo_value: float = 0
    new_rent_cost: float = 0
    inmo_strategy: str = "portfolio"
    annuity_rate: float = 0.05 

class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config
        self.assets = assets
        self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year
        self.total_steps = int(config.horizon_years * config.steps_per_year)
        self.corr_matrix = np.eye(len(assets))

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        n_assets = len(self.assets)
        
        capital_paths = np.zeros((n_sims, n_steps + 1))
        capital_paths[:, 0] = self.cfg.initial_capital
        cpi_paths = np.ones((n_sims, n_steps + 1))
        ruin_indices = np.full(n_sims, -1)
        debug_net_flow = np.zeros(n_steps + 1)
        
        asset_values = np.zeros((n_sims, n_assets))
        for i, a in enumerate(self.assets):
            asset_values[:, i] = self.cfg.initial_capital * a.weight

        try: L = np.linalg.cholesky(self.corr_matrix)
        except: L = np.eye(n_assets)

        try: rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
        except: rv_idx = 0
        try: rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
        except: rf_idx = 1

        p_crisis_manual = 1 - (1 - self.cfg.prob_crisis)**self.dt
        use_crisis_vol = not self.cfg.use_fat_tails
        in_crisis = np.zeros(n_sims, dtype=bool)
        shadow_rv_index = np.ones(n_sims); shadow_rv_peak = np.ones(n_sims)
        
        annuity_monthly = 0
        if self.cfg.sell_year > 0 and self.cfg.inmo_strategy == 'annuity':
            months = (self.cfg.horizon_years - self.cfg.sell_year) * 12
            if months > 0:
                r_m = self.cfg.annuity_rate / 12
                if r_m > 0:
                    annuity_monthly = self.cfg.net_inmo_value * (r_m * (1+r_m)**months) / ((1+r_m)**months - 1)
                else: annuity_monthly = self.cfg.net_inmo_value / months

        # CRÍTICO: Inicialización fuera del loop
        max_real_wealth = np.full(n_sims, self.cfg.initial_capital)

        for t in range(1, n_steps + 1):
            inf_shock = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), n_sims)
            cpi_paths[:, t] = cpi_paths[:, t-1] * (1 + np.maximum(inf_shock, -0.99))
            
            if p_crisis_manual > 0:
                new_c = np.random.rand(n_sims) < p_crisis_manual
                in_crisis = np.logical_or(in_crisis, new_c)
                in_crisis[np.random.rand(n_sims) < 0.15] = False

            if self.cfg.use_fat_tails:
                z_uncorr = np.random.standard_t(5, (n_sims, n_assets)) * np.sqrt(0.6)
            else:
                z_uncorr = np.random.normal(0, 1, (n_sims, n_assets))
            z_corr = np.dot(z_uncorr, L.T)
            
            step_rets = np.zeros((n_sims, n_assets))
            for i, a in enumerate(self.assets):
                mu, sig = a.mu_nominal, a.sigma_nominal
                if p_crisis_manual > 0 and np.any(in_crisis): 
                    mu *= self.cfg.crisis_drift 
                    if use_crisis_vol: sig *= self.cfg.crisis_vol
                step_rets[:, i] = (mu - 0.5 * sig**2) * self.dt + sig * np.sqrt(self.dt) * z_corr[:, i]

            # SAFETY CLIP: Evita explosiones exponenciales
            step_rets = np.clip(step_rets, -0.5, 0.5)
            asset_values *= np.exp(step_rets)
            
            shadow_rv_index *= np.exp(step_rets[:, rv_idx])
            shadow_rv_peak = np.maximum(shadow_rv_peak, shadow_rv_index)
            mkt_dd = (shadow_rv_peak - shadow_rv_index) / np.maximum(shadow_rv_peak, 1e-9)
            sim_in_trouble = np.logical_or(in_crisis, mkt_dd > 0.15)

            spy = self.cfg.steps_per_year
            current_year = t / spy

            if self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * spy):
                if self.cfg.inmo_strategy == 'portfolio':
                    asset_values[:, 0] += self.cfg.net_inmo_value * cpi_paths[:, t]

            total_cap = np.sum(asset_values, axis=1)
            current_real_wealth = total_cap / cpi_paths[:, t]
            max_real_wealth = np.maximum(max_real_wealth, current_real_wealth)

            living_base = 0
            for w in self.withdrawals:
                if w.from_year <= current_year < w.to_year:
                    living_base = w.amount_nominal_monthly_start; break
            
            living_nom = np.zeros(n_sims)
            if self.cfg.use_guardrails:
                dd_port = (max_real_wealth - current_real_wealth) / np.maximum(max_real_wealth, 1.0)
                in_tr_g = dd_port > self.cfg.guardrail_trigger
                living_nom[~in_tr_g] = living_base * cpi_paths[~in_tr_g, t]
                living_nom[in_tr_g] = (living_base * cpi_paths[in_tr_g, t]) * (1.0 - self.cfg.guardrail_cut)
            else: living_nom = np.full(n_sims, living_base) * cpi_paths[:, t]

            rent_nom = np.zeros(n_sims); ann_nom = np.zeros(n_sims)
            if self.cfg.sell_year > 0 and current_year >= self.cfg.sell_year:
                rent_nom = np.full(n_sims, self.cfg.new_rent_cost) * cpi_paths[:, t]
                if self.cfg.inmo_strategy == 'annuity':
                    ann_nom = np.full(n_sims, annuity_monthly) * cpi_paths[:, t]

            net_cashflow = ann_nom - (living_nom + rent_nom)
            debug_net_flow[t] = np.median(net_cashflow)
            port_adj = -net_cashflow 
            
            mask_surp = port_adj <= 0
            if np.any(mask_surp):
                surp = -port_adj[mask_surp]
                for i, asset in enumerate(self.assets):
                    asset_values[mask_surp, i] += surp * asset.weight

            mask_def = port_adj > 0
            if np.any(mask_def):
                wd_req = port_adj[mask_def]
                # SAFETY CLIP: No retirar más de lo que existe
                wd_req = np.minimum(wd_req, total_cap[mask_def])
                
                if self.cfg.use_smart_buckets:
                    rf_bal = np.maximum(asset_values[mask_def, rf_idx], 0)
                    take_rf = np.minimum(wd_req, rf_bal)
                    take_rv = wd_req - take_rf 
                    asset_values[mask_def, rf_idx] -= take_rf
                    asset_values[mask_def, rv_idx] -= take_rv
                else:
                    tot_d = total_cap[mask_def]
                    ratio = wd_req / np.where(tot_d!=0, tot_d, 1.0)
                    asset_values[mask_def] *= (1 - np.minimum(ratio, 1.0)[:, np.newaxis])

            if (t % spy == 0) or (self.cfg.sell_year > 0 and t == int(self.cfg.sell_year * spy)):
                do_reb = ~sim_in_trouble if self.cfg.use_smart_buckets else np.full(n_sims, True)
                if np.any(do_reb):
                    tot = np.sum(asset_values[do_reb], axis=1)
                    alive = tot > 0
                    if np.any(alive):
                        for i, a in enumerate(self.assets):
                            asset_values[do_reb, i] = tot * a.weight
            
            asset_values = np.maximum(asset_values, 0)
            capital_paths[:, t] = np.sum(asset_values, axis=1)
            ruin_indices[(capital_paths[:, t-1] > 0) & (capital_paths[:, t] <= 1000)] = t
            
        return capital_paths, cpi_paths, ruin_indices, annuity_monthly, debug_net_flow

def clean(lbl, d, k): 
    v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
    return int(re.sub(r'\D', '', v)) if v else 0
def fmt(v): return f"{int(v):,}".replace(",", ".")

# --- INTERFAZ DEL SIMULADOR ---
def app(default_rf=0, default_mx=0, default_rv=0, default_usd_nominal=0, default_tc=930, default_ret_rf=6.0, default_ret_rv=10.0, default_inmo_neto=0):
    
    SCENARIOS = {
        "Pesimista 🌧️": {"rf": 5.0, "rv": 8.0, "inf": 4.5, "vol": 20.0, "crisis": 10},
        "Estable (Base) ☁️": {"rf": 6.5, "rv": 10.5, "inf": 3.0, "vol": 16.0, "crisis": 5},
        "Optimista ☀️": {"rf": 7.5, "rv": 13.0, "inf": 2.5, "vol": 14.0, "crisis": 2},
        "Mis Datos 🏠": {"rf": default_ret_rf, "rv": default_ret_rv, "inf": 3.5, "vol": 18.0, "crisis": 5}
    }

    with st.sidebar:
        st.header("1. Escenario")
        sel = st.selectbox("Preset:", list(SCENARIOS.keys()), index=1)
        vals = SCENARIOS[sel]
        with st.expander("Variables", expanded=True):
            p_inf = st.number_input("Inflación (%)", value=vals["inf"], step=0.1)
            p_rf = st.number_input("Retorno RF (%)", value=vals["rf"], step=0.1)
            p_rv = st.number_input("Retorno RV (%)", value=vals["rv"], step=0.1)
            p_vol = st.slider("Volatilidad RV", 10.0, 30.0, vals["vol"])
            p_cris = st.slider("Prob. Crisis (%)", 0, 20, vals["crisis"])

        st.divider()
        st.markdown("### 🏡 Estrategia Inmobiliaria")
        sell_prop = st.checkbox("Vender Propiedad Futura", value=False)
        if sell_prop:
            val_inmo = st.number_input("Valor Neto Hoy ($)", value=int(default_inmo_neto))
            sale_year = st.slider("Año de Venta", 1, 40, 10)
            rent_cost = st.number_input("Nuevo Arriendo ($/mes)", value=1500000, step=100000)
            strat = st.radio("Destino:", ["Invertir", "Anualidad"], index=1)
            inmo_strat = 'portfolio' if strat == "Invertir" else 'annuity'
            annuity_r = 5.0 if inmo_strat == 'annuity' else 0.0
        else:
            val_inmo, sale_year, rent_cost, inmo_strat, annuity_r = 0, 0, 0, 'portfolio', 0

        st.divider()
        st.markdown("### 🧠 Seguridad")
        use_smart = st.checkbox("🥛 Smart Buckets", True)
        use_guard = st.checkbox("🛡️ Guardrails", True)
        use_fat = st.checkbox("📉 Fat Tails", True)
        
        n_sims = st.slider("Sims", 500, 5000, 1000)
        horiz = st.slider("Horizonte", 10, 60, 40)

    st.markdown("### 💰 Capital Inversión")
    ini_def = default_rf + default_mx + default_rv + (default_usd_nominal * default_tc)
    if ini_def == 0: ini_def = 1000000
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        cap_input = clean("Capital Líquido ($)", ini_def, "cap")
        if sell_prop: st.success(f"Año {sale_year}: +${fmt(val_inmo)}")
    
    has_mx = default_mx > 0
    with c2: 
        if has_mx:
            st.caption("Mix Detectado:")
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
    
    if sell_prop: st.info(f"ℹ️ El arriendo (${fmt(rent_cost)}) se sumará automáticamente desde el año {sale_year}.")

    if st.button("🚀 EJECUTAR ANÁLISIS PRO", type="primary"):
        assets = [
            AssetBucket("RV", pct_rv/100, p_rv/100, p_vol/100, False),
            AssetBucket("RF", (100-pct_rv)/100, p_rf/100, 0.05, True)
        ]
        wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
        
        cfg = SimulationConfig(
            horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims, 
            inflation_mean=p_inf/100, prob_crisis=vals["crisis"]/100,
            use_guardrails=use_guard, use_fat_tails=use_fat, use_smart_buckets=use_smart,
            sell_year=sale_year, net_inmo_value=val_inmo, new_rent_cost=rent_cost,
            inmo_strategy=inmo_strat, annuity_rate=annuity_r/100.0
        )
        
        sim = InstitutionalSimulator(cfg, assets, wds)
        sim.corr_matrix = np.array([[1.0, 0.25], [0.25, 1.0]])
        
        with st.spinner("Simulando..."):
            paths, cpi, ruin_idx, ann_val, deb_net = sim.run()
            final = paths[:, -1]
            success = np.mean(final > 0) * 100
            legacy = np.median(final / cpi[:, -1])
            
            fails = ruin_idx[ruin_idx > -1]
            risk_start = np.percentile(fails/12, 20) if len(fails) > 0 else 0
            
            clr = "#10b981" if success > 90 else "#ef4444"
            st.markdown(f"""<div style="text-align:center; padding:20px; border:2px solid {clr}; border-radius:10px;">
                <h2 style="color:{clr}; margin:0;">Probabilidad de Éxito: {success:.1f}%</h2>
                <p style="margin:0;">Herencia Real Mediana: <b>${fmt(legacy)}</b></p></div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilidad Ruina", f"{100-success:.1f}%")
            c2.metric("Inicio Riesgo (80%)", f"Año {risk_start:.1f}" if len(fails)>0 else "Nunca")
            if sell_prop and inmo_strat == 'annuity':
                delta = ann_val - rent_cost
                c3.metric("Flujo Inmobiliario", f"${fmt(delta)}", delta="Superávit" if delta>0 else "Déficit")

            y = np.arange(paths.shape[1])/12
            # SAFETY CLIP VISUAL:
            max_y = np.percentile(paths[:,-1], 95) * 1.5
            paths_safe = np.clip(paths, 0, max_y)
            
            p10, p50, p90 = np.percentile(paths_safe, [10, 50, 90], axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y, y=p50, line=dict(color='#3b82f6', width=3), name='Mediana'))
            fig.add_trace(go.Scatter(x=y, y=p10, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=y, y=p90, fill='tonexty', fillcolor='rgba(59, 130, 246, 0.1)', line=dict(width=0), name='Rango 80%'))
            if sell_prop: fig.add_vline(x=sale_year, line_dash="dash", line_color="green", annotation_text="Venta")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("🔎 Flujos de Caja"):
                fig_f = go.Figure()
                fig_f.add_trace(go.Scatter(x=y, y=deb_net, fill='tozeroy', name='Flujo Neto'))
                fig_f.add_hline(y=0, line_dash="dot", line_color="white")
                if sell_prop: fig_f.add_vline(x=sale_year, line_dash="dash", line_color="green")
                st.plotly_chart(fig_f, use_container_width=True)
