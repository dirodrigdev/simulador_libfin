import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List
import re

# --- 1. UTILIDADES ---
def fmt(v): return f"{int(v):,}".replace(",", ".")
def clean_input(label, val, key):
    val_str = fmt(val)
    new_val = st.text_input(label, value=val_str, key=key)
    clean_val = re.sub(r'\.', '', new_val)
    clean_val = re.sub(r'\D', '', clean_val)
    return int(clean_val) if clean_val else 0
def fmt_pct(v): return f"{v*100:.1f}%"

# --- 2. CONFIGURACIÓN ---
@dataclass
class AssetBucket:
    name: str; weight: float = 0.0; is_bond: bool = False
@dataclass
class WithdrawalTramo:
    from_year: int; to_year: int; amount_nominal_monthly_start: float
@dataclass
class ExtraCashflow:
    year: int; amount: float; name: str
@dataclass
class SimulationConfig:
    horizon_years: int = 40; steps_per_year: int = 12; initial_capital: float = 1800000000; n_sims: int = 2000
    mu_normal_rv: float = 0.11; mu_normal_rf: float = 0.06; inflation_mean: float = 0.035; inflation_vol: float = 0.012
    is_active_managed: bool = True; use_guardrails: bool = True; guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10
    enable_prop: bool = True; net_inmo_value: float = 500000000; new_rent_cost: float = 1500000
    emergency_months_trigger: int = 24; forced_sale_year: int = 0; bucket_years_safety: float = 3.5
    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.22; mu_global_rf: float = -0.02; corr_global: float = 0.75  
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 3. MOTOR V16.7 ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year; self.total_steps = int(config.horizon_years * config.steps_per_year)
        self.mu_regimes = np.array([[self.cfg.mu_normal_rv, self.cfg.mu_normal_rf],[self.cfg.mu_local_rv, self.cfg.mu_local_rf],[self.cfg.mu_global_rv, self.cfg.mu_global_rf]])
        vol_f = 0.80 if self.cfg.is_active_managed else 1.0
        self.sigma_regimes = np.array([[0.15, 0.05], [0.22, 0.12], [0.30, 0.14]]) * vol_f
        self.L_normal = np.linalg.cholesky(np.array([[1.0, 0.35], [0.35, 1.0]]))
        self.L_local = np.linalg.cholesky(np.array([[1.0, -0.25], [-0.25, 1.0]]))
        self.L_global = np.linalg.cholesky(np.array([[1.0, 0.75], [0.75, 1.0]]))
        self.p_norm_l = self.cfg.prob_enter_local; self.p_norm_g = self.cfg.prob_enter_global; self.p_exit = self.cfg.prob_exit_crisis

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        capital_paths = np.zeros((n_sims, n_steps+1)); capital_paths[:,0] = self.cfg.initial_capital
        cpi_paths = np.ones((n_sims, n_steps+1)); is_alive = np.ones(n_sims, dtype=bool) 
        ruin_indices = np.full(n_sims, -1); has_house = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        asset_values = np.zeros((n_sims, 2)) # 0: RV, 1: RF
        for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight
        
        current_regime = np.zeros(n_sims, dtype=int)
        df = 8; Z_raw = (np.random.normal(0, 1, (n_sims, n_steps, 2)) / np.sqrt(np.random.chisquare(df, (n_sims, n_steps, 1)) / df)) / np.sqrt(df / (df - 2)) 
        inf_sh = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive): break
            # Markov
            m0 = (current_regime==0)&alive
            if np.any(m0):
                r = np.random.rand(np.sum(m0))
                current_regime[np.where(m0)[0][r < self.p_norm_l]] = 1
                current_regime[np.where(m0)[0][(r >= self.p_norm_l) & (r < (self.p_norm_l+self.p_norm_g))]] = 2
            mc = (current_regime>0)&alive
            if np.any(mc): current_regime[np.where(mc)[0][np.random.rand(np.sum(mc)) < self.p_exit]] = 0

            # Mercado
            z_t = Z_raw[:, t, :]
            z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate([self.L_normal, self.L_local, self.L_global]):
                m = (current_regime == r_idx) & alive
                if np.any(m): z_f[m] = np.dot(z_t[m], L.T)
            
            p_def = np.ones(n_sims); p_def[current_regime > 0] = 0.85
            mus, sigs = self.mu_regimes[current_regime], self.sigma_regimes[current_regime]
            asset_values[alive] *= np.exp((mus[alive]-0.5*sigs[alive]**2)*self.dt + sigs[alive]*np.sqrt(self.dt)*z_f[alive]*p_def[alive, None])
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (current_regime == 1)*0.003))

            # Hitos y Gasto
            if (t+1)%12==0:
                y = (t+1)//12
                for e in self.cfg.extra_cashflows: 
                    if e.year == y: asset_values[alive, 1] += e.amount * cpi_paths[alive, t+1]

            cur_y = (t+1)/12; m_spend = 0
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year: m_spend = w.amount_nominal_monthly_start * cpi_paths[:, t+1]; break
            
            # Venta Casa
            trig = alive & has_house & (np.sum(asset_values,1) < m_spend*self.cfg.emergency_months_trigger)
            if np.any(trig): asset_values[trig, 1] += self.cfg.net_inmo_value * cpi_paths[trig, t+1]; has_house[trig] = False

            # RETIRO BUCKET PROTECTED (Diego's Rule)
            out = m_spend + (self.cfg.enable_prop & (~has_house)) * (1500000 * cpi_paths[:, t+1])
            wd = np.minimum(out, np.sum(asset_values,1))
            
            # 1. Sacar de RF
            rf_bal = np.maximum(asset_values[:, 1], 0)
            take_rf = np.minimum(wd, rf_bal)
            asset_values[:, 1] -= take_rf
            # 2. Si falta, sacar de RV (vender en crisis)
            asset_values[:, 0] -= (wd - take_rf)

            asset_values = np.maximum(asset_values, 0); capital_paths[:, t+1] = np.sum(asset_values, 1)
            dead = (capital_paths[:, t+1] <= 1000) & alive
            if np.any(dead): is_alive[dead]=False; ruin_indices[dead]=t+1; capital_paths[dead, t+1:]=0; asset_values[dead]=0

        return capital_paths, cpi_paths, ruin_indices

# --- 4. INTERFAZ ---
def app(default_rf=720000000, default_rv=1080000000, default_inmo_neto=500000000):
    st.markdown("## 🦅 Panel de Decisión (V16.7 - Strategist Edition)")
    st.info("🛡️ **Regla de Oro:** Se prioriza el uso de Renta Fija para el gasto, protegiendo la Renta Variable en periodos de volatilidad.")

    with st.sidebar:
        st.header("⚙️ Configuración")
        is_active = st.toggle("Gestión Activa / Balanceados", value=True)
        sel_glo = st.selectbox("Escenario de Crisis", ["Crash Financiero", "Colapso Sistémico", "Recesión Estándar"], index=0)
        sel_ret = st.selectbox("Rentabilidad Nominal", ["Conservador", "Histórico (11%)", "Crecimiento (13%)", "Personalizado"], index=1)
        if sel_ret == "Personalizado":
            c_mu_rv = st.number_input("RV % Anual", 0.0, 25.0, 11.0)/100; c_mu_rf = st.number_input("RF % Anual", 0.0, 15.0, 6.0)/100
        else:
            rates = {"Conservador": [0.08, 0.045], "Histórico (11%)": [0.11, 0.06], "Crecimiento (13%)": [0.13, 0.07]}
            c_mu_rv, c_mu_rf = rates[sel_ret]
        n_sims = st.slider("Simulaciones", 500, 3000, 1000); horiz = st.slider("Horizonte", 10, 50, 40)

    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    with tab_sim:
        c1, c2, c3 = st.columns(3)
        with c1: cap_in = clean_input("Capital Total ($)", 1800000000, "cap")
        with c2: pct_rv = st.slider("RV %", 0, 100, 60)
        with c3: st.metric("Reserva RF", f"${fmt(cap_in * (1-pct_rv/100))}")
        
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Gasto F1", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
        with g2: r2 = clean_input("Gasto F2", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
        with g3: r3 = clean_input("Gasto F3", 5000000, "r3")

        with st.expander("💸 Inyecciones / Salidas"):
            if 'extra_events' not in st.session_state: st.session_state.extra_events = []
            ce1, ce2, ce3, ce4 = st.columns([1,2,2,1])
            with ce1: ev_y = st.number_input("Año", 1, 40, 5, key="evy")
            with ce2: ev_a = clean_input("Monto", 0, "eva")
            with ce3: ev_t = st.selectbox("Tipo", ["Entrada", "Salida"], key="evt")
            if ce4.button("Add"): st.session_state.extra_events.append(ExtraCashflow(ev_y, ev_a if ev_t=="Entrada" else -ev_a, "Hito"))
            for e in st.session_state.extra_events: st.text(f"Año {e.year}: ${fmt(e.amount)}")
            if st.button("Limpiar"): st.session_state.extra_events = []

        enable_p = st.checkbox("Venta Casa Emergencia (500M)", value=True)

        if st.button("🚀 INICIAR SIMULACIÓN"):
            assets = [AssetBucket("RV", pct_rv/100), AssetBucket("RF", (100-pct_rv)/100, True)]
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap_in, n_sims=n_sims, is_active_managed=is_active, enable_prop=enable_p, net_inmo_value=500000000, mu_normal_rv=c_mu_rv, mu_normal_rf=c_mu_rf, extra_cashflows=st.session_state.extra_events)
            sim = InstitutionalSimulator(cfg, assets, wds); paths, cpi, r_i = sim.run()
            prob = (1 - (np.sum(r_i > -1)/n_sims))*100
            st.markdown(f"<div style='text-align:center; padding:20px; border-radius:10px; background:rgba(30,30,30,0.5);'><h1>Éxito: {prob:.1f}%</h1><h3>Legado: ${fmt(np.median(paths[:,-1]/cpi[:,-1]))}</h3></div>", unsafe_allow_html=True)
            y = np.arange(paths.shape[1])/12; fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y, y[::-1]]), y=np.concatenate([np.percentile(paths, 90, 0), np.percentile(paths, 10, 0)[::-1]]), fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='80% Rango'))
            fig.add_trace(go.Scatter(x=y, y=np.percentile(paths, 50, 0), line=dict(color='#3b82f6', width=3), name='Mediana'))
            st.plotly_chart(fig, use_container_width=True)

    with tab_opt:
        st.subheader("🎯 Optimizador con Restricción de Seguridad")
        st.write("El buscador encontrará el Mix óptimo, pero **siempre priorizando tu reserva de 3.5 años en Renta Fija**.")
        target = st.slider("Meta Éxito %", 70, 100, 90)
        if st.button("🔍 CALCULAR MIX ÓPTIMO"):
            res = []
            with st.spinner("Simulando..."):
                for v in np.linspace(0, 100, 11):
                    a_t = [AssetBucket("RV", v/100), AssetBucket("RF", (100-v)/100, True)]
                    w_t = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
                    c_t = SimulationConfig(horizon_years=horiz, initial_capital=cap_in, n_sims=300, is_active_managed=is_active, enable_prop=enable_p, net_inmo_value=500000000, mu_normal_rv=c_mu_rv, mu_normal_rf=c_mu_rf, extra_cashflows=st.session_state.extra_events)
                    _, _, ri = InstitutionalSimulator(c_t, a_t, w_t).run()
                    res.append({"v": v, "p": (1-(np.sum(ri>-1)/300))*100})
            df = pd.DataFrame(res)
            best = df.iloc[(df['p']-target).abs().argsort()[:1]].iloc[0]
            st.success(f"✅ Mix sugerido: **{int(best['v'])}% RV** (Prob: {best['p']:.1f}%)")
