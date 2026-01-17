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

# --- 2. CLASES ---
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
    mu_normal_rv: float = 0.10; mu_normal_rf: float = 0.06; inflation_mean: float = 0.035; inflation_vol: float = 0.012
    is_active_managed: bool = True 
    use_guardrails: bool = True; guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10; use_smart_buckets: bool = True
    enable_prop: bool = True; net_inmo_value: float = 500000000; new_rent_cost: float = 1500000
    emergency_months_trigger: int = 24; forced_sale_year: int = 0 
    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.35; mu_global_rf: float = -0.06; corr_global: float = 0.90   
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 3. MOTOR V16 ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year; self.total_steps = int(config.horizon_years * config.steps_per_year)

        mu_drag = 0.0075 if self.cfg.is_active_managed else 0.0
        self.mu_regimes = np.array([
            [self.cfg.mu_normal_rv - mu_drag, self.cfg.mu_normal_rf - (mu_drag/2)],
            [self.cfg.mu_local_rv, self.cfg.mu_local_rf],
            [self.cfg.mu_global_rv, self.cfg.mu_global_rf]
        ])
        
        vol_factor = 0.80 if self.cfg.is_active_managed else 1.0
        base_sigma = np.array([[0.15, 0.05], [0.22, 0.12], [0.30, 0.14]])
        self.sigma_regimes = base_sigma * vol_factor
        
        cn = np.array([[1.0, 0.35], [0.35, 1.0]]); self.L_normal = np.linalg.cholesky(cn)
        cl = np.clip(self.cfg.corr_local, -0.99, 0.99); self.L_local = np.linalg.cholesky(np.array([[1.0, cl], [cl, 1.0]]))
        cg = np.clip(self.cfg.corr_global, -0.99, 0.99); self.L_global = np.linalg.cholesky(np.array([[1.0, cg], [cg, 1.0]]))
        self.p_norm_to_local = self.cfg.prob_enter_local; self.p_norm_to_global = self.cfg.prob_enter_global; self.p_exit = self.cfg.prob_exit_crisis

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        n_assets = len(self.assets)
        capital_paths = np.zeros((n_sims, n_steps + 1)); capital_paths[:, 0] = self.cfg.initial_capital
        cpi_paths = np.ones((n_sims, n_steps + 1)); is_alive = np.ones(n_sims, dtype=bool) 
        ruin_indices = np.full(n_sims, -1); has_house = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        asset_values = np.zeros((n_sims, n_assets))
        for i, a in enumerate(self.assets): asset_values[:, i] = self.cfg.initial_capital * a.weight
        
        try: rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
        except: rv_idx = 0
        try: rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
        except: rf_idx = 1
        
        current_regime = np.zeros(n_sims, dtype=int)
        df = 8; G = np.random.normal(0, 1, (n_sims, n_steps, n_assets))
        W = np.random.chisquare(df, (n_sims, n_steps, 1)) / df
        Z_raw = (G / np.sqrt(W)) / np.sqrt(df / (df - 2)) 
        inf_shocks = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))
        z_final = np.zeros((n_sims, n_assets))

        for t in range(n_steps):
            alive = is_alive
            if np.any(alive):
                m0_alive = (current_regime == 0) & alive
                if np.any(m0_alive):
                    r_ = np.random.rand(np.sum(m0_alive))
                    new_l = r_ < self.p_norm_to_local; new_g = (r_ >= self.p_norm_to_local) & (r_ < (self.p_norm_to_local+self.p_norm_to_global))
                    current_regime[np.where(m0_alive)[0][new_l]] = 1; current_regime[np.where(m0_alive)[0][new_g]] = 2
                mc_alive = (current_regime > 0) & alive
                if np.any(mc_alive):
                    r_ = np.random.rand(np.sum(mc_alive)); back = r_ < self.p_exit
                    current_regime[np.where(mc_alive)[0][back]] = 0

                z_final.fill(0.0); z_t = Z_raw[:, t, :]
                mask0 = (current_regime == 0) & alive; mask1 = (current_regime == 1) & alive; mask2 = (current_regime == 2) & alive
                if np.any(mask0): z_final[mask0] = np.dot(z_t[mask0], self.L_normal.T)
                if np.any(mask1): z_final[mask1] = np.dot(z_t[mask1], self.L_local.T)
                if np.any(mask2): z_final[mask2] = np.dot(z_t[mask2], self.L_global.T)
                
                path_defense = np.ones(n_sims)
                if self.cfg.is_active_managed: path_defense[current_regime > 0] = 0.85 

                mus_t = self.mu_regimes[current_regime]; sigs_t = self.sigma_regimes[current_regime]
                step_rets = (mus_t - 0.5 * sigs_t**2) * self.dt + (sigs_t * np.sqrt(self.dt) * z_final * path_defense[:, None])
                asset_values[alive] *= np.exp(step_rets[alive])

                cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + (inf_shocks[:, t] + (current_regime == 1)*0.003))

                if (t+1) % 12 == 0:
                    y_at = (t+1)//12
                    for evt in self.cfg.extra_cashflows:
                        if evt.year == y_at:
                            amt = evt.amount * cpi_paths[alive, t+1]
                            tot = np.sum(asset_values[alive], axis=1, keepdims=True); tot[tot==0]=1.0
                            asset_values[alive] += amt[:, None] * (asset_values[alive]/tot)

                current_year = (t+1)/12; m_spend_nom = 0
                for w in self.withdrawals:
                    if w.from_year <= current_year < w.to_year: m_spend_nom = w.amount_nominal_monthly_start * cpi_paths[:, t+1]; break
                
                trig_mask = alive & has_house & (np.sum(asset_values,1) < m_spend_nom*self.cfg.emergency_months_trigger)
                if np.any(trig_mask):
                    asset_values[trig_mask, rf_idx] += self.cfg.net_inmo_value * cpi_paths[trig_mask, t+1]
                    has_house[trig_mask] = False

                outflow = m_spend_nom + (self.cfg.enable_prop & (~has_house)) * (self.cfg.new_rent_cost * cpi_paths[:, t+1])
                if self.cfg.use_guardrails:
                    real_w = np.sum(asset_values,1)/cpi_paths[:, t+1]
                    outflow[( (self.cfg.initial_capital - real_w)/self.cfg.initial_capital ) > 0.20] *= (1 - self.cfg.guardrail_cut)
                
                wd = np.minimum(outflow, np.sum(asset_values,1))
                if self.cfg.use_smart_buckets:
                    rf_b = np.maximum(asset_values[:, rf_idx], 0); t_rf = np.minimum(wd, rf_b)
                    asset_values[:, rf_idx] -= t_rf; asset_values[:, rv_idx] -= (wd - t_rf)
                else:
                    rat = wd / np.maximum(np.sum(asset_values,1), 1.0)
                    asset_values *= (1 - rat[:, None])

                asset_values = np.maximum(asset_values, 0); capital_paths[:, t+1] = np.sum(asset_values, 1)
                dead = (capital_paths[:, t+1] <= 1000) & alive
                if np.any(dead):
                    is_alive[dead] = False; ruin_indices[dead] = t+1
                    capital_paths[dead, t+1:] = 0; asset_values[dead] = 0

        return capital_paths, cpi_paths, ruin_indices, 0, np.zeros(n_steps)

# --- 4. INTERFAZ ---
def app(default_rf=720000000, default_rv=1080000000, default_inmo_neto=500000000):
    st.markdown("## 🦅 Panel de Decisión (V16 Sovereign Alpha)")
    
    SCENARIOS_GLOBAL = {
        "Colapso Sistémico (PÉSIMO)": {"corr": 0.92, "rf_ret": -0.06, "rv_ret": -0.30},
        "Crash Financiero (Recomendado)": {"corr": 0.75, "rf_ret": -0.02, "rv_ret": -0.22},
        "Recesión Estándar (OPTIMISTA)": {"corr": 0.55, "rf_ret": 0.01, "rv_ret": -0.15}
    }
    SCENARIOS_LOCAL = {
        "Falla del Hedge (PÉSIMO)": {"corr": 0.20, "rf_ret": 0.0},
        "Protección Estándar (Recomendado)": {"corr": -0.25, "rf_ret": 0.08},
        "Dólar Blindado (OPTIMISTA)": {"corr": -0.35, "rf_ret": 0.12}
    }
    SCENARIOS_RENTABILIDAD = {
        "Conservador": {"rv": 0.08, "rf": 0.045},
        "Histórico (Recomendado)": {"rv": 0.11, "rf": 0.06},
        "Crecimiento (Optimista)": {"rv": 0.13, "rf": 0.07},
        "Personalizado": {"rv": 0.0, "rf": 0.0}
    }

    with st.sidebar:
        st.header("⚙️ Configuración")
        is_active = st.toggle("Gestión Activa / Balanceados", value=True)
        st.divider()
        sel_glo = st.selectbox("🌎 Crisis Global", list(SCENARIOS_GLOBAL.keys()), index=1)
        sel_loc = st.selectbox("🇨🇱 Crisis Local", list(SCENARIOS_LOCAL.keys()), index=1)
        sel_ret = st.selectbox("📈 Rentabilidad", list(SCENARIOS_RENTABILIDAD.keys()), index=1)
        if sel_ret == "Personalizado":
            chosen_mu_rv = st.number_input("RV % Nom.", 0.0, 20.0, 11.0)/100; chosen_mu_rf = st.number_input("RF % Nom.", 0.0, 15.0, 6.0)/100
        else:
            chosen_mu_rv = SCENARIOS_RENTABILIDAD[sel_ret]["rv"]; chosen_mu_rf = SCENARIOS_RENTABILIDAD[sel_ret]["rf"]
        n_sims = st.slider("Simulaciones", 500, 3000, 1000); horiz = st.slider("Horizonte", 10, 50, 40)
        use_guard = st.checkbox("🛡️ Modo Austeridad", True)

    total_ini = default_rf + default_rv; pct_rv_input = 60 # Forzado a 60% por defecto
    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    with tab_sim:
        st.subheader("1. Estructura")
        c1, c2, c3 = st.columns(3)
        with c1: cap_input = clean_input("Capital Total ($)", 1800000000, "cap_total")
        with c2: pct_rv_user = st.slider("Motor (RV)", 0, 100, int(pct_rv_input))
        with c3: st.metric("Mix", f"{100-pct_rv_user}% Def / {pct_rv_user}% Mot")
        
        st.subheader("2. Gastos")
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Fase 1", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
        with g2: r2 = clean_input("Fase 2", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
        with g3: r3 = clean_input("Fase 3", 5000000, "r3")
        
        st.subheader("3. Respaldo Inmobiliario")
        enable_prop = st.checkbox("Activar Venta Emergencia", value=True) # Activado por defecto
        if enable_prop:
            val_inmo = clean_input("Valor Neto ($)", 500000000, "v_i")
            trigger_m = st.slider("Vender si quedan X meses vida", 6, 60, 24)
        else: val_inmo, trigger_m = 0, 0

        if st.button("🚀 INICIAR SIMULACIÓN V16", type="primary"):
            p_glo = SCENARIOS_GLOBAL[sel_glo]; p_loc = SCENARIOS_LOCAL[sel_loc]
            assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims, use_guardrails=use_guard, is_active_managed=is_active, enable_prop=enable_prop, net_inmo_value=val_inmo, emergency_months_trigger=trigger_m, mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf, mu_local_rv=-0.15, mu_local_rf=p_loc["rf_ret"], corr_local=p_loc["corr"], mu_global_rv=p_glo["rv_ret"], mu_global_rf=p_glo["rf_ret"], corr_global=p_glo["corr"])
            sim = InstitutionalSimulator(cfg, assets, wds); paths, cpi, ruin_indices, _, _ = sim.run()
            success_prob = (1 - (np.sum(ruin_indices > -1)/n_sims))*100
            legacy = np.median(paths[:,-1]/cpi[:,-1])
            clr = "#10b981" if success_prob > 90 else "#f59e0b" if success_prob > 75 else "#ef4444"
            st.markdown(f"<div style='text-align:center; padding:20px; border-left:10px solid {clr}; background:rgba(30,30,30,0.5);'><h1>Probabilidad Éxito: {success_prob:.1f}%</h1><hr><h3>Legado Real: ${fmt(legacy)}</h3></div>", unsafe_allow_html=True)
            
            y_ax = np.arange(paths.shape[1])/12; fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_ax, y_ax[::-1]]), y=np.concatenate([np.percentile(paths, 90, 0), np.percentile(paths, 10, 0)[::-1]]), fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_ax, y=np.percentile(paths, 50, 0), line=dict(color='#3b82f6', width=3), name='Mediana'))
            fig.update_layout(title="Evolución Patrimonio (CLP Nominal)", template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)

    with tab_opt:
        st.write("Configuración de optimización lista para usar con el motor V16.")
