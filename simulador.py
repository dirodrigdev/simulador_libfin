import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List
import re

# --- 1. UTILIDADES DE FORMATO ---
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
    mu_normal_rv: float = 0.11; mu_normal_rf: float = 0.06; inflation_mean: float = 0.035; inflation_vol: float = 0.012
    is_active_managed: bool = True; use_guardrails: bool = True; guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10
    enable_prop: bool = True; net_inmo_value: float = 500000000; new_rent_cost: float = 1500000
    emergency_months_trigger: int = 24; forced_sale_year: int = 0 
    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.22; mu_global_rf: float = -0.02; corr_global: float = 0.75  
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 3. MOTOR V16.8 (BUCKET PROTECTED) ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year; self.total_steps = int(config.horizon_years * config.steps_per_year)
        # RF y RV indexados por el motor
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
        # Inicializar buckets
        asset_values[:, 0] = self.cfg.initial_capital * 0.60 # Default RV
        asset_values[:, 1] = self.cfg.initial_capital * 0.40 # Default RF
        
        current_regime = np.zeros(n_sims, dtype=int)
        df = 8; Z_raw = (np.random.normal(0, 1, (n_sims, n_steps, 2)) / np.sqrt(np.random.chisquare(df, (n_sims, n_steps, 1)) / df)) / np.sqrt(df / (df - 2)) 
        inf_sh = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive): break
            # 1. Markov
            m0 = (current_regime==0)&alive
            if np.any(m0):
                r = np.random.rand(np.sum(m0))
                current_regime[np.where(m0)[0][r < self.p_norm_l]] = 1
                current_regime[np.where(m0)[0][(r >= self.p_norm_l) & (r < (self.p_norm_l+self.p_norm_g))]] = 2
            mc = (current_regime>0)&alive
            if np.any(mc): current_regime[np.where(mc)[0][np.random.rand(np.sum(mc)) < self.p_exit]] = 0

            # 2. Shocks
            z_t = Z_raw[:, t, :]
            z_f = np.zeros_like(z_t)
            for r_idx, L in enumerate([self.L_normal, self.L_local, self.L_global]):
                mask = (current_regime == r_idx) & alive
                if np.any(mask): z_f[mask] = np.dot(z_t[mask], L.T)
            
            p_def = np.ones(n_sims); p_def[current_regime > 0] = 0.85
            mus, sigs = self.mu_regimes[current_regime], self.sigma_regimes[current_regime]
            asset_values[alive] *= np.exp((mus[alive]-0.5*sigs[alive]**2)*self.dt + sigs[alive]*np.sqrt(self.dt)*z_f[alive]*p_def[alive, None])
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + (inf_sh[:, t] + (current_regime == 1)*0.003))

            # 3. Extra Cashflows (Hitos)
            if (t+1)%12==0:
                y = (t+1)//12
                for e in self.cfg.extra_cashflows: 
                    if e.year == y: asset_values[alive, 1] += e.amount * cpi_paths[alive, t+1]

            # 4. Gastos
            cur_y = (t+1)/12; m_spend = 0
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year: m_spend = w.amount_nominal_monthly_start * cpi_paths[:, t+1]; break
            
            # Venta Casa Emergencia
            trig = alive & has_house & (np.sum(asset_values,1) < m_spend*self.cfg.emergency_months_trigger)
            if np.any(trig): 
                asset_values[trig, 1] += self.cfg.net_inmo_value * cpi_paths[trig, t+1]
                has_house[trig] = False

            # 5. RETIRO BUCKET PROTECTED (Diego's Logic)
            out = m_spend + (self.cfg.enable_prop & (~has_house)) * (1500000 * cpi_paths[:, t+1])
            if self.cfg.use_guardrails:
                real_w = np.sum(asset_values,1)/cpi_paths[:, t+1]
                out[( (self.cfg.initial_capital - real_w)/self.cfg.initial_capital ) > 0.20] *= (1 - self.cfg.guardrail_cut)
            
            wd = np.minimum(out, np.sum(asset_values,1))
            rf_bal = np.maximum(asset_values[:, 1], 0)
            take_rf = np.minimum(wd, rf_bal)
            asset_values[:, 1] -= take_rf
            asset_values[:, 0] -= (wd - take_rf) # Solo saca de RV si RF es 0

            # 6. Final mes
            asset_values = np.maximum(asset_values, 0); capital_paths[:, t+1] = np.sum(asset_values, 1)
            dead = (capital_paths[:, t+1] <= 1000) & alive
            if np.any(dead): is_alive[dead]=False; ruin_indices[dead]=t+1; capital_paths[dead, t+1:]=0; asset_values[dead]=0

        return capital_paths, cpi_paths, ruin_indices

# --- 4. INTERFAZ ---
def app():
    st.markdown("## 🦅 Panel de Decisión (V16.8 Sovereign Final)")
    
    SCENARIOS_RET = {
        "Conservador": [0.08, 0.045], "Histórico (11%)": [0.11, 0.06], "Crecimiento (13%)": [0.13, 0.07]
    }
    SCENARIOS_GLO = {
        "Crash Financiero": [-0.22, -0.02, 0.75], "Colapso Sistémico": [-0.30, -0.06, 0.92], "Recesión Estándar": [-0.15, 0.01, 0.55]
    }

    with st.sidebar:
        st.header("⚙️ Escenarios (Nominales)")
        is_active = st.toggle("Gestión Activa / Balanceados", value=True)
        sel_glo = st.selectbox("Crisis Global", list(SCENARIOS_GLO.keys()), index=0)
        p_glo = SCENARIOS_GLO[sel_glo]
        st.caption(f"Tasa Stress: RV {p_glo[0]*100}% | RF {p_glo[1]*100}%")
        
        st.divider()
        sel_ret = st.selectbox("Rentabilidad Normal", list(SCENARIOS_RET.keys()) + ["Personalizado"], index=1)
        if sel_ret == "Personalizado":
            c_rv = st.number_input("RV %", 0.0, 25.0, 11.0)/100; c_rf = st.number_input("RF %", 0.0, 15.0, 6.0)/100
        else:
            c_rv, c_rf = SCENARIOS_RET[sel_ret]
            st.info(f"RV: {c_rv*100}% | RF: {c_rf*100}%")
        
        n_sims = st.slider("Simulaciones", 500, 3000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 50, 40)

    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    with tab_sim:
        c1, c2, c3 = st.columns(3)
        with c1: cap_val = clean_input("Capital Total ($)", 1800000000, "cap")
        with c2: rv_slider = st.slider("Motor (RV %)", 0, 100, 60)
        with c3: st.metric("Reserva RF (Defensa)", f"${fmt(cap_val * (1-rv_slider/100))}")
        
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Gasto Fase 1", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
        with g2: r2 = clean_input("Gasto Fase 2", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
        with g3: r3 = clean_input("Gasto Fase 3", 5000000, "r3")

        with st.expander("💸 Inyecciones o Salidas de Capital"):
            if 'extra_events' not in st.session_state: st.session_state.extra_events = []
            ce1, ce2, ce3, ce4 = st.columns([1,2,2,1])
            with ce1: ev_y = st.number_input("Año", 1, 40, 5, key="evy")
            with ce2: ev_a = clean_input("Monto ($)", 0, "eva")
            with ce3: ev_t = st.selectbox("Tipo", ["Entrada", "Salida"], key="evt")
            if ce4.button("Add"): st.session_state.extra_events.append(ExtraCashflow(ev_y, ev_a if ev_t=="Entrada" else -ev_a, "Hito"))
            for e in st.session_state.extra_events: st.text(f"Año {e.year}: ${fmt(e.amount)}")
            if st.button("Limpiar Hitos"): st.session_state.extra_events = []

        enable_p = st.checkbox("Venta Casa Emergencia (500M)", value=True)

        if st.button("🚀 INICIAR SIMULACIÓN", type="primary"):
            assets = [AssetBucket("RV", rv_slider/100), AssetBucket("RF", (100-rv_slider)/100, True)]
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap_val, n_sims=n_sims, is_active_managed=is_active, enable_prop=enable_p, net_inmo_value=500000000, mu_normal_rv=c_rv, mu_normal_rf=c_rf, extra_cashflows=st.session_state.extra_events, mu_global_rv=p_glo[0], mu_global_rf=p_glo[1], corr_global=p_glo[2])
            sim = InstitutionalSimulator(cfg, assets, wds); paths, cpi, r_i = sim.run()
            
            prob = (1 - (np.sum(r_i > -1)/n_sims))*100
            legacy = np.median(paths[:,-1]/cpi[:,-1])
            clr = "#10b981" if prob > 90 else "#f59e0b" if prob > 75 else "#ef4444"
            st.markdown(f"<div style='text-align:center; padding:20px; border-radius:10px; border-left:10px solid {clr}; background:rgba(30,30,30,0.5);'><h1>Éxito: {prob:.1f}%</h1><h3>Herencia Real: ${fmt(legacy)}</h3></div>", unsafe_allow_html=True)
            
            y_ax = np.arange(paths.shape[1])/12; fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_ax, y_ax[::-1]]), y=np.concatenate([np.percentile(paths, 90, 0), np.percentile(paths, 10, 0)[::-1]]), fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_ax, y=np.percentile(paths, 50, 0), line=dict(color='#3b82f6', width=3), name='Mediana'))
            fig.update_layout(title="Patrimonio Nominal", template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)

    with tab_opt:
        st.subheader("🎯 Buscador de Soluciones Estratégicas")
        st.write("Calcula el valor ideal bajo la regla de protección de cubos.")
        target = st.slider("Meta Éxito %", 70, 100, 95)
        opt_var = st.selectbox("Optimizar:", ["Mix RV %", "Gasto Fase 1", "Gasto Fase 2", "Gasto Fase 3"])
        
        if st.button("🔍 CALCULAR ÓPTIMO"):
            res = []
            with st.spinner("Ejecutando..."):
                if "Mix" in opt_var: vals = np.linspace(0, 100, 11)
                else: 
                    b = r1 if "1" in opt_var else (r2 if "2" in opt_var else r3)
                    vals = np.linspace(b*0.4, b*1.6, 11)
                for v in vals:
                    if "Mix" in opt_var:
                        a_t = [AssetBucket("RV", v/100), AssetBucket("RF", (100-v)/100, True)]
                        w_t = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
                    else:
                        a_t = [AssetBucket("RV", rv_slider/100), AssetBucket("RF", (100-rv_slider)/100, True)]
                        tw1, tw2, tw3 = r1, r2, r3
                        if "1" in opt_var: tw1 = v
                        elif "2" in opt_var: tw2 = v
                        else: tw3 = v
                        w_t = [WithdrawalTramo(0, d1, tw1), WithdrawalTramo(d1, d1+d2, tw2), WithdrawalTramo(d1+d2, horiz, tw3)]
                    
                    c_t = SimulationConfig(horizon_years=horiz, initial_capital=cap_val, n_sims=400, is_active_managed=is_active, enable_prop=enable_p, net_inmo_value=500000000, mu_normal_rv=c_rv, mu_normal_rf=c_rf, extra_cashflows=st.session_state.extra_events, mu_global_rv=p_glo[0], mu_global_rf=p_glo[1], corr_global=p_glo[2])
                    _, _, ri = InstitutionalSimulator(c_t, a_t, w_t).run()
                    res.append({"v": v, "p": (1-(np.sum(ri>-1)/400))*100})
            
            df = pd.DataFrame(res)
            best = df.iloc[(df['p']-target).abs().argsort()[:1]].iloc[0]
            st.success(f"✅ Valor ideal: **{fmt(best['v']) if 'Gasto' in opt_var else str(int(best['v']))+'%'}**")
            fig_o = go.Figure(); fig_o.add_trace(go.Scatter(x=df['v'], y=df['p'], mode='lines+markers'))
            fig_o.add_hline(y=target, line_dash="dash", line_color="green")
            st.plotly_chart(fig_o, use_container_width=True)

if __name__ == "__main__":
    app()
