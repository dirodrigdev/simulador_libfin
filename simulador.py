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
    mu_normal_rv: float = 0.10; mu_normal_rf: float = 0.06; inflation_mean: float = 0.035; inflation_vol: float = 0.012
    is_active_managed: bool = True 
    use_guardrails: bool = True; guardrail_trigger: float = 0.15; guardrail_cut: float = 0.10; use_smart_buckets: bool = True
    enable_prop: bool = True; net_inmo_value: float = 500000000; new_rent_cost: float = 1500000
    emergency_months_trigger: int = 24; forced_sale_year: int = 0 
    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.35; mu_global_rf: float = -0.06; corr_global: float = 0.90   
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 3. MOTOR V16.5 (BUCKET PROTECTED + OPTIMIZER CORE) ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config; self.assets = assets; self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year; self.total_steps = int(config.horizon_years * config.steps_per_year)
        self.mu_regimes = np.array([[self.cfg.mu_normal_rv, self.cfg.mu_normal_rf],[self.cfg.mu_local_rv, self.cfg.mu_local_rf],[self.cfg.mu_global_rv, self.cfg.mu_global_rf]])
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
        
        try: 
            rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
            rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
        except: rv_idx, rf_idx = 0, 1
        
        current_regime = np.zeros(n_sims, dtype=int)
        df = 8; Z_raw = (np.random.normal(0, 1, (n_sims, n_steps, n_assets)) / np.sqrt(np.random.chisquare(df, (n_sims, n_steps, 1)) / df)) / np.sqrt(df / (df - 2)) 
        inf_shocks = np.random.normal(self.cfg.inflation_mean * self.dt, self.cfg.inflation_vol * np.sqrt(self.dt), (n_sims, n_steps))
        z_final = np.zeros((n_sims, n_assets))

        for t in range(n_steps):
            alive = is_alive
            if not np.any(alive): break
            
            # Markov
            m0 = (current_regime == 0) & alive
            if np.any(m0):
                r_ = np.random.rand(np.sum(m0))
                current_regime[np.where(m0)[0][r_ < self.p_norm_to_local]] = 1
                current_regime[np.where(m0)[0][(r_ >= self.p_norm_to_local) & (r_ < (self.p_norm_to_local+self.p_norm_to_global))]] = 2
            mc = (current_regime > 0) & alive
            if np.any(mc):
                r_ = np.random.rand(np.sum(mc)); current_regime[np.where(mc)[0][r_ < self.p_exit]] = 0

            # Mercado con Protección Táctica
            z_t = Z_raw[:, t, :]
            mask0, mask1, mask2 = (current_regime==0)&alive, (current_regime==1)&alive, (current_regime==2)&alive
            if np.any(mask0): z_final[mask0] = np.dot(z_t[mask0], self.L_normal.T)
            if np.any(mask1): z_final[mask1] = np.dot(z_t[mask1], self.L_local.T)
            if np.any(mask2): z_final[mask2] = np.dot(z_t[mask2], self.L_global.T)
            
            p_def = np.ones(n_sims); p_def[current_regime > 0] = 0.85 
            mus_t, sigs_t = self.mu_regimes[current_regime], self.sigma_regimes[current_regime]
            asset_values[alive] *= np.exp((mus_t[alive]-0.5*sigs_t[alive]**2)*self.dt + sigs_t[alive]*np.sqrt(self.dt)*z_final[alive]*p_def[alive, None])
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + (inf_shocks[:, t] + (current_regime == 1)*0.003))

            # Cashflows extraordinarios
            if (t+1) % 12 == 0:
                y = (t+1)//12
                for evt in self.cfg.extra_cashflows:
                    if evt.year == y: asset_values[alive, rf_idx] += evt.amount * cpi_paths[alive, t+1]

            # Gastos y Gatillo Inmobiliario
            cur_y = (t+1)/12; m_spend = 0
            for w in self.withdrawals:
                if w.from_year <= cur_y < w.to_year: m_spend = w.amount_nominal_monthly_start * cpi_paths[:, t+1]; break
            
            trig = alive & has_house & (np.sum(asset_values,1) < m_spend*self.cfg.emergency_months_trigger)
            if np.any(trig):
                asset_values[trig, rf_idx] += self.cfg.net_inmo_value * cpi_paths[trig, t+1]; has_house[trig] = False

            # Retiro Estratégico (Prioridad RF)
            out = m_spend + (self.cfg.enable_prop & (~has_house)) * (1500000 * cpi_paths[:, t+1])
            if self.cfg.use_guardrails:
                out[( (self.cfg.initial_capital - np.sum(asset_values,1)/cpi_paths[:, t+1])/self.cfg.initial_capital ) > 0.20] *= (1 - self.cfg.guardrail_cut)
            
            wd = np.minimum(out, np.sum(asset_values,1))
            rf_b = np.maximum(asset_values[:, rf_idx], 0); t_rf = np.minimum(wd, rf_b)
            asset_values[:, rf_idx] -= t_rf; asset_values[:, rv_idx] -= (wd - t_rf)

            asset_values = np.maximum(asset_values, 0); capital_paths[:, t+1] = np.sum(asset_values, 1)
            dead = (capital_paths[:, t+1] <= 1000) & alive
            if np.any(dead):
                is_alive[dead] = False; ruin_indices[dead] = t+1; capital_paths[dead, t+1:] = 0; asset_values[dead] = 0

        return capital_paths, cpi_paths, ruin_indices

# --- 4. INTERFAZ ---
def app(default_rf=720000000, default_rv=1080000000, default_inmo_neto=500000000):
    st.markdown("## 🦅 Panel de Decisión Patrimonial")
    
    # Escenarios Transparentes
    SCENARIOS_RENTABILIDAD = {
        "Conservador": {"rv": 0.08, "rf": 0.045, "label": "RV 8.0% | RF 4.5%"},
        "Histórico (Recomendado)": {"rv": 0.11, "rf": 0.06, "label": "RV 11.0% | RF 6.0%"},
        "Crecimiento (Optimista)": {"rv": 0.13, "rf": 0.07, "label": "RV 13.0% | RF 7.0%"},
        "Personalizado": {"rv": 0.0, "rf": 0.0, "label": "Definir manualmente"}
    }
    
    SCENARIOS_GLOBAL = {
        "Crash Financiero (Recomendado)": {"rv": -0.22, "rf": -0.02, "corr": 0.75, "label": "Caída RV -22%"},
        "Colapso Sistémico (PÉSIMO)": {"rv": -0.30, "rf": -0.06, "corr": 0.92, "label": "Caída RV -30%"},
        "Recesión Estándar (OPTIMISTA)": {"rv": -0.15, "rf": 0.01, "corr": 0.55, "label": "Caída RV -15%"}
    }

    with st.sidebar:
        st.header("⚙️ Configuración")
        is_active = st.toggle("Gestión Activa / Balanceados", value=True)
        st.divider()
        sel_glo = st.selectbox("Escenario de Crisis", list(SCENARIOS_GLOBAL.keys()), index=0)
        st.caption(f"Tasa Stress: {SCENARIOS_GLOBAL[sel_glo]['label']}")
        
        st.divider()
        st.markdown("### 📈 Rentabilidad Nominal")
        sel_ret = st.selectbox("Proyección", list(SCENARIOS_RENTABILIDAD.keys()), index=1)
        if sel_ret == "Personalizado":
            chosen_mu_rv = st.number_input("RV % Anual", 0.0, 25.0, 11.0)/100
            chosen_mu_rf = st.number_input("RF % Anual", 0.0, 15.0, 6.0)/100
        else:
            chosen_mu_rv = SCENARIOS_RENTABILIDAD[sel_ret]["rv"]
            chosen_mu_rf = SCENARIOS_RENTABILIDAD[sel_ret]["rf"]
            st.info(f"Usando: {SCENARIOS_RENTABILIDAD[sel_ret]['label']}")
        
        st.divider()
        n_sims = st.slider("Simulaciones", 500, 3000, 1000)
        horiz = st.slider("Horizonte (Años)", 10, 50, 40)

    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    total_ini = 1800000000; pct_rv_def = 60 

    with tab_sim:
        st.subheader("1. Capital y Estructura")
        c1, c2, c3 = st.columns(3)
        with c1: cap_input = clean_input("Capital Total ($)", total_ini, "cap_total")
        with c2: pct_rv_user = st.slider("Motor (RV %)", 0, 100, pct_rv_def)
        with c3: st.metric("Mix", f"{100-pct_rv_user}% RF / {pct_rv_user}% RV")
        
        st.subheader("2. Gastos e Hitos")
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Fase 1 (Inicio)", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
        with g2: r2 = clean_input("Fase 2 (Intermedia)", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
        with g3: r3 = clean_input("Fase 3 (Vejez)", 5000000, "r3")

        with st.expander("💸 Inyecciones o Salidas Extraordinarias"):
            if 'extra_events' not in st.session_state: st.session_state.extra_events = []
            ce1, ce2, ce3, ce4 = st.columns([1,2,2,1])
            with ce1: ev_y = st.number_input("Año", 1, 40, 5, key="evy")
            with ce2: ev_a = clean_input("Monto ($)", 0, "eva")
            with ce3: ev_t = st.selectbox("Tipo", ["Entrada", "Salida"], key="evt")
            with ce4: 
                if st.button("Add"): st.session_state.extra_events.append(ExtraCashflow(ev_y, ev_a if ev_t=="Entrada" else -ev_a, "Hito"))
            for e in st.session_state.extra_events: st.text(f"Año {e.year}: ${fmt(e.amount)}")
            if st.button("Limpiar"): st.session_state.extra_events = []

        st.subheader("3. Respaldo Inmobiliario")
        enable_prop = st.checkbox("Activar Venta Emergencia", value=True) 
        if enable_prop:
            val_inmo = clean_input("Valor Neto Casa ($)", 500000000, "v_i")
            trigger_m = st.slider("Gatillo (Meses Vida)", 6, 60, 24)
        else: val_inmo, trigger_m = 0, 0

        if st.button("🚀 INICIAR SIMULACIÓN", type="primary"):
            assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            cfg = SimulationConfig(horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims, is_active_managed=is_active, enable_prop=enable_prop, net_inmo_value=val_inmo, emergency_months_trigger=trigger_m, extra_cashflows=st.session_state.extra_events, mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf, mu_local_rv=-0.15, mu_local_rf=0.08, corr_local=-0.25, mu_global_rv=SCENARIOS_GLOBAL[sel_glo]['rv'], mu_global_rf=SCENARIOS_GLOBAL[sel_glo]['rf'], corr_global=SCENARIOS_GLOBAL[sel_glo]['corr'])
            sim = InstitutionalSimulator(cfg, assets, wds); paths, cpi, ruin_indices = sim.run()
            
            success_prob = (1 - (np.sum(ruin_indices > -1)/n_sims))*100
            legacy = np.median(paths[:,-1]/cpi[:,-1])
            clr = "#10b981" if success_prob > 90 else "#f59e0b" if success_prob > 75 else "#ef4444"
            st.markdown(f"<div style='text-align:center; padding:20px; border-left:10px solid {clr}; background:rgba(30,30,30,0.5);'><h1>Probabilidad Éxito: {success_prob:.1f}%</h1><hr><h3>Legado Real (Valor Hoy): ${fmt(legacy)}</h3></div>", unsafe_allow_html=True)
            
            y_ax = np.arange(paths.shape[1])/12; fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_ax, y_ax[::-1]]), y=np.concatenate([np.percentile(paths, 90, 0), np.percentile(paths, 10, 0)[::-1]]), fill='toself', fillcolor='rgba(59,130,246,0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_ax, y=np.percentile(paths, 50, 0), line=dict(color='#3b82f6', width=3), name='Mediana'))
            fig.update_layout(title="Evolución Patrimonio", template="plotly_dark"); st.plotly_chart(fig, use_container_width=True)

    with tab_opt:
        st.subheader("🎯 Buscador de Soluciones")
        target_prob = st.slider("Meta de Éxito Objetivo (%)", 60, 100, 90)
        var_to_opt = st.selectbox("¿Qué quieres calcular?", ["Gasto Fase 1", "Gasto Fase 2", "Gasto Fase 3", "Mix RV Ideal"])
        
        if st.button("🔍 CALCULAR VALOR ÓPTIMO"):
            results = []
            with st.spinner("Ejecutando iteraciones de mercado..."):
                # Rango de búsqueda
                if "Mix" in var_to_opt: vals = np.linspace(0, 100, 11)
                else: 
                    base = r1 if "1" in var_to_opt else (r2 if "2" in var_to_opt else r3)
                    vals = np.linspace(base*0.4, base*1.6, 11)

                for v in vals:
                    if "Mix" in var_to_opt:
                        t_assets = [AssetBucket("Motor", v/100, False), AssetBucket("Defensa", (100-v)/100, True)]
                        t_wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
                    else:
                        t_assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
                        tw1, tw2, tw3 = r1, r2, r3
                        if "1" in var_to_opt: tw1 = v
                        elif "2" in var_to_opt: tw2 = v
                        else: tw3 = v
                        t_wds = [WithdrawalTramo(0, d1, tw1), WithdrawalTramo(d1, d1+d2, tw2), WithdrawalTramo(d1+d2, horiz, tw3)]

                    c_opt = SimulationConfig(horizon_years=horiz, initial_capital=cap_input, n_sims=500, is_active_managed=is_active, enable_prop=enable_prop, net_inmo_value=val_inmo, emergency_months_trigger=trigger_m, extra_cashflows=st.session_state.extra_events, mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf, mu_local_rv=-0.15, mu_local_rf=0.08, corr_local=-0.25, mu_global_rv=SCENARIOS_GLOBAL[sel_glo]['rv'], mu_global_rf=SCENARIOS_GLOBAL[sel_glo]['rf'], corr_global=SCENARIOS_GLOBAL[sel_glo]['corr'])
                    s_t = InstitutionalSimulator(c_opt, t_assets, t_wds)
                    _, _, r_i = s_t.run()
                    results.append({"val": v, "prob": (1-(np.sum(r_i > -1)/500))*100})
            
            df_opt = pd.DataFrame(results)
            best = df_opt.iloc[(df_opt['prob']-target_prob).abs().argsort()[:1]]
            st.success(f"✅ Para lograr {target_prob}% de éxito, el valor ideal de **{var_to_opt}** es: **{fmt(best.iloc[0]['val']) if 'Gasto' in var_to_opt else str(int(best.iloc[0]['val'])) + '%'}**")
            
            fig_opt = go.Figure()
            fig_opt.add_trace(go.Scatter(x=df_opt['val'], y=df_opt['prob'], mode='lines+markers', name='Resultados'))
            fig_opt.add_hline(y=target_prob, line_dash="dash", line_color="green", annotation_text="Meta")
            fig_opt.update_layout(title="Análisis de Sensibilidad", xaxis_title=var_to_opt, yaxis_title="Éxito %", template="plotly_dark")
            st.plotly_chart(fig_opt, use_container_width=True)
