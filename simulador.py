import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- 1. CONFIGURACIÓN ---
@dataclass
class AssetBucket:
    name: str
    weight: float = 0.0
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
    
    # Parámetros Base (Se llenan desde la UI)
    mu_normal_rv: float = 0.10
    mu_normal_rf: float = 0.06
    inflation_mean: float = 0.035
    inflation_vol: float = 0.012
    
    use_guardrails: bool = True
    guardrail_trigger: float = 0.15
    guardrail_cut: float = 0.10
    use_smart_buckets: bool = True
    
    # Inmobiliario
    enable_prop: bool = False
    net_inmo_value: float = 0
    new_rent_cost: float = 0
    emergency_months_trigger: int = 24  
    forced_sale_year: int = 0 
    
    # Crisis
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.35; mu_global_rf: float = -0.06; corr_global: float = 0.90   
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 2. MOTOR DE SIMULACIÓN V11 (TRANSPARENTE) ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config
        self.assets = assets
        self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year
        self.total_steps = int(config.horizon_years * config.steps_per_year)

        self.mu_regimes = np.array([
            [self.cfg.mu_normal_rv, self.cfg.mu_normal_rf],
            [self.cfg.mu_local_rv, self.cfg.mu_local_rf],
            [self.cfg.mu_global_rv, self.cfg.mu_global_rf]
        ])
        # Volatilidad ajustada para no "castigar" excesivamente la RV en tiempos normales
        self.sigma_regimes = np.array([[0.15, 0.05], [0.22, 0.12], [0.30, 0.14]])
        
        cn = np.array([[1.0, 0.35], [0.35, 1.0]])
        self.L_normal = np.linalg.cholesky(cn)
        
        cl = np.clip(self.cfg.corr_local, -0.99, 0.99)
        self.L_local = np.linalg.cholesky(np.array([[1.0, cl], [cl, 1.0]]))
        
        cg = np.clip(self.cfg.corr_global, -0.99, 0.99)
        self.L_global = np.linalg.cholesky(np.array([[1.0, cg], [cg, 1.0]]))
        
        self.p_norm_to_local = self.cfg.prob_enter_local
        self.p_norm_to_global = self.cfg.prob_enter_global
        self.p_exit = self.cfg.prob_exit_crisis

    def run(self):
        n_sims, n_steps = self.cfg.n_sims, self.total_steps
        n_assets = len(self.assets)
        
        capital_paths = np.zeros((n_sims, n_steps + 1))
        capital_paths[:, 0] = self.cfg.initial_capital
        cpi_paths = np.ones((n_sims, n_steps + 1))
        
        is_alive = np.ones(n_sims, dtype=bool) 
        ruin_indices = np.full(n_sims, -1) 
        has_house = np.full(n_sims, self.cfg.enable_prop, dtype=bool)
        
        asset_values = np.zeros((n_sims, n_assets))
        for i, a in enumerate(self.assets):
            asset_values[:, i] = self.cfg.initial_capital * a.weight
            
        try: rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
        except: rv_idx = 0
        try: rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
        except: rf_idx = 1
        
        current_regime = np.zeros(n_sims, dtype=int)

        df = 8 
        G = np.random.normal(0, 1, (n_sims, n_steps, n_assets))
        W = np.random.chisquare(df, (n_sims, n_steps, 1)) / df
        Z_raw = G / np.sqrt(W)
        Z_raw /= np.sqrt(df / (df - 2)) 

        inf_shocks = np.random.normal(self.cfg.inflation_mean * self.dt, 
                                      self.cfg.inflation_vol * np.sqrt(self.dt), 
                                      (n_sims, n_steps))

        z_final = np.zeros((n_sims, n_assets))

        for t in range(n_steps):
            if np.any(is_alive):
                mask_0 = (current_regime == 0) & is_alive
                if np.any(mask_0):
                    n_alive_0 = np.sum(mask_0)
                    rand_0 = np.random.rand(n_alive_0)
                    p_L = self.p_norm_to_local; p_G = self.p_norm_to_global
                    new_0_to_1 = rand_0 < p_L
                    new_0_to_2 = (rand_0 >= p_L) & (rand_0 < (p_L + p_G))
                    idx_0 = np.where(mask_0)[0]
                    current_regime[idx_0[new_0_to_1]] = 1
                    current_regime[idx_0[new_0_to_2]] = 2
                
                mask_crisis = (current_regime > 0) & is_alive
                if np.any(mask_crisis):
                    n_alive_c = np.sum(mask_crisis)
                    rand_c = np.random.rand(n_alive_c)
                    back_to_norm = rand_c < self.p_exit
                    idx_c = np.where(mask_crisis)[0]
                    current_regime[idx_c[back_to_norm]] = 0

            spy = self.cfg.steps_per_year
            current_year = (t+1) / spy
            
            monthly_spend_base = 0
            for w in self.withdrawals:
                if w.from_year <= current_year < w.to_year:
                    monthly_spend_base = w.amount_nominal_monthly_start; break
            
            current_inf_shock = inf_shocks[:, t]
            current_inf_shock[current_regime == 1] += 0.003 
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + np.maximum(current_inf_shock, -0.02))
            
            monthly_spend_nom = monthly_spend_base * cpi_paths[:, t+1]

            z_final.fill(0.0)
            z_t = Z_raw[:, t, :]
            
            m0 = (current_regime == 0) & is_alive
            m1 = (current_regime == 1) & is_alive
            m2 = (current_regime == 2) & is_alive
            
            if np.any(m0): z_final[m0] = np.dot(z_t[m0], self.L_normal.T)
            if np.any(m1): z_final[m1] = np.dot(z_t[m1], self.L_local.T)
            if np.any(m2): z_final[m2] = np.dot(z_t[m2], self.L_global.T)
            
            mus_t = self.mu_regimes[current_regime]
            sigs_t = self.sigma_regimes[current_regime]
            step_rets = (mus_t - 0.5 * sigs_t**2) * self.dt + sigs_t * np.sqrt(self.dt) * z_final
            step_rets = np.clip(step_rets, -0.6, 0.6) 
            asset_values[is_alive] *= np.exp(step_rets[is_alive])

            total_cap_now = np.sum(asset_values, axis=1)
            panic_threshold = monthly_spend_nom * self.cfg.emergency_months_trigger
            
            forced_time = (self.cfg.forced_sale_year > 0) and ((t+1) >= int(self.cfg.forced_sale_year * spy))
            trigger_mask = is_alive & has_house & ((total_cap_now < panic_threshold) | forced_time)
            
            if np.any(trigger_mask):
                sale_value = self.cfg.net_inmo_value * cpi_paths[trigger_mask, t+1]
                asset_values[trigger_mask, 0] += sale_value
                has_house[trigger_mask] = False

            living_nom = np.full(n_sims, monthly_spend_nom)
            
            if self.cfg.use_guardrails:
                 curr_real_wealth = np.sum(asset_values, axis=1) / cpi_paths[:, t+1]
                 dd_initial = (self.cfg.initial_capital - curr_real_wealth) / self.cfg.initial_capital
                 living_nom[dd_initial > 0.20] *= (1.0 - self.cfg.guardrail_cut)

            rent_nom = np.zeros(n_sims)
            if self.cfg.enable_prop:
                payers_mask = is_alive & (~has_house)
                rent_nom[payers_mask] = self.cfg.new_rent_cost * cpi_paths[payers_mask, t+1]
            
            total_outflow = living_nom + rent_nom
            
            total_cap_pre_wd = np.sum(asset_values, axis=1)
            wd_actual = np.minimum(total_outflow, total_cap_pre_wd)
            
            mask_wd = (wd_actual > 0) & is_alive
            if np.any(mask_wd):
                if self.cfg.use_smart_buckets:
                    rf_bal = np.maximum(asset_values[mask_wd, rf_idx], 0)
                    take_rf = np.minimum(wd_actual[mask_wd], rf_bal)
                    take_rv = wd_actual[mask_wd] - take_rf
                    asset_values[mask_wd, rf_idx] -= take_rf
                    asset_values[mask_wd, rv_idx] -= take_rv
                else:
                    ratio = wd_actual[mask_wd] / total_cap_pre_wd[mask_wd]
                    asset_values[mask_wd] *= (1 - ratio[:, np.newaxis])

            asset_values = np.maximum(asset_values, 0)
            capital_paths[:, t+1] = np.sum(asset_values, axis=1)
            
            died_now = (capital_paths[:, t+1] <= 1000) & is_alive
            if np.any(died_now):
                is_alive[died_now] = False
                ruin_indices[died_now] = t+1
                capital_paths[died_now, t+1:] = 0.0
                asset_values[died_now] = 0.0

        return capital_paths, cpi_paths, ruin_indices, 0, np.zeros(n_steps)

# --- 3. UTILIDADES Y FORMATO ---
def fmt(v): 
    """Formato chileno: 10.000.000"""
    return f"{int(v):,}".replace(",", ".")

def clean_input(label, val, key):
    val_str = fmt(val)
    new_val = st.text_input(label, value=val_str, key=key)
    clean_val = re.sub(r'\.', '', new_val)
    return int(clean_val) if clean_val.isdigit() else 0

def fmt_pct(v):
    return f"{v*100:.1f}%"

# --- 4. INTERFAZ PRINCIPAL ---
def app(default_rf=363000000, default_rv=1368000000, default_inmo_neto=0):
    
    st.markdown("## 🦅 Panel de Decisión (Transparente)")
    st.info("💡 **Novedad:** Ahora puedes ver y editar las rentabilidades base para ajustar el 'optimismo' del modelo.")

    # --- DEFINICIÓN DE ESCENARIOS ---
    # Global y Local se mantienen igual...
    SCENARIOS_GLOBAL = {
        "Colapso Sistémico (PÉSIMO)": {"corr": 0.92, "rf_ret": -0.06, "rv_ret": -0.35, "desc": "Inflación + Recesión."},
        "Crash Financiero (Recomendado)": {"corr": 0.70, "rf_ret": -0.02, "rv_ret": -0.30, "desc": "Caída fuerte, cash aguanta."},
        "Recesión Estándar (OPTIMISTA)": {"corr": 0.50, "rf_ret": 0.0, "rv_ret": -0.20, "desc": "Bonos protegen."}
    }
    
    SCENARIOS_LOCAL = {
        "Falla del Hedge (PÉSIMO)": {"corr": 0.20, "rf_ret": 0.0, "desc": "Dólar NO sube en crisis."},
        "Protección Estándar (Recomendado)": {"corr": -0.25, "rf_ret": 0.08, "desc": "Dólar sube y protege."},
        "Dólar Blindado (OPTIMISTA)": {"corr": -0.35, "rf_ret": 0.10, "desc": "Peso colapsa, USD vuela."}
    }

    # Rentabilidades Base (Más Opciones + Custom)
    SCENARIOS_RENTABILIDAD = {
        "Conservador": {"rv": 0.07, "rf": 0.04, "desc": "Pesimista. Asume bajo crecimiento mundial."},
        "Histórico (S&P500 50y)": {"rv": 0.10, "rf": 0.05, "desc": "Promedios históricos de largo plazo."},
        "Crecimiento (Optimista)": {"rv": 0.12, "rf": 0.06, "desc": "Asume buen desempeño bursátil (Equity Premium)."},
        "Personalizado": {"rv": 0.0, "rf": 0.0, "desc": "Define tus propios números."}
    }

    # --- SIDEBAR CONFIGURACIÓN ---
    with st.sidebar:
        st.header("1. Configurar Escenarios")
        
        # Selectores de Crisis
        st.markdown("**A. Escenarios de Crisis**")
        sel_glo = st.selectbox("🌎 Mundo", list(SCENARIOS_GLOBAL.keys()), index=1)
        sel_loc = st.selectbox("🇨🇱 Chile", list(SCENARIOS_LOCAL.keys()), index=1)
        
        st.divider()
        
        # Selector de Rentabilidad (EL MOTOR)
        st.markdown("**B. Motor de Crecimiento (Años Normales)**")
        sel_ret = st.selectbox("Proyección Rentabilidad", list(SCENARIOS_RENTABILIDAD.keys()), index=1)
        
        # Lógica para mostrar/editar números
        if sel_ret == "Personalizado":
            c_cust1, c_cust2 = st.columns(2)
            mu_rv_user = c_cust1.number_input("Retorno RV %", 0.0, 20.0, 10.0, 0.5) / 100.0
            mu_rf_user = c_cust2.number_input("Retorno RF %", 0.0, 15.0, 5.0, 0.5) / 100.0
            chosen_mu_rv = mu_rv_user
            chosen_mu_rf = mu_rf_user
            st.caption("Ingresa retornos nominales anuales.")
        else:
            chosen_mu_rv = SCENARIOS_RENTABILIDAD[sel_ret]["rv"]
            chosen_mu_rf = SCENARIOS_RENTABILIDAD[sel_ret]["rf"]
            
            # VISUALIZADOR DE NÚMEROS (Lo que pediste)
            col_kpi1, col_kpi2 = st.columns(2)
            col_kpi1.metric("RV (Motor)", fmt_pct(chosen_mu_rv))
            col_kpi2.metric("RF (Defensa)", fmt_pct(chosen_mu_rf))
            st.caption(SCENARIOS_RENTABILIDAD[sel_ret]["desc"])
        
        st.divider()
        n_sims = st.slider("Precisión", 500, 3000, 1000)
        horiz = st.slider("Horizonte", 10, 50, 40)
        use_guard = st.checkbox("🛡️ Activar 'Modo Austeridad'", True)

    # --- CUERPO PRINCIPAL ---
    # 1. CAPITAL
    total_ini = default_rf + default_rv
    pct_rv_input = (default_rv / total_ini) * 100 if total_ini > 0 else 0
    
    st.markdown("### 💰 Tu Capital y Estructura")
    c1, c2, c3 = st.columns(3)
    with c1: cap_input = clean_input("Capital Total ($)", total_ini, "cap_total")
    with c2: pct_rv_user = st.slider("% Motor (RV)", 0, 100, int(pct_rv_input))
    with c3: st.metric("Mix Efectivo", f"{100-pct_rv_user}% Def / {pct_rv_user}% Mot")

    # 2. GASTOS
    st.markdown("### 💸 Tus Gastos Mensuales (Estimados)")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean_input("Fase 1 (Inicio)", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
    with g2: r2 = clean_input("Fase 2 (Intermedia)", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
    with g3: r3 = clean_input("Fase 3 (Vejez)", 5000000, "r3"); st.caption("Resto")

    # 3. INMOBILIARIO
    st.markdown("### 🏡 Estrategia Inmobiliaria (Salvavidas)")
    enable_prop = st.checkbox("Tengo una Propiedad para vender en caso de emergencia", value=False)
    
    if enable_prop:
        c_inmo1, c_inmo2 = st.columns(2)
        with c_inmo1:
            val_inmo = clean_input("Valor Neto Propiedad ($)", default_inmo_neto, "v_inmo")
            rent_cost = clean_input("Costo Arriendo (si vendo) ($)", 1500000, "v_rent")
        with c_inmo2:
            trigger_months = st.slider("Gatillo: Vender si me quedan X meses de vida", 6, 60, 24)
            forced_sale = st.checkbox("Forzar venta en año específico", value=False)
            forced_year = st.slider("Año Venta Forzada", 1, 40, 30) if forced_sale else 0
    else:
        val_inmo, rent_cost, trigger_months, forced_year = 0, 0, 0, 0

    # --- EJECUCIÓN ---
    if st.button("🚀 EJECUTAR SIMULACIÓN", type="primary"):
        p_glo = SCENARIOS_GLOBAL[sel_glo]
        p_loc = SCENARIOS_LOCAL[sel_loc]
        
        assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
        wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
        
        cfg = SimulationConfig(
            horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims,
            use_guardrails=use_guard, 
            
            enable_prop=enable_prop,
            net_inmo_value=val_inmo, 
            new_rent_cost=rent_cost, 
            emergency_months_trigger=trigger_months,
            forced_sale_year=forced_year,
            
            # AQUÍ USAMOS LOS NÚMEROS ELEGIDOS O CUSTOM
            mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf,
            
            mu_local_rv=-0.15, mu_local_rf=p_loc["rf_ret"], corr_local=p_loc["corr"],
            mu_global_rv=p_glo["rv_ret"], mu_global_rf=p_glo["rf_ret"], corr_global=p_glo["corr"], 
            prob_enter_local=0.005, prob_enter_global=0.004, prob_exit_crisis=0.085
        )
        
        sim = InstitutionalSimulator(cfg, assets, wds)
        
        with st.spinner(f"Simulando {n_sims} futuros posibles..."):
            paths, cpi, ruin_indices, _, _ = sim.run()
            
            failures = np.sum(ruin_indices > -1)
            success_prob = (1 - (failures / n_sims)) * 100
            
            final_wealth = paths[:, -1]
            median_legacy = np.median(final_wealth / cpi[:, -1])
            
            if success_prob >= 90: clr, msg, icon = "#10b981", "EXCELENTE", "✅"
            elif success_prob >= 75: clr, msg, icon = "#f59e0b", "ACEPTABLE", "⚠️"
            else: clr, msg, icon = "#ef4444", "CRÍTICO", "🛑"

            st.markdown(f"""
            <div style="background-color:rgba(30,30,30,0.5); padding:20px; border-radius:10px; border-left: 10px solid {clr}; text-align: center;">
                <h1 style="color:{clr}; margin:0;">{icon} {msg}</h1>
                <h2 style="margin:10px 0;">Probabilidad de Éxito: {success_prob:.1f}%</h2>
                <hr style="border-color: #555;">
                <p style="margin:0; font-size:1.1em;">Herencia Estimada (Valor Hoy): <b>${fmt(median_legacy)}</b></p>
                <p style="font-size:0.9em; opacity:0.8;">Parametros: RV {fmt_pct(chosen_mu_rv)} | RF {fmt_pct(chosen_mu_rf)}</p>
            </div>""", unsafe_allow_html=True)
            
            y_axis = np.arange(paths.shape[1])/12
            upper_bound = np.percentile(paths, 90, axis=0)
            lower_bound = np.percentile(paths, 10, axis=0)
            median_path = np.percentile(paths, 50, axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_axis, y_axis[::-1]]), y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_axis, y=median_path, line=dict(color='#3b82f6', width=3), name='Mediana'))
            
            fig.update_layout(title="Evolución de tu Patrimonio", xaxis_title="Años", yaxis_title="Patrimonio ($)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fails = ruin_indices[ruin_indices > -1]
            if len(fails) > 0:
                first_fail = np.percentile(fails/12, 10)
                st.error(f"💀 Ruina temprana: {first_fail:.1f} años.")
            else:
                st.success("🎉 Plan Sostenible en todas las simulaciones.")
