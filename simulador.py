import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List
import re

# --- 1. UTILIDADES DE FORMATO (BLINDADAS) ---
def fmt(v): 
    """Formato: 10.000.000"""
    if v is None: return "0"
    try:
        return f"{int(v):,}".replace(",", ".")
    except:
        return str(v)

def clean_input(label, val, key):
    val_str = fmt(val)
    new_val = st.text_input(label, value=val_str, key=key)
    clean_val = re.sub(r'\.', '', new_val)
    clean_val = re.sub(r'\D', '', clean_val)
    return int(clean_val) if clean_val else 0

def fmt_pct(v):
    return f"{v*100:.1f}%"

# --- 2. CONFIGURACIÓN ---
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
class ExtraCashflow:
    year: int
    amount: float
    name: str

@dataclass
class SimulationConfig:
    horizon_years: int = 40
    steps_per_year: int = 12
    initial_capital: float = 1_000_000
    n_sims: int = 2000
    
    # Parámetros Base
    mu_normal_rv: float = 0.10
    mu_normal_rf: float = 0.06
    inflation_mean: float = 0.035
    inflation_vol: float = 0.012
    
    # Perfil del Instrumento (NUEVO)
    is_active_managed: bool = True # ¿Son fondos balanceados/activos?
    
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
    
    extra_cashflows: List[ExtraCashflow] = field(default_factory=list)
    
    # Crisis
    mu_local_rv: float = -0.15; mu_local_rf: float = 0.08; corr_local: float = -0.25  
    mu_global_rv: float = -0.35; mu_global_rf: float = -0.06; corr_global: float = 0.90   
    prob_enter_local: float = 0.005; prob_enter_global: float = 0.004; prob_exit_crisis: float = 0.085  

# --- 3. MOTOR DE SIMULACIÓN V15 (ESTABILIZADO) ---
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
        
        # --- AJUSTE DE GESTIÓN ACTIVA ---
        # Si los fondos son gestionados/balanceados, la volatilidad es MENOR que un índice puro.
        # Un fondo activo "actúa rápido", amortiguando caídas (pero también recortando picos).
        vol_factor = 0.75 if self.cfg.is_active_managed else 1.0
        
        # Volatilidades Base [RV, RF]
        base_sigma = np.array([[0.15, 0.05], [0.22, 0.12], [0.28, 0.14]])
        self.sigma_regimes = base_sigma * vol_factor
        
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
            current_year_float = (t+1) / spy
            current_year_int = int(current_year_float)
            
            monthly_spend_base = 0
            for w in self.withdrawals:
                if w.from_year <= current_year_float < w.to_year:
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
            
            # --- PROTECCIÓN GESTIÓN ACTIVA EN CRISIS ---
            # Si hay gestión activa y estamos en crisis, suavizamos la caída (Beta < 1)
            # Esto simula que el fondo balanceado "actúa rápido" y pasa a caja.
            active_defense = 1.0
            if self.cfg.is_active_managed and np.any(current_regime > 0):
                # Reducimos el shock negativo un 15% (Defensa táctica)
                active_defense = 0.85 
            
            shock = sigs_t * np.sqrt(self.dt) * z_final * active_defense
            drift = (mus_t - 0.5 * sigs_t**2) * self.dt
            
            step_rets = drift + shock
            step_rets = np.clip(step_rets, -0.6, 0.6) 
            asset_values[is_alive] *= np.exp(step_rets[is_alive])

            # EVENTOS EXTRA
            if (t+1) % spy == 1: 
                for evt in self.cfg.extra_cashflows:
                    if evt.year == current_year_int:
                        amount_real = evt.amount * cpi_paths[is_alive, t+1]
                        total_assets = np.sum(asset_values[is_alive], axis=1, keepdims=True)
                        total_assets[total_assets==0] = 1.0 
                        weights = asset_values[is_alive] / total_assets
                        asset_values[is_alive] += amount_real[:, np.newaxis] * weights

            # GATILLO INMOBILIARIO
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

# --- 4. INTERFAZ PRINCIPAL ---
def app(default_rf=363000000, default_rv=1368000000, default_inmo_neto=0):
    
    st.markdown("## 🦅 Panel de Decisión Patrimonial")
    st.info("💡 **Nota:** Valores en miles ($ 00.000.000). Rentabilidades NOMINALES (incluyen inflación).")

    # --- ESCENARIOS (V15 Ajustados) ---
    SCENARIOS_GLOBAL = {
        "Colapso Sistémico (PÉSIMO)": {"corr": 0.92, "rf_ret": -0.06, "rv_ret": -0.30, "desc": "Peor que Recomendado. Crisis total."},
        "Crash Financiero (Recomendado)": {"corr": 0.70, "rf_ret": -0.02, "rv_ret": -0.25, "desc": "Escenario Base. Caída fuerte pero con recuperación."},
        "Recesión Estándar (OPTIMISTA)": {"corr": 0.50, "rf_ret": 0.0, "rv_ret": -0.15, "desc": "Mejor que Recomendado. Corrección normal."}
    }
    
    SCENARIOS_LOCAL = {
        "Falla del Hedge (PÉSIMO)": {"corr": 0.20, "rf_ret": 0.0, "desc": "Dólar NO protege."},
        "Protección Estándar (Recomendado)": {"corr": -0.25, "rf_ret": 0.08, "desc": "Dólar protege."},
        "Dólar Blindado (OPTIMISTA)": {"corr": -0.35, "rf_ret": 0.10, "desc": "USD vuela."}
    }

    # Rentabilidades Base (V15 - Ajuste Equilibrado)
    SCENARIOS_RENTABILIDAD = {
        "Conservador": {"rv": 0.08, "rf": 0.04, "desc": "Bajo crecimiento."},
        "Histórico (Recomendado)": {"rv": 0.11, "rf": 0.06, "desc": "Promedios balanceados."},
        "Crecimiento (Optimista)": {"rv": 0.13, "rf": 0.07, "desc": "Buen desempeño bursátil."},
        "Personalizado": {"rv": 0.0, "rf": 0.0, "desc": "Manual."}
    }

    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # EL INTERRUPTOR CLAVE
        st.markdown("### 🏦 Tipo de Instrumentos")
        is_active = st.toggle("Gestión Activa / Balanceados", value=True, 
                             help="Actívalo si tus fondos NO son 100% acciones puras, sino fondos mixtos que 'actúan rápido' ante cambios.")
        if is_active:
            st.caption("✅ Modo Estabilizado: Se asume menor volatilidad y defensa en crisis.")
        else:
            st.caption("⚠️ Modo Pasivo: Se asume volatilidad total de mercado (ETFs puros).")

        st.divider()
        sel_glo = st.selectbox("🌎 Crisis Global", list(SCENARIOS_GLOBAL.keys()), index=1)
        sel_loc = st.selectbox("🇨🇱 Crisis Local", list(SCENARIOS_LOCAL.keys()), index=1)
        sel_ret = st.selectbox("📈 Rentabilidad", list(SCENARIOS_RENTABILIDAD.keys()), index=1)
        
        if sel_ret == "Personalizado":
            c1, c2 = st.columns(2)
            mu_rv_user = c1.number_input("RV % Nom.", 0.0, 20.0, 10.0, 0.5) / 100.0
            mu_rf_user = c2.number_input("RF % Nom.", 0.0, 15.0, 5.0, 0.5) / 100.0
            chosen_mu_rv = mu_rv_user; chosen_mu_rf = mu_rf_user
        else:
            chosen_mu_rv = SCENARIOS_RENTABILIDAD[sel_ret]["rv"]
            chosen_mu_rf = SCENARIOS_RENTABILIDAD[sel_ret]["rf"]
            st.caption(f"RV: {fmt_pct(chosen_mu_rv)} | RF: {fmt_pct(chosen_mu_rf)}")
        
        n_sims = st.slider("Simulaciones", 500, 3000, 1000)
        horiz = st.slider("Horizonte", 10, 50, 40)
        use_guard = st.checkbox("🛡️ Activar 'Modo Austeridad'", True)

    tab_sim, tab_opt = st.tabs(["📊 Simulador", "🎯 Optimizador"])

    total_ini = default_rf + default_rv
    pct_rv_input = (default_rv / total_ini) * 100 if total_ini > 0 else 0
    
    # --- TAB 1 ---
    with tab_sim:
        st.subheader("1. Capital y Estructura")
        c1, c2, c3 = st.columns(3)
        with c1: cap_input = clean_input("Capital Total ($)", total_ini, "cap_total")
        with c2: pct_rv_user = st.slider("Motor (RV)", 0, 100, int(pct_rv_input), key="slider_mix")
        with c3: st.metric("Mix", f"{100-pct_rv_user}% RF / {pct_rv_user}% RV")

        st.subheader("2. Flujos de Vida")
        g1, g2, g3 = st.columns(3)
        with g1: r1 = clean_input("Fase 1 (Inicio)", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7, key="d1")
        with g2: r2 = clean_input("Fase 2 (Intermedia)", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13, key="d2")
        with g3: r3 = clean_input("Fase 3 (Vejez)", 5000000, "r3"); st.caption("Resto")

        with st.expander("💸 Movimientos Extraordinarios (Hitos)"):
            if 'extra_events' not in st.session_state: st.session_state.extra_events = []
            ce1, ce2, ce3, ce4 = st.columns([2, 2, 2, 1])
            with ce1: ev_year = st.number_input("Año", 1, 40, 5, key="ev_y")
            with ce2: ev_amt = clean_input("Monto ($)", 0, "ev_a")
            with ce3: ev_type = st.selectbox("Tipo", ["Entrada (+)", "Gasto (-)"], key="ev_t")
            with ce4: 
                if st.button("Add"):
                    final_amt = ev_amt if ev_type == "Entrada (+)" else -ev_amt
                    st.session_state.extra_events.append(ExtraCashflow(ev_year, final_amt, "Hito"))
            if st.session_state.extra_events:
                for e in st.session_state.extra_events: st.text(f"Año {e.year}: ${fmt(e.amount)}")
                if st.button("Limpiar"): st.session_state.extra_events = []

        st.subheader("3. Inmobiliario")
        enable_prop = st.checkbox("Propiedad de Respaldo", value=False, key="chk_prop")
        if enable_prop:
            c1, c2 = st.columns(2)
            with c1:
                val_inmo = clean_input("Valor Neto ($)", default_inmo_neto, "v_inmo")
                rent_cost = clean_input("Costo Arriendo ($)", 1500000, "v_rent")
            with c2:
                trigger_months = st.slider("Vender si quedan X meses", 6, 60, 24, key="trig")
                forced_sale = st.checkbox("Forzar venta", value=False, key="chk_force")
                forced_year = st.slider("Año Venta", 1, 40, 30, key="sl_force") if forced_sale else 0
        else:
            val_inmo, rent_cost, trigger_months, forced_year = 0, 0, 0, 0

        if st.button("🚀 EJECUTAR SIMULACIÓN", type="primary"):
            p_glo = SCENARIOS_GLOBAL[sel_glo]; p_loc = SCENARIOS_LOCAL[sel_loc]
            
            assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
            wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            
            cfg = SimulationConfig(
                horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims, use_guardrails=use_guard, 
                is_active_managed=is_active, # <-- V15 FEATURE
                enable_prop=enable_prop, net_inmo_value=val_inmo, new_rent_cost=rent_cost, 
                emergency_months_trigger=trigger_months, forced_sale_year=forced_year,
                extra_cashflows=st.session_state.extra_events,
                mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf,
                mu_local_rv=-0.15, mu_local_rf=p_loc["rf_ret"], corr_local=p_loc["corr"],
                mu_global_rv=p_glo["rv_ret"], mu_global_rf=p_glo["rf_ret"], corr_global=p_glo["corr"], 
                prob_enter_local=0.005, prob_enter_global=0.004, prob_exit_crisis=0.085
            )
            
            sim = InstitutionalSimulator(cfg, assets, wds)
            
            with st.spinner("Simulando..."):
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
                    <p style="font-size:1.1em; margin-top:5px;">Herencia Mediana: <b>${fmt(median_legacy)}</b></p>
                </div>""", unsafe_allow_html=True)

                if failures > 0:
                    ruin_years = ruin_indices[ruin_indices > -1] / 12
                    p25, p75 = np.percentile(ruin_years, [25, 75])
                    st.warning(f"💀 **Zona Crítica:** El 50% de las quiebras ocurren entre el año **{p25:.1f}** y el **{p75:.1f}**.")

                y_axis = np.arange(paths.shape[1])/12
                upper = np.percentile(paths, 90, axis=0); lower = np.percentile(paths, 10, axis=0); median = np.percentile(paths, 50, axis=0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=np.concatenate([y_axis, y_axis[::-1]]), y=np.concatenate([upper, lower[::-1]]),
                    fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
                fig.add_trace(go.Scatter(x=y_axis, y=median, line=dict(color='#3b82f6', width=3), name='Mediana'))
                if enable_prop and forced_sale: fig.add_vline(x=forced_year, line_dash="dash", line_color="green", annotation_text="Venta")
                fig.update_layout(title="Evolución Patrimonial", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2 ---
    with tab_opt:
        st.markdown("### 🎯 Optimizador")
        target_prob = st.slider("Meta de Éxito (%)", 50, 100, 90, key="opt_target")
        opt_var = st.selectbox("Variable a Optimizar", ["Mix Inversión (RV/RF)", "Gasto Fase 1", "Gasto Fase 2", "Gasto Fase 3"], key="opt_var")
        
        if st.button("🔍 BUSCAR ÓPTIMO", type="primary"):
            p_glo = SCENARIOS_GLOBAL[sel_glo]; p_loc = SCENARIOS_LOCAL[sel_loc]
            
            assets_base = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
            wds_base = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
            
            results = []
            with st.spinner("Calculando escenarios..."):
                if "Mix" in opt_var:
                    vals = list(range(0, 101, 10))
                    for v in vals:
                        a_test = [AssetBucket("Motor", v/100, False), AssetBucket("Defensa", (100-v)/100, True)]
                        c_test = SimulationConfig(
                            horizon_years=horiz, initial_capital=cap_input, n_sims=500,
                            use_guardrails=use_guard, is_active_managed=is_active,
                            enable_prop=enable_prop, net_inmo_value=val_inmo, new_rent_cost=rent_cost, 
                            emergency_months_trigger=trigger_months, forced_sale_year=forced_year,
                            extra_cashflows=st.session_state.extra_events,
                            mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf,
                            mu_local_rv=-0.15, mu_local_rf=p_loc["rf_ret"], corr_local=p_loc["corr"],
                            mu_global_rv=p_glo["rv_ret"], mu_global_rf=p_glo["rf_ret"], corr_global=p_glo["corr"], 
                            prob_enter_local=0.005, prob_enter_global=0.004, prob_exit_crisis=0.085
                        )
                        s = InstitutionalSimulator(c_test, a_test, wds_base)
                        _, _, r, _, _ = s.run()
                        results.append({"x": v, "y": (1-(np.sum(r>-1)/500))*100})
                    
                    best = min(results, key=lambda x: abs(x['y'] - target_prob))
                    st.success(f"✅ Mix sugerido: **{best['x']}% RV** (Prob: {best['y']:.1f}%)")

                elif "Gasto" in opt_var:
                    base = r1 if "Fase 1" in opt_var else (r2 if "Fase 2" in opt_var else r3)
                    vals = np.linspace(base*0.5, base*1.5, 10)
                    for v in vals:
                        w_test = wds_base.copy()
                        if "Fase 1" in opt_var: w_test[0] = WithdrawalTramo(0, d1, v)
                        elif "Fase 2" in opt_var: w_test[1] = WithdrawalTramo(d1, d1+d2, v)
                        else: w_test[2] = WithdrawalTramo(d1+d2, horiz, v)
                        
                        c_test = SimulationConfig(
                            horizon_years=horiz, initial_capital=cap_input, n_sims=500,
                            use_guardrails=use_guard, is_active_managed=is_active,
                            enable_prop=enable_prop, net_inmo_value=val_inmo, new_rent_cost=rent_cost, 
                            emergency_months_trigger=trigger_months, forced_sale_year=forced_year,
                            extra_cashflows=st.session_state.extra_events,
                            mu_normal_rv=chosen_mu_rv, mu_normal_rf=chosen_mu_rf,
                            mu_local_rv=-0.15, mu_local_rf=p_loc["rf_ret"], corr_local=p_loc["corr"],
                            mu_global_rv=p_glo["rv_ret"], mu_global_rf=p_glo["rf_ret"], corr_global=p_glo["corr"], 
                            prob_enter_local=0.005, prob_enter_global=0.004, prob_exit_crisis=0.085
                        )
                        s = InstitutionalSimulator(c_test, assets_base, w_test)
                        _, _, r, _, _ = s.run()
                        results.append({"x": v, "y": (1-(np.sum(r>-1)/500))*100})
                    
                    valid = [r for r in results if r['y'] >= target_prob]
                    if valid:
                        best = max(valid, key=lambda k: k['x'])
                        st.success(f"✅ Gasto Máximo Sostenible: **${fmt(best['x'])}**")
                    else:
                        st.error("❌ Meta no alcanzable en este rango.")

            df = pd.DataFrame(results)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines+markers'))
            fig.add_hline(y=target_prob, line_dash="dash", line_color="green")
            fig.update_layout(title="Curva de Optimización", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
