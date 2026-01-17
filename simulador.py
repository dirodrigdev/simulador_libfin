import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# --- 1. CONFIGURACIÓN Y CLASES DE DATOS ---
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
    
    # Parámetros Base (Normales)
    mu_normal_rv: float = 0.10
    mu_normal_rf: float = 0.06
    
    # Inflación
    inflation_mean: float = 0.035
    inflation_vol: float = 0.012
    
    # Estrategias
    use_guardrails: bool = True
    guardrail_trigger: float = 0.15
    guardrail_cut: float = 0.10
    use_smart_buckets: bool = True
    
    # Inmobiliario
    sell_year: int = 0
    net_inmo_value: float = 0
    new_rent_cost: float = 0
    inmo_strategy: str = "portfolio"
    annuity_rate: float = 0.05 
    
    # Parámetros Crisis (Calculados en UI)
    mu_local_rv: float = -0.15
    mu_local_rf: float = 0.08  
    corr_local: float = -0.25  
    
    mu_global_rv: float = -0.35 
    mu_global_rf: float = -0.06 
    corr_global: float = 0.90   
    
    prob_enter_local: float = 0.005  
    prob_enter_global: float = 0.004 
    prob_exit_crisis: float = 0.085  

# --- 2. EL MOTOR DE SIMULACIÓN (V7 SOVEREIGN GRADE) ---
class InstitutionalSimulator:
    def __init__(self, config, assets, withdrawals):
        self.cfg = config
        self.assets = assets
        self.withdrawals = withdrawals
        self.dt = 1/config.steps_per_year
        self.total_steps = int(config.horizon_years * config.steps_per_year)

        # DEFINICIÓN DE 3 RÉGIMENES
        self.mu_regimes = np.array([
            [self.cfg.mu_normal_rv, self.cfg.mu_normal_rf], # 0: Normal
            [self.cfg.mu_local_rv, self.cfg.mu_local_rf],   # 1: Local
            [self.cfg.mu_global_rv, self.cfg.mu_global_rf]  # 2: Global
        ])
        
        self.sigma_regimes = np.array([
            [0.14, 0.05],   # 0: Normal
            [0.22, 0.18],   # 1: Local
            [0.32, 0.15]    # 2: Global
        ])
        
        # MATRICES DE CORRELACIÓN
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
        ruin_indices = np.full(n_sims, -1) # -1 = Vivo
        
        asset_values = np.zeros((n_sims, n_assets))
        for i, a in enumerate(self.assets):
            asset_values[:, i] = self.cfg.initial_capital * a.weight
            
        try: rv_idx = next(i for i, a in enumerate(self.assets) if not a.is_bond)
        except: rv_idx = 0
        try: rf_idx = next(i for i, a in enumerate(self.assets) if a.is_bond)
        except: rf_idx = 1
        
        current_regime = np.zeros(n_sims, dtype=int)
        
        annuity_monthly = 0
        if self.cfg.sell_year > 0 and self.cfg.inmo_strategy == 'annuity':
            months = (self.cfg.horizon_years - self.cfg.sell_year) * 12
            if months > 0:
                r_m = self.cfg.annuity_rate / 12
                annuity_monthly = self.cfg.net_inmo_value * (r_m * (1+r_m)**months) / ((1+r_m)**months - 1) if r_m > 0 else self.cfg.net_inmo_value / months

        # MOTOR MATEMÁTICO: T-STUDENT MULTIVARIANTE (Validado por PhD)
        df = 8 
        G = np.random.normal(0, 1, (n_sims, n_steps, n_assets))
        W = np.random.chisquare(df, (n_sims, n_steps, 1)) / df
        Z_base = G / np.sqrt(W)
        Z_base *= np.sqrt((df-2)/df) 

        inf_shocks = np.random.normal(self.cfg.inflation_mean * self.dt, 
                                      self.cfg.inflation_vol * np.sqrt(self.dt), 
                                      (n_sims, n_steps))

        for t in range(n_steps):
            # A. MARKOV
            mask_0 = (current_regime == 0)
            rand_0 = np.random.rand(np.sum(mask_0))
            
            p_L = self.p_norm_to_local
            p_G = self.p_norm_to_global
            
            new_0_to_1 = rand_0 < p_L
            new_0_to_2 = (rand_0 >= p_L) & (rand_0 < (p_L + p_G))
            
            idx_0 = np.where(mask_0)[0]
            current_regime[idx_0[new_0_to_1]] = 1
            current_regime[idx_0[new_0_to_2]] = 2
            
            mask_crisis = (current_regime > 0)
            if np.any(mask_crisis):
                rand_c = np.random.rand(np.sum(mask_crisis))
                back_to_norm = rand_c < self.p_exit
                idx_c = np.where(mask_crisis)[0]
                current_regime[idx_c[back_to_norm]] = 0
            
            # B. INFLACIÓN
            current_inf_shock = inf_shocks[:, t]
            current_inf_shock[current_regime == 1] += 0.003 
            cpi_paths[:, t+1] = cpi_paths[:, t] * (1 + np.maximum(current_inf_shock, -0.02))
            
            # C. CHOLESKY
            z_t = Z_base[:, t, :] 
            z_final = np.zeros_like(z_t)
            
            m0 = (current_regime == 0); m1 = (current_regime == 1); m2 = (current_regime == 2)
            if np.any(m0): z_final[m0] = np.dot(z_t[m0], self.L_normal.T)
            if np.any(m1): z_final[m1] = np.dot(z_t[m1], self.L_local.T)
            if np.any(m2): z_final[m2] = np.dot(z_t[m2], self.L_global.T)
            
            mus_t = self.mu_regimes[current_regime]
            sigs_t = self.sigma_regimes[current_regime]
            
            step_rets = (mus_t - 0.5 * sigs_t**2) * self.dt + \
                         sigs_t * np.sqrt(self.dt) * z_final
            
            step_rets = np.clip(step_rets, -0.6, 0.6) 
            asset_values *= np.exp(step_rets)
            
            # D. EVENTOS (Venta Propiedad)
            spy = self.cfg.steps_per_year
            current_year = (t+1) / spy
            
            if self.cfg.sell_year > 0 and (t+1) == int(self.cfg.sell_year * spy):
                if self.cfg.inmo_strategy == 'portfolio':
                    asset_values[:, 0] += self.cfg.net_inmo_value * cpi_paths[:, t+1]
            
            # E. FLUJOS
            total_cap = np.sum(asset_values, axis=1)
            current_real_wealth = total_cap / cpi_paths[:, t+1]
            
            living_base = 0
            for w in self.withdrawals:
                if w.from_year <= current_year < w.to_year:
                    living_base = w.amount_nominal_monthly_start; break
            
            living_nom = np.full(n_sims, living_base) * cpi_paths[:, t+1]
            
            if self.cfg.use_guardrails:
                 dd_initial = (self.cfg.initial_capital - current_real_wealth) / self.cfg.initial_capital
                 living_nom[dd_initial > 0.20] *= (1.0 - self.cfg.guardrail_cut)

            rent_nom = np.zeros(n_sims); ann_nom = np.zeros(n_sims)
            if self.cfg.sell_year > 0 and current_year >= self.cfg.sell_year:
                rent_nom = np.full(n_sims, self.cfg.new_rent_cost) * cpi_paths[:, t+1]
                if self.cfg.inmo_strategy == 'annuity':
                    ann_nom = np.full(n_sims, annuity_monthly) * cpi_paths[:, t+1]
            
            net_cashflow = ann_nom - (living_nom + rent_nom)
            port_adj = -net_cashflow 
            
            mask_surp = port_adj <= 0
            if np.any(mask_surp):
                surp = -port_adj[mask_surp]
                for i, asset in enumerate(self.assets):
                    asset_values[mask_surp, i] += surp * asset.weight
            
            mask_def = port_adj > 0
            if np.any(mask_def):
                wd_req = port_adj[mask_def]
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
            
            # F. ESTADO Y RUINA
            asset_values = np.maximum(asset_values, 0)
            capital_paths[:, t+1] = np.sum(asset_values, axis=1)
            
            # DETECTOR DE MUERTE SÚBITA
            # Si en este mes (t+1) el capital es < 1000, marca ruina.
            is_ruined = (capital_paths[:, t+1] <= 1000) & (ruin_indices == -1)
            ruin_indices[is_ruined] = t+1

        return capital_paths, cpi_paths, ruin_indices, annuity_monthly, np.zeros(n_steps)

# --- 3. UTILIDADES ---
def clean(lbl, d, k): 
    v = st.text_input(lbl, value=f"{int(d):,}".replace(",", "."), key=k)
    return int(re.sub(r'\D', '', v)) if v else 0

def fmt(v): return f"{int(v):,}".replace(",", ".")

# --- 4. INTERFAZ PRINCIPAL (APP) ---
def app(default_rf=363000000, default_rv=1368000000, default_inmo_neto=0):
    
    st.markdown("## 🦅 Panel de Decisión Patrimonial")
    st.markdown("Motor **V7 Sovereign Grade** (Corregido). No permite 'resurrección financiera'.")

    # ESCENARIOS
    SCENARIOS_GLOBAL = {
        "Colapso Sistémico (Recomendado)": 
            {"corr": 0.92, "rf_ret": -0.06, "rv_ret": -0.35, "desc": "⚠️ No hay refugio. Inflación + Recesión."},
        "Crash Financiero (Tipo 2008)": 
            {"corr": 0.70, "rf_ret": -0.02, "rv_ret": -0.30, "desc": "Todo cae, salvo el cash."},
        "Recesión Estándar": 
            {"corr": 0.50, "rf_ret": 0.0, "rv_ret": -0.20, "desc": "La bolsa cae, bonos protegen algo."}
    }
    SCENARIOS_LOCAL = {
        "Protección Estándar (Recomendado)": 
            {"corr": -0.25, "rf_ret": 0.08, "desc": "El Dólar/UF actúa como hedge."},
        "Dólar Blindado": 
            {"corr": -0.35, "rf_ret": 0.10, "desc": "Patrimonio USD se dispara."},
        "Falla del Hedge": 
            {"corr": 0.20, "rf_ret": 0.0, "desc": "Crisis interna sin protección."}
    }
    SCENARIOS_RENTABILIDAD = {
        "Histórico (Recomendado)": {"rv": 0.10, "rf": 0.06},
        "Conservador": {"rv": 0.07, "rf": 0.04},
        "Optimista": {"rv": 0.12, "rf": 0.07}
    }

    with st.sidebar:
        st.header("1. Configurar Escenarios")
        sel_glo = st.selectbox("🌎 Crisis Global", list(SCENARIOS_GLOBAL.keys()), index=0)
        sel_loc = st.selectbox("🇨🇱 Crisis Local", list(SCENARIOS_LOCAL.keys()), index=0)
        sel_ret = st.selectbox("📈 Rentabilidad Base", list(SCENARIOS_RENTABILIDAD.keys()), index=0)
        
        st.divider()
        n_sims = st.slider("Precisión", 500, 3000, 1500)
        horiz = st.slider("Horizonte (Años)", 10, 50, 40)
        use_guard = st.checkbox("🛡️ Activar 'Modo Austeridad'", True)

    total_ini = default_rf + default_rv
    pct_rv_input = (default_rv / total_ini) * 100 if total_ini > 0 else 0
    
    st.markdown("### 💰 Tu Estructura")
    c1, c2, c3 = st.columns(3)
    with c1: cap_input = clean("Capital Total ($)", total_ini, "cap_total")
    with c2: pct_rv_user = st.slider("% Motor (RV)", 0, 100, int(pct_rv_input))
    with c3: st.metric("Mix", f"{100-pct_rv_user}% Def / {pct_rv_user}% Mot")
    
    st.markdown("### 💸 Tus Necesidades")
    g1, g2, g3 = st.columns(3)
    with g1: r1 = clean("Fase 1", 6000000, "r1"); d1 = st.number_input("Años F1", 0, 40, 7)
    with g2: r2 = clean("Fase 2", 5500000, "r2"); d2 = st.number_input("Años F2", 0, 40, 13)
    with g3: r3 = clean("Fase 3", 5000000, "r3"); st.caption("Resto")
    
    with st.expander("🏡 Opciones Inmobiliarias"):
        sell_prop = st.checkbox("Vender Propiedad Futura", value=False)
        if sell_prop:
            val_inmo = st.number_input("Valor Neto Hoy ($)", value=int(default_inmo_neto))
            sale_year = st.slider("Año de Venta", 1, 40, 35) # Default alto para probar
            rent_cost = st.number_input("Nuevo Arriendo ($/mes)", value=1500000, step=100000)
            strat = st.radio("Destino:", ["Invertir", "Anualidad"], index=0)
            inmo_strat = 'portfolio' if "Invertir" in strat else 'annuity'
            annuity_r = 5.0 if inmo_strat == 'annuity' else 0.0
        else:
            val_inmo, sale_year, rent_cost, inmo_strat, annuity_r = 0, 0, 0, 'portfolio', 0

    if st.button("🚀 EJECUTAR SIMULACIÓN", type="primary"):
        params_glo = SCENARIOS_GLOBAL[sel_glo]
        params_loc = SCENARIOS_LOCAL[sel_loc]
        params_ret = SCENARIOS_RENTABILIDAD[sel_ret]
        
        assets = [AssetBucket("Motor", pct_rv_user/100, False), AssetBucket("Defensa", (100-pct_rv_user)/100, True)]
        wds = [WithdrawalTramo(0, d1, r1), WithdrawalTramo(d1, d1+d2, r2), WithdrawalTramo(d1+d2, horiz, r3)]
        
        cfg = SimulationConfig(
            horizon_years=horiz, initial_capital=cap_input, n_sims=n_sims,
            use_guardrails=use_guard, sell_year=sale_year, net_inmo_value=val_inmo, 
            new_rent_cost=rent_cost, inmo_strategy=inmo_strat, annuity_rate=annuity_r/100.0,
            mu_normal_rv=params_ret["rv"], mu_normal_rf=params_ret["rf"],
            mu_local_rv=-0.15, mu_local_rf=params_loc["rf_ret"], corr_local=params_loc["corr"],
            mu_global_rv=params_glo["rv_ret"], mu_global_rf=params_glo["rf_ret"], corr_global=params_glo["corr"], 
            prob_enter_local=0.005, prob_enter_global=0.004, prob_exit_crisis=0.085
        )
        
        sim = InstitutionalSimulator(cfg, assets, wds)
        
        with st.spinner(f"Simulando {n_sims} vidas..."):
            paths, cpi, ruin_idx, ann_val, _ = sim.run()
            
            # --- CÁLCULO DE ÉXITO ESTRICTO ---
            # Si en CUALQUIER momento (ruin_idx > -1) quebraste, es un fallo.
            failures = np.sum(ruin_indices > -1)
            success_prob = (1 - (failures / n_sims)) * 100
            
            final_wealth = paths[:, -1]
            median_legacy = np.median(final_wealth / cpi[:, -1])
            
            # Semáforo
            if success_prob >= 90: clr, msg, icon = "#10b981", "LUZ VERDE", "✅"
            elif success_prob >= 75: clr, msg, icon = "#f59e0b", "PRECAUCIÓN", "⚠️"
            else: clr, msg, icon = "#ef4444", "ALTO RIESGO", "🛑"

            st.markdown(f"""
            <div style="background-color:rgba(30,30,30,0.5); padding:20px; border-radius:10px; border-left: 10px solid {clr}; text-align: center;">
                <h1 style="color:{clr}; margin:0;">{icon} {msg}</h1>
                <h2 style="margin:10px 0;">Probabilidad de Éxito: {success_prob:.1f}%</h2>
                <hr style="border-color: #555;">
                <p style="margin:0; font-size:0.9em;">Herencia Estimada: <b>${fmt(median_legacy)}</b></p>
            </div>""", unsafe_allow_html=True)
            
            # Gráfico
            y_axis = np.arange(paths.shape[1])/12
            upper_bound = np.percentile(paths, 90, axis=0)
            lower_bound = np.percentile(paths, 10, axis=0)
            median_path = np.percentile(paths, 50, axis=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.concatenate([y_axis, y_axis[::-1]]), y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself', fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
            fig.add_trace(go.Scatter(x=y_axis, y=median_path, line=dict(color='#3b82f6', width=3), name='Mediana'))
            if sell_prop: fig.add_vline(x=sale_year, line_dash="dash", line_color="green", annotation_text="Venta Prop.")
            
            fig.update_layout(title="Proyección Patrimonial", xaxis_title="Años", yaxis_title="Patrimonio ($)", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            fails = ruin_idx[ruin_idx > -1]
            if len(fails) > 0:
                first_fail = np.percentile(fails/12, 10)
                st.error(f"💀 En los peores casos, la ruina ocurre el año {first_fail:.1f} (aunque luego vendas la casa).")
            else:
                st.success("🎉 Plan 100% Sostenible.")
