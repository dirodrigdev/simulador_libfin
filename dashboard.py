import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import json

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control V26 (Analytics)", layout="wide", page_icon="🔬")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; margin-bottom: 150px; }
    
    .main-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }
    
    /* Footer Flotante */
    .floating-footer {
        position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #ffffff; border-top: 3px solid #64748b;
        box-shadow: 0px -4px 15px rgba(0,0,0,0.1); z-index: 9999;
        padding: 10px 0px; display: flex; justify-content: center; align-items: center; gap: 40px;
    }
    .footer-item { text-align: center; min-width: 140px; }
    .footer-label { font-size: 0.65rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .footer-value { font-size: 1.3rem; font-weight: 800; color: #0f172a; font-family: 'JetBrains Mono'; }
    
    .status-green { color: #16a34a; } .status-yellow { color: #d97706; } .status-red { color: #dc2626; }
    
    /* Tablas Análisis */
    .metric-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }
    .metric-name { font-weight: 600; color: #475569; }
    .metric-val { font-family: 'JetBrains Mono'; color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# --- UTILIDADES ---
def fmt(valor): return f"{int(valor):,}".replace(",", ".")
def parse(texto): return int(re.sub(r'\D', '', texto)) if texto else 0

def input_dinero(label, default, key, disabled=False):
    val_str = st.text_input(label, value=fmt(default), key=key, disabled=disabled)
    return parse(val_str)

# --- CLASIFICADOR JSON ---
def procesar_gems_json(data_raw):
    try:
        if isinstance(data_raw, dict) and "registros" in data_raw:
             lista = data_raw["registros"][0]["instrumentos"]
             fecha = data_raw["registros"][0].get("fecha_dato", "N/A")
        elif isinstance(data_raw, list):
             lista = data_raw
             fecha = "Lista Manual"
        else: return 0, 0, pd.DataFrame(), "Error"
    except: return 0, 0, pd.DataFrame(), "Error"

    total_rv, total_rf, total_clp = 0, 0, 0
    df_rows = []
    kw_rv = ["agresivo", "fondo a", "gestión activa", "moneda renta", "equity", "accion", "etf", "sp500"]
    
    for item in lista:
        nom = item.get("nombre", "").lower()
        tipo = item.get("tipo", "").lower()
        sub = str(item.get("subtipo", "")).lower()
        saldo = item.get("saldo_clp", 0)
        
        if "pasivo" in tipo or "hipotecario" in nom:
             df_rows.append({"Instrumento": item.get("nombre"), "Monto": fmt(saldo), "Categoría": "🔴 PASIVO"})
             continue

        es_rv = any(k in nom or k in sub for k in kw_rv)
        cat = "Renta Variable" if es_rv else "Renta Fija"
        if es_rv: total_rv += saldo
        else: total_rf += saldo
        
        total_clp += saldo
        df_rows.append({"Instrumento": item.get("nombre"), "Monto": fmt(saldo), "Categoría": cat})
        
    pct_rv = total_rv / total_clp if total_clp > 0 else 0
    return total_clp, pct_rv, pd.DataFrame(df_rows), fecha

# --- MATEMÁTICA ANUALIDAD ---
def calc_pmt(pv, r, n):
    if n <= 0: return 0
    r_m = r / 12
    n_m = n * 12
    if r_m == 0: return pv / n_m
    return pv * (r_m * (1 + r_m)**n_m) / ((1 + r_m)**n_m - 1)

# --- MOTOR SIMULACIÓN CORE ---
def simulacion_core(n_sims, months, cap_ini, eventos_dict, g1, d1, g2, d2, g3, 
                   alloc_rv, ret_rv, vol_rv, ret_rf, vol_rf, corr, use_guard, dd_trig, m_cut, drag, inflacion):
    
    # Ajuste de Tasas Reales
    mu_rv, sigma_rv = (ret_rv - inflacion - drag)/12, vol_rv/np.sqrt(12)
    mu_rf, sigma_rf = (ret_rf - inflacion - drag)/12, vol_rf/np.sqrt(12)
    
    df = 5 # T-Student
    std_adj = np.sqrt((df - 2) / df)
    z1 = np.random.standard_t(df, (months, n_sims)) * std_adj
    z2 = np.random.standard_t(df, (months, n_sims)) * std_adj
    eps_rf = corr * z1 + np.sqrt(1 - corr**2) * z2
    
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = cap_ini
    peak = np.full(n_sims, cap_ini)
    ruin_month = np.zeros(n_sims)
    
    # Tracking de supervivencia mensual
    survival_curve = np.ones(months + 1) * 100
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * alloc_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - alloc_rv)
        
        mask_alive = wealth[t-1] > 0
        wealth[t, mask_alive] = wealth[t-1, mask_alive] * (1 + ret[mask_alive])
        
        if t in eventos_dict:
            wealth[t, mask_alive] += eventos_dict[t]
        
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak[mask_alive] = np.maximum(peak[mask_alive], wealth[t, mask_alive])
        
        gasto_real = np.full(n_sims, target)
        if use_guard:
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = (peak - wealth[t]) / peak
                dd = np.nan_to_num(dd)
            mask_crisis = dd > dd_trig
            gasto_real[mask_crisis] -= m_cut
            
        wealth[t, mask_alive] -= gasto_real[mask_alive]
        
        new_ruins = (wealth[t] <= 0) & (wealth[t-1] > 0)
        wealth[t, wealth[t] <= 0] = 0
        ruin_month[new_ruins] = t
        
        # Calcular supervivencia al cierre del mes
        survival_curve[t] = (np.sum(wealth[t] > 0) / n_sims) * 100
        
    is_fail = wealth[-1] <= 0
    return wealth, is_fail, ruin_month, survival_curve

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio("Fuente de Datos", ["Manual", "Pegar JSON"], horizontal=True)
    json_data = None
    if modo == "Pegar JSON":
        txt = st.text_area("JSON Raw", height=100)
        if txt: 
            try: json_data = json.loads(txt)
            except: st.error("JSON inválido")

    with st.expander("📉 Variables Macroeconómicas", expanded=True):
        n_sims = st.select_slider("Sims", [1000, 2000, 5000], 2000)
        inflacion = st.number_input("Inflación (%)", 3.0)/100
        drag = st.number_input("Costos/Impuestos (%)", 1.0)/100
        
    with st.expander("🏠 Plan Z", expanded=False):
        val_casa = input_dinero("Valor Casa ($)", 250000000, "z_v")
        z_arr = input_dinero("Arriendo Futuro ($)", 800000, "z_a")
        z_tasa = st.number_input("Tasa Anualidad (%)", 4.0)/100

    with st.expander("🌪️ Reglas Crisis", expanded=False):
        use_g = st.checkbox("Activar Recortes", True)
        dd_t = st.slider("Gatillo (%)", 10, 50, 30)/100
        m_cut = input_dinero("Monto Recorte ($)", 1200000, "cut")

# --- UI PRINCIPAL ---
st.title("🛡️ Diego FIRE Control V26")

# 1. CAPITAL
cap_final, alloc_final = 0, 0.6
if modo == "Pegar JSON" and json_data:
    cap_calc, alloc_calc, df_audit, f_dat = procesar_gems_json(json_data)
    cap_final, alloc_final = cap_calc, alloc_calc
    st.success(f"Datos Cargados: $ {fmt(cap_final)} ({alloc_final*100:.1f}% RV)")
else:
    c1, c2 = st.columns([2, 1])
    with c1: cap_final = input_dinero("💰 Capital Líquido ($)", 1800000000, "c_man")
    with c2: alloc_final = st.slider("% RV", 0, 100, 60)/100.0

# 2. EVENTOS
st.markdown("### 📅 Eventos de Capital")
with st.expander("Ver / Editar Eventos (Auto, Herencia)", expanded=False):
    c_e1, c_e2 = st.columns(2)
    ev1_m = input_dinero("Inyección ($)", 0, "ev1"); ev1_y = c_e1.number_input("Año Iny.", 10)
    ev2_m = input_dinero("Gasto Extra ($)", -45000000, "ev2"); ev2_y = c_e2.number_input("Año Gasto", 3)

eventos = {}
if ev1_m!=0: eventos[ev1_y*12]=ev1_m
if ev2_m!=0: eventos[ev2_y*12]=ev2_m

# 3. GASTOS
st.markdown("### 💸 Gasto Regular")
c1, c2, c3 = st.columns(3)
with c1: g1 = input_dinero("Fase 1 ($)", 6000000, "g1"); d1 = st.number_input("Años F1", 7)
with c2: g2 = input_dinero("Fase 2 ($)", 5500000, "g2"); d2 = st.number_input("Años F2", 13)
with c3: g3 = input_dinero("Fase 3 ($)", 5000000, "g3")

# --- INDICADORES "DÓNDE ESTOY" ---
swr = (g1 * 12) / cap_final * 100 if cap_final > 0 else 0
runway = cap_final / (g1 * 12) if g1 > 0 else 0

st.markdown("#### 🧭 Tu Ubicación Actual")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tasa de Retiro (SWR)", f"{swr:.2f}%", help="Regla general segura es < 4%")
k2.metric("Pista de Aterrizaje", f"{runway:.1f} Años", help="Años que dura el dinero sin ninguna rentabilidad (colchón neto)")
k3.metric("Capital Objetivo 4%", f"$ {fmt(g1*12*25)}", help="Capital necesario para que el retiro sea el 4%")
diff = cap_final - (g1*12*25)
k4.metric("Brecha FIRE", f"$ {fmt(diff)}", delta_color="normal")

# --- EJECUCIÓN ---
if st.button("🚀 INICIAR DIAGNÓSTICO PROFUNDO", type="primary", use_container_width=True):
    # Parametros Base
    r_rv, v_rv = 0.065, 0.18
    r_rf, v_rf = 0.015, 0.05
    corr_base = 0.8
    
    # 1. SIMULACIÓN BASE
    wealth, is_fail, ruin_months, survival_curve = simulacion_core(
        n_sims, 40*12, cap_final, eventos, g1, d1, g2, d2, g3,
        alloc_final, r_rv, v_rv, r_rf, v_rf, corr_base, use_g, dd_t, m_cut, drag, inflacion
    )
    
    prob = (1 - np.mean(is_fail)) * 100
    herencia = np.median(wealth[-1])
    
    # Footer
    c_s = "status-green" if prob >= 90 else "status-red"
    st.markdown(f"""<div class="floating-footer">
        <div class="footer-item"><div class="footer-label">Éxito</div><div class="footer-value {c_s}">{prob:.1f}%</div></div>
        <div class="footer-item"><div class="footer-label">Herencia</div><div class="footer-value">${fmt(herencia)}</div></div>
    </div>""", unsafe_allow_html=True)

    # --- PESTAÑAS ANALÍTICAS ---
    t1, t2, t3, t4 = st.tabs(["📊 Proyección", "💀 Riesgo Temporal", "🌪️ Sensibilidad", "🏠 Plan Z"])
    
    with t1:
        p10, p50, p90 = np.percentile(wealth, [10, 50, 90], axis=1)
        x = np.arange(len(p50))/12
        fig = go.Figure([
            go.Scatter(x=x, y=p90, line=dict(width=0), showlegend=False),
            go.Scatter(x=x, y=p10, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', name='Rango'),
            go.Scatter(x=x, y=p50, line=dict(color='#0f172a', width=3), name='Mediana')
        ])
        fig.update_layout(template="plotly_white", yaxis=dict(tickformat=",.0f", tickprefix="$ "), height=450, title="Evolución Patrimonio")
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        # SUPERVIVENCIA + HISTOGRAMA
        c_risk1, c_risk2 = st.columns([2, 1])
        
        with c_risk1:
            # Curva de Supervivencia
            st.markdown("##### 📉 Curva de Supervivencia")
            df_surv = pd.DataFrame({"Mes": np.arange(len(survival_curve)), "Año": np.arange(len(survival_curve))/12, "Probabilidad": survival_curve})
            fig_s = px.line(df_surv, x="Año", y="Probabilidad", range_y=[0, 105])
            fig_s.add_hline(y=90, line_dash="dot", line_color="green", annotation_text="Zona Segura (90%)")
            fig_s.add_hline(y=75, line_dash="dot", line_color="orange", annotation_text="Zona Alerta (75%)")
            st.plotly_chart(fig_s, use_container_width=True)
            
        with c_risk2:
            st.markdown("##### 📍 ¿Cuándo empieza el peligro?")
            # Buscar el primer mes donde la supervivencia baja del 95% y del 80%
            idx_95 = np.argmax(survival_curve < 95)
            idx_80 = np.argmax(survival_curve < 80)
            
            if survival_curve[-1] > 95:
                st.success("✅ **Riesgo Nulo:** Tu plan no baja del 95% de éxito en todo el periodo.")
            else:
                year_risk = idx_95 / 12
                st.warning(f"⚠️ **Inicio del Riesgo:** Año {year_risk:.1f}")
                st.caption("Aquí la probabilidad de éxito cae bajo el 95%.")
                
                if survival_curve[-1] < 80:
                    year_crit = idx_80 / 12
                    st.error(f"🚨 **Zona Crítica:** Año {year_crit:.1f}")
                    st.caption("Probabilidad cae bajo el 80%.")

    with t3:
        st.markdown("#### 🌪️ Ranking de Variables: ¿Qué afecta más a tu plan?")
        st.caption("Comparamos tu escenario base contra escenarios estresados (Simulaciones rápidas de 500 iteraciones).")
        
        # Escenarios
        scenarios = [
            ("Base", 0, 0, 0),
            ("Inflación (+1.5%)", 0.015, 0, 0),
            ("Bolsa (-1.5% Retorno)", 0, -0.015, 0),
            ("Gastos (+10%)", 0, 0, 1.10)
        ]
        
        results_sens = []
        bar = st.progress(0)
        
        # Ejecución rápida
        for i, (name, d_inf, d_ret, d_spend) in enumerate(scenarios):
            # Modificadores
            inf_test = inflacion + d_inf
            ret_rv_test = r_rv + d_ret
            # Gasto
            g1_t = g1 * d_spend if d_spend > 0 else g1
            g2_t = g2 * d_spend if d_spend > 0 else g2
            g3_t = g3 * d_spend if d_spend > 0 else g3
            
            _, is_fail_s, _, _ = simulacion_core(
                500, 40*12, cap_final, eventos, g1_t, d1, g2_t, d2, g3_t,
                alloc_final, ret_rv_test, v_rv, r_rf, v_rf, corr_base, use_g, dd_t, m_cut, drag, inf_test
            )
            prob_s = (1 - np.mean(is_fail_s)) * 100
            results_sens.append({"Escenario": name, "Éxito": prob_s})
            bar.progress((i+1)/4)
            
        base_val = results_sens[0]["Éxito"]
        
        # Tabla de Deltas
        for res in results_sens[1:]:
            delta = res["Éxito"] - base_val
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-name">{res['Escenario']}</span>
                <span class="metric-val" style="color: {'red' if delta < -5 else 'orange'};">
                    {res['Éxito']:.1f}% (Impacto: {delta:.1f} pp)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
        # Conclusión Sensibilidad
        deltas = [r["Éxito"] - base_val for r in results_sens[1:]]
        worst_idx = np.argmin(deltas)
        worst_name = results_sens[1+worst_idx]["Escenario"]
        st.info(f"💡 **Conclusión:** Tu plan es más sensible a: **{worst_name}**.")

    with t4:
        # PLAN Z (Cálculo)
        fails = ruin_months[ruin_months > 0] / 12
        y_fail = np.percentile(fails, 20) if len(fails)>0 else 40 # Usamos P20 para ser conservadores
        
        neto = val_casa * 0.96
        years_left = max(1, 40 - y_fail)
        income_z = calc_pmt(neto, z_tasa, years_left)
        final_money = income_z - z_arr
        
        c_z1, c_z2 = st.columns(2)
        with c_z1:
            st.metric("Año Activación (P20)", f"Año {y_fail:.1f}")
            st.metric("Capital Venta", f"$ {fmt(neto)}")
        with c_z2:
            st.metric("Disponible para Vivir", f"$ {fmt(final_money)}", delta="Mensual")
            if final_money < 1000000:
                st.error("Plan Z Insuficiente")
            else:
                st.success("Plan Z Viable")
