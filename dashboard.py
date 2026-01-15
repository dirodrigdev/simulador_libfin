import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import json

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control V25 (Master)", layout="wide", page_icon="💎")

# --- ESTILOS CSS (ESTÉTICA V19 + V24) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; margin-bottom: 120px; }
    
    .main-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
    /* Footer Flotante */
    .floating-footer {
        position: fixed; bottom: 0; left: 0; width: 100%;
        background-color: #ffffff; border-top: 3px solid #cbd5e1;
        box-shadow: 0px -4px 12px rgba(0,0,0,0.08); z-index: 9999;
        padding: 12px 0px; display: flex; justify-content: center; align-items: center; gap: 40px;
    }
    .footer-item { text-align: center; min-width: 140px; }
    .footer-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .footer-value { font-size: 1.4rem; font-weight: 800; color: #0f172a; font-family: 'JetBrains Mono'; }
    
    .status-green { color: #16a34a; } 
    .status-yellow { color: #d97706; } 
    .status-red { color: #dc2626; }
    
    /* Plan Z Card */
    .z-card { background-color: #eff6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 4px; }
    .z-highlight { font-weight: bold; color: #1e3a8a; font-size: 1.1rem; }
    
    /* TextArea JSON */
    .stTextArea textarea { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- UTILIDADES DE FORMATO (Rescatado de V19.2) ---
def fmt(valor): 
    """1000000 -> 1.000.000"""
    return f"{int(valor):,}".replace(",", ".")

def parse(texto): 
    """1.000.000 -> 1000000"""
    return int(re.sub(r'\D', '', texto)) if texto else 0

def input_dinero(label, default, key, disabled=False):
    """Input de texto que fuerza formato de miles"""
    val_str = st.text_input(label, value=fmt(default), key=key, disabled=disabled)
    return parse(val_str)

# --- CLASIFICADOR INTELIGENTE (Rescatado de V22/V23) ---
def procesar_gems_json(data_raw):
    try:
        if isinstance(data_raw, dict) and "registros" in data_raw:
             lista = data_raw["registros"][0]["instrumentos"]
             fecha = data_raw["registros"][0].get("fecha_dato", "N/A")
        elif isinstance(data_raw, list):
             lista = data_raw
             fecha = "Lista Manual"
        else: return 0, 0, pd.DataFrame(), "Error Estructura"
    except: return 0, 0, pd.DataFrame(), "Error Lectura"

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

# --- MATEMÁTICA FINANCIERA (Plan Z - V24) ---
def calc_pmt(principal, rate_anual, years):
    if years <= 0: return 0
    r_mensual = rate_anual / 12
    n_meses = years * 12
    if r_mensual == 0: return principal / n_meses
    return principal * (r_mensual * (1 + r_mensual)**n_meses) / ((1 + r_mensual)**n_meses - 1)

# --- MOTOR SIMULACIÓN CORE (V24 - Eventos Flexibles) ---
def simulacion_core(n_sims, months, cap_ini, eventos_dict, g1, d1, g2, d2, g3, 
                   alloc_rv, ret_rv, vol_rv, ret_rf, vol_rf, corr, use_guard, dd_trig, m_cut, drag):
    
    mu_rv, sigma_rv = (ret_rv-drag)/12, vol_rv/np.sqrt(12)
    mu_rf, sigma_rf = (ret_rf-drag)/12, vol_rf/np.sqrt(12)
    
    df = 5 # T-Student
    std_adj = np.sqrt((df - 2) / df)
    z1 = np.random.standard_t(df, (months, n_sims)) * std_adj
    z2 = np.random.standard_t(df, (months, n_sims)) * std_adj
    eps_rf = corr * z1 + np.sqrt(1 - corr**2) * z2
    
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = cap_ini
    peak = np.full(n_sims, cap_ini)
    ruin_month = np.zeros(n_sims)
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * alloc_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - alloc_rv)
        
        mask_alive = wealth[t-1] > 0
        wealth[t, mask_alive] = wealth[t-1, mask_alive] * (1 + ret[mask_alive])
        
        # Eventos Dinámicos
        if t in eventos_dict:
            wealth[t, mask_alive] += eventos_dict[t]
        
        # Gasto
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
        
        # Chequeo Ruina
        new_ruins = (wealth[t] <= 0) & (wealth[t-1] > 0)
        wealth[t, wealth[t] <= 0] = 0
        ruin_month[new_ruins] = t
        
    is_fail = wealth[-1] <= 0
    return wealth, is_fail, ruin_month

# --- SIDEBAR: TODA LA CONFIGURACIÓN ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # 1. FUENTE DE DATOS
    modo = st.radio("Fuente de Datos", ["Ingreso Manual", "Pegar JSON"], horizontal=True)
    json_data = None
    if modo == "Pegar JSON":
        st.caption("Pega el JSON de tu Gems aquí:")
        txt = st.text_area("JSON Raw", height=100)
        if txt: 
            try: json_data = json.loads(txt)
            except: st.error("JSON inválido")

    # 2. MERCADO
    with st.expander("📉 Mercado & Inflación", expanded=False):
        n_sims = st.select_slider("Simulaciones", [1000, 2000, 5000], 2000)
        drag = st.number_input("Costos Anuales (%)", 1.0)/100
        
    # 3. PLAN Z (Configuración para el cálculo final)
    with st.expander("🏠 Configuración Plan Z", expanded=False):
        val_casa = input_dinero("Valor Propiedad ($)", 250000000, "z_val")
        z_arr = input_dinero("Arriendo Futuro ($)", 800000, "z_arr")
        z_tasa = st.number_input("Tasa Anualidad (%)", 4.0)/100
        
    # 4. REGLAS CRISIS
    with st.expander("🌪️ Reglas de Crisis", expanded=False):
        use_g = st.checkbox("Activar Recortes", True)
        dd_t = st.slider("Gatillo (%)", 10, 50, 30)/100
        m_cut = input_dinero("Monto Recorte ($)", 1200000, "z_cut")

# --- UI PRINCIPAL ---
st.title("🛡️ Diego FIRE Control V25")

# 1. CAPITAL (HÍBRIDO: JSON O MANUAL)
cap_final, alloc_final = 0, 0.6
if modo == "Pegar JSON" and json_data:
    cap_calc, alloc_calc, df_audit, f_dato = procesar_gems_json(json_data)
    cap_final, alloc_final = cap_calc, alloc_calc
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.metric("Capital Líquido (Detectado)", f"$ {fmt(cap_final)}")
        st.caption(f"Fuente: JSON ({f_dato}) - Sin Deuda")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.metric("Perfil Riesgo", f"{alloc_final*100:.1f}% RV")
        st.progress(alloc_final)
        st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Ver detalle de cuentas detectadas"):
        st.dataframe(df_audit, use_container_width=True)
else:
    c1, c2 = st.columns([2, 1])
    with c1: 
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        cap_final = input_dinero("💰 Capital Líquido Manual ($)", 1800000000, "c_man")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2: 
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.caption("Asset Allocation")
        alloc_final = st.slider("% RV", 0, 100, 60)/100.0
        st.markdown('</div>', unsafe_allow_html=True)

# 2. EVENTOS (V24)
st.markdown("### 📅 Eventos de Capital")
with st.expander("Configurar Inyecciones o Gastos (Auto, Herencias)", expanded=False):
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    ev1_m = input_dinero("Evento 1 ($)", 0, "e1"); ev1_y = col_ev1.number_input("Año", 10, key="y1")
    ev2_m = input_dinero("Evento 2 ($)", -45000000, "e2"); ev2_y = col_ev2.number_input("Año", 3, key="y2")
    ev3_m = input_dinero("Evento 3 ($)", 0, "e3"); ev3_y = col_ev3.number_input("Año", 20, key="y3")

eventos = {}
for m, y in [(ev1_m, ev1_y), (ev2_m, ev2_y), (ev3_m, ev3_y)]:
    if m != 0: eventos[y*12] = m

# 3. GASTOS (V19.2 Style)
st.markdown("### 💸 Gasto Regular")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g1 = input_dinero("Fase 1 ($)", 6000000, "g1"); d1 = st.number_input("Años F1", 7)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g2 = input_dinero("Fase 2 ($)", 5500000, "g2"); d2 = st.number_input("Años F2", 13)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g3 = input_dinero("Fase 3 ($)", 5000000, "g3"); st.caption("Resto vida")
    st.markdown('</div>', unsafe_allow_html=True)

# --- EJECUCIÓN ---
if st.button("🚀 ANALIZAR PLAN COMPLETO", type="primary", use_container_width=True):
    # Parámetros Base
    r_rv, v_rv = 0.065, 0.18
    r_rf, v_rf = 0.015, 0.05
    corr = 0.8
    
    wealth, is_fail, ruin_months = simulacion_core(
        n_sims, 40*12, cap_final, eventos, g1, d1, g2, d2, g3,
        alloc_final, r_rv, v_rv, r_rf, v_rf, corr, use_g, dd_t, m_cut, drag
    )
    
    # KPIs
    prob = (1 - np.mean(is_fail)) * 100
    herencia = np.median(wealth[-1])
    riesgo = 100 - prob
    
    # FOOTER FLOTANTE (V19.3)
    stat = "status-green" if prob >= 90 else ("status-yellow" if prob >= 75 else "status-red")
    st.markdown(f"""
    <div class="floating-footer">
        <div class="footer-item"><div class="footer-label">Éxito</div><div class="footer-value {stat}">{prob:.1f}%</div></div>
        <div class="footer-item"><div class="footer-label">Riesgo</div><div class="footer-value">{riesgo:.1f}%</div></div>
        <div class="footer-item"><div class="footer-label">Herencia</div><div class="footer-value">${fmt(herencia)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # TABS DIAGNÓSTICO (V24)
    t1, t2, t3 = st.tabs(["📊 Proyección", "🔥 Mapa de Ruina", "🏠 Plan Z"])
    
    with t1:
        p10, p50, p90 = np.percentile(wealth, [10, 50, 90], axis=1)
        x = np.arange(len(p50))/12
        fig = go.Figure([
            go.Scatter(x=x, y=p90, line=dict(width=0), showlegend=False, hoverinfo='skip'),
            go.Scatter(x=x, y=p10, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', name='Rango P10-P90'),
            go.Scatter(x=x, y=p50, line=dict(color='#0f172a', width=3), name='Mediana')
        ])
        fig.update_layout(template="plotly_white", yaxis=dict(tickformat=",.0f", tickprefix="$ "), height=450, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        fails = ruin_months[ruin_months > 0] / 12
        if len(fails) > 0:
            st.warning(f"Se registraron {len(fails)} quiebras en 2000 escenarios.")
            fig_h = px.histogram(x=fails, nbins=20, labels={'x': 'Año de Quiebra'}, title="Distribución Temporal de la Ruina")
            fig_h.update_layout(bargap=0.1, template="plotly_white")
            st.plotly_chart(fig_h, use_container_width=True)
            st.caption("Si las barras están a la izquierda, es peligroso. A la derecha, es longevidad.")
        else:
            st.success("✅ Cero quiebras registradas.")

    with t3:
        st.markdown("#### Simulación de Venta de Propiedad")
        # Año mediano de quiebra
        y_fail = np.median(fails) if len(fails)>0 else 40
        neto = val_casa * 0.96 # 4% costo venta
        years_left = max(1, 40 - y_fail)
        
        # Anualidad
        income_z = calc_pmt(neto, z_tasa, years_left)
        final_money = income_z - z_arr
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("Año Activación (Mediano)", f"Año {y_fail:.1f}")
            st.metric("Capital por Venta Casa", f"$ {fmt(neto)}")
        with c_b:
            st.markdown(f"""
            <div class="z-card">
                <div>+ Anualidad Casa: <b>$ {fmt(income_z)}</b></div>
                <div>- Arriendo Nuevo: <b>$ {fmt(z_arr)}</b></div>
                <hr>
                <div class="z-highlight">Disponible Vida: $ {fmt(final_money)} / mes</div>
            </div>
            """, unsafe_allow_html=True)
