import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import json

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control V23", layout="wide", page_icon="🛡️")

# --- ESTILOS CSS ---
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
    .footer-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 700; }
    .footer-value { font-size: 1.4rem; font-weight: 800; color: #0f172a; font-family: 'JetBrains Mono'; }
    .status-green { color: #16a34a; } .status-yellow { color: #d97706; } .status-red { color: #dc2626; }
    
    /* Area de Texto para JSON */
    .stTextArea textarea { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# --- UTILIDADES DE FORMATO ---
def fmt(valor): return f"{int(valor):,}".replace(",", ".")
def parse(texto): return int(re.sub(r'\D', '', texto)) if texto else 0

def input_dinero(label, default, key, disabled=False):
    val_str = st.text_input(label, value=fmt(default), key=key, disabled=disabled)
    return parse(val_str)

# --- CLASIFICADOR INTELIGENTE (JSON) ---
def procesar_gems_json(data_raw):
    """
    Procesa el JSON pegado. Robusto a errores de estructura.
    """
    try:
        # Detectar estructura (si es lista directa o dict con 'registros')
        if isinstance(data_raw, dict) and "registros" in data_raw:
             lista_instrumentos = data_raw["registros"][0]["instrumentos"]
             fecha_dato = data_raw["registros"][0].get("fecha_dato", "N/A")
        elif isinstance(data_raw, list):
             lista_instrumentos = data_raw
             fecha_dato = "Manual/Lista"
        else:
             return 0, 0, pd.DataFrame(), "Formato No Reconocido"
    except:
        return 0, 0, pd.DataFrame(), "Error Estructura"

    total_rv = 0
    total_rf = 0
    total_clp_liquido = 0
    df_rows = []
    
    # Palabras clave
    kw_rv = ["agresivo", "fondo a", "gestión activa", "moneda renta", "equity", "accion", "etf", "sp500"]
    
    for item in lista_instrumentos:
        nombre = item.get("nombre", "").lower()
        tipo = item.get("tipo", "").lower()
        subtipo = str(item.get("subtipo", "")).lower()
        saldo = item.get("saldo_clp", 0)
        
        # 1. FILTRO DE PASIVOS
        if "pasivo" in tipo or "hipotecario" in nombre:
            df_rows.append({"Instrumento": item.get("nombre"), "Monto": fmt(saldo), "Categoría": "🔴 PASIVO (Excluido)"})
            continue

        # 2. CLASIFICACIÓN
        es_rv = False
        if any(k in nombre or k in subtipo for k in kw_rv): es_rv = True
        
        if es_rv:
            total_rv += saldo
            cat = "Renta Variable"
        else:
            total_rf += saldo
            cat = "Renta Fija"
            if "dolar" in nombre or "usd" in nombre: cat = "Caja/RF USD"
        
        total_clp_liquido += saldo
        df_rows.append({"Instrumento": item.get("nombre"), "Monto": fmt(saldo), "Categoría": cat})
        
    pct_rv = total_rv / total_clp_liquido if total_clp_liquido > 0 else 0
    return total_clp_liquido, pct_rv, pd.DataFrame(df_rows), fecha_dato

# --- MOTOR MATEMÁTICO ---
def simulacion_core(n_sims, months, cap_ini, inj_m, inj_a, g1, d1, g2, d2, g3, 
                   alloc_rv, ret_rv, vol_rv, ret_rf, vol_rf, corr, use_guard, dd_trig, m_cut, drag):
    
    mu_rv, sigma_rv = (ret_rv-drag)/12, vol_rv/np.sqrt(12)
    mu_rf, sigma_rf = (ret_rf-drag)/12, vol_rf/np.sqrt(12)
    
    df = 5
    std_adj = np.sqrt((df - 2) / df)
    z1 = np.random.standard_t(df, (months, n_sims)) * std_adj
    z2 = np.random.standard_t(df, (months, n_sims)) * std_adj
    eps_rf = corr * z1 + np.sqrt(1 - corr**2) * z2
    
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = cap_ini
    peak = np.full(n_sims, cap_ini)
    is_fail = np.zeros(n_sims, dtype=bool)
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * alloc_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - alloc_rv)
        wealth[t] = wealth[t-1] * (1 + ret)
        
        if t == inj_a * 12: wealth[t] += inj_m
        if t == 36: wealth[t] -= 45000000 
        
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak = np.maximum(peak, wealth[t])
        
        gasto_real = np.full(n_sims, target)
        if use_guard and np.any((peak - wealth[t])/peak > dd_trig):
            mask = (peak - wealth[t])/peak > dd_trig
            gasto_real[mask] -= m_cut
            
        wealth[t] -= gasto_real
        wealth[t] = np.maximum(wealth[t], 0)
        is_fail |= (wealth[t] == 0)
        
    return wealth, is_fail

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # SELECCIÓN DE MODO
    modo_ingreso = st.radio("Fuente de Datos", ["Ingreso Manual", "Pegar JSON (Gems)"], index=0)
    
    json_data = None
    if modo_ingreso == "Pegar JSON (Gems)":
        st.info("Pega aquí el contenido de tu archivo JSON:")
        txt_json = st.text_area("JSON Raw", height=200, placeholder='{"registros": [...] }')
        if txt_json:
            try:
                json_data = json.loads(txt_json)
                st.success("JSON procesado correctamente")
            except:
                st.warning("El texto no es un JSON válido aún.")

    with st.expander("📉 Mercado", expanded=False):
        n_sims = st.select_slider("Simulaciones", [1000, 2000, 5000], value=2000)
        drag = st.number_input("Costos (%)", 1.0) / 100
    with st.expander("🌪️ Reglas de Crisis", expanded=False):
        use_g = st.checkbox("Recortar Gasto", True)
        dd_t = st.slider("Gatillo Caída (%)", 10, 50, 30) / 100
        m_cut = input_dinero("Monto Recorte ($)", 1200000, "cut_k")
    with st.expander("📊 Parametros RV/RF", expanded=False):
        ret_rv = st.number_input("Retorno RV (%)", 6.5)/100
        ret_rf = st.number_input("Retorno RF (%)", 1.5)/100
        vol_rv = st.number_input("Volatilidad RV (%)", 18.0)/100
        vol_rf = st.number_input("Volatilidad RF (%)", 5.0)/100
        corr = st.number_input("Correlación", 0.8)

# --- UI PRINCIPAL ---
st.title("🛡️ Diego FIRE Control V23")

# LÓGICA DE DATOS
cap_final = 0
alloc_final = 0.6

if modo_ingreso == "Pegar JSON (Gems)" and json_data:
    # MODO AUTOMÁTICO
    cap_calc, alloc_calc, df_audit, fecha_json = procesar_gems_json(json_data)
    cap_final = cap_calc
    alloc_final = alloc_calc
    
    st.success(f"📂 **Datos Procesados**: Se detectaron instrumentos del **{fecha_json}**.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.metric("Capital Líquido (Sin Deuda)", f"$ {fmt(cap_final)}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.metric("Perfil de Riesgo (RV)", f"{alloc_final*100:.1f}%")
        st.progress(alloc_final)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with st.expander("🔍 Auditoría de Clasificación (Detalle)"):
        st.dataframe(df_audit, use_container_width=True)
        
else:
    # MODO MANUAL
    col_cap, col_risk = st.columns(2)
    with col_cap:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        cap_final = input_dinero("💰 Capital Líquido Total ($)", 1800000000, "cap_manual")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_risk:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.caption("⚖️ Asset Allocation")
        alloc_final = st.slider("% Renta Variable", 0, 100, 60, key="slider_manual") / 100.0
        st.markdown('</div>', unsafe_allow_html=True)

# GASTOS Y OTROS
st.markdown("### 💸 Plan de Gasto")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g1 = input_dinero("Fase 1 ($)", 6000000, "g1")
    d1 = st.number_input("Años Fase 1", 7)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g2 = input_dinero("Fase 2 ($)", 5500000, "g2")
    d2 = st.number_input("Años Fase 2", 13)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g3 = input_dinero("Fase 3 ($)", 5000000, "g3")
    st.caption("Resto de la vida")
    st.markdown('</div>', unsafe_allow_html=True)

# --- EJECUCIÓN ---
if st.button("🚀 EJECUTAR ANÁLISIS", type="primary", use_container_width=True):
    inj_m = 0 
    inj_a = 10
    
    wealth, is_fail = simulacion_core(n_sims, 40*12, cap_final, inj_m, inj_a, g1, d1, g2, d2, g3,
                                     alloc_final, ret_rv, vol_rv, ret_rf, vol_rf, corr, use_g, dd_t, m_cut, drag)
    
    prob = (1 - np.mean(is_fail)) * 100
    herencia = np.median(wealth[-1])
    riesgo = np.mean(is_fail) * 100
    
    # FOOTER
    color = "status-green" if prob >= 90 else ("status-yellow" if prob >= 75 else "status-red")
    st.markdown(f"""
    <div class="floating-footer">
        <div class="footer-item"><div class="footer-label">Éxito</div><div class="footer-value {color}">{prob:.1f}%</div></div>
        <div class="footer-item"><div class="footer-label">Herencia P50</div><div class="footer-value">${fmt(herencia)}</div></div>
    </div>
    """, unsafe_allow_html=True)

    # GRÁFICO
    p10, p50, p90 = np.percentile(wealth, [10, 50, 90], axis=1)
    x = np.arange(len(p50))/12
    fig = go.Figure([
        go.Scatter(x=x, y=p90, line=dict(width=0), showlegend=False),
        go.Scatter(x=x, y=p10, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', name='Rango'),
        go.Scatter(x=x, y=p50, line=dict(color='#0f172a', width=3), name='Mediana')
    ])
    fig.update_layout(template="plotly_white", yaxis=dict(tickformat=",.0f", tickprefix="$ "), hovermode="x unified", height=400)
    st.plotly_chart(fig, use_container_width=True)
