import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control V19.1", layout="wide", page_icon="🏦")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f1f5f9; }
    .main-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #e2e8f0; margin-bottom: 20px; }
    .kpi-card { background: white; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center; border-top: 5px solid #3b82f6; }
    .kpi-val { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 800; color: #0f172a; }
    .kpi-lbl { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; margin-top: 5px; }
    .success { border-top-color: #10b981; }
    .warning { border-top-color: #f59e0b; }
    .danger { border-top-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# --- UTILIDADES DE FORMATO ---
def fmt_puntos(numero):
    """Formatea números con puntos para miles: 1.000.000"""
    return f"{numero:,.0f}".replace(",", ".")

# --- MOTOR MATEMÁTICO ---
def generate_returns(n_sims, n_months, mu, sigma, df=5):
    std_adj = np.sqrt((df - 2) / df) if df > 2 else 1.0
    return mu + sigma * np.random.standard_t(df, (n_months, n_sims)) * std_adj

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración Técnica")
    
    with st.expander("📈 Mercado e Inflación", expanded=False):
        n_sims = st.select_slider("Simulaciones", options=[1000, 2000, 5000], value=2000)
        inflacion_anual = st.number_input("Inflación Anual (%)", value=3.5, step=0.1) / 100
        drag_anual = st.number_input("Costos/Impuestos (%)", value=1.0, step=0.1) / 100
        df_student = st.slider("Grados de Libertad (Crisis)", 3, 20, 5)

    with st.expander("🎯 Riesgo y Retorno", expanded=True):
        alloc_rv = st.slider("% Renta Variable", 0, 100, 60) / 100
        ret_rv = st.number_input("Retorno RV Real (%)", value=6.5, step=0.1) / 100
        vol_rv = st.number_input("Volatilidad RV (%)", value=18.0, step=0.1) / 100
        ret_rf = st.number_input("Retorno RF Real (%)", value=1.5, step=0.1) / 100
        vol_rf = st.number_input("Volatilidad RF (%)", value=5.0, step=0.1) / 100
        correlacion = st.slider("Correlación en Crisis", 0.0, 1.0, 0.8)

    with st.expander("🌪️ Reglas de Crisis", expanded=True):
        use_guardrails = st.checkbox("Activar Recorte de Gasto", value=True)
        dd_trigger = st.slider("Gatillo de Caída (%)", 10, 50, 30) / 100
        monto_recorte = st.number_input("Monto Recorte ($)", value=1200000, step=100000)

# --- CUERPO PRINCIPAL ---
st.title("🛡️ Diego FIRE Control center")

col_cap, col_inj = st.columns(2)
with col_cap:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    # Formateamos el valor inicial con puntos para que sea legible al usuario
    cap_inicial = st.number_input("💰 Capital Líquido Inicial ($)", value=1800000000, step=1000000, format="%d", help="Ingresa el monto sin puntos, se visualizará correctamente al simular.")
    st.caption(f"Valor ingresado: **$ {fmt_puntos(cap_inicial)}**")
    st.markdown('</div>', unsafe_allow_html=True)
with col_inj:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    inj_monto = st.number_input("💉 Inyección Futura ($)", value=0, step=1000000, format="%d")
    inj_anio = st.number_input("Año de Inyección", value=10, min_value=1)
    st.caption(f"Inyección: **$ {fmt_puntos(inj_monto)}**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 💸 Plan de Gasto Mensual (Formatos con puntos)")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g1 = st.number_input("Fase 1: Mensual ($)", value=6000000, step=100000, format="%d")
    d1 = st.number_input("Fase 1: Años", value=7)
    st.caption(f"Gasto: **$ {fmt_puntos(g1)}**")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g2 = st.number_input("Fase 2: Mensual ($)", value=5500000, step=100000, format="%d")
    d2 = st.number_input("Fase 2: Años", value=13)
    st.caption(f"Gasto: **$ {fmt_puntos(g2)}**")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    g3 = st.number_input("Fase 3: Mensual ($)", value=5000000, step=100000, format="%d")
    st.caption(f"Gasto: **$ {fmt_puntos(g3)}**")
    st.markdown('</div>', unsafe_allow_html=True)

# --- SIMULACIÓN ---
if st.button("🚀 EJECUTAR ESCENARIOS", type="primary", use_container_width=True):
    months = 40 * 12
    mu_rv, mu_rf = (ret_rv - drag_anual)/12, (ret_rf - drag_anual)/12
    sigma_rv, sigma_rf = vol_rv/np.sqrt(12), vol_rf/np.sqrt(12)
    
    z1 = generate_returns(n_sims, months, 0, 1, df_student)
    z2 = generate_returns(n_sims, months, 0, 1, df_student)
    eps_rf = correlacion * z1 + np.sqrt(1 - correlacion**2) * z2
    
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = cap_inicial
    is_fail = np.zeros(n_sims, dtype=bool)
    peak = np.full(n_sims, cap_inicial)
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * alloc_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - alloc_rv)
        wealth[t] = wealth[t-1] * (1 + ret)
        if t == inj_anio * 12: wealth[t] += inj_monto
        if t == 36: wealth[t] -= 45000000 # Auto
        
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak = np.maximum(peak, wealth[t])
        if use_guardrails and np.any((peak - wealth[t])/peak > dd_trigger):
            target -= monto_recorte
            
        wealth[t] -= target
        wealth[t] = np.maximum(wealth[t], 0)
        is_fail |= (wealth[t] == 0)

    # --- RESULTADOS ---
    res_liq = (1 - np.mean(is_fail)) * 100
    med_final = np.median(wealth[-1])
    
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi-card {"success" if res_liq > 85 else "warning"}"><div class="kpi-val">{res_liq:.1f}%</div><div class="kpi-lbl">Éxito Líquido</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card danger"><div class="kpi-val">{np.mean(is_fail)*100:.1f}%</div><div class="kpi-lbl">Riesgo Plan Z</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card money"><div class="kpi-val">${fmt_puntos(med_final)}</div><div class="kpi-lbl">Herencia Mediana (P50)</div></div>', unsafe_allow_html=True)

    # Gráfico con puntos en los ejes
    p10, p50, p90 = np.percentile(wealth, [10, 50, 90], axis=1)
    x = np.arange(len(p50))/12
    fig = go.Figure([
        go.Scatter(x=x, y=p90, line=dict(width=0), showlegend=False, hoverinfo='skip'),
        go.Scatter(x=x, y=p10, fill='tonexty', fillcolor='rgba(59,130,246,0.1)', name='Rango P10-P90'),
        go.Scatter(x=x, y=p50, line=dict(color='#0f172a', width=3), name='Escenario Central')
    ])
    fig.update_layout(
        title="Evolución de Patrimonio (Formato con puntos)", 
        template="plotly_white", 
        yaxis=dict(tickformat=",.0f", tickprefix="$ "),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
