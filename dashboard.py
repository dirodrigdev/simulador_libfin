import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import json
from datetime import date

# --- 1. CONFIGURACIÓN PREMIUM ---
st.set_page_config(page_title="Diego Family Office", layout="wide", page_icon="🏛️")

# --- 2. ESTÉTICA "BANKING GRADE" (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700;800&family=Roboto+Mono:wght@400;500&display=swap');
    
    /* General */
    body { font-family: 'Manrope', sans-serif; background-color: #f4f6f9; color: #1e293b; margin-bottom: 100px; }
    h1, h2, h3 { color: #0f172a; font-weight: 800; letter-spacing: -0.5px; }
    
    /* Tarjetas Métricas (KPIs) */
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }
    .metric-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 700; margin-bottom: 8px; }
    .metric-value { font-family: 'Manrope', sans-serif; font-size: 2rem; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; display: flex; align-items: center; gap: 5px; }
    
    /* Colores Semánticos */
    .positive { color: #10b981; } .negative { color: #ef4444; } .neutral { color: #f59e0b; }
    
    /* Inputs */
    .stTextInput input { font-family: 'Roboto Mono', monospace; font-weight: 500; }
    
    /* Footer Fijo Elegante */
    .executive-footer {
        position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.95); color: white; backdrop-filter: blur(10px);
        padding: 12px 40px; border-radius: 50px; display: flex; gap: 40px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); z-index: 9999;
        white-space: nowrap;
    }
    .footer-stat { text-align: center; }
    .footer-stat span { display: block; font-size: 0.7rem; opacity: 0.7; text-transform: uppercase; }
    .footer-stat strong { font-size: 1.2rem; font-weight: 700; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --- 3. UTILIDADES INTELIGENTES ---
def fmt(v): return f"{int(v):,}".replace(",", ".")
def parse(t): return int(re.sub(r'\D', '', t)) if t else 0
def input_dinero(lbl, d, k): return parse(st.text_input(lbl, value=fmt(d), key=k))

# --- 4. MOTOR ACTUARIAL & FINANCIERO ---
def procesar_gems(json_raw):
    """Interpreta la cartera automáticamente."""
    try:
        # Detectar estructura (lista directa o diccionario con registros)
        if isinstance(json_raw, dict) and "registros" in json_raw:
            data = json_raw["registros"][0]["instrumentos"]
        elif isinstance(json_raw, list):
            data = json_raw
        else:
            return 0, 0, pd.DataFrame()
    except: return 0, 0, pd.DataFrame()

    total_rv, total_rf, total_liq = 0, 0, 0
    rows = []
    # Palabras clave ampliadas según tu JSON real
    kw_rv = ["agresivo", "fondo a", "equity", "accion", "etf", "sp500", "nasdaq", "gestión activa", "moneda renta"]
    
    for i in data:
        nom = i.get("nombre", "").lower()
        tipo = i.get("tipo", "").lower()
        sub = str(i.get("subtipo", "")).lower()
        saldo = i.get("saldo_clp", 0)
        
        if "pasivo" in tipo or "hipotecario" in nom: continue # Ignorar deuda
        
        # Lógica de clasificación
        es_rv = any(k in nom or k in sub for k in kw_rv)
        cat = "Renta Variable" if es_rv else "Renta Fija"
        
        if es_rv: total_rv += saldo
        else: total_rf += saldo
        total_liq += saldo
        rows.append({"Activo": i.get("nombre"), "Monto": fmt(saldo), "Clase": cat})
        
    return total_liq, (total_rv/total_liq if total_liq else 0), pd.DataFrame(rows)

def proyeccion_montecarlo(n_sims, months, cap, g1, d1, g2, d2, g3, pct_rv, params):
    # Desempaquetar
    mu_rv, sig_rv = (params['r_rv'] - params['inf'] - params['cost'])/12, params['v_rv']/np.sqrt(12)
    mu_rf, sig_rf = (params['r_rf'] - params['inf'] - params['cost'])/12, params['v_rf']/np.sqrt(12)
    
    # Motor T-Student (Cisnes Negros)
    df = 5
    std_adj = np.sqrt((df-2)/df)
    z1 = np.random.standard_t(df, (months, n_sims)) * std_adj
    z2 = np.random.standard_t(df, (months, n_sims)) * std_adj
    eps = params['corr'] * z1 + np.sqrt(1 - params['corr']**2) * z2
    
    # Trayectorias
    wealth = np.zeros((months+1, n_sims))
    wealth[0] = cap
    ruin_idx = np.full(n_sims, -1) # -1 significa "no quebró"
    
    peak = np.full(n_sims, cap)
    
    for t in range(1, months+1):
        # Retornos Mercado
        ret = (mu_rv + sig_rv * z1[t-1]) * pct_rv + (mu_rf + sig_rf * eps[t-1]) * (1 - pct_rv)
        
        # Flujo de Caja
        mask_ok = wealth[t-1] > 0
        wealth[t, mask_ok] = wealth[t-1, mask_ok] * (1 + ret[mask_ok])
        
        # Gasto Dinámico
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak[mask_ok] = np.maximum(peak[mask_ok], wealth[t, mask_ok])
        
        # Regla de Crisis (Guardrail)
        if params['guardrail']:
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = (peak - wealth[t]) / peak
                dd = np.nan_to_num(dd)
            
            mask_cut = (dd > 0.30) & mask_ok
            current_g = np.full(n_sims, target)
            current_g[mask_cut] -= 1200000 # Recorte duro
            wealth[t, mask_ok] -= current_g[mask_ok]
        else:
            wealth[t, mask_ok] -= target

        # Registro de Ruina
        just_died = (wealth[t] <= 0) & (wealth[t-1] > 0)
        wealth[t, wealth[t] < 0] = 0
        ruin_idx[just_died] = t
        
    return wealth, ruin_idx

# --- 5. INTERFAZ DE USUARIO (DASHBOARD) ---

# HEADER
c_logo, c_title = st.columns([1, 6])
with c_logo: st.markdown("## 🏛️")
with c_title: st.markdown("# Diego Family Office \n ### Informe de Solvencia Patrimonial")

# SIDEBAR: DATOS & SUPUESTOS
with st.sidebar:
    st.markdown("### 1. Perfil del Cliente")
    # CORRECCIÓN AQUÍ: Rango ampliado desde 1950 y default 1978
    birth_year = st.number_input("Año de Nacimiento", 1950, 2010, 1978)
    current_year = date.today().year
    age = current_year - birth_year
    life_expectancy = st.slider("Esperanza de Vida (Planificación)", 85, 100, 95)
    horizon_years = life_expectancy - age
    
    st.info(f"👤 **Edad:** {age} años \n\n 🏁 **Horizonte:** {horizon_years} años (hasta {current_year + horizon_years})")

    st.markdown("### 2. Origen de Datos")
    source = st.radio("Fuente", ["Pegar JSON (Gems)", "Manual"], label_visibility="collapsed")
    
    cap_liq, pct_rv = 0, 0.6
    
    if source == "Pegar JSON (Gems)":
        txt = st.text_area("JSON Raw", height=150, placeholder="Pega el JSON de Gems aquí...")
        if txt:
            try:
                c, p, df = procesar_gems(json.loads(txt))
                cap_liq, pct_rv = c, p
                if cap_liq > 0:
                    st.success(f"✅ Carteras Procesadas ($ {fmt(cap_liq)})")
                else:
                    st.warning("JSON válido pero sin saldo líquido detectado")
            except: st.error("❌ Error en formato JSON")
    else:
        cap_liq = input_dinero("Capital Líquido ($)", 1800000000, "man_cap")
        pct_rv = st.slider("% Renta Variable", 0, 100, 60)/100.0

    st.markdown("### 3. Supuestos Macro")
    with st.expander("Ver Parámetros Avanzados"):
        inf = st.number_input("Inflación Est. (%)", 3.0)/100
        cost = st.number_input("Costos Gestión (%)", 1.0)/100
        guard = st.checkbox("Activar 'Guardrails' (Recorte en Crisis)", True)

# MAIN: TARJETAS SUPERIORES
st.markdown("---")
col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    st.markdown("#### 💸 Fase 1: Lifestyle Actual")
    g1 = input_dinero("Gasto Mensual ($)", 6000000, "g1")
    d1 = st.number_input("Duración (Años)", 7, key="d1")

with col_g2:
    st.markdown("#### 📉 Fase 2: Transición")
    g2 = input_dinero("Gasto Mensual ($)", 5500000, "g2")
    d2 = st.number_input("Duración (Años)", 13, key="d2")

with col_g3:
    st.markdown("#### 👴 Fase 3: Madurez (Salud)")
    g3 = input_dinero("Gasto Mensual ($)", 5000000, "g3")
    st.caption(f"Desde los {age + d1 + d2} años en adelante")

# EJECUCIÓN
if cap_liq > 0:
    # Parametros Mercado
    params = {
        'r_rv': 0.075, 'v_rv': 0.18, # 7.5% retorno, 18% volatilidad
        'r_rf': 0.035, 'v_rf': 0.05, # 3.5% retorno, 5% volatilidad
        'corr': 0.8, 'inf': inf, 'cost': cost, 'guardrail': guard
    }
    
    # Simulación
    sims = 2000
    months = horizon_years * 12
    # CORRECCIÓN: Asegurar que months > 0
    if months <= 0: months = 12 
    
    paths, ruin_idx = proyeccion_montecarlo(sims, months, cap_liq, g1, d1, g2, d2, g3, pct_rv, params)
    
    # KPIs
    success_rate = (np.sum(ruin_idx == -1) / sims) * 100
    median_legacy = np.median(paths[-1])
    
    # Análisis de "Zona de Muerte"
    ruin_years = (ruin_idx[ruin_idx > -1] / 12) + age 
    
    risk_start_age = "N/A"
    most_dangerous_age = "N/A"
    
    if len(ruin_years) > 0:
        risk_start_age = f"{np.percentile(ruin_years, 10):.0f} Años"
        most_dangerous_age = f"{int(np.median(ruin_years))} Años"

    # --- RENDERIZADO VISUAL ---
    
    # 1. TARJETAS DE IMPACTO
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Solvencia del Plan</div>
            <div class="metric-value {'positive' if success_rate > 90 else 'negative'}">{success_rate:.1f}%</div>
            <div class="metric-sub">Probabilidad de Éxito</div>
        </div>""", unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Capital Líquido</div>
            <div class="metric-value">${fmt(cap_liq/1e6)}M</div>
            <div class="metric-sub">Disponible Hoy</div>
        </div>""", unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Herencia (P50)</div>
            <div class="metric-value neutral">${fmt(median_legacy/1e6)}M</div>
            <div class="metric-sub">A los {age + horizon_years} años</div>
        </div>""", unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Zona de Riesgo</div>
            <div class="metric-value negative">{risk_start_age}</div>
            <div class="metric-sub">Edad de 1er fallo</div>
        </div>""", unsafe_allow_html=True)

    # 2. GRÁFICO
    st.markdown("### 🔭 Proyección Patrimonial")
    
    years_axis = np.arange(months + 1) / 12 + age
    p10 = np.percentile(paths, 10, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p90 = np.percentile(paths, 90, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([years_axis, years_axis[::-1]]),
        y=np.concatenate([p90, p10[::-1]]),
        fill='toself', fillcolor='rgba(148, 163, 184, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Rango 80%', hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(x=years_axis, y=p50, line=dict(color='#0f172a', width=4), name='Mediana'))
    fig.add_shape(type="line", x0=age, x1=age+horizon_years, y0=0, y1=0, line=dict(color="#ef4444", width=2, dash="dash"))
    
    fig.update_layout(
        template="plotly_white", height=500, hovermode="x unified",
        xaxis=dict(title="Tu Edad", showgrid=False),
        yaxis=dict(title="Patrimonio ($)", showgrid=True, gridcolor='#f1f5f9', tickformat=",.0f"),
        legend=dict(orientation="h", y=1.02, x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. EXECUTIVE SUMMARY
    st.markdown("### 📝 Executive Summary")
    
    analysis_text = f"""
    **Estimado Diego:**
    
    Basado en los parámetros actuales, tu cartera de **${fmt(cap_liq)}** tiene una probabilidad del **{success_rate:.1f}%** de sostener tu estilo de vida hasta los **{age + horizon_years} años**.
    """
    
    if success_rate > 95:
        analysis_text += f"\n\n🟢 **Conclusión:** El plan es **excesivamente sólido**. Tienes capital excedente. Podrías aumentar tu gasto mensual en la Fase 1 o reducir tu exposición a Renta Variable."
    elif success_rate > 80:
        analysis_text += f"\n\n🟡 **Conclusión:** El plan es **viable pero requiere disciplina**. La zona de peligro comienza a los **{risk_start_age}**. Es crucial monitorear la inflación."
    else:
        analysis_text += f"\n\n🔴 **Conclusión:** El plan presenta **riesgo estructural**. Existe una alta probabilidad de agotar la liquidez alrededor de los **{most_dangerous_age}**. Se recomienda urgentemente ajustar gastos."

    st.info(analysis_text)
    
    # Footer Flotante
    st.markdown(f"""
    <div class="executive-footer">
        <div class="footer-stat"><span>Solvencia</span><strong>{success_rate:.1f}%</strong></div>
        <div class="footer-stat"><span>Herencia Est.</span><strong>${fmt(median_legacy/1e6)}M</strong></div>
        <div class="footer-stat"><span>Riesgo Inicio</span><strong>{risk_start_age}</strong></div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("👈 Carga los datos en la barra lateral (Pegar JSON o Manual) para generar el informe.")
