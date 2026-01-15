import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import json
from datetime import date

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Diego Family Office V30", layout="wide", page_icon="🏛️")

if 'reporte_generado' not in st.session_state: st.session_state.reporte_generado = False
if 'resultados' not in st.session_state: st.session_state.resultados = {}

# --- 2. ESTILOS (BANKING GRADE) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;700;800&family=Roboto+Mono:wght@400;500&display=swap');
    body { font-family: 'Manrope', sans-serif; background-color: #f4f6f9; color: #1e293b; margin-bottom: 120px; }
    h1, h2, h3 { color: #0f172a; font-weight: 800; letter-spacing: -0.5px; }
    
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05); }
    .metric-title { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; font-weight: 700; margin-bottom: 8px; }
    .metric-value { font-family: 'Manrope', sans-serif; font-size: 1.8rem; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    
    .recommendation-card {
        background-color: #f0fdf4; border-left: 4px solid #16a34a; padding: 15px; border-radius: 4px; margin-bottom: 10px;
    }
    .rec-title { font-weight: 700; color: #166534; font-size: 1rem; }
    .rec-val { font-family: 'Roboto Mono', monospace; font-weight: 700; color: #15803d; }
    
    .positive { color: #10b981; } .negative { color: #ef4444; } .neutral { color: #f59e0b; }
    
    .executive-footer {
        position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
        background: rgba(15, 23, 42, 0.95); color: white; backdrop-filter: blur(10px);
        padding: 12px 40px; border-radius: 50px; display: flex; gap: 40px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); z-index: 9999;
        white-space: nowrap;
    }
    .footer-stat { text-align: center; }
    .footer-stat span { display: block; font-size: 0.7rem; opacity: 0.7; text-transform: uppercase; color: #cbd5e1; }
    .footer-stat strong { font-size: 1.2rem; font-weight: 700; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --- 3. UTILIDADES ---
def fmt(v): return f"{int(v):,}".replace(",", ".")
def parse(t): return int(re.sub(r'\D', '', t)) if t else 0
def input_dinero(lbl, d, k): return parse(st.text_input(lbl, value=fmt(d), key=k))

# --- 4. MOTOR LÓGICO ---
def procesar_gems(json_raw):
    try:
        if isinstance(json_raw, dict) and "registros" in json_raw:
            data = json_raw["registros"][0]["instrumentos"]
        elif isinstance(json_raw, list):
            data = json_raw
        else: return 0, 0
    except: return 0, 0

    total_rv, total_liq = 0, 0
    kw_rv = ["agresivo", "fondo a", "equity", "accion", "etf", "sp500", "nasdaq", "gestión activa", "moneda renta"]
    
    for i in data:
        nom = i.get("nombre", "").lower()
        tipo = i.get("tipo", "").lower()
        sub = str(i.get("subtipo", "")).lower()
        saldo = i.get("saldo_clp", 0)
        
        if "pasivo" in tipo or "hipotecario" in nom: continue 
        
        es_rv = any(k in nom or k in sub for k in kw_rv)
        if es_rv: total_rv += saldo
        total_liq += saldo
        
    return total_liq, (total_rv/total_liq if total_liq else 0)

def proyeccion_montecarlo(n_sims, months, cap, g1, d1, g2, d2, g3, pct_rv, params):
    mu_rv, sig_rv = (params['r_rv'] - params['inf'] - params['cost'])/12, params['v_rv']/np.sqrt(12)
    mu_rf, sig_rf = (params['r_rf'] - params['inf'] - params['cost'])/12, params['v_rf']/np.sqrt(12)
    
    df = 5
    std_adj = np.sqrt((df-2)/df)
    z1 = np.random.standard_t(df, (months, n_sims)) * std_adj
    z2 = np.random.standard_t(df, (months, n_sims)) * std_adj
    eps = params['corr'] * z1 + np.sqrt(1 - params['corr']**2) * z2
    
    wealth = np.zeros((months+1, n_sims))
    wealth[0] = cap
    ruin_idx = np.full(n_sims, -1)
    peak = np.full(n_sims, cap)
    
    for t in range(1, months+1):
        ret = (mu_rv + sig_rv * z1[t-1]) * pct_rv + (mu_rf + sig_rf * eps[t-1]) * (1 - pct_rv)
        
        mask_ok = wealth[t-1] > 0
        wealth[t, mask_ok] = wealth[t-1, mask_ok] * (1 + ret[mask_ok])
        
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak[mask_ok] = np.maximum(peak[mask_ok], wealth[t, mask_ok])
        
        if params['guardrail']:
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = (peak - wealth[t]) / peak
                dd = np.nan_to_num(dd)
            mask_cut = (dd > 0.30) & mask_ok
            current_g = np.full(n_sims, target)
            current_g[mask_cut] -= 1200000 
            wealth[t, mask_ok] -= current_g[mask_ok]
        else:
            wealth[t, mask_ok] -= target

        just_died = (wealth[t] <= 0) & (wealth[t-1] > 0)
        wealth[t, wealth[t] < 0] = 0
        ruin_idx[just_died] = t
        
    return wealth, ruin_idx

# --- 5. MOTOR DE RECOMENDACIONES (INTELIGENCIA) ---
def buscar_mejora(target_prob, base_prob, params_base, cap, g1, d1, g2, d2, g3, pct_rv, horizon_months):
    """
    Busca qué cambios logran aumentar la probabilidad de éxito en X puntos.
    Retorna lista de sugerencias.
    """
    if base_prob >= 99.0: return []
    
    target = min(100.0, base_prob + 5.0) # Buscamos +5%
    recs = []
    
    # 1. Reducir Gasto Fase 1
    # Búsqueda binaria simple
    low, high = 0, g1
    best_g1 = g1
    found = False
    
    # Intentamos 10 iteraciones para encontrar el G1 que da el target
    for _ in range(10):
        mid = (low + high) / 2
        # Simulacion rápida (500 sims)
        _, r_idx = proyeccion_montecarlo(500, horizon_months, cap, mid, d1, g2, d2, g3, pct_rv, params_base)
        p = (np.sum(r_idx == -1) / 500) * 100
        if p >= target:
            best_g1 = mid
            low = mid # Queremos el gasto más alto posible que cumpla
            found = True
        else:
            high = mid # Necesitamos bajar más el gasto
            
    # Si logramos mejorar algo significativo
    if found and best_g1 < g1:
        delta = g1 - best_g1
        recs.append({
            "tipo": "Gasto Mensual",
            "accion": f"Reducir Gasto Fase 1 en **$ {fmt(delta)}**",
            "impacto": f"+5% Prob. Éxito"
        })

    # 2. Inyección de Capital
    # Cuánto capital extra necesito hoy
    low_c, high_c = cap, cap * 1.5
    best_cap = cap
    found_c = False
    for _ in range(10):
        mid = (low_c + high_c) / 2
        _, r_idx = proyeccion_montecarlo(500, horizon_months, mid, g1, d1, g2, d2, g3, pct_rv, params_base)
        p = (np.sum(r_idx == -1) / 500) * 100
        if p >= target:
            best_cap = mid
            high_c = mid # Queremos el capital mínimo necesario
            found_c = True
        else:
            low_c = mid
            
    if found_c and best_cap > cap:
        delta_c = best_cap - cap
        recs.append({
            "tipo": "Capital",
            "accion": f"Inyectar Capital Hoy: **$ {fmt(delta_c)}**",
            "impacto": f"+5% Prob. Éxito"
        })
        
    return recs

# --- 6. INTERFAZ ---

c_logo, c_title = st.columns([1, 6])
with c_logo: st.markdown("## 🏛️")
with c_title: st.markdown("# Diego Family Office \n ### Informe de Solvencia & Recomendaciones")

# SIDEBAR
with st.sidebar:
    st.markdown("### 1. Perfil")
    birth_year = st.number_input("Año Nacimiento", 1950, 2010, 1978)
    age = date.today().year - birth_year
    life_expectancy = st.slider("Esperanza de Vida", 85, 100, 95)
    horizon_years = life_expectancy - age
    st.info(f"👤 **Edad:** {age} años | 🏁 **Horizonte:** {horizon_years} años")

    st.markdown("### 2. Datos")
    source = st.radio("Fuente", ["Pegar JSON", "Manual"], horizontal=True)
    cap_liq, pct_rv = 0, 0.6
    
    if source == "Pegar JSON":
        txt = st.text_area("JSON de Gems", height=100)
        if txt:
            try:
                c, p = procesar_gems(json.loads(txt))
                cap_liq, pct_rv = c, p
                if c > 0: st.success(f"✅ Detectado: $ {fmt(c)}")
                else: st.warning("JSON sin saldo válido")
            except: st.error("JSON inválido")
    else:
        cap_liq = input_dinero("Capital Líquido ($)", 1800000000, "c_man")
        pct_rv = st.slider("% Renta Variable", 0, 100, 60)/100.0
    
    # VISUALIZADOR DE MIX INMEDIATO
    pct_rf = 1.0 - pct_rv
    st.markdown("---")
    st.markdown("### ⚖️ Mix de Inversión")
    c_pie1, c_pie2 = st.columns(2)
    c_pie1.metric("Renta Fija", f"{pct_rf*100:.0f}%")
    c_pie2.metric("Renta Variable", f"{pct_rv*100:.0f}%")
    st.progress(pct_rv)
    st.caption("Barra indica % de Renta Variable (Riesgo)")

    st.markdown("### 3. Supuestos")
    with st.expander("Parámetros Macro"):
        inf = st.number_input("Inflación (%)", 3.0)/100
        cost = st.number_input("Costos (%)", 1.0)/100
        guard = st.checkbox("Guardrails", True)

# INPUTS GASTO
st.markdown("---")
col_g1, col_g2, col_g3 = st.columns(3)
with col_g1:
    st.markdown("#### 💸 Fase 1: Lifestyle")
    g1 = input_dinero("Gasto Mensual ($)", 6000000, "g1")
    d1 = st.number_input("Años", 7, key="d1")
with col_g2:
    st.markdown("#### 📉 Fase 2: Transición")
    g2 = input_dinero("Gasto Mensual ($)", 5500000, "g2")
    d2 = st.number_input("Años", 13, key="d2")
with col_g3:
    st.markdown("#### 👴 Fase 3: Madurez")
    g3 = input_dinero("Gasto Mensual ($)", 5000000, "g3")
    st.caption("Resto de la vida")

st.markdown("---")

# EJECUCIÓN
def ejecutar_simulacion():
    if cap_liq <= 0:
        st.error("⚠️ Error: Faltan datos de capital.")
        return

    with st.spinner("Analizando escenarios y buscando optimizaciones..."):
        params = {'r_rv': 0.075, 'v_rv': 0.18, 'r_rf': 0.035, 'v_rf': 0.05, 'corr': 0.8, 'inf': inf, 'cost': cost, 'guardrail': guard}
        months = max(12, horizon_years * 12)
        paths, ruin_idx = proyeccion_montecarlo(2000, months, cap_liq, g1, d1, g2, d2, g3, pct_rv, params)
        
        prob = (np.sum(ruin_idx == -1) / 2000) * 100
        
        # Calcular Recomendaciones
        recs = buscar_mejora(prob, prob, params, cap_liq, g1, d1, g2, d2, g3, pct_rv, months)
        
        st.session_state.resultados = {
            "paths": paths, "ruin_idx": ruin_idx, "months": months, "cap": cap_liq, 
            "prob": prob, "recs": recs, "pct_rv": pct_rv
        }
        st.session_state.reporte_generado = True

if st.button("🚀 GENERAR INFORME EJECUTIVO", type="primary", use_container_width=True):
    ejecutar_simulacion()

if st.session_state.reporte_generado:
    res = st.session_state.resultados
    paths = res["paths"]
    ruin_idx = res["ruin_idx"]
    prob = res["prob"]
    recs = res["recs"]
    
    median_legacy = np.median(paths[-1])
    ruin_years = (ruin_idx[ruin_idx > -1] / 12) + age 
    risk_start = f"{np.percentile(ruin_years, 10):.0f} Años" if len(ruin_years) > 0 else "N/A"
    
    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Solvencia</div><div class="metric-value {'positive' if prob > 90 else 'negative'}">{prob:.1f}%</div><div class="metric-sub">Probabilidad Éxito</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Capital</div><div class="metric-value">${fmt(res['cap']/1e6)}M</div><div class="metric-sub">{res['pct_rv']*100:.0f}% RV / {(1-res['pct_rv'])*100:.0f}% RF</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Herencia</div><div class="metric-value neutral">${fmt(median_legacy/1e6)}M</div><div class="metric-sub">Estimada P50</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card"><div class="metric-title">Riesgo</div><div class="metric-value negative">{risk_start}</div><div class="metric-sub">Edad 1er Fallo</div></div>""", unsafe_allow_html=True)

    # RECOMENDACIONES INTELIGENTES
    if prob < 99.0 and len(recs) > 0:
        st.markdown("### 💡 ¿Cómo mejorar mi plan?")
        st.info(f"Para aumentar tu probabilidad de éxito en **5 puntos (al {min(100, prob+5):.1f}%)**, podrías considerar:")
        
        c_rec1, c_rec2 = st.columns(2)
        for i, rec in enumerate(recs):
            with (c_rec1 if i % 2 == 0 else c_rec2):
                st.markdown(f"""
                <div class="recommendation-card">
                    <div class="rec-title">{rec['tipo']}</div>
                    <div class="rec-val">{rec['accion']}</div>
                </div>
                """, unsafe_allow_html=True)

    # GRÁFICO
    st.markdown("### 🔭 Proyección Patrimonial")
    years_axis = np.arange(res["months"] + 1) / 12 + age
    p10, p50, p90 = np.percentile(paths, 10, axis=1), np.percentile(paths, 50, axis=1), np.percentile(paths, 90, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([years_axis, years_axis[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(148, 163, 184, 0.2)', line=dict(color='rgba(0,0,0,0)'), name='Rango 80%'))
    fig.add_trace(go.Scatter(x=years_axis, y=p50, line=dict(color='#0f172a', width=4), name='Mediana'))
    fig.add_shape(type="line", x0=age, x1=age+horizon_years, y0=0, y1=0, line=dict(color="#ef4444", width=2, dash="dash"))
    fig.update_layout(template="plotly_white", height=500, hovermode="x unified", xaxis=dict(title="Tu Edad"), yaxis=dict(tickformat=",.0f"))
    st.plotly_chart(fig, use_container_width=True)

    # FOOTER
    st.markdown(f"""<div class="executive-footer"><div class="footer-stat"><span>Solvencia</span><strong>{prob:.1f}%</strong></div><div class="footer-stat"><span>Herencia</span><strong>${fmt(median_legacy/1e6)}M</strong></div></div>""", unsafe_allow_html=True)
    
elif cap_liq == 0:
    st.info("👈 Ingresa tu capital y presiona **GENERAR INFORME**.")
    
