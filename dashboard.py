import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re
import json

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control V24 (Diagnóstico)", layout="wide", page_icon="🏗️")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; margin-bottom: 120px; }
    .main-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #e2e8f0; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    
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
    
    /* Estilo para Plan Z */
    .z-card { background-color: #eff6ff; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 4px; margin-top: 10px; }
    .z-metric { font-size: 1.2rem; font-weight: bold; color: #1e3a8a; }
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
        
        if "pasivo" in tipo or "hipotecario" in nom: continue

        es_rv = any(k in nom or k in sub for k in kw_rv)
        cat = "Renta Variable" if es_rv else "Renta Fija"
        if es_rv: total_rv += saldo
        else: total_rf += saldo
        
        total_clp += saldo
        df_rows.append({"Instrumento": item.get("nombre"), "Monto": fmt(saldo), "Categoría": cat})
        
    pct_rv = total_rv / total_clp if total_clp > 0 else 0
    return total_clp, pct_rv, pd.DataFrame(df_rows), fecha

# --- MATEMÁTICA ANUALIDAD ---
def calc_pmt(principal, rate_anual, years):
    # Calcula cuota fija mensual que agota el capital en N años
    if years <= 0: return 0
    r_mensual = rate_anual / 12
    n_meses = years * 12
    if r_mensual == 0: return principal / n_meses
    return principal * (r_mensual * (1 + r_mensual)**n_meses) / ((1 + r_mensual)**n_meses - 1)

# --- MOTOR SIMULACIÓN ---
def simulacion_core(n_sims, months, cap_ini, eventos, g1, d1, g2, d2, g3, 
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
    
    # Array para guardar el mes exacto de ruina (0 si no hubo ruina)
    ruin_month = np.zeros(n_sims) 
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * alloc_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - alloc_rv)
        
        # Solo aplicamos retorno a quienes aún tienen dinero
        mask_alive = wealth[t-1] > 0
        wealth[t, mask_alive] = wealth[t-1, mask_alive] * (1 + ret[mask_alive])
        
        # Eventos (Inyecciones/Retiros)
        if t in eventos:
            wealth[t, mask_alive] += eventos[t]
        
        # Gasto
        target = g1 if t <= d1*12 else (g2 if t <= (d1+d2)*12 else g3)
        peak[mask_alive] = np.maximum(peak[mask_alive], wealth[t, mask_alive])
        
        gasto_real = np.full(n_sims, target)
        if use_guard:
            # Drawdown check
            with np.errstate(divide='ignore', invalid='ignore'):
                dd = (peak - wealth[t]) / peak
                dd = np.nan_to_num(dd) # Fix div by zero
            mask_crisis = dd > dd_trig
            gasto_real[mask_crisis] -= m_cut
            
        wealth[t, mask_alive] -= gasto_real[mask_alive]
        
        # Detectar nuevas ruinas en este mes
        new_ruins = (wealth[t] <= 0) & (wealth[t-1] > 0)
        wealth[t, wealth[t] <= 0] = 0 # Piso cero
        ruin_month[new_ruins] = t # Registramos el mes de la muerte financiera
        
    is_fail = wealth[-1] <= 0
    return wealth, is_fail, ruin_month

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuración")
    modo = st.radio("Datos", ["Manual", "Pegar JSON"], horizontal=True)
    
    json_data = None
    if modo == "Pegar JSON":
        txt = st.text_area("Pegar JSON", height=100)
        if txt: 
            try: json_data = json.loads(txt)
            except: st.error("JSON inválido")

    with st.expander("📉 Mercado", expanded=False):
        n_sims = st.select_slider("Sims", [1000, 2000, 5000], 2000)
        drag = st.number_input("Costos (%)", 1.0)/100
        
    with st.expander("🏠 Plan Z (Inmobiliario)", expanded=True):
        st.caption("¿Qué pasa si me quedo en cero?")
        val_casa = input_dinero("Valor Casa ($)", 250000000, "z_val")
        costo_vta = st.number_input("Costo Venta (%)", 4.0) / 100
        arriendo_futuro = input_dinero("Arriendo Futuro ($)", 800000, "z_arr")
        tasa_anualidad = st.number_input("Tasa Retorno Anualidad (%)", 4.0) / 100

# --- UI PRINCIPAL ---
st.title("🛡️ Diego FIRE Control V24")

# 1. CAPITAL
cap_final, alloc_final = 0, 0.6
if modo == "Pegar JSON" and json_data:
    cap_calc, alloc_calc, df_audit, _ = procesar_gems_json(json_data)
    cap_final, alloc_final = cap_calc, alloc_calc
    st.success(f"Datos Cargados: $ {fmt(cap_final)} ({alloc_final*100:.1f}% RV)")
else:
    c1, c2 = st.columns(2)
    with c1: cap_final = input_dinero("💰 Capital Líquido ($)", 1800000000, "cap_m")
    with c2: alloc_final = st.slider("% RV", 0, 100, 60)/100.0

# 2. EVENTOS (Inyecciones/Salidas)
st.markdown("### 📅 Eventos de Capital")
with st.expander("Configurar Inyecciones o Gastos Extraordinarios", expanded=False):
    col_ev1, col_ev2, col_ev3 = st.columns(3)
    # Evento 1
    ev1_m = input_dinero("Evento 1 ($)", 0, "ev1_m")
    ev1_y = col_ev1.number_input("Año", 10, key="ev1_y")
    # Evento 2 (Ej. Auto)
    ev2_m = input_dinero("Evento 2 ($) (Negativo = Gasto)", -45000000, "ev2_m")
    ev2_y = col_ev2.number_input("Año", 3, key="ev2_y")
    # Evento 3
    ev3_m = input_dinero("Evento 3 ($)", 0, "ev3_m")
    ev3_y = col_ev3.number_input("Año", 20, key="ev3_y")

# Diccionario de eventos para el motor
eventos_dict = {}
if ev1_m != 0: eventos_dict[ev1_y*12] = ev1_m
if ev2_m != 0: eventos_dict[ev2_y*12] = ev2_m
if ev3_m != 0: eventos_dict[ev3_y*12] = ev3_m

# 3. GASTOS REGULARES
st.markdown("### 💸 Gasto Regular")
c1, c2, c3 = st.columns(3)
with c1:
    g1 = input_dinero("Fase 1 ($)", 6000000, "g1"); d1 = st.number_input("Años F1", 7)
with c2:
    g2 = input_dinero("Fase 2 ($)", 5500000, "g2"); d2 = st.number_input("Años F2", 13)
with c3:
    g3 = input_dinero("Fase 3 ($)", 5000000, "g3")

# --- EJECUCIÓN ---
if st.button("🚀 DIAGNOSTICAR PLAN", type="primary", use_container_width=True):
    # Parametros hardcodeados para demo (se pueden mover a sidebar)
    r_rv, v_rv = 0.065, 0.18
    r_rf, v_rf = 0.015, 0.05
    corr = 0.8
    
    wealth, is_fail, ruin_months = simulacion_core(
        n_sims, 40*12, cap_final, eventos_dict, g1, d1, g2, d2, g3,
        alloc_final, r_rv, v_rv, r_rf, v_rf, corr, True, 0.3, 1200000, drag
    )
    
    # KPIs Generales
    prob = (1 - np.mean(is_fail)) * 100
    herencia = np.median(wealth[-1])
    
    # Footer
    c_stat = "status-green" if prob >= 90 else "status-red"
    st.markdown(f"""<div class="floating-footer">
        <div class="footer-item"><div class="footer-label">Éxito Líquido</div><div class="footer-value {c_stat}">{prob:.1f}%</div></div>
        <div class="footer-item"><div class="footer-label">Herencia P50</div><div class="footer-value">${fmt(herencia)}</div></div>
    </div>""", unsafe_allow_html=True)

    # --- PESTAÑAS DE ANÁLISIS ---
    tab1, tab2, tab3 = st.tabs(["📊 Proyección", "💀 Zona de Peligro", "🏠 Análisis Plan Z"])
    
    with tab1:
        # Gráfico clásico
        p10, p50, p90 = np.percentile(wealth, [10, 50, 90], axis=1)
        x = np.arange(len(p50))/12
        fig = go.Figure([
            go.Scatter(x=x, y=p90, line=dict(width=0), showlegend=False),
            go.Scatter(x=x, y=p10, fill='tonexty', fillcolor='rgba(37, 99, 235, 0.1)', name='Rango'),
            go.Scatter(x=x, y=p50, line=dict(color='#0f172a', width=3), name='Mediana')
        ])
        fig.update_layout(template="plotly_white", yaxis=dict(tickformat=",.0f", tickprefix="$ "), height=400, title="Evolución Patrimonio Líquido")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # ANÁLISIS DE RUINA
        ruin_indices = ruin_months[ruin_months > 0] # Filtramos solo los que quebraron
        if len(ruin_indices) > 0:
            ruin_years = ruin_indices / 12
            
            c_izq, c_der = st.columns([2, 1])
            with c_izq:
                fig_hist = px.histogram(x=ruin_years, nbins=20, labels={'x': 'Año de Ruina'}, title="¿Cuándo ocurre el desastre?")
                fig_hist.update_layout(bargap=0.1, template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with c_der:
                st.markdown("#### 🕵️‍♂️ Informe Forense")
                avg_ruin = np.median(ruin_years)
                early_ruin = np.percentile(ruin_years, 10) # El 10% más mala suerte
                
                st.metric("Total Escenarios Fallidos", f"{len(ruin_indices)} ({len(ruin_indices)/n_sims*100:.1f}%)")
                st.metric("Año Mediano de Quiebra", f"Año {avg_ruin:.1f}")
                st.warning(f"⚠️ **Riesgo Temprano:** En el peor de los casos, la liquidez se acaba el **Año {early_ruin:.1f}**.")
                
                if early_ruin < 10:
                    st.error("¡Cuidado! Hay riesgo significativo de quiebra en la primera década.")
                else:
                    st.success("La quiebra temprana es muy rara. El riesgo está en la longevidad.")
        else:
            st.success("🎉 ¡Increíble! En 2000 escenarios, nunca te quedaste sin dinero líquido.")

    with tab3:
        # PLAN Z MATH
        st.markdown("### 🏠 Ejecución del Plan Z (Venta de Propiedad)")
        st.write("Si tu liquidez llega a cero, activamos este protocolo de emergencia.")
        
        # Usamos el año mediano de ruina calculado arriba (o el final si no hubo ruina)
        year_activacion = np.median(ruin_indices)/12 if len(ruin_indices) > 0 else 40
        
        neto_venta = val_casa * (1 - costo_vta)
        years_left = max(1, 40 - year_activacion) # Años que quedan por vivir desde la quiebra
        
        # Calculamos anualidad
        anualidad_mensual = calc_pmt(neto_venta, tasa_anualidad, years_left)
        disponible_vivir = anualidad_mensual - arriendo_futuro
        
        c_z1, c_z2 = st.columns(2)
        with c_z1:
            st.info(f"📅 **Momento Esperado:** Año {year_activacion:.1f}")
            st.metric("Neto a Recibir (Venta)", f"$ {fmt(neto_venta)}")
            st.caption(f"Valor Casa ${fmt(val_casa)} - Costos {costo_vta*100}%")
            
        with c_z2:
            st.markdown(f"""
            <div class="z-card">
                <div>Ingreso por Anualidad (RF): <b>$ {fmt(anualidad_mensual)}</b></div>
                <div>(-) Costo Arriendo: <b>$ {fmt(arriendo_futuro)}</b></div>
                <hr>
                <div class="z-metric">Para Vivir: $ {fmt(disponible_vivir)} / mes</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Veredicto Plan Z
        if disponible_vivir > 3000000:
            st.success("✅ **Plan Z Sólido:** Incluso vendiendo la casa, mantienes un estilo de vida muy digno.")
        elif disponible_vivir > 1500000:
            st.warning("⚠️ **Plan Z Ajustado:** Podrás vivir, pero con austeridad comparado a tu vida actual.")
        else:
            st.error("🚨 **Plan Z Insuficiente:** Vender la casa no alcanza para cubrir arriendo y vida básica.")
