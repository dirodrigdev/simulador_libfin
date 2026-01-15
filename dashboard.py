import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Centro de Comando FIRE V18", layout="wide", page_icon="🏦")

# --- ESTILOS CSS MODERNOS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    
    /* Sidebar Styling */
    .stSidebar .st-emotion-cache-16txtl3 { padding-top: 2rem; }
    .streamlit-expanderHeader { font-weight: 600; color: #1e293b; background-color: #f1f5f9; border-radius: 8px; }
    
    /* KPI Cards */
    .kpi-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
    .kpi-card { 
        background: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center; 
    }
    .kpi-val { 
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; 
        font-weight: 800; color: #0f172a; margin: 5px 0;
    }
    .kpi-lbl { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; }
    .kpi-sub { font-size: 0.7rem; color: #94a3b8; }

    /* Colors */
    .liquid { border-top: 4px solid #3b82f6; }
    .total { border-top: 4px solid #10b981; }
    .danger { border-top: 4px solid #ef4444; }
    .money { border-top: 4px solid #f59e0b; }

</style>
""", unsafe_allow_html=True)

# --- FUNCIONES MATEMÁTICAS ---

def calculate_pmt(pv, r, n):
    """Calcula la cuota de una anualidad (amortización inversa)"""
    if n <= 0: return 0
    if r <= 0: return pv / n
    return pv * (r * (1 + r)**n) / ((1 + r)**n - 1)

def generate_returns(n_sims, n_months, mu, sigma, dist_type, df):
    """Genera retornos matriciales según distribución"""
    if dist_type == "Normal":
        return np.random.normal(mu, sigma, (n_months, n_sims))
    else: # T-Student (Colas Gordas)
        # Ajuste: t-student tiene varianza df/(df-2). Normalizamos para que sigma sea comparable.
        if df > 2:
            std_adj = np.sqrt((df - 2) / df)
        else:
            std_adj = 1.0 
        
        t_samples = np.random.standard_t(df, (n_months, n_sims))
        return mu + sigma * t_samples * std_adj

# --- SIDEBAR: CENTRO DE COMANDO ---

with st.sidebar:
    st.header("🎛️ Centro de Control")
    
    # --- A. CAPITAL & TIEMPO ---
    with st.expander("💰 A. Capital, Tiempo e Inyección", expanded=True):
        initial_capital = st.number_input("Capital Inicial ($)", value=1_731_194_681, step=1_000_000, format="%d")
        years = st.number_input("Horizonte (Años)", value=40)
        current_age = st.number_input("Edad Actual", value=45)
        
        st.markdown("---")
        st.caption("💉 Inyección Futura (Herencia/Bono)")
        inject_amt = st.number_input("Monto Inyección ($)", value=0, step=1_000_000)
        inject_year = st.number_input("Año Inyección", value=10)

    # --- B. ESTRATEGIA DE GASTO ---
    with st.expander("💸 B. Estrategia de Gasto (Fases)", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            p1_amt = st.number_input("Fase 1: Monto Mensual", value=6_000_000, step=100_000)
            p1_yrs = st.number_input("Fase 1: Duración (Años)", value=7)
        with col2:
            p2_amt = st.number_input("Fase 2: Monto Mensual", value=5_500_000, step=100_000)
            p2_yrs = st.number_input("Fase 2: Duración (Años)", value=13)
        
        p3_amt = st.number_input("Fase 3 (Resto): Monto Mensual", value=4_500_000, step=100_000)
        
        st.markdown("---")
        use_crisis = st.checkbox("Activar Regla de Crisis (Guardrail)", value=True)
        if use_crisis:
            dd_trigger = st.slider("Gatillo Drawdown (%)", 10, 50, 30) / 100.0
            crisis_cut = st.number_input("Recorte Mensual ($)", value=1_200_000, step=100_000)
        else:
            dd_trigger = 1.0 # Imposible de alcanzar
            crisis_cut = 0

    # --- C. MERCADO & ESTADÍSTICA ---
    with st.expander("📉 C. Mercado & Motor Estadístico", expanded=False):
        n_sims = st.selectbox("N° Simulaciones", [1000, 2000, 5000], index=1)
        
        dist_type = st.selectbox("Tipo de Distribución", ["T-Student (Colas Gordas)", "Normal (Gaussiana)"])
        df_student = 5
        if "T-Student" in dist_type:
            df_student = st.slider("Grados de Libertad (df)", 3, 30, 5, help="Menor número = Eventos extremos más frecuentes")
        
        st.markdown("##### Asset Allocation")
        alloc_rv = st.slider("% Renta Variable", 0, 100, 60) / 100.0
        alloc_rf = 1.0 - alloc_rv
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mu_rv_ann = st.number_input("Retorno RV (%)", value=6.5, step=0.1) / 100.0
            sigma_rv_ann = st.number_input("Volatilidad RV (%)", value=18.0, step=0.1) / 100.0
        with col_m2:
            mu_rf_ann = st.number_input("Retorno RF (%)", value=1.5, step=0.1) / 100.0
            sigma_rf_ann = st.number_input("Volatilidad RF (%)", value=5.0, step=0.1) / 100.0
            
        corr = st.slider("Correlación RV/RF", -1.0, 1.0, 0.6)
        drag = st.number_input("Costo Fricción/Impuestos (%)", value=1.0, step=0.1) / 100.0

    # --- D. EVENTOS & PLAN Z ---
    with st.expander("🚗 D. Auto & Plan Z (Rescate)", expanded=False):
        st.markdown("##### Compra de Auto")
        buy_car = st.checkbox("Incluir Compra Auto", value=True)
        if buy_car:
            car_cost = st.number_input("Costo Auto ($)", value=45_400_000)
            car_month = st.number_input("Mes de Compra", value=36)
        else:
            car_cost = 0
            car_month = -1

        st.markdown("##### 🏠 Plan Z (Inmobiliario)")
        use_plan_z = st.checkbox("Activar Plan Z", value=True)
        z_value = st.number_input("Valor Neto Venta Casa ($)", value=250_000_000)
        z_rent = st.number_input("Costo Arriendo Post-Venta ($)", value=0)
        z_alloc_rv = st.slider("Riesgo Inversión Plan Z (% RV)", 0, 100, 20) / 100.0

# --- LÓGICA DE SIMULACIÓN (BACKEND VECTORIZADO) ---

def run_simulation():
    months = years * 12
    
    # 1. Ajuste de Tasas Mensuales (con Drag)
    mu_rv_m = (mu_rv_ann - drag) / 12
    mu_rf_m = (mu_rf_ann - drag) / 12
    sigma_rv_m = sigma_rv_ann / np.sqrt(12)
    sigma_rf_m = sigma_rf_ann / np.sqrt(12)
    
    # 2. Generación de Ruido Correlacionado
    # Generamos dos series independientes
    dist_name = "T-Student" if "T-Student" in dist_type else "Normal"
    
    z1 = generate_returns(n_sims, months, 0, 1, dist_name, df_student)
    z2 = generate_returns(n_sims, months, 0, 1, dist_name, df_student)
    
    # Correlación de Cholesky para 2 activos
    # r_rv = z1
    # r_rf = rho * z1 + sqrt(1 - rho^2) * z2
    eps_rv = z1
    eps_rf = corr * z1 + np.sqrt(1 - corr**2) * z2
    
    # Retornos finales
    ret_rv = mu_rv_m + sigma_rv_m * eps_rv
    ret_rf = mu_rf_m + sigma_rf_m * eps_rf
    
    # 3. Inicialización de Arrays
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = initial_capital
    peak_wealth = np.full(n_sims, initial_capital)
    
    # Flags de estado
    is_ruined_liquid = np.zeros(n_sims, dtype=bool) # Perdió liquidez (vendió casa)
    is_ruined_total = np.zeros(n_sims, dtype=bool)  # Perdió todo (incluso casa)
    months_in_crisis = np.zeros(n_sims)
    
    # Pre-cálculo de tasa Plan Z (para anualidad)
    z_rate_m = (mu_rv_m * z_alloc_rv) + (mu_rf_m * (1 - z_alloc_rv))
    
    # Tiempos de fases
    m_p1 = p1_yrs * 12
    m_p2 = (p1_yrs + p2_yrs) * 12
    m_inject = inject_year * 12

    # --- BUCLE TEMPORAL ---
    # Usamos un bucle for sobre el tiempo porque las decisiones (gastos, crisis) dependen del estado anterior
    
    current_wealth = wealth[0].copy()
    current_alloc_rv = np.full(n_sims, alloc_rv) # Vector de allocation (cambia si activan Plan Z)
    
    for t in range(1, months + 1):
        # A. Retorno de Mercado (Vectorizado)
        # Portafolio ponderado dinámico
        port_ret = ret_rv[t-1] * current_alloc_rv + ret_rf[t-1] * (1 - current_alloc_rv)
        current_wealth *= (1 + port_ret)
        
        # B. Inyecciones / Eventos
        if t == m_inject:
            current_wealth += inject_amt
        if buy_car and t == car_month:
            # Solo restamos si aun no estamos en ruina total, o si estamos en plan Z pero con saldo
            current_wealth -= car_cost
            
        # C. Definir Gasto Base
        if t <= m_p1: base_spend = p1_amt
        elif t <= m_p2: base_spend = p2_amt
        else: base_spend = p3_amt
        
        # D. Regla de Crisis & Plan Z Check
        
        # Separamos lógica para quienes AUN tienen liquidez vs quienes YA están en Plan Z
        
        # --- GRUPO 1: PORTAFOLIO NORMAL (No Ruined Liquid) ---
        mask_ok = ~is_ruined_liquid
        if np.any(mask_ok):
            # Actualizar picos solo para los vivos
            peak_wealth[mask_ok] = np.maximum(peak_wealth[mask_ok], current_wealth[mask_ok])
            drawdown = (peak_wealth[mask_ok] - current_wealth[mask_ok]) / peak_wealth[mask_ok]
            
            # Vector de gasto
            spend_vec = np.full(np.sum(mask_ok), base_spend)
            
            # Aplicar recorte
            crisis_mask = (drawdown > dd_trigger)
            spend_vec[crisis_mask] -= crisis_cut
            
            # Registrar tiempo en crisis (usamos indices globales)
            # Truco: mapear indices de mask_ok a indices globales es complejo en numpy puro dentro de loop
            # Simplificación: Sumamos 1 a todos los que cumplieron la condición
            # Para exactitud estricta, necesitariamos mantener indices. 
            # Aproximación: sumamos al contador global filtrando
            
            # Aplicar gasto
            current_wealth[mask_ok] -= spend_vec
            
            # Check Ruina Liquida
            new_ruins = (current_wealth < 0) & mask_ok
            if np.any(new_ruins) and use_plan_z:
                is_ruined_liquid[new_ruins] = True
                
                # ACTIVAR PLAN Z
                current_wealth[new_ruins] = z_value # Reset a valor casa
                current_alloc_rv[new_ruins] = z_alloc_rv # Cambiar perfil riesgo
                
                # Calcular Cuota Anualidad para estos nuevos arruinados
                # Nota: En una simulación vectorizada pura, calcular PMT individual es difícil.
                # Haremos una aproximación: Retiramos una cuota fija recalculada al momento de quiebra
                # y esa cuota se mantiene fija (ajustada por inflación implícita en tasas reales).
                
                # Simplificación para velocidad: 
                # El modelo asume que al entrar en Plan Z, se calcula una anualidad PERFECTA 
                # que consume el capital hasta el mes 480.
                # En cada paso siguiente, retiraremos esa cuota calculada dinámicamente o fija.
                # Para hacerlo simple: En modo Plan Z, el gasto NO es p1/p2/p3.
                # El gasto es: (Wealth / Meses Restantes) * Factor Ajuste, o PMT.
        
        # --- GRUPO 2: EN PLAN Z (Ruined Liquid pero !Ruined Total) ---
        mask_z = is_ruined_liquid & ~is_ruined_total
        if np.any(mask_z):
            # Calcular PMT dinámico para vaciar la cuenta al final
            months_left = max(1, months - t)
            # Formula vectorizada de PMT
            r = z_rate_m
            pv = current_wealth[mask_z]
            pmt = pv * (r * (1 + r)**months_left) / ((1 + r)**months_left - 1)
            
            # Retirar cuota (que cubre vida + arriendo)
            current_wealth[mask_z] -= pmt
            
            # Check Ruina Total (Si incluso con la casa se acaba la plata)
            # Esto pasa si la volatilidad del mix Z es muy alta y tienes mala suerte
            sub_zero = (current_wealth < 0) & mask_z
            is_ruined_total[sub_zero] = True
            current_wealth[sub_zero] = 0 # Game Over

        # Guardar historia
        wealth[t] = current_wealth
        
        # Contabilizar crisis (aproximado para KPI)
        # Si drawdown > trigger Y aun no liquidado
        active_crisis = (~is_ruined_liquid) & ((peak_wealth - current_wealth)/peak_wealth > dd_trigger)
        months_in_crisis[active_crisis] += 1

    return wealth, is_ruined_liquid, is_ruined_total, months_in_crisis, months

# --- INTERFAZ PRINCIPAL ---

st.title("🛡️ Centro de Comando FIRE V18")
st.markdown("Simulación de Montecarlo con **Gestión de Crisis Dinámica** y **Análisis de Colas Gordas**.")

if st.button("🚀 EJECUTAR SIMULACIÓN MAESTRA", type="primary"):
    with st.spinner("Procesando miles de universos paralelos..."):
        # Ejecutar Motor
        wealth_paths, liquid_fail, total_fail, crisis_months, total_months = run_simulation()
        
        final_wealth = wealth_paths[-1]
        
        # Cálculo de KPIs
        pct_liquid_success = (1 - np.mean(liquid_fail)) * 100
        pct_total_success = (1 - np.mean(total_fail)) * 100
        pct_plan_z = np.mean(liquid_fail) * 100 # Probabilidad de tener que vender la casa
        
        avg_crisis_percent = (np.mean(crisis_months) / total_months) * 100
        median_legacy = np.median(final_wealth)
        
        # --- DISPLAY KPIs ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        kpi1.markdown(f"""
        <div class="kpi-card liquid">
            <div class="kpi-val">{pct_liquid_success:.1f}%</div>
            <div class="kpi-lbl">Éxito Líquido</div>
            <div class="kpi-sub">Sin vender activos fijos</div>
        </div>
        """, unsafe_allow_html=True)
        
        kpi2.markdown(f"""
        <div class="kpi-card total">
            <div class="kpi-val">{pct_total_success:.1f}%</div>
            <div class="kpi-lbl">Éxito Total</div>
            <div class="kpi-sub">Supervivencia Final</div>
        </div>
        """, unsafe_allow_html=True)
        
        kpi3.markdown(f"""
        <div class="kpi-card money">
            <div class="kpi-val">${median_legacy/1e6:,.0f} M</div>
            <div class="kpi-lbl">Herencia (P50)</div>
            <div class="kpi-sub">Saldo Final Esperado</div>
        </div>
        """, unsafe_allow_html=True)
        
        kpi4.markdown(f"""
        <div class="kpi-card danger">
            <div class="kpi-val">{avg_crisis_percent:.1f}%</div>
            <div class="kpi-lbl">Tiempo en Crisis</div>
            <div class="kpi-sub">% Meses con Recortes</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- GRÁFICOS ---
        tab1, tab2 = st.tabs(["📈 Proyección Patrimonial", "📊 Distribución Final"])
        
        with tab1:
            # Calcular percentiles para el gráfico
            p10 = np.percentile(wealth_paths, 10, axis=1)
            p50 = np.percentile(wealth_paths, 50, axis=1)
            p90 = np.percentile(wealth_paths, 90, axis=1)
            x_axis = np.arange(total_months + 1) / 12 # Años
            
            fig = go.Figure()
            
            # Área sombreada
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_axis, x_axis[::-1]]),
                y=np.concatenate([p90, p10[::-1]]),
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Rango de Incertidumbre (P10-P90)',
                hoverinfo="skip"
            ))
            
            # Mediana
            fig.add_trace(go.Scatter(
                x=x_axis, y=p50,
                line=dict(color='#0f172a', width=3),
                name='Mediana (P50)'
            ))
            
            # Línea Cero
            fig.add_hline(y=0, line_dash="dash", line_color="#ef4444", annotation_text="Ruina Total")
            
            # Evento Auto
            if buy_car:
                fig.add_vline(x=car_month/12, line_dash="dot", line_color="gray", annotation_text="🚗")
            
            # Evento Inyección
            if inject_amt > 0:
                fig.add_vline(x=inject_year, line_dash="dot", line_color="green", annotation_text="💰")

            fig.update_layout(
                title="Evolución del Patrimonio en el Tiempo",
                xaxis_title="Años",
                yaxis_title="Patrimonio ($)",
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Histograma
            # Filtramos outliers extremos para mejor visualización (máximo 3 veces la mediana)
            cutoff = median_legacy * 3
            filtered_data = final_wealth[final_wealth < cutoff]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=filtered_data,
                nbinsx=50,
                marker_color='#3b82f6',
                opacity=0.75,
                name='Escenarios'
            ))
            fig_hist.add_vline(x=0, line_color="#ef4444", line_width=3, annotation_text="Ruina")
            fig_hist.add_vline(x=median_legacy, line_color="#0f172a", line_dash="dash", annotation_text="Mediana")
            
            fig_hist.update_layout(
                title="Distribución de Resultados Finales",
                xaxis_title="Saldo Final ($)",
                yaxis_title="Frecuencia",
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

else:
    st.info("👈 Configura los parámetros en el menú lateral y presiona 'EJECUTAR SIMULACIÓN' para comenzar.")
