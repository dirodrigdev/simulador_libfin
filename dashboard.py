import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diego FIRE Control", layout="wide", page_icon="🛡️")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@500&display=swap');
    body { font-family: 'Inter', sans-serif; background-color: #f8fafc; }
    .kpi-card { 
        background: white; padding: 20px; border-radius: 12px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; text-align: center; 
    }
    .kpi-val { 
        font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; 
        font-weight: 800; color: #0f172a; margin: 5px 0;
    }
    .kpi-lbl { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; }
    .liquid { border-top: 4px solid #3b82f6; }
    .total { border-top: 4px solid #10b981; }
    .danger { border-top: 4px solid #ef4444; }
    .money { border-top: 4px solid #f59e0b; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIONES ---
def calculate_pmt(pv, r, n):
    if n <= 0: return 0
    if r <= 0: return pv / n
    return pv * (r * (1 + r)**n) / ((1 + r)**n - 1)

def generate_returns(n_sims, n_months, mu, sigma, dist_type, df):
    if dist_type == "Normal":
        return np.random.normal(mu, sigma, (n_months, n_sims))
    else:
        std_adj = np.sqrt((df - 2) / df) if df > 2 else 1.0
        return mu + sigma * np.random.standard_t(df, (n_months, n_sims)) * std_adj

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Centro de Control")
    
    with st.expander("💰 Capital e Inyección", expanded=True):
        initial_capital = st.number_input("Capital Inicial ($)", value=1800000000, step=10000000)
        years = st.number_input("Años Plan", value=40)
        st.caption("Inyección Futura")
        inject_amt = st.number_input("Monto Inyección ($)", value=0, step=1000000)
        inject_year = st.number_input("Año Inyección", value=10)

    with st.expander("💸 Estrategia de Gasto", expanded=True):
        p1_amt = st.number_input("Fase 1: Mes ($)", value=6000000, step=100000)
        p1_yrs = st.number_input("Fase 1: Años", value=7)
        p2_amt = st.number_input("Fase 2: Mes ($)", value=5500000, step=100000)
        p2_yrs = st.number_input("Fase 2: Años", value=13)
        p3_amt = st.number_input("Fase 3: Mes ($)", value=5000000, step=100000)
        
        use_crisis = st.checkbox("Regla de Crisis", value=True)
        dd_trigger = st.slider("Gatillo Caída (%)", 10, 50, 30) / 100.0 if use_crisis else 1.0
        crisis_cut = st.number_input("Recorte ($)", value=1200000) if use_crisis else 0

    with st.expander("📉 Mercado & Riesgo", expanded=False):
        n_sims = st.selectbox("Simulaciones", [1000, 2000, 5000], index=1)
        dist_type = st.selectbox("Distribución", ["T-Student (Realista)", "Normal"])
        df_student = st.slider("Grados Libertad", 3, 20, 5) if "T-Student" in dist_type else 5
        alloc_rv = st.slider("% Renta Variable", 0, 100, 60) / 100.0
        drag = st.number_input("Costo/Impuestos (%)", value=1.0) / 100.0

    with st.expander("🚗 Auto & Plan Z", expanded=False):
        buy_car = st.checkbox("Compra Auto", value=True)
        car_cost = st.number_input("Costo Auto ($)", value=45000000) if buy_car else 0
        use_plan_z = st.checkbox("Activar Plan Z", value=True)
        z_value = st.number_input("Venta Casa ($)", value=250000000)

# --- MOTOR DE SIMULACIÓN ---
def run_sim():
    months = years * 12
    mu_rv = (0.065 - drag) / 12
    mu_rf = (0.015 - drag) / 12
    sigma_rv = 0.18 / np.sqrt(12)
    sigma_rf = 0.05 / np.sqrt(12)
    
    z1 = generate_returns(n_sims, months, 0, 1, "T-Student" if "T-Student" in dist_type else "Normal", df_student)
    z2 = generate_returns(n_sims, months, 0, 1, "T-Student" if "T-Student" in dist_type else "Normal", df_student)
    eps_rf = 0.8 * z1 + np.sqrt(1 - 0.64) * z2 # Correlación realista 0.8
    
    wealth = np.zeros((months + 1, n_sims))
    wealth[0] = initial_capital
    is_liquid_fail = np.zeros(n_sims, dtype=bool)
    is_total_fail = np.zeros(n_sims, dtype=bool)
    peak = np.full(n_sims, initial_capital)
    
    curr_w = wealth[0].copy()
    curr_rv = np.full(n_sims, alloc_rv)
    
    for t in range(1, months + 1):
        ret = (mu_rv + sigma_rv * z1[t-1]) * curr_rv + (mu_rf + sigma_rf * eps_rf[t-1]) * (1 - curr_rv)
        curr_w *= (1 + ret)
        
        if t == inject_year * 12: curr_w += inject_amt
        if t == 36 and buy_car: curr_w -= car_cost
        
        # Gasto y Crisis
        target = p1_amt if t <= p1_yrs*12 else (p2_amt if t <= (p1_yrs+p2_yrs)*12 else p3_amt)
        
        mask_ok = ~is_liquid_fail
        peak[mask_ok] = np.maximum(peak[mask_ok], curr_w[mask_ok])
        crisis = (peak[mask_ok] - curr_w[mask_ok])/peak[mask_ok] > dd_trigger
        
        current_spend = np.full(np.sum(mask_ok), target)
        current_spend[crisis] -= crisis_cut
        curr_w[mask_ok] -= current_spend
        
        # Plan Z
        new_fails = (curr_w < 0) & mask_ok
        if np.any(new_fails) and use_plan_z:
            is_liquid_fail[new_fails] = True
            curr_w[new_fails] = z_value
            curr_rv[new_fails] = 0.2 # Modo seguro
            
        # Ruina Total
        mask_z = is_liquid_fail & ~is_total_fail
        if np.any(mask_z):
            rem = max(1, months - t)
            pmt = calculate_pmt(curr_w[mask_z], (mu_rv*0.2 + mu_rf*0.8), rem)
            curr_w[mask_z] -= pmt
            is_total_fail[(curr_w < 0) & mask_z] = True
            
        curr_w[is_total_fail] = 0
        wealth[t] = curr_w
        
    return wealth, is_liquid_fail, is_total_fail

# --- UI ---
st.title("🛡️ Diego FIRE Control center")

if st.button("🚀 EJECUTAR ESCENARIOS", type="primary"):
    w_paths, liq_f, tot_f = run_sim()
    
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi-card liquid"><div class="kpi-val">{ (1-np.mean(liq_f))*100:.1f}%</div><div class="kpi-lbl">Éxito Líquido</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card total"><div class="kpi-val">{ (1-np.mean(tot_f))*100:.1f}%</div><div class="kpi-lbl">Éxito Total</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card money"><div class="kpi-val">${ np.median(w_paths[-1])/1e6:,.0f}M</div><div class="kpi-lbl">Herencia Mediana</div></div>', unsafe_allow_html=True)

    # Gráfico
    p10, p50, p90 = np.percentile(w_paths, [10, 50, 90], axis=1)
    x = np.arange(len(p50))/12
    fig = go.Figure([
        go.Scatter(x=x, y=p90/1e6, line=dict(width=0), showlegend=False),
        go.Scatter(x=x, y=p10/1e6, fill='tonexty', fillcolor='rgba(59,130,246,0.1)', name='Rango P10-P90'),
        go.Scatter(x=x, y=p50/1e6, line=dict(color='#0f172a', width=3), name='Mediana')
    ])
    fig.update_layout(title="Evolución Patrimonio (En Millones $)", template="plotly_white", yaxis_title="Millones CLP")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Configura y presiona Ejecutar.")
