import streamlit as st
import json
import os
import pandas as pd
import simulador  # Asegúrate de que simulador.py tenga el código V7 Sovereign Grade

st.set_page_config(page_title="Gestión Patrimonial", layout="wide", page_icon="🏦")

# --- RUTAS ---
DATA_FOLDER = "Data"
FILE_MENSUAL = os.path.join(DATA_FOLDER, "macro_instrumentos_2025-11.json")

# --- ESTADO ---
if 'vista_actual' not in st.session_state: st.session_state.vista_actual = 'HOME'
if 'datos_cargados' not in st.session_state: st.session_state.datos_cargados = {}

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_datos():
    if not os.path.exists(FILE_MENSUAL): return None, {}, "Error: Archivo no encontrado"
    try:
        with open(FILE_MENSUAL, "r", encoding="utf-8") as f: data = json.load(f)
        reg = data["registros"][0]
        df = pd.DataFrame(reg.get("instrumentos", []))
        
        if not df.empty:
            cols_num = ["saldo_clp", "saldo_nominal"]
            for col in cols_num:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            def bucket_sim(row):
                t = str(row.get("tipo", "")).lower(); n = str(row.get("nombre", "")).lower()
                if "pasivo" in t or "hipoteca" in t: return "PASIVO"
                if "inmobiliario" in t: return "INMO"
                if row.get("moneda") == "USD" and ("caja" in t or "liquidez" in n): return "USD"
                if "acciones" in n or "equity" in n or "riesgo" in n or "fondo a" in n: return "RV"
                if "mixto" in n or "moderado" in n: return "MX"
                return "RF"
            
            df["bucket_sim"] = df.apply(bucket_sim, axis=1)

        return df, reg.get("macro", {}), reg.get("fecha_dato", "N/A")
    except Exception as e: return None, {}, "Error"

df_cartera, macro_data, fecha_corte = cargar_datos()
uf_val = macro_data.get("uf_promedio", 39600)
tc_val = macro_data.get("dolar_observado_promedio", 930)

# --- CÁLCULOS ---
valor_prop_uf_default = 14500
deuda_hipo_clp_default = 0

if df_cartera is not None and not df_cartera.empty:
    deuda_hipo_uf = df_cartera[df_cartera["bucket_sim"] == "PASIVO"]["saldo_nominal"].sum()
    if deuda_hipo_uf == 0: 
         deuda_hipo_uf = df_cartera[df_cartera["tipo"] == "Pasivo Inmobiliario"]["saldo_nominal"].sum()
    deuda_hipo_clp_default = deuda_hipo_uf * uf_val

neto_inmo_estimado = (valor_prop_uf_default * uf_val) - deuda_hipo_clp_default

if df_cartera is not None and not df_cartera.empty:
    grp = df_cartera.groupby("bucket_sim")["saldo_clp"].sum()
    st.session_state.datos_cargados = {
        'rf': grp.get("RF", 0), 
        'mx': grp.get("MX", 0), 
        'rv': grp.get("RV", 0),
        'usd': df_cartera[df_cartera["bucket_sim"]=="USD"]["saldo_nominal"].sum(),
        'tc': tc_val,
        'inmo_neto': neto_inmo_estimado 
    }

def fmt(v): return f"${int(v):,}".replace(",", ".")

# --- VISTAS ---
def render_home():
    st.title("Panel Gestión Patrimonial")
    st.caption(f"Corte: {fecha_corte}")
    
    if df_cartera is not None:
        activos = df_cartera[df_cartera["bucket_sim"]!="PASIVO"]["saldo_clp"].sum()
        pasivos = df_cartera[df_cartera["bucket_sim"]=="PASIVO"]["saldo_clp"].sum()
        
        c1, c2 = st.columns(2)
        c1.metric("Patrimonio Financiero", fmt(activos - pasivos))
        c2.metric("Inmobiliario Neto", fmt(neto_inmo_estimado))

    if st.button("🔮 IR AL SIMULADOR", type="primary"): 
        st.session_state.vista_actual = 'SIMULADOR'; st.rerun()

def render_simulador():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    d = st.session_state.datos_cargados
    
    # --- CONSOLIDACIÓN PARA MODELO V7 ---
    # La nueva tesis agrupa todo lo defensivo (RF + Mixto + USD) vs Motor (RV)
    
    # 1. Calcular saldos
    saldo_rf = d.get('rf', 0)
    saldo_mx = d.get('mx', 0)
    saldo_usd_clp = d.get('usd', 0) * d.get('tc', 930)
    
    # 2. Asignar a Buckets V7
    # DEFENSA: Sumamos RF pura + Mixto (asumimos componente defensivo) + Caja USD
    total_defensa = saldo_rf + saldo_mx + saldo_usd_clp
    
    # MOTOR: Renta Variable Pura
    total_motor = d.get('rv', 0)

    # 3. LLAMADA AL SIMULADOR V7 (Solo 3 argumentos clave)
    simulador.app(
        default_rf=total_defensa,      # Pasa todo el bloque defensivo
        default_rv=total_motor,        # Pasa el bloque de riesgo
        default_inmo_neto=d.get('inmo_neto', 0)
    )

if st.session_state.vista_actual == 'HOME': render_home()
elif st.session_state.vista_actual == 'SIMULADOR': render_simulador()
