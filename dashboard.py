import streamlit as st
import json
import os
import pandas as pd
import simulador 

st.set_page_config(page_title="Gestión Patrimonial", layout="wide", page_icon="🏦")

# --- RUTAS Y CARGA ---
DATA_FOLDER = "Data"
FILE_MENSUAL = os.path.join(DATA_FOLDER, "macro_instrumentos_2025-11.json")

if 'vista_actual' not in st.session_state: st.session_state.vista_actual = 'HOME'
if 'datos_cargados' not in st.session_state: st.session_state.datos_cargados = {}

@st.cache_data
def cargar_datos():
    if not os.path.exists(FILE_MENSUAL): return None, {}, "Error"
    try:
        with open(FILE_MENSUAL, "r", encoding="utf-8") as f: data = json.load(f)
        reg = data["registros"][0]
        df = pd.DataFrame(reg.get("instrumentos", []))
        if not df.empty:
            cols = ["saldo_clp", "saldo_nominal"]
            for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
            def bkt(row):
                t = str(row.get("tipo", "")).lower(); n = str(row.get("nombre", "")).lower()
                if "pasivo" in t or "hipoteca" in t: return "PASIVO"
                if "inmobiliario" in t: return "INMO"
                if row.get("moneda") == "USD" and ("caja" in t or "liquidez" in n): return "USD"
                if "acciones" in n or "equity" in n or "riesgo" in n or "fondo a" in n: return "RV"
                if "mixto" in n or "moderado" in n: return "MX"
                return "RF"
            df["bucket_sim"] = df.apply(bkt, axis=1)
        return df, reg.get("macro", {}), reg.get("fecha_dato", "N/A")
    except: return None, {}, "Error"

df_cartera, macro_data, fecha_corte = cargar_datos()
uf_val = macro_data.get("uf_promedio", 39600)
tc_val = macro_data.get("dolar_observado_promedio", 930)

# Calculos Inmobiliarios
val_prop_uf = 14500
deuda_clp = 0
if df_cartera is not None and not df_cartera.empty:
    pasivo = df_cartera[df_cartera["bucket_sim"] == "PASIVO"]["saldo_nominal"].sum()
    if pasivo == 0: pasivo = df_cartera[df_cartera["tipo"] == "Pasivo Inmobiliario"]["saldo_nominal"].sum()
    deuda_clp = pasivo * uf_val

neto_inmo = (val_prop_uf * uf_val) - deuda_clp

if df_cartera is not None:
    grp = df_cartera.groupby("bucket_sim")["saldo_clp"].sum()
    st.session_state.datos_cargados = {
        'rf': grp.get("RF", 0), 'mx': grp.get("MX", 0), 'rv': grp.get("RV", 0),
        'usd': df_cartera[df_cartera["bucket_sim"]=="USD"]["saldo_nominal"].sum(),
        'tc': tc_val, 'inmo_neto': neto_inmo
    }

def fmt(v): return f"${int(v):,}".replace(",", ".")

def render_home():
    st.title("Panel Gestión Patrimonial")
    st.caption(f"Datos al: {fecha_corte}")
    if df_cartera is not None:
        activos = df_cartera[df_cartera["bucket_sim"]!="PASIVO"]["saldo_clp"].sum()
        pasivos = df_cartera[df_cartera["bucket_sim"]=="PASIVO"]["saldo_clp"].sum()
        c1, c2 = st.columns(2)
        c1.metric("Patrimonio Financiero", fmt(activos - pasivos))
        c2.metric("Inmobiliario Neto", fmt(neto_inmo))
    
    if st.button("🔮 IR AL SIMULADOR", type="primary"): 
        st.session_state.vista_actual = 'SIMULADOR'; st.rerun()

def render_simulador():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    d = st.session_state.datos_cargados
    
    # PREPARACIÓN DE DATOS PARA V7
    saldo_rf = d.get('rf', 0)
    saldo_mx = d.get('mx', 0)
    saldo_usd_clp = d.get('usd', 0) * d.get('tc', 930)
    
    # Consolidación: Defensa vs Motor
    total_defensa = saldo_rf + saldo_mx + saldo_usd_clp
    total_motor = d.get('rv', 0)

    # LLAMADA AL MOTOR V7 (Argumentos corregidos)
    simulador.app(
        default_rf=total_defensa,      
        default_rv=total_motor,        
        default_inmo_neto=d.get('inmo_neto', 0)
    )

if st.session_state.vista_actual == 'HOME': render_home()
elif st.session_state.vista_actual == 'SIMULADOR': render_simulador()
