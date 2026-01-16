import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import simulador 
import shutil
from datetime import datetime

st.set_page_config(page_title="Gestión Patrimonial", layout="wide", page_icon="🏦")

# --- RUTAS ---
DATA_FOLDER = "Data"
FILE_MENSUAL = os.path.join(DATA_FOLDER, "macro_instrumentos_2025-11.json")

# --- ESTADO ---
if 'vista_actual' not in st.session_state: st.session_state.vista_actual = 'HOME'
if 'datos_cargados' not in st.session_state: st.session_state.datos_cargados = {}

# --- CARGA DE DATOS ---
@st.cache_data
def cargar_y_clasificar_datos():
    if not os.path.exists(FILE_MENSUAL): return None, {}, "Error: Archivo no encontrado"
    try:
        with open(FILE_MENSUAL, "r", encoding="utf-8") as f: data = json.load(f)
        reg = data["registros"][0]
        df = pd.DataFrame(reg.get("instrumentos", []))
        
        if not df.empty:
            cols_num = ["saldo_clp", "saldo_nominal", "rentabilidad_mensual_pct", "valor_cuota"]
            for col in cols_num:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            def clasificar_global(row):
                t = str(row.get("tipo", "")).lower(); n = str(row.get("nombre", "")).lower()
                if "pasivo" in t or "hipoteca" in t: return "PASIVO"
                if "inmobiliario" in t: return "INMOBILIARIO" 
                if "riesgo" in n or "crypto" in n: return "INV. RIESGO (*)"
                if "caja" in t or "cuenta" in t or "dap" in t or "liquidez" in n: return "LIQUIDEZ"
                if "afp" in t or "apv" in t: return "RENTA VARIABLE"
                if "acciones" in n or "equity" in n or "agresivo" in n or "fondo a" in n: return "RENTA VARIABLE"
                if "bonos" in n or "deuda" in n or "conservador" in n or "uf" in n: return "RENTA FIJA (INV)"
                if "mixto" in n or "moderado" in n: return "RENTA VARIABLE"
                return "OTROS ACTIVOS"

            df["Categoria_Global"] = df.apply(clasificar_global, axis=1)
            
            def bucket_sim(row):
                cat = row["Categoria_Global"]
                if row["moneda"] == "USD" and cat == "LIQUIDEZ": return "USD"
                if cat == "RENTA VARIABLE" or cat == "INV. RIESGO (*)": return "RV"
                if "mixto" in str(row.get("nombre","")).lower(): return "MX"
                return "RF"
            
            df["bucket_sim"] = df.apply(bucket_sim, axis=1)

        return df, reg.get("macro", {}), reg.get("fecha_dato", "N/A")
    except Exception as e:
        return None, {}, "Error"

df_cartera, macro_data, fecha_corte = cargar_y_clasificar_datos()
uf_val = macro_data.get("uf_promedio", 39600)
tc_val = macro_data.get("dolar_observado_promedio", 930)

# CÁLCULO PATRIMONIAL PREVIO PARA CARGAR DATOS
# Necesitamos calcular el Neto Inmobiliario para pasarlo al simulador
valor_prop_uf_default = 14500 # Default si no hay input
deuda_hipo_clp_default = 0

if df_cartera is not None and not df_cartera.empty:
    deuda_hipo_uf = df_cartera[df_cartera["tipo"] == "Pasivo Inmobiliario"]["saldo_nominal"].sum()
    deuda_hipo_clp_default = deuda_hipo_uf * uf_val

neto_inmo_estimado = (valor_prop_uf_default * uf_val) - deuda_hipo_clp_default

if df_cartera is not None and not df_cartera.empty:
    grp = df_cartera.groupby("bucket_sim")["saldo_clp"].sum()
    st.session_state.datos_cargados = {
        'rf': grp.get("RF", 0), 'mx': grp.get("MX", 0), 'rv': grp.get("RV", 0),
        'usd': df_cartera[df_cartera["moneda"]=="USD"]["saldo_nominal"].sum(),
        'tc': tc_val,
        'inmo_neto': neto_inmo_estimado # Nuevo dato
    }

def calcular_totales(df, val_prop_uf, deuda_tc_clp):
    if df is None: return {}
    deuda_hipo_uf = df[df["tipo"] == "Pasivo Inmobiliario"]["saldo_nominal"].sum()
    deuda_hipo_clp = deuda_hipo_uf * uf_val
    pasivos_totales = deuda_hipo_clp + deuda_tc_clp
    df_act = df[df["Categoria_Global"] != "PASIVO"]
    mapa_activos = df_act.groupby("Categoria_Global")["saldo_clp"].sum().to_dict()
    riesgo_json = mapa_activos.get("INV. RIESGO (*)", 0)
    riesgo_manual = 380000 * tc_val if riesgo_json == 0 else 0
    total_riesgo = riesgo_json + riesgo_manual
    activo_inmo_clp = val_prop_uf * uf_val
    mapa_activos["INMOBILIARIO"] = activo_inmo_clp
    mapa_activos["INV. RIESGO (*)"] = total_riesgo
    activos_brutos = sum(mapa_activos.values())
    patrimonio_neto = activos_brutos - pasivos_totales
    
    return {
        "activos_brutos": activos_brutos,
        "pasivos_totales": pasivos_totales,
        "patrimonio_neto": patrimonio_neto,
        "pn_sin_especulativos": patrimonio_neto - total_riesgo,
        "mapa_activos": mapa_activos,
        "neto_inmo": activo_inmo_clp - deuda_hipo_clp,
        "neto_liq": mapa_activos.get("LIQUIDEZ", 0) - deuda_tc_clp,
        "deuda_hipo_clp": deuda_hipo_clp,
        "dividendo_uf": df[df["tipo"]=="Pasivo Inmobiliario"]["valor_cuota"].sum()
    }

def fmt(v, s="$"): return f"{s} {v:,.0f}".replace(",", ".")

def render_home():
    res = calcular_totales(df_cartera, 14500, 93500000)
    pn = res.get("patrimonio_neto", 0)
    
    # Actualizar el neto inmo en session state con el cálculo preciso
    st.session_state.datos_cargados['inmo_neto'] = res.get("neto_inmo", 0)

    st.title("Panel Gestión Patrimonial")
    st.caption(f"📅 Corte: {fecha_corte}")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Patrimonio Neto Real", fmt(pn))
    k2.metric("En Dólares", fmt(pn/tc_val, "US$ "))
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: 
        if st.button("📊 MIX PATRIMONIAL", use_container_width=True): st.session_state.vista_actual = 'SEGUIMIENTO'; st.rerun()
    with c2: 
        if st.button("🔮 SIMULADOR PRO", type="primary", use_container_width=True): st.session_state.vista_actual = 'SIMULADOR'; st.rerun()
    with c3: 
        if st.button("✅ VALIDACIÓN", use_container_width=True): st.session_state.vista_actual = 'VALIDACION'; st.rerun()

def render_seguimiento():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    with st.sidebar:
        val_prop_uf = st.number_input("Valor Prop (UF)", value=14500)
        deuda_tc = st.number_input("Deuda TC", value=93500000)
    
    res = calcular_totales(df_cartera, val_prop_uf, deuda_tc)
    st.title("📑 Informe Mix")
    st.dataframe(pd.DataFrame([
        ["Activos", fmt(res["activos_brutos"])],
        ["Pasivos", fmt(res["pasivos_totales"])],
        ["Patrimonio Neto", fmt(res["patrimonio_neto"])]
    ], columns=["Concepto", "Monto"]), use_container_width=True)

def render_simulador():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    d = st.session_state.datos_cargados
    # Pasamos el neto inmobiliario
    simulador.app(
        default_rf=d.get('rf',0), 
        default_rv=d.get('rv',0), 
        default_usd_nominal=d.get('usd',0), 
        default_tc=d.get('tc',930),
        default_inmo_neto=d.get('inmo_neto', 0) # <--- NUEVO
    )

def render_validacion():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    st.title("🔍 Validación")
    st.dataframe(df_cartera)

if st.session_state.vista_actual == 'HOME': render_home()
elif st.session_state.vista_actual == 'SEGUIMIENTO': render_seguimiento()
elif st.session_state.vista_actual == 'SIMULADOR': render_simulador()
elif st.session_state.vista_actual == 'VALIDACION': render_validacion()
