import streamlit as st
import json
import os
import pandas as pd
import simulador 

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
            cols_num = ["saldo_clp", "saldo_nominal", "rentabilidad_mensual_pct", "valor_cuota"]
            for col in cols_num:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            # Clasificador
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
            
            # Buckets para Simulador
            def bucket_sim(row):
                cat = row["Categoria_Global"]
                if row["moneda"] == "USD" and cat == "LIQUIDEZ": return "USD"
                if cat == "RENTA VARIABLE" or cat == "INV. RIESGO (*)": return "RV"
                if "mixto" in str(row.get("nombre","")).lower(): return "MX"
                return "RF"
            
            df["bucket_sim"] = df.apply(bucket_sim, axis=1)

        return df, reg.get("macro", {}), reg.get("fecha_dato", "N/A")
    except Exception as e: return None, {}, "Error"

df_cartera, macro_data, fecha_corte = cargar_datos()
uf_val = macro_data.get("uf_promedio", 39600)
tc_val = macro_data.get("dolar_observado_promedio", 930)

# --- CÁLCULO PATRIMONIAL ---
valor_prop_uf_default = 14500
deuda_hipo_clp_default = 0

if df_cartera is not None and not df_cartera.empty:
    deuda_hipo_uf = df_cartera[df_cartera["tipo"] == "Pasivo Inmobiliario"]["saldo_nominal"].sum()
    deuda_hipo_clp_default = deuda_hipo_uf * uf_val

neto_inmo_estimado = (valor_prop_uf_default * uf_val) - deuda_hipo_clp_default

if df_cartera is not None and not df_cartera.empty:
    grp = df_cartera.groupby("bucket_sim")["saldo_clp"].sum()
    st.session_state.datos_cargados = {
        'rf': grp.get("RF", 0), 
        'mx': grp.get("MX", 0), 
        'rv': grp.get("RV", 0),
        'usd': df_cartera[df_cartera["moneda"]=="USD"]["saldo_nominal"].sum(),
        'tc': tc_val,
        'inmo_neto': neto_inmo_estimado 
    }

def fmt(v, s="$"): return f"{s} {v:,.0f}".replace(",", ".")

# --- VISTAS ---
def render_home():
    st.title("Panel Gestión Patrimonial")
    st.caption(f"📅 Corte: {fecha_corte}")
    
    if df_cartera is not None:
        activos = df_cartera[df_cartera["Categoria_Global"]!="PASIVO"]["saldo_clp"].sum()
        pasivos = df_cartera[df_cartera["Categoria_Global"]=="PASIVO"]["saldo_clp"].sum()
        
        k1, k2 = st.columns(2)
        k1.metric("Patrimonio Financiero (Liq)", fmt(activos - pasivos))
        k2.metric("Inmobiliario Neto Est.", fmt(neto_inmo_estimado))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1: 
        if st.button("🔮 IR AL SIMULADOR", type="primary", use_container_width=True): 
            st.session_state.vista_actual = 'SIMULADOR'; st.rerun()
    with c2:
        if st.button("📊 VER DETALLE", use_container_width=True):
            st.info("Próximamente")

def render_simulador():
    if st.sidebar.button("🏠 Volver"): st.session_state.vista_actual = 'HOME'; st.rerun()
    d = st.session_state.datos_cargados
    # PASO DE PARAMETROS AL SIMULADOR (CORREGIDO)
    simulador.app(
        default_rf=d.get('rf',0), 
        default_mx=d.get('mx', 0), 
        default_rv=d.get('rv',0), 
        default_usd_nominal=d.get('usd',0), 
        default_tc=d.get('tc',930),
        default_inmo_neto=d.get('inmo_neto', 0)
    )

if st.session_state.vista_actual == 'HOME': render_home()
elif st.session_state.vista_actual == 'SIMULADOR': render_simulador()
