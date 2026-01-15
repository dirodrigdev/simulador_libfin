# NOMBRE DEL ARCHIVO: Home.py
import streamlit as st
import json
import os
import pandas as pd
import simulador # Importamos el motor

st.set_page_config(page_title="Gestión Patrimonial V3", layout="wide", page_icon="🏦")

# --- CONFIGURACIÓN DE RUTAS ---
# Asegúrate de que esta ruta coincida con tu carpeta real
DATA_FOLDER = "Data"
FILE_MENSUAL = os.path.join(DATA_FOLDER, "macro_instrumentos_2025-11.json")

# --- ESTADO DE SESIÓN ---
if 'vista_actual' not in st.session_state: st.session_state.vista_actual = 'HOME'
if 'datos_cargados' not in st.session_state: st.session_state.datos_cargados = {}

# --- CARGA DE DATOS (Tu lógica original mejorada) ---
@st.cache_data
def cargar_datos():
    if not os.path.exists(FILE_MENSUAL): return None, {}, "Archivo no encontrado"
    try:
        with open(FILE_MENSUAL, "r", encoding="utf-8") as f: data = json.load(f)
        reg = data["registros"][0]
        df = pd.DataFrame(reg.get("instrumentos", []))
        
        if not df.empty:
            # Limpieza numérica
            cols = ["saldo_clp", "saldo_nominal", "rentabilidad_mensual_pct"]
            for c in cols: 
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
            # Clasificador de Buckets (Heurística)
            def bucket(r):
                n = str(r.get("nombre","")).lower(); t = str(r.get("tipo","")).lower()
                if "pasivo" in t: return "PASIVO"
                # Regla simple: Si dice Fondo A, Accion, Equity o Riesgo -> RV
                if "fondo a" in n or "accion" in n or "agresivo" in n or "equity" in n or "riesgo" in n: return "RV"
                # Dólares líquidos
                if "dolar" in n or "usd" in n: return "USD" 
                return "RF" # El resto es Renta Fija

            df["bucket"] = df.apply(bucket, axis=1)
            
            # CÁLCULO DE RETORNO NOMINAL ANUALIZADO
            # Fórmula: ((1 + mensual/100)^12 - 1) * 100
            # Esto convierte tu "1.5% mensual" en un "19.5% anual" nominal
            df["ret_anual_nominal"] = df["rentabilidad_mensual_pct"].apply(lambda x: ((1+x/100)**12 - 1)*100 if x!=0 else 0)
            
        return df, reg.get("macro", {}), reg.get("fecha_dato", "N/A")
    except Exception as e: return None, {}, str(e)

df, macro, fecha = cargar_datos()
tc = macro.get("dolar_observado_promedio", 930)

# --- PROCESAMIENTO DE PROMEDIOS PONDERADOS ---
if df is not None and not df.empty:
    df_a = df[df["bucket"] != "PASIVO"]
    
    def w_avg(sub_df):
        tot = sub_df["saldo_clp"].sum()
        if tot == 0: return 0
        # Promedio ponderado: Suma(Saldo * Retorno) / Total Saldo
        return (sub_df["saldo_clp"] * sub_df["ret_anual_nominal"]).sum() / tot

    # Calculamos TUS rentabilidades reales promedio
    r_rf_real = w_avg(df_a[df_a["bucket"]=="RF"])
    r_rv_real = w_avg(df_a[df_a["bucket"]=="RV"])
    
    # Defaults de seguridad si el JSON viene vacío de rentabilidades
    if r_rf_real == 0: r_rf_real = 6.0
    if r_rv_real == 0: r_rv_real = 10.0

    st.session_state.datos_cargados = {
        'rf': df_a[df_a["bucket"]=="RF"]["saldo_clp"].sum(),
        'rv': df_a[df_a["bucket"]=="RV"]["saldo_clp"].sum(),
        'usd': df_a[df_a["bucket"]=="USD"]["saldo_clp"].sum(), # En CLP
        'usd_nom': df_a[df_a["bucket"]=="USD"]["saldo_nominal"].sum(), # En USD
        'tc': tc,
        'ret_rf': r_rf_real,
        'ret_rv': r_rv_real
    }

# --- VISTAS ---
def render_home():
    st.title("Gestión Patrimonial - Vista General")
    st.caption(f"📅 Datos cargados al: {fecha} | Dólar: ${tc:,.0f}")
    
    if df is not None:
        # Mostramos tus fondos y cuánto rinden ANUALMENTE (Nominal)
        st.markdown("### 📊 Tus Rentabilidades Anualizadas (Nominales)")
        st.dataframe(
            df[df["bucket"]!="PASIVO"][["nombre", "bucket", "saldo_clp", "ret_anual_nominal"]]
            .style.format({"saldo_clp": "${:,.0f}", "ret_anual_nominal": "{:.2f}%"}),
            use_container_width=True,
            height=250
        )
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Capital Líquido Total", f"$ {st.session_state.datos_cargados['rf'] + st.session_state.datos_cargados['rv'] + st.session_state.datos_cargados['usd']:,.0f}")
        c2.metric("Tu Retorno RF Promedio", f"{st.session_state.datos_cargados['ret_rf']:.2f}%")
        c3.metric("Tu Retorno RV Promedio", f"{st.session_state.datos_cargados['ret_rv']:.2f}%")
        
        st.info("💡 Estos promedios se enviarán al simulador como base para el escenario 'Personalizado'.")

    st.markdown("---")
    if st.button("🔮 IR AL SIMULADOR", type="primary", use_container_width=True):
        st.session_state.vista_actual = 'SIMULADOR'; st.rerun()

def render_simulador():
    if st.sidebar.button("🏠 Volver a Home"): st.session_state.vista_actual = 'HOME'; st.rerun()
    
    d = st.session_state.datos_cargados
    # Llamamos al motor simulador.py pasándole tus datos
    simulador.app(
        default_rf=d.get('rf', 0), 
        default_mx=0, # Asumimos MX integrado en los otros o 0
        default_rv=d.get('rv', 0),
        default_usd_nominal=d.get('usd_nom', 0), 
        default_tc=d.get('tc', 930),
        default_ret_rf=d.get('ret_rf', 6.0), 
        default_ret_rv=d.get('ret_rv', 10.0)
    )

# --- ROUTER ---
if st.session_state.vista_actual == 'HOME': render_home()
elif st.session_state.vista_actual == 'SIMULADOR': render_simulador()
