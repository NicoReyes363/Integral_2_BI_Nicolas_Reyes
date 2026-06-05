import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Configuración de la página web
st.set_page_config(
    page_title="ConecTel - Predicción de Morosidad",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📊 Sistema de Alerta Temprana — ConecTel S.A.")
st.markdown("""
Esta aplicación evalúa proactivamente el riesgo de que un cliente caiga en **mora severa (>90 días)** en los próximos 6 meses utilizando el modelo predictivo optimizado de la compañía.
""")

# 2. Carga y desempaquetado de tus componentes de entrenamiento
@st.cache_resource
def cargar_todo():
    try:
        # Cargamos tu archivo exacto
        dict_componentes = joblib.load("componentes_conectel_rf.joblib")
        modelo = dict_componentes['modelo']
        scaler = dict_componentes['scaler']
        umbral = dict_componentes.get('umbral_optimo', 0.5)
        return modelo, scaler, umbral
    except Exception as e:
        st.error(f"❌ Error al cargar los componentes: {e}")
        return None, None, 0.5

model, scaler, umbral_optimo = cargar_todo()

if model is None:
    st.error("Por favor, asegúrate de que 'componentes_conectel_rf.joblib' esté en la raíz de tu repositorio de GitHub.")
    st.stop()
else:
    st.sidebar.success(f"✅ Random Forest cargado (Umbral Óptimo: {umbral_optimo:.4f})")

# Nombres exactos de las 40 columnas en el orden estricto que espera tu modelo
cols_modelo = [
    'edad', 'antiguedad_meses', 'tiene_internet', 'velocidad_mbps', 'tiene_tv',
    'tiene_linea_movil', 'num_servicios', 'factura_mensual_clp', 'dias_mora_hist',
    'reclamos_12m', 'llamadas_soporte_6m', 'nps', 'descuento_activo',
    'meses_sin_reajuste', 'ingreso_estimado_clp', 'cambios_plan_12m',
    'ratio_factura_ingreso', 'indice_conflictividad', 'antiguedad_categoria',
    'region_Araucanía', 'region_Atacama', 'region_Biobío', 'region_Coquimbo',
    'region_Los Lagos', 'region_Maule', 'region_Metropolitana', "region_O'Higgins",
    'region_Valparaíso', 'genero_Femenino', 'genero_Masculino', 'genero_No binario',
    'genero_Prefiero no decir', 'tipo_contrato_Bianual', 'tipo_contrato_Mensual',
    'plan_Estándar', 'plan_Premium', 'metodo_pago_Débito automático',
    'metodo_pago_Efectivo', 'metodo_pago_Transferencia', 'metodo_pago_WebPay'
]

# Las 18 columnas numéricas que tu objeto 'scaler' sabe procesar
cols_scaler = [
    'edad', 'antiguedad_meses', 'tiene_internet', 'velocidad_mbps', 'tiene_tv',
    'tiene_linea_movil', 'num_servicios', 'factura_mensual_clp', 'dias_mora_hist',
    'reclamos_12m', 'llamadas_soporte_6m', 'nps', 'descuento_activo',
    'meses_sin_reajuste', 'ingreso_estimado_clp', 'cambios_plan_12m',
    'ratio_factura_ingreso', 'indice_conflictividad'
]

# 3. Formulario UI para el Usuario
st.header("📋 Variables Críticas de Evaluación")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)
    
    with col1:
        dias_mora_hist = st.number_input(
            "Días de mora históricos:", min_value=0, max_value=365, value=0,
            help="Historial de días de retraso en pagos anteriores."
        )
        contrato = st.selectbox(
            "Tipo de Contrato:", options=["Mensual", "Anual", "Bianual"], index=0
        )
        
    with col2:
        indice_conflictividad = st.slider(
            "Índice de Conflictividad:", min_value=0, max_value=10, value=0,
            help="Cantidad de reclamos y fricción técnica acumulada."
        )
        plan = st.selectbox(
            "Tipo de Plan:", options=["Básico", "Estándar", "Premium"], index=0
        )
        
    # Variables Complementarias colapsables (para evitar dañar el diseño de tu entrega)
    with st.expander("⚙️ Ver variables complementarias del perfil (Valores Base)"):
        c3, c4 = st.columns(2)
        with c3:
            edad = st.number_input("Edad:", min_value=18, max_value=100, value=40)
            antiguedad_meses = st.number_input("Antigüedad (meses):", min_value=0, value=24)
            factura_mensual = st.number_input("Factura Mensual (CLP):", min_value=0, value=25000)
            ingreso_estimado = st.number_input("Ingreso Estimado (CLP):", min_value=0, value=650000)
            region = st.selectbox("Región:", options=["Metropolitana", "Valparaíso", "Biobío", "Araucanía", "Atacama", "Coquimbo", "Los Lagos", "Maule", "O'Higgins"])
        with c4:
            llamadas_soporte = st.number_input("Llamadas Soporte (6m):", min_value=0, value=1)
            reclamos_12m = st.number_input("Reclamos (12m):", min_value=0, value=0)
            nps = st.slider("Nota NPS asignada:", min_value=1, max_value=10, value=8)
            genero = st.selectbox("Género:", options=["Masculino", "Femenino", "No binario", "Prefiero no decir"])
            metodo_pago = st.selectbox("Método de Pago:", options=["WebPay", "Débito automático", "Efectivo", "Transferencia"])

    entregado = st.form_submit_button("🎯 Evaluar Riesgo de Morosidad")

# 4. Construcción matemática del vector y predicción
if entregado:
    # Creamos la fila base inicializada completamente en ceros
    df_input = pd.DataFrame(0.0, index=[0], columns=cols_modelo)
    
    # Asignamos los valores numéricos capturados
    df_input['edad'] = float(edad)
    df_input['antiguedad_meses'] = float(antiguedad_meses)
    df_input['tiene_internet'] = 1.0  # Por defecto activo
    df_input['velocidad_mbps'] = 100.0
    df_input['tiene_tv'] = 0.0
    df_input['tiene_linea_movil'] = 0.0
    df_input['num_servicios'] = 1.0
    df_input['factura_mensual_clp'] = float(factura_mensual)
    df_input['dias_mora_hist'] = float(dias_mora_hist)
    df_input['reclamos_12m'] = float(reclamos_12m)
    df_input['llamadas_soporte_6m'] = float(llamadas_soporte)
    df_input['nps'] = float(nps)
    df_input['descuento_activo'] = 0.0
    df_input['meses_sin_reajuste'] = 6.0
    df_input['ingreso_estimado_clp'] = float(ingreso_estimado)
    df_input['cambios_plan_12m'] = 0.0
    df_input['ratio_factura_ingreso'] = float(factura_mensual / (ingreso_estimado if ingreso_estimado > 0 else 1))
    df_input['indice_conflictividad'] = float(indice_conflictividad)
    df_input['antiguedad_categoria'] = 1.0
    
    # Escalamiento estricto con tu objeto entrenado para las 18 variables numéricas
    try:
        df_input[cols_scaler] = scaler.transform(df_input[cols_scaler])
    except Exception as e_scale:
        st.error(f"Error en el escalador numérico: {e_scale}")
        st.stop()
        
    # Mapeo manual del One-Hot Encoding según las selecciones del usuario
    if f"region_{region}" in df_input.columns:
        df_input[f"region_{region}"] = 1.0
    if f"genero_{genero}" in df_input.columns:
        df_input[f"genero_{genero}"] = 1.0
    if f"metodo_pago_{metodo_pago}" in df_input.columns:
        df_input[f"metodo_pago_{metodo_pago}"] = 1.0
        
    if contrato == "Mensual":
        df_input['tipo_contrato_Mensual'] = 1.0
    elif contrato == "Bianual":
        df_input['tipo_contrato_Bianual'] = 1.0
        
    if plan == "Estándar":
        df_input['plan_Estándar'] = 1.0
    elif plan == "Premium":
        df_input['plan_Premium'] = 1.0

    # Ejecución técnica de la predicción probabilística
    try:
        probabilidad_mora = model.predict_proba(df_input)[0][1]
        
        # Clasificamos usando tu umbral óptimo personalizado del diccionario
        alerta_activa = probabilidad_mora >= umbral_optimo
        
        st.subheader("🔍 Resultado de la Evaluación Técnica")
        
        if alerta_activa:
            st.error(f"🚨 **ALTO RIESGO DE MOROSIDAD SEVERA** (Probabilidad: {probabilidad_mora:.2%})")
            st.markdown(f"""
            **Plan de Acción Proactivo (ConecTel S.A.):**
            1. **Ingreso Automático:** El cliente ha superado el umbral de tolerancia estipulado ({umbral_optimo:.4f}). Debe incorporarse a la bitácora de alertas tempranas.
            2. **Contención Comercial:** Si el contrato es comercializado como *Mensual*, activar campaña inmediata para migrarlo a modalidad *Anual*.
            3. **Gestión Operacional:** Mitigar de forma prioritaria su nivel de fricción actual (`indice_conflictividad`) antes del cierre del ciclo de facturación.
            """)
        else:
            st.success(f"✅ **CLIENTE DE BAJO RIESGO** (Probabilidad: {probabilidad_mora:.2%})")
            st.markdown(f"**Nota:** La probabilidad estimada se encuentra por debajo del umbral crítico de control ({umbral_optimo:.4f}). Realizar seguimiento estándar.")
            
    except Exception as error_pred:
        st.error(f"Error interno al calcular la predicción en el modelo: {error_pred}")