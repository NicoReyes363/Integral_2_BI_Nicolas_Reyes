import streamlit as st
import pandas as pd
import joblib

# 1. Configuración de la página web
st.set_page_config(
    page_title="ConecTel - Predicción de Morosidad",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📊 Sistema de Alerta Temprana — ConecTel S.A.")
st.markdown("""
Esta aplicación permite al equipo de cobranza y al área comercial evaluar proactivamente 
el riesgo de que un cliente caiga en **mora severa (>90 días)** en los próximos 6 meses.
""")

# 2. Función para cargar el archivo
@st.cache_resource
def cargar_componentes_modelo():
    try:
        # Cargamos tu archivo exacto
        componentes = joblib.load("componentes_conectel_rf.joblib")
        
        # Como tu archivo es un diccionario estructurado, extraemos sus componentes
        if isinstance(componentes, dict):
            modelo = componentes.get('modelo', None)
            # Intentamos extraer escalador o codificador si los guardaste con estas llaves comunes
            scaler = componentes.get('scaler', componentes.get('escalador', None))
            encoder = componentes.get('encoder', componentes.get('codificador', None))
            return modelo, scaler, encoder
        else:
            # Si por alguna razón se guardó el modelo o un Pipeline directo
            return componentes, None, None
            
    except Exception as e:
        st.error(f"❌ Error crítico al leer 'componentes_conectel_rf.joblib': {e}")
        return None, None, None

# Extraemos el modelo entrenado (RandomForestClassifier) y sus componentes asociados
model, scaler, encoder = cargar_componentes_modelo()

if model is None:
    st.error("No se pudo inicializar la aplicación. Asegúrate de que 'componentes_conectel_rf.joblib' esté subido en la raíz de tu repositorio de GitHub.")
    st.stop()
else:
    st.sidebar.success("✅ Componentes de ConecTel RF cargados con éxito.")

# 3. Formulario de entrada de datos para el usuario (Variables de tu entrega)
st.header("📋 Datos del Cliente a Evaluar")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)
    
    with col1:
        dias_mora_hist = st.number_input(
            "Días de mora históricos (`dias_mora_hist`):", 
            min_value=0, max_value=365, value=0,
            help="Historial de días de retraso en pagos anteriores."
        )
        contrato = st.selectbox(
            "Tipo de Contrato:", 
            options=["Mensual", "Anual", "Dos Anos"], 
            index=0
        )
        
    with col2:
        indice_conflictividad = st.slider(
            "Índice de Conflictividad:", 
            min_value=0, max_value=10, value=0,
            help="Cantidad de reclamos y llamadas al soporte técnico."
        )
        plan = st.selectbox(
            "Tipo de Plan:", 
            options=["Básico", "Premium", "Intermedio"], 
            index=0
        )

    # Botón de envío del formulario
    entregado = st.form_submit_button("Evaluar Riesgo de Morosidad")

# 4. Procesamiento de datos y predicción con el Random Forest
if entregado:
    # Creamos un diccionario con el mismo formato y nombres de columnas que usaste al entrenar
    datos_cliente = {
        'dias_mora_hist': [dias_mora_hist],
        'contrato': [contrato],
        'plan': [plan],
        'indice_conflictividad': [indice_conflictividad]
    }
    
    df_input = pd.DataFrame(datos_cliente)
    
    try:
        # NOTA IMPORTANTE DE EJECUCIÓN:
        # Si guardaste tu modelo usando un Pipeline de scikit-learn en la llave 'modelo', 
        # las siguientes dos líneas procesarán y predecirán todo automáticamente:
        prediccion = model.predict(df_input)[0]
        probabilidad_mora = model.predict_proba(df_input)[0][1]
        
        st.subheader("🔍 Resultado del Análisis Técnico")
        
        # Despliegue de resultados basados en tus conclusiones estratégicas
        if prediccion == 1:
            st.error(f"🚨 **ALTO RIESGO DE MOROSIDAD** (Probabilidad: {probabilidad_mora:.2%})")
            st.markdown("""
            **Recomendaciones Estratégicas para el Equipo de Cobranza:**
            1. **Gestión Preventiva Activa:** Este cliente debe ingresar de inmediato a la bitácora de alertas tempranas debido a sus antecedentes.
            2. **Acción Comercial Directa:** Si su contrato actual es *Mensual*, ofrecer incentivos o promociones urgentes para migrarlo a una modalidad *Anual* para mitigar el riesgo.
            3. **Intervención Temprana:** Monitorear de cerca su *Índice de Conflictividad*. Resolver sus reclamos técnicos antes de que se transformen en una causa de no pago.
            """)
        else:
            st.success(f"✅ **CLIENTE DE BAJO RIESGO** (Probabilidad de mora: {probabilidad_mora:.2%})")
            st.markdown("**Recomendación:** Mantener las condiciones comerciales estándar y realizar un monitoreo trimestral preventivo regular.")
            
    except Exception as error_pred:
        st.error(f"Hubo un problema al procesar los datos para la predicción: {error_pred}")
        st.info("""
        💡 **Consejo Técnico:** Si el error indica problemas con variables de texto o dimensiones de la matriz (ej. *ValueError*), 
        significa que en tu notebook no usaste un `Pipeline` unificado, sino que aplicaste el escalador y el codificador por separado. 
        Si es el caso, debes usar los objetos `scaler` y `encoder` cargados en la línea 26 para transformar `df_input` antes de pasarlo al `model.predict()`.
        """)