import streamlit as st
import pandas as pd
import joblib

# Configurar pagina
st.set_page_config(page_title="IA Sanarate", layout="wide")
st.image("assets/Inteligencia-artificial.jpg", use_container_width=True)
st.markdown("<h1 style='text-align:center;'>Inteligencia artificial</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>9no. Ingenieria en Sistemas Sanarate</h3>", unsafe_allow_html=True)
st.write("---")

# Menu lateral
with st.sidebar.expander("🔧 Menu", expanded=True):
    seccion = st.radio("Elige la aplicacion", ["Prediccion de Gastos", "Proyecto Deep Learning"])

# Cargar pipeline entrenado
@st.cache_resource
def load_model():
    return joblib.load("models/expenses_model.pkl")

# App de prediccion
if seccion == "Prediccion de Gastos":
    try:
        model = load_model()

        st.subheader("🧮 Parametros de Entrada")

        # Inputs numericos
        comidas_uni = st.number_input("Comidas en la Universidad", min_value=0, step=1)
        edad = st.number_input("Edad", min_value=18, step=1)
        cursos_dia = st.number_input("Cursos en el Dia", min_value=0, step=1)

        # Inputs categoricos
        datos = {
            "lugar_de_origen": st.selectbox("Lugar de Origen", ["Sansare", "Jalapa", "Guatemala", "Guastatoya", "Sanarate", "Agua Caliente", "San Antonio La Paz"]),
            "transporte_en_el_que_viaja": st.selectbox("Transporte en el que Viaja", ["Bus", "Moto", "Carro", "A pie"]),
            "compra_snacks": st.selectbox("Compra Snacks", ["Si", "No"]),
            "actividades_extra": st.selectbox("Actividades Extra en la Uni", ["Si", "No"]),
            "lleva_almuerzo": st.selectbox("Lleva Almuerzo", ["Si", "No"]),
            "compra_almuerzo": st.selectbox("Compra Almuerzo", ["Si", "No"]),
            "ocupacion": st.selectbox("Ocupacion", ["Estudia", "Trabaja", "Ambas"]),
            "desayuno_casa": st.selectbox("Desayuno en Casa", ["Si", "No"]),
            "compra_desayuno": st.selectbox("Compra Desayuno", ["Si", "No"]),
            "comparte_transporte": st.selectbox("Comparte Transporte", ["Si", "No"]),
            "hecha_o_da_dinero_para_gasolina": st.selectbox("Hecha o da dinero para gasolina", ["Si", "No"]),
            "comidas_en_la_universidad": comidas_uni,
            "edad": edad,
            "cursos_dia": cursos_dia
        }

        if st.button("▶️ Calcular gasto"):
            df_input = pd.DataFrame([datos])
            pred = model.predict(df_input)[0]
            st.success(f"💰 Gasto estimado: Q{pred:.2f}")

        st.subheader("📊 Resultados")
        with st.expander("📄 Informacion del Proyecto"):
            st.markdown("""
                Este proyecto predice cuanto gasta un estudiante universitario el dia **domingo** cuando asiste a clases.

                **Datos considerados:**
                - Lugar de origen
                - Medio de transporte
                - Snacks, comidas, desayuno
                - Edad, cursos del dia, gasolina
                - Ocupacion (trabaja, estudia, ambas)
                - Comparte transporte 
                - Hecha o da dinero para gasolina

                **Modelo utilizado:** Regresion RidgeCV con escalado numerico y codificacion ordinal.
                
                **Plataforma:** Streamlit Cloud + GitHub (correo universitario).
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ocurrio un error al cargar el pipeline: {e}")

# Placeholder
else:
    st.header("🤖 Proyecto Deep Learning")
    st.info("Proximamente se integrara aqui el modelo basado en redes neuronales.")
