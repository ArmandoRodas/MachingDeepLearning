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
    seccion = st.radio("Elige la aplicacion", ["Prediccion de Gastos", "Prediccion de Abandono"])

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
elif seccion == "Prediccion de Abandono":
    st.header("🔍 Predicción de Abandono Universitario")
    st.markdown("Completa el siguiente formulario con tus datos:")

    estudios_previos = st.selectbox("¿Usted tiene estudios universitarios?", ["Sí", "No"])
    inscrito = st.selectbox("¿Está inscrito en la Universidad actualmente?", ["Sí", "No"])
    reprobado = st.selectbox("¿Ha reprobado alguna materia?", ["Sí", "No"])
    solvente = st.selectbox("¿Está solvente actualmente con la Universidad?", ["Sí", "No"])
    empleo = st.selectbox("¿Tienes empleo actualmente?", ["Sí", "No"])
    traslado = st.slider("¿Cuánto tiempo tardas en llegar a la universidad? (en horas)", 0.0, 5.0, 1.0, step=0.5)

    if st.button("▶️ Predecir"):
        try:
            model = load_model("models_abandono/modelo_abandono.h5")
            scaler = joblib.load("models_abandono/scaler.pkl")

            map_si_no = {"Sí": 1, "No": 0}
            datos = np.array([[map_si_no[estudios_previos], map_si_no[inscrito], map_si_no[reprobado],
                               map_si_no[solvente], map_si_no[empleo], traslado]])
            datos_escalados = scaler.transform(datos)
            prob = model.predict(datos_escalados)[0][0]

            st.write(f"📊 Probabilidad de abandono: **{prob:.2%}**")
            if prob > 0.5:
                st.error("🔴 Riesgo ALTO de abandono universitario.")
            else:
                st.success("🟢 Riesgo BAJO de abandono universitario.")
        except Exception as e:
            st.error(f"❌ Error al cargar modelo o predecir: {e}")
