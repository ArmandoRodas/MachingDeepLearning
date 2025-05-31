import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

# Cargar los datos
df = pd.read_excel("data/datos_gasto_ampliado.xlsx", engine="openpyxl")
df.columns = df.columns.str.strip().str.lower()

# Renombrar columnas
df = df.rename(columns={
    'lugar de origen': 'lugar_de_origen',
    'transporte en el que viaja': 'transporte_en_el_que_viaja',
    'comidas en la uni': 'comidas_en_la_universidad',
    'compra snacks': 'compra_snacks',
    'actividades extra en la uni': 'actividades_extra',
    'lleva almuerzo': 'lleva_almuerzo',
    'compra almuerzo': 'compra_almuerzo',
    'ocupacion': 'ocupacion',
    'edad': 'edad',
    'cursos en el dia': 'cursos_dia',
    'desayuno en casa': 'desayuno_casa',
    'compra desayuno': 'compra_desayuno',
    'comparte transporte': 'comparte_transporte',
    'hecha o da dinero para gasolina': 'hecha_o_da_dinero_para_gasolina',
    'gasto_total_q': 'gasto_total'
})

# Columnas categóricas y numéricas
cat_cols = [
    "lugar_de_origen", "transporte_en_el_que_viaja", "compra_snacks", "actividades_extra",
    "lleva_almuerzo", "compra_almuerzo", "ocupacion", "desayuno_casa",
    "compra_desayuno", "comparte_transporte", "hecha_o_da_dinero_para_gasolina"
]

num_cols = ["comidas_en_la_universidad", "edad", "cursos_dia"]

# Separar variables
X = df[cat_cols + num_cols]
y = df["gasto_total"]

# Preprocesamiento
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols)
])

# Pipeline completo
pipeline = Pipeline([
    ("prep", preprocessor),
    ("reg", RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

# Entrenar
pipeline.fit(X, y)

# Crear carpeta si no existe
os.makedirs("models", exist_ok=True)

# Guardar modelo
joblib.dump(pipeline, "models/expenses_model.pkl")
print("✔ Modelo entrenado y guardado correctamente.")
