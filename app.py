import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import tensorflow as tf

# Configurar la página
st.set_page_config(page_title="Predicción Cognitiva", layout="centered")

# Mostrar archivos en directorio para debug
st.write("Archivos en el directorio actual:", os.listdir())

# Cargar datos y modelo
@st.cache_resource
def cargar_datos_y_modelo():
    df = pd.read_csv("human_cognitive_performance.csv")

    # Codificar y escalar
    df_encoded = df.copy()
    label_encoders = {}
    for col in ['Gender', 'Diet_Type', 'Exercise_Frequency']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    columnas_a_escalar = [
        'Age', 'Sleep_Duration', 'Stress_Level', 'Daily_Screen_Time',
        'Caffeine_Intake', 'Reaction_Time', 'Memory_Test_Score'
    ]
    df_encoded[columnas_a_escalar] = scaler.fit_transform(df_encoded[columnas_a_escalar])

    df_encoded = df_encoded.drop(columns=['User_ID', 'AI_Predicted_Score'])

    X = df_encoded.drop(columns=['Cognitive_Score'])
    y = df_encoded['Cognitive_Score']

    model_ann = load_model("modelo_cognitivo")

    # Entrenar modelo de regresión
    reg = LinearRegression()
    reg.fit(X, y)

    return df, model_ann, reg, scaler, label_encoders, columnas_a_escalar, X.columns.tolist()

# Cargar datos y modelo
df, model_ann, reg_model, scaler, label_encoders, columnas_a_escalar, input_features = cargar_datos_y_modelo()

# --- Interfaz ---
st.title("🧠 Predicción de Puntuación Cognitiva")
st.markdown("Ingresa tus datos para estimar tu puntuación cognitiva.")

# Entradas del usuario
user_input = {}
user_input['Age'] = st.slider("Edad", 18, 90, 30)
user_input['Gender'] = st.selectbox("Género", ['Male', 'Female', 'Other'])
user_input['Sleep_Duration'] = st.slider("Horas de sueño por día", 0.0, 12.0, 7.0)
user_input['Stress_Level'] = st.slider("Nivel de estrés (1-10)", 1, 10, 5)
user_input['Diet_Type'] = st.selectbox("Tipo de dieta", ['Vegetarian', 'Non-Vegetarian', 'Vegan'])
user_input['Exercise_Frequency'] = st.selectbox("Frecuencia de ejercicio", ['Low', 'Medium', 'High'])
user_input['Daily_Screen_Time'] = st.slider("Horas frente a pantallas", 0, 16, 6)
user_input['Caffeine_Intake'] = st.slider("Consumo de cafeína (mg/día)", 0, 500, 100)
user_input['Reaction_Time'] = st.slider("Tiempo de reacción (ms)", 100, 1000, 300)
user_input['Memory_Test_Score'] = st.slider("Puntaje de test de memoria", 0, 100, 70)

# Convertir entradas
df_user = pd.DataFrame([user_input])

# Antes de codificar, chequea que el valor existe en LabelEncoder para evitar error
for col, le in label_encoders.items():
    if df_user[col].iloc[0] not in le.classes_:
        st.error(f"El valor '{df_user[col].iloc[0]}' no está en las clases del encoder para '{col}'.")
    else:
        df_user[col] = le.transform(df_user[col])

df_user[columnas_a_escalar] = scaler.transform(df_user[columnas_a_escalar])

# Reordenar columnas
df_user = df_user[input_features]
st.write("=== DEBUG INPUT df_user ===")
st.write(df_user.values)              # muestra el dataframe completo con datos codificados y escalados




# Predicciones
if st.button("Predecir puntuación cognitiva"):
    X_pred = df_user.values.astype(np.float32)
    infer = model_ann.signatures["serving_default"]
    result = infer(tf.constant(X_pred))
    pred_ann = list(result.values())[0].numpy()[0][0]
    pred_reg = reg_model.predict(df_user)[0]


    st.subheader("🧾 Resultados de Predicción:")
    st.write(f"🔹 **Red Neuronal:** {pred_ann:.2f}")
    st.write(f"🔹 **Regresión Lineal:** {pred_reg:.2f}")

# Visualizaciones
st.subheader("📊 Exploración de Datos")

tab1, tab2, tab3 = st.tabs(["Cognitive Score", "Relaciones", "Distribuciones"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Cognitive_Score'], kde=True, ax=ax1)
    ax1.set_title("Distribución del Cognitive Score")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Stress_Level', y='Cognitive_Score', data=df, ax=ax2)
    ax2.set_title("Cognitive Score vs Stress Level")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Exercise_Frequency', y='Cognitive_Score', data=df, ax=ax3)
    ax3.set_title("Cognitive Score vs Exercise Frequency")
    st.pyplot(fig3)

with tab3:
    fig4, ax4 = plt.subplots()
    sns.histplot(df['Age'], ax=ax4)
    ax4.set_title("Distribución de Edad")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots()
    sns.countplot(x='Gender', hue='Diet_Type', data=df, ax=ax5)
    ax5.set_title("Género vs Dieta")
    st.pyplot(fig5)
