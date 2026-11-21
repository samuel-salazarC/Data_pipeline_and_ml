import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_pipeline import DataPipeline

# =====================================
# Configuración de página
# =====================================
st.set_page_config(
    page_title="Sistema de Detección de Intrusiones",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# Estilos CSS profesionales
# =====================================
st.markdown("""
<style>

/* Fondo general */
body {
    background-color: #f5f7fa;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1e293b;
    color: white;
}

/* Títulos */
h1, h2, h3 {
    color: #0f172a;
    font-weight: 600;
}

/* Botones */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 14px;
    border: none;
}
.stButton > button:hover {
    background-color: #1d4ed8;
}

/* Cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# Sidebar
# =====================================
st.sidebar.markdown("## Panel de Control")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Módulos",
    [
        "Inicio",
        "Data Pipeline",
        "Visualización",
        "Modelo"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto de Ciencia de Datos")

# =====================================
# Función para cargar datasets
# =====================================
def load_dataset_from_github(filename):
    """Carga el dataset desde la carpeta local datos/ o, si no existe, desde GitHub."""
    local_path = os.path.join("datos", filename)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    url = f"https://raw.githubusercontent.com/samuel-salazarC/Data_pipeline_and_ml/main/datos/{filename}"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        st.error(f"No se pudo cargar el dataset {filename}")
        return None

# =====================================
# Pipeline
# =====================================
# Permite definir el token en .streamlit/secrets.toml (GITHUB_TOKEN = "...")
try:
    if "GITHUB_TOKEN" in st.secrets and not os.environ.get("GITHUB_TOKEN"):
        os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

pipeline = DataPipeline(
    dataset_name="dnkumars/cybersecurity-intrusion-detection-dataset",
    github_user="samuel-salazarC",
    github_repo="Data_pipeline_and_ml",
    github_branch="main"
)

# =====================================
# Inicio
# =====================================
if section == "Inicio":
    st.title("Detección de Intrusiones Cibernéticas")

    col1, col2, col3 = st.columns(3)

    col1.metric("Estado del Pipeline", "Activo")
    col2.metric("Modelo", "Random Forest")
    col3.metric("Versión", "1.0")

    st.markdown("---")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Descripción del Sistema")
    st.write(
        "Plataforma para la ejecución de pipelines de datos y modelos de machine learning "
        "orientados a la detección de intrusiones en redes."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Equipo")
    st.write("""
    - Nathan Ghenassia — Ciencia de Datos  
    - María Fernanda Camacho — Desarrollo Web  
    - Samuel Salazar — Ingeniería de Datos  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# Data Pipeline
# =====================================
elif section == "Data Pipeline":
    st.title("Ejecución de Pipeline")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(
        "Ejecuta el flujo completo de procesamiento de datos: ingesta, limpieza, transformación y almacenamiento."
    )

    if st.button("Ejecutar Pipeline"):
        with st.spinner("Procesando datos..."):
            logs = pipeline.run_pipeline()
        st.success("Ejecución completada")
        st.text_area("Logs", logs, height=250)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================
# Visualización
# =====================================
elif section == "Visualización":
    st.title("Análisis de Datos")

    df_procesado = load_dataset_from_github("dataset_procesado.csv")

    if df_procesado is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución de Ataques")
            fig, ax = plt.subplots()
            sns.countplot(x='attack_detected', data=df_procesado, ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Protocolos")
            protocol_counts = df_procesado['protocol_type'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

        st.subheader("Correlación")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df_procesado.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# =====================================
# Modelo
# =====================================
elif section == "Modelo":
    st.title("Modelo de Machine Learning")

    df_procesado = load_dataset_from_github("dataset_procesado.csv")

    if df_procesado is not None and st.button("Ejecutar Modelo"):

        X = df_procesado.drop(columns=['attack_detected'])
        y = df_procesado['attack_detected']

        # Separar entrenamiento y prueba para evaluar con datos no vistos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Cargar o entrenar
        if os.path.exists("scaler.pkl") and os.path.exists("best_model.pkl"):
            scaler = joblib.load("scaler.pkl")
            model = joblib.load("best_model.pkl")
            st.info("Modelo cargado desde best_model.pkl. Si se entrenó con todos los datos, la métrica puede estar inflada.")
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(model, "best_model.pkl")

        # Predicción sobre el conjunto de prueba
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)

        resultados = X_test.copy()
        resultados["Real"] = y_test.values
        resultados["Predicción"] = predictions

        # Métricas
        accuracy = accuracy_score(y_test, predictions)

        col1, col2 = st.columns(2)
        col1.metric("Exactitud (test)", f"{accuracy:.2%}")
        col2.metric("Registros de prueba", len(X_test))

        st.subheader("Predicciones")
        st.dataframe(resultados)

        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, predictions)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=["No Ataque", "Ataque"],
                    yticklabels=["No Ataque", "Ataque"],
                    ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

        st.pyplot(fig)