import os

import altair as alt
import pandas as pd
import requests
import streamlit as st
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from data_pipeline import DataPipeline

# =====================================
# Configuración general
# =====================================
st.set_page_config(
    page_title="Detección de Intrusiones",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET = "dnkumars/cybersecurity-intrusion-detection-dataset"
GITHUB_USER = "samuel-salazarC"
GITHUB_REPO = "Data_pipeline_and_ml"
GITHUB_BRANCH = "main"
TARGET = "attack_detected"

# Códigos que genera el LabelEncoder del pipeline (orden alfabético)
CATEGORIAS = {
    "protocol_type": {0: "ICMP", 1: "TCP", 2: "UDP"},
    "encryption_used": {0: "AES", 1: "DES"},
    "browser_type": {0: "Chrome", 1: "Edge", 2: "Firefox", 3: "Safari", 4: "Unknown"},
}

# Nombres legibles de cada variable
NOMBRES = {
    "network_packet_size": "Tamaño del paquete (bytes)",
    "protocol_type": "Protocolo",
    "login_attempts": "Intentos de inicio de sesión",
    "session_duration": "Duración de la sesión (s)",
    "encryption_used": "Cifrado",
    "ip_reputation_score": "Riesgo de la IP (0 a 1)",
    "failed_logins": "Inicios de sesión fallidos",
    "browser_type": "Navegador",
    "unusual_time_access": "Acceso en horario inusual",
}

CLASES = ["Normal", "Ataque"]
COLORES = ["#2a78d6", "#eb6834"]
ESCALA_CLASE = alt.Scale(domain=CLASES, range=COLORES)
GRIS_EJE = "#898781"


# =====================================
# Datos y modelo
# =====================================
@st.cache_data(show_spinner="Cargando datos...")
def cargar_dataset(filename, _version=None):
    """Lee el CSV de la carpeta datos/ o, si no existe, del repositorio en GitHub."""
    local_path = os.path.join("datos", filename)
    if os.path.exists(local_path):
        return pd.read_csv(local_path)
    url = (
        f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/"
        f"{GITHUB_BRANCH}/datos/{filename}"
    )
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException:
        return None
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    return None


def version_datos(filename):
    """Cambia cuando el archivo local se regenera, para refrescar la caché."""
    path = os.path.join("datos", filename)
    return os.path.getmtime(path) if os.path.exists(path) else None


def con_etiquetas(df):
    """Devuelve una copia con las categorías y la clase en texto legible."""
    out = df.copy()
    for col, mapa in CATEGORIAS.items():
        if col in out.columns:
            out[col] = out[col].map(mapa).fillna(out[col].astype(str))
    out["Clase"] = out[TARGET].map({0: "Normal", 1: "Ataque"})
    return out


@st.cache_resource(show_spinner="Entrenando el modelo...")
def entrenar_modelo(df, n_arboles, test_size):
    """Entrena un Random Forest y lo evalúa con datos que no vio al entrenar."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    modelo = RandomForestClassifier(n_estimators=n_arboles, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    return {
        "modelo": modelo,
        "columnas": list(X.columns),
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "n_train": len(X_train),
        "metricas": {
            "Exactitud": accuracy_score(y_test, pred),
            "Precisión": precision_score(y_test, pred),
            "Sensibilidad": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        },
    }


def obtener_datos():
    df = cargar_dataset("dataset_procesado.csv", version_datos("dataset_procesado.csv"))
    if df is None:
        st.error(
            "No se encontró `datos/dataset_procesado.csv`. "
            "Ve a **1 · Datos** y ejecuta el pipeline para generarlo."
        )
        st.stop()
    return df


def pct(valor):
    return f"{valor:.1%}".replace(".", ",")


def miles(valor):
    return f"{valor:,.0f}".replace(",", " ")


# =====================================
# Gráficos
# =====================================
def estilo(chart, alto=280):
    return (
        chart.properties(height=alto)
        .configure_view(stroke=None)
        .configure_axis(
            gridColor="#e1e0d9",
            domainColor="#c3c2b7",
            tickColor="#c3c2b7",
            labelColor=GRIS_EJE,
            titleColor="#52514e",
            titleFontWeight="normal",
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            orient="top", title=None, labelColor="#52514e", labelFontSize=12, symbolType="square"
        )
    )


def barras_tasa(df, columna, titulo_x, orden=None, alto=280):
    """Porcentaje de sesiones con ataque para cada valor de una variable."""
    resumen = (
        df.groupby(columna, observed=True)[TARGET]
        .agg(tasa="mean", sesiones="size")
        .reset_index()
    )
    orden = orden if orden is not None else "-y"
    chart = (
        alt.Chart(resumen)
        .mark_bar(
            color=COLORES[1],
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
            size=28 if len(resumen) <= 6 else 16,
        )
        .encode(
            x=alt.X(f"{columna}:N", title=titulo_x, sort=orden, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "tasa:Q",
                title="Sesiones con ataque",
                axis=alt.Axis(format=".0%", tickCount=5),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip(f"{columna}:N", title=titulo_x),
                alt.Tooltip("tasa:Q", title="Con ataque", format=".1%"),
                alt.Tooltip("sesiones:Q", title="Sesiones", format=","),
            ],
        )
    )
    return estilo(chart, alto)


def histograma_por_clase(df, columna, titulo_x, bins=30, alto=280):
    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.75, binSpacing=1)
        .encode(
            x=alt.X(f"{columna}:Q", bin=alt.Bin(maxbins=bins), title=titulo_x),
            y=alt.Y("count():Q", title="Sesiones", stack=None),
            color=alt.Color("Clase:N", scale=ESCALA_CLASE),
            tooltip=[
                alt.Tooltip("Clase:N"),
                alt.Tooltip(f"{columna}:Q", bin=alt.Bin(maxbins=bins), title=titulo_x),
                alt.Tooltip("count():Q", title="Sesiones", format=","),
            ],
        )
    )
    return estilo(chart, alto)


def barras_horizontales(datos, campo_valor, campo_nombre, titulo_x, formato, alto=300):
    base = alt.Chart(datos).encode(
        y=alt.Y(f"{campo_nombre}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(f"{campo_valor}:Q", title=titulo_x, axis=alt.Axis(format=formato, tickCount=5)),
        tooltip=[
            alt.Tooltip(f"{campo_nombre}:N", title="Variable"),
            alt.Tooltip(f"{campo_valor}:Q", title=titulo_x, format=formato),
        ],
    )
    barras = base.mark_bar(
        color=COLORES[0], cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=16
    )
    return estilo(barras, alto)


def matriz_confusion(y_real, y_pred):
    cm = confusion_matrix(y_real, y_pred)
    filas = [
        {"Real": CLASES[i], "Predicción": CLASES[j], "Sesiones": int(cm[i, j])}
        for i in range(2)
        for j in range(2)
    ]
    datos = pd.DataFrame(filas)
    base = alt.Chart(datos).encode(
        x=alt.X("Predicción:N", sort=CLASES, title="Lo que predijo el modelo",
                axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Real:N", sort=CLASES, title="Lo que realmente pasó"),
    )
    celdas = base.mark_rect(stroke="#fcfcfb", strokeWidth=3, cornerRadius=4).encode(
        color=alt.Color(
            "Sesiones:Q",
            scale=alt.Scale(range=["#cde2fb", "#184f95"]),
            legend=None,
        ),
        tooltip=["Real:N", "Predicción:N", alt.Tooltip("Sesiones:Q", format=",")],
    )
    umbral = int(cm.max() * 0.5)
    texto = base.mark_text(fontSize=20, fontWeight=600).encode(
        text=alt.Text("Sesiones:Q", format=","),
        color=alt.condition(
            alt.datum.Sesiones > umbral, alt.value("#ffffff"), alt.value("#0b0b0b")
        ),
    )
    return estilo(celdas + texto, 260), cm


# =====================================
# Barra lateral
# =====================================
PAGINAS = ["Inicio", "1 · Datos", "2 · Exploración", "3 · Modelo", "4 · Probar una sesión"]

if "pagina" not in st.session_state:
    st.session_state.pagina = PAGINAS[0]


def ir_a(pagina):
    st.session_state.pagina = pagina


with st.sidebar:
    st.markdown("### 🛡️ Detección de Intrusiones")
    st.caption("Pipeline de datos y modelo de machine learning")
    st.radio("Secciones", PAGINAS, key="pagina")
    st.divider()
    st.caption(
        "Sigue los pasos en orden: prepara los datos, explóralos, "
        "revisa el modelo y pruébalo con una sesión."
    )

# Token opcional desde .streamlit/secrets.toml (GITHUB_TOKEN = "...")
try:
    if "GITHUB_TOKEN" in st.secrets and not os.environ.get("GITHUB_TOKEN"):
        os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]
except Exception:
    pass

pagina = st.session_state.pagina

# =====================================
# Inicio
# =====================================
if pagina == "Inicio":
    st.title("Detección de intrusiones en redes")
    st.write(
        "Esta aplicación prepara un conjunto de datos de sesiones de red, permite explorarlo "
        "y entrena un modelo que estima si una sesión es un ataque."
    )

    df = obtener_datos()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sesiones analizadas", miles(len(df)))
    c2.metric("Sesiones con ataque", pct(df[TARGET].mean()))
    c3.metric("Variables por sesión", df.shape[1] - 1)
    c4.metric("Exactitud del modelo", pct(entrenar_modelo(df, 100, 0.2)["metricas"]["Exactitud"]))

    st.subheader("Cómo usar la aplicación")
    pasos = [
        ("1 · Datos", "Prepara los datos",
         "Descarga y limpia el dataset de Kaggle. Es opcional: ya hay una copia lista."),
        ("2 · Exploración", "Explora los datos",
         "Mira qué características tienen las sesiones con ataque frente a las normales."),
        ("3 · Modelo", "Revisa el modelo",
         "Consulta qué tan bien acierta el modelo y qué variables pesan más."),
        ("4 · Probar una sesión", "Prueba una sesión",
         "Ingresa los datos de una sesión y obtén la probabilidad de que sea un ataque."),
    ]
    columnas = st.columns(4)
    for col, (destino, titulo, texto) in zip(columnas, pasos):
        with col:
            with st.container(border=True):
                st.markdown(f"**{titulo}**")
                st.caption(texto)
                st.button("Abrir", key=f"ir_{destino}", on_click=ir_a, args=(destino,),
                          width="stretch")

    with st.expander("Equipo del proyecto"):
        st.markdown(
            "- **Nathan Ghenassia**: Ciencia de Datos\n"
            "- **María Fernanda Camacho**: Desarrollo Web\n"
            "- **Samuel Salazar**: Ingeniería de Datos"
        )

# =====================================
# 1 · Datos
# =====================================
elif pagina == "1 · Datos":
    st.title("Datos")
    st.write(
        "El pipeline descarga el dataset desde Kaggle, cifra el identificador de sesión, "
        "limpia los datos y guarda el resultado en la carpeta `datos/`."
    )

    etapas = [
        ("Descarga", "Obtiene el dataset desde Kaggle."),
        ("Cifrado", "Cifra la columna `session_id`."),
        ("Limpieza", "Quita filas incompletas y convierte las categorías a números."),
        ("Guardado", "Guarda los CSV en `datos/` y, si hay token, los sube a GitHub."),
    ]
    for col, (i, (nombre, detalle)) in zip(st.columns(4), enumerate(etapas, start=1)):
        with col:
            with st.container(border=True):
                st.markdown(f"**{i}. {nombre}**")
                st.caption(detalle)

    st.info(
        "Ejecutar el pipeline requiere credenciales de Kaggle configuradas en este equipo. "
        "Si no las tienes, puedes seguir con los datos que ya están guardados.",
        icon="ℹ️",
    )

    if st.button("Ejecutar pipeline", type="primary"):
        pipeline = DataPipeline(
            dataset_name=DATASET,
            github_user=GITHUB_USER,
            github_repo=GITHUB_REPO,
            github_branch=GITHUB_BRANCH,
        )
        with st.spinner("Descargando y procesando los datos..."):
            logs = pipeline.run_pipeline()
        if "Ocurrió un error" in logs or "Acceso denegado" in logs:
            st.error("El pipeline no terminó. Revisa el registro para ver el motivo.")
        else:
            st.success("Pipeline completado. Los datos de `datos/` están actualizados.")
            cargar_dataset.clear()
        with st.expander("Registro de la ejecución", expanded=True):
            st.code(logs or "Sin mensajes.", language=None)

    st.subheader("Vista previa de los datos")
    tab_proc, tab_orig = st.tabs(["Datos procesados", "Datos originales"])
    with tab_proc:
        df = obtener_datos()
        st.caption(
            f"{miles(len(df))} sesiones y {df.shape[1]} columnas. "
            "Las categorías se muestran con su nombre; el modelo las recibe como números."
        )
        vista = con_etiquetas(df).drop(columns=[TARGET]).rename(columns=NOMBRES)
        st.dataframe(vista.head(200), width="stretch", hide_index=True)
    with tab_orig:
        original = cargar_dataset("dataset_original.csv", version_datos("dataset_original.csv"))
        if original is None:
            st.warning("No se encontró `datos/dataset_original.csv`.")
        else:
            st.caption(
                f"{miles(len(original))} sesiones y {original.shape[1]} columnas, "
                "con `session_id` cifrado."
            )
            st.dataframe(original.head(200), width="stretch", hide_index=True)

# =====================================
# 2 · Exploración
# =====================================
elif pagina == "2 · Exploración":
    st.title("Exploración de los datos")
    df = con_etiquetas(obtener_datos())

    f1, f2, f3 = st.columns(3)
    protocolos = f1.multiselect("Protocolo", sorted(df["protocol_type"].unique()),
                                placeholder="Todos")
    navegadores = f2.multiselect("Navegador", sorted(df["browser_type"].unique()),
                                 placeholder="Todos")
    cifrados = f3.multiselect("Cifrado", sorted(df["encryption_used"].unique()),
                              placeholder="Todos")
    if protocolos:
        df = df[df["protocol_type"].isin(protocolos)]
    if navegadores:
        df = df[df["browser_type"].isin(navegadores)]
    if cifrados:
        df = df[df["encryption_used"].isin(cifrados)]

    if df.empty:
        st.warning("Ninguna sesión cumple con los filtros elegidos.")
        st.stop()

    ataques = int(df[TARGET].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Sesiones", miles(len(df)))
    c2.metric("Con ataque", miles(ataques))
    c3.metric("Porcentaje con ataque", pct(df[TARGET].mean()))

    st.subheader("Qué distingue a un ataque")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Ataques según los inicios de sesión fallidos**")
            st.caption("Porcentaje de sesiones con ataque para cada cantidad de fallos.")
            st.altair_chart(
                barras_tasa(df, "failed_logins", "Inicios de sesión fallidos", orden="ascending"),
                width="stretch",
            )
    with col2:
        with st.container(border=True):
            st.markdown("**Riesgo de la IP en sesiones normales y ataques**")
            st.caption("Cantidad de sesiones por nivel de riesgo de la IP.")
            st.altair_chart(
                histograma_por_clase(df, "ip_reputation_score", "Riesgo de la IP (0 a 1)"),
                width="stretch",
            )

    col3, col4 = st.columns(2)
    with col3:
        with st.container(border=True):
            st.markdown("**Ataques según los intentos de inicio de sesión**")
            st.caption("Porcentaje de sesiones con ataque para cada cantidad de intentos.")
            st.altair_chart(
                barras_tasa(df, "login_attempts", "Intentos de inicio de sesión",
                            orden="ascending"),
                width="stretch",
            )
    with col4:
        with st.container(border=True):
            st.markdown("**Ataques por categoría**")
            opciones = {"Navegador": "browser_type", "Protocolo": "protocol_type",
                        "Cifrado": "encryption_used"}
            eleccion = st.radio("Agrupar por", list(opciones), horizontal=True,
                                label_visibility="collapsed")
            st.altair_chart(
                barras_tasa(df, opciones[eleccion], eleccion, alto=250),
                width="stretch",
            )

    with st.container(border=True):
        st.markdown("**Relación de cada variable con los ataques**")
        st.caption(
            "Correlación con la presencia de ataque. Cuanto más alto el valor, "
            "más aumenta la variable junto con los ataques."
        )
        numerico = obtener_datos().loc[df.index]
        corr = (
            numerico.corr(numeric_only=True)[TARGET]
            .drop(TARGET)
            .dropna()
            .rename(index=NOMBRES)
            .reset_index()
        )
        corr.columns = ["Variable", "Correlación"]
        st.altair_chart(
            barras_horizontales(corr, "Correlación", "Variable", "Correlación con ataque", ".2f"),
            width="stretch",
        )

# =====================================
# 3 · Modelo
# =====================================
elif pagina == "3 · Modelo":
    st.title("Modelo")
    st.write(
        "Un Random Forest aprende de una parte de las sesiones y se evalúa con el resto, "
        "que no vio durante el entrenamiento."
    )
    df = obtener_datos()

    with st.expander("Ajustes del entrenamiento"):
        a1, a2 = st.columns(2)
        n_arboles = a1.slider("Cantidad de árboles", 50, 300, 100, step=50)
        test_pct = a2.slider("Porcentaje de datos para prueba", 10, 40, 20, step=5)

    r = entrenar_modelo(df, n_arboles, test_pct / 100)
    m = r["metricas"]

    st.caption(
        f"Entrenado con {miles(r['n_train'])} sesiones y evaluado con {miles(len(r['y_test']))}."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exactitud", pct(m["Exactitud"]),
              help="De todas las sesiones de prueba, cuántas clasificó bien.")
    c2.metric("Precisión", pct(m["Precisión"]),
              help="De las sesiones que marcó como ataque, cuántas lo eran.")
    c3.metric("Sensibilidad", pct(m["Sensibilidad"]),
              help="De los ataques reales, cuántos detectó.")
    c4.metric("F1", pct(m["F1"]),
              help="Balance entre precisión y sensibilidad.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**Aciertos y errores**")
            grafico, cm = matriz_confusion(r["y_test"], r["pred"])
            st.altair_chart(grafico, width="stretch")
            st.caption(
                f"El modelo dejó pasar {miles(cm[1, 0])} ataques como sesiones normales "
                f"y dio {miles(cm[0, 1])} falsas alarmas."
            )
    with col2:
        with st.container(border=True):
            st.markdown("**Variables que más pesan en la decisión**")
            imp = pd.DataFrame({
                "Variable": [NOMBRES.get(c, c) for c in r["columnas"]],
                "Importancia": r["modelo"].feature_importances_,
            })
            st.altair_chart(
                barras_horizontales(imp, "Importancia", "Variable", "Importancia", ".0%", 285),
                width="stretch",
            )

    with st.expander("Ver las predicciones sobre los datos de prueba"):
        tabla = con_etiquetas(r["X_test"].assign(**{TARGET: r["y_test"]}))
        tabla = tabla.drop(columns=[TARGET]).rename(columns={"Clase": "Real"})
        tabla["Predicción"] = pd.Series(r["pred"], index=tabla.index).map(
            {0: "Normal", 1: "Ataque"}
        )
        tabla["Acertó"] = tabla["Real"] == tabla["Predicción"]
        solo_errores = st.checkbox("Mostrar solo los errores")
        if solo_errores:
            tabla = tabla[~tabla["Acertó"]]
        st.dataframe(tabla.rename(columns=NOMBRES), width="stretch", hide_index=True)

# =====================================
# 4 · Probar una sesión
# =====================================
elif pagina == "4 · Probar una sesión":
    st.title("Probar una sesión")
    st.write(
        "Ingresa las características de una sesión de red y el modelo estima "
        "la probabilidad de que sea un ataque."
    )
    df = obtener_datos()
    r = entrenar_modelo(df, 100, 0.2)
    inverso = {col: {v: k for k, v in mapa.items()} for col, mapa in CATEGORIAS.items()}

    with st.form("sesion"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**Conexión**")
            protocolo = st.selectbox("Protocolo", list(inverso["protocol_type"]), index=1)
            cifrado = st.selectbox("Cifrado", list(inverso["encryption_used"]))
            navegador = st.selectbox("Navegador", list(inverso["browser_type"]))
        with b:
            st.markdown("**Sesión**")
            paquete = st.number_input(
                "Tamaño del paquete (bytes)", 64, 1500, int(df["network_packet_size"].median())
            )
            duracion = st.number_input(
                "Duración de la sesión (s)", 0.5, 7200.0,
                float(round(df["session_duration"].median())), step=10.0,
            )
            horario = st.toggle("Acceso en horario inusual")
        with c:
            st.markdown("**Autenticación**")
            intentos = st.slider("Intentos de inicio de sesión", 1, 15, 4)
            fallidos = st.slider("Inicios de sesión fallidos", 0, 5, 1)
            riesgo = st.slider(
                "Riesgo de la IP", 0.0, 1.0, 0.30, step=0.01,
                help="0 es una IP confiable y 1 es una IP de alto riesgo.",
            )
        enviado = st.form_submit_button("Evaluar sesión", type="primary")

    if enviado:
        sesion = pd.DataFrame([{
            "network_packet_size": paquete,
            "protocol_type": inverso["protocol_type"][protocolo],
            "login_attempts": intentos,
            "session_duration": duracion,
            "encryption_used": inverso["encryption_used"][cifrado],
            "ip_reputation_score": riesgo,
            "failed_logins": fallidos,
            "browser_type": inverso["browser_type"][navegador],
            "unusual_time_access": int(horario),
        }])[r["columnas"]]
        prob = float(r["modelo"].predict_proba(sesion)[0, 1])

        with st.container(border=True):
            izq, der = st.columns([1, 2])
            izq.metric("Probabilidad de ataque", pct(prob))
            with der:
                if prob >= 0.5:
                    st.error("**Posible ataque.** El modelo clasifica esta sesión como intrusión.",
                             icon="🚨")
                else:
                    st.success("**Sesión normal.** El modelo no detecta señales de intrusión.",
                               icon="✅")
                st.progress(prob)
        st.caption(
            f"Es una estimación de un modelo con {pct(r['metricas']['Exactitud'])} de exactitud "
            "en datos de prueba; no reemplaza una revisión de seguridad."
        )
    else:
        st.caption("Ajusta los valores y presiona **Evaluar sesión**.")
