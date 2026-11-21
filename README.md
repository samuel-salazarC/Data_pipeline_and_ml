# Detección de Intrusiones Cibernéticas: Data Pipeline y ML

Este proyecto implementa un pipeline ETL que descarga desde Kaggle el dataset [`dnkumars/cybersecurity-intrusion-detection-dataset`](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset), cifra la columna `session_id`, limpia y codifica los datos, y los guarda en `datos/` (o los sube a GitHub si se define `GITHUB_TOKEN`). Incluye además una aplicación en Streamlit para ejecutar el pipeline, visualizar los datos y entrenar un clasificador Random Forest que detecta ataques.

### Contexto
Proyecto universitario desarrollado en equipo por:
- **Nathan Ghenassia**: Ciencia de Datos
- **María Fernanda Camacho**: Desarrollo Web
- **Samuel Salazar**: Ingeniería de Datos

### Estructura del repositorio
- `data_pipeline.py`: clase `DataPipeline` (descarga desde Kaggle, cifrado de `session_id`, limpieza, codificación, guardado local y subida opcional a GitHub).
- `app.py`: aplicación Streamlit con las secciones Inicio, Datos, Exploración, Modelo y Probar una sesión.
- `.streamlit/config.toml`: tema visual de la aplicación.
- `ml_model.ipynb`: notebook de comparación de modelos con búsqueda genética de hiperparámetros.
- `datos/`: datasets generados por el pipeline (`dataset_original.csv` y `dataset_procesado.csv`).
- `roles.json`: roles de usuario y sus niveles de acceso al pipeline.
- `requirements.txt`: dependencias de Python.
- `Informe DataPipeline_ML.pdf`: informe del proyecto.

### Resultados
- El Random Forest de la app logra **~89 % de exactitud** sobre el 20 % de datos reservado para prueba.
- El notebook `ml_model.ipynb` compara SVC, árbol de decisión y MLP con búsqueda genética de hiperparámetros.

### Tecnologías
Python, pandas, scikit-learn, Streamlit, Kaggle API, cryptography.

---

# Guía de Instalación y Ejecución del Proyecto

## Requerimientos
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes herramientas:

### Software necesario
- Python 3.10 o superior
- Pip (gestor de paquetes de Python)
- Kaggle API configurada (para descargar datasets)
- Git (para clonar repositorios y autenticarse en GitHub)

### Librerías de Python
Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

Si no tienes `requirements.txt`, instala manualmente:

```bash
pip install pandas numpy scikit-learn sklearn-genetic-opt joblib matplotlib seaborn requests cryptography kaggle six streamlit
```

### Credenciales y archivos necesarios
1. **Token de Kaggle:**  
   - Crea una cuenta en [Kaggle](https://www.kaggle.com/).
   - Descarga tu `kaggle.json` desde `Account > API > Create New API Token`.
   - Ubica `kaggle.json` en la ruta `~/.kaggle/kaggle.json` (Linux/Mac) o `C:\Users\TU_USUARIO\.kaggle\kaggle.json` (Windows).

2. **Token de GitHub (opcional):**  
   - Solo es necesario si quieres que el pipeline suba los CSV al repositorio. Sin token, los datos se guardan en `datos/` y la subida se omite.
   - Genera un [token de GitHub](https://github.com/settings/tokens) con permiso de escritura sobre este repo y defínelo como variable de entorno:

     ```bash
     export GITHUB_TOKEN="tu_token"        # Linux/Mac
     $env:GITHUB_TOKEN="tu_token"          # Windows PowerShell
     ```

   - En Streamlit también puedes ponerlo en `.streamlit/secrets.toml` como `GITHUB_TOKEN = "tu_token"` (este archivo está en `.gitignore`).

3. **Roles de usuario:**  
   - Los roles y permisos se definen en `roles.json`, incluido en el repositorio.

---

## Orden de Ejecución
Sigue estos pasos para ejecutar correctamente todo el proyecto:

### 1. Ejecutar el pipeline de datos
```bash
python data_pipeline.py
```
Descarga los datos de Kaggle, los limpia, los guarda en `datos/` y, si hay `GITHUB_TOKEN`, los sube a GitHub.

### 2. Entrenar y comparar modelos (opcional)
Abre y ejecuta el notebook `ml_model.ipynb`.
Compara SVC, árbol de decisión y MLP con búsqueda genética de hiperparámetros.

### 3. Iniciar la aplicación web
```bash
streamlit run app.py
```
Inicia la interfaz en Streamlit para visualizar datos y probar el modelo.

---

## Solución de Problemas
- **No se encuentra `kaggle.json`**  
  Asegúrate de que el archivo está en `~/.kaggle/` o `C:\Users\TU_USUARIO\.kaggle\`.

- **Error de permisos en `roles.json`**  
  Asegúrate de que el usuario tiene `access_level: "full"` en `roles.json`.

- **El modelo da una precisión distinta a la esperada**  
  La app entrena el modelo al abrirse y lo evalúa sobre los datos de prueba (20 % por defecto). Puedes cambiar ese porcentaje en la sección Modelo.

---
