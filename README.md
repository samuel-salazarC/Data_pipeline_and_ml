# 📌 **Guía de Instalación y Ejecución del Proyecto**

## 🛠 **Requerimientos**
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes herramientas:

### 🔹 **Software Necesario**
- Python 3.10 o superior
- Pip (gestor de paquetes de Python)
- Kaggle API configurada (para descargar datasets)
- Git (para clonar repositorios y autenticarse en GitHub)

### 🔹 **Librerías de Python**
Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

Si no tienes `requirements.txt`, instala manualmente:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn requests cryptography kaggle streamlit
```

### 🔹 **Credenciales y Archivos Necesarios**
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

## 🚀 **Orden de Ejecución**
Sigue estos pasos para ejecutar correctamente todo el proyecto:

### **1️⃣ Ejecutar el Pipeline de Datos**
```bash
python data_pipeline.py
```
✔️ Descarga los datos de Kaggle, los limpia, los guarda en `datos/` y, si hay `GITHUB_TOKEN`, los sube a GitHub.

### **2️⃣ Entrenar y Comparar Modelos (opcional)**
Abre y ejecuta el notebook `ml_model.ipynb`.
✔️ Compara SVC, árbol de decisión y MLP con búsqueda genética de hiperparámetros.

### **3️⃣ Iniciar la Aplicación Web**
```bash
streamlit run app.py
```
✔️ Inicia la interfaz en Streamlit para visualizar datos y probar el modelo.

---

## 🔧 **Solución de Problemas**
- **No se encuentra `kaggle.json`**  
  ➜ Asegúrate de que el archivo está en `~/.kaggle/` o `C:\Users\TU_USUARIO\.kaggle\`.
  
- **Error de permisos en `roles.json`**  
  ➜ Asegúrate de que el usuario tiene `access_level: "full"` en `roles.json`.

- **El modelo da una precisión distinta a la esperada**  
  ➜ La app evalúa sobre un 20 % de datos de prueba. Si existe un `best_model.pkl` antiguo, bórralo junto con `scaler.pkl` para reentrenar.

---
