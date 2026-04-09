# 📌 **Guía de Instalación y Ejecución del Proyecto**

## 🛠 **Requerimientos**
Antes de ejecutar el código, asegúrate de tener instaladas las siguientes herramientas:

### 🔹 **Software Necesario**
- Python 3.8 o superior
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

2. **Token de GitHub:**  
   - Genera un [Token de GitHub](https://github.com/settings/tokens) con permisos de `repo`.
   - Guárdalo en `token_encrypt.py` y ejecuta este script para encriptarlo:

     ```bash
     python token_encrypt.py
     ```

3. **Roles de usuario:**  
   - Define roles y permisos en un archivo `roles.json` con el siguiente formato:

     ```json
     {
       "admin": { "access_level": "full" },
       "user": { "access_level": "restricted" }
     }
     ```

---

## 🚀 **Orden de Ejecución**
Sigue estos pasos para ejecutar correctamente todo el proyecto:

### **1️⃣ Encriptar Token de GitHub**
```bash
python token_encrypt.py
```
✔️ Esto creará `github_token.enc` con el token encriptado.

### **2️⃣ Ejecutar el Pipeline de Datos**
```bash
python data_pipeline.py
```
✔️ Descarga los datos de Kaggle, los limpia y los sube a GitHub.

### **3️⃣ Entrenar y Guardar el Modelo de Machine Learning**
```bash
python ml_model.py
```
✔️ Entrena el modelo y guarda `best_model.pkl` y `scaler.pkl`.

### **4️⃣ Iniciar la Aplicación Web**
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

- **No se ejecuta `app.py`**  
  ➜ Verifica que `best_model.pkl` y `scaler.pkl` existen en el directorio.

---
