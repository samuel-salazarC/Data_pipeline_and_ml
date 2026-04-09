import os
import json
import logging
import base64
import requests
import pandas as pd
from io import StringIO
from kaggle.api.kaggle_api_extended import KaggleApi
from cryptography.fernet import Fernet
from sklearn.preprocessing import LabelEncoder

class DataPipeline:
    def __init__(self, dataset_name, github_user, github_repo, github_branch, token_file="github_token.enc"):
        """
        Inicializa el pipeline de datos.
        """
        self.dataset_name = dataset_name
        self.github_user = github_user
        self.github_repo = github_repo
        self.github_branch = github_branch
        self.token_file = token_file
        self.encryptor = self.generate_or_load_key()
        self.logs = []
        
        # Configuración de logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("pipeline.log"),
                                logging.StreamHandler()
                            ])
        self.logger = logging.getLogger(__name__)
        
    def log(self, message, level=logging.INFO):
        """Guarda los logs del pipeline."""
        self.logs.append(message)
        self.logger.log(level, message)
    
    def generate_or_load_key(self):
        """Genera o carga una clave de encriptación."""
        key_file = "secret.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
        return Fernet(key)
    
    def encrypt_data(self, value):
        """Encripta un valor si no es NaN."""
        return self.encryptor.encrypt(value.encode()).decode() if pd.notna(value) else value
    
    def decrypt_data(self, value):
        """Desencripta un valor si no es NaN."""
        return self.encryptor.decrypt(value).decode() if pd.notna(value) else value
    
    def check_user_role(self, user_role):
        """Verifica si el usuario tiene permisos de acceso al pipeline."""
        with open("roles.json") as f:
            roles = json.load(f)
        if user_role in roles and roles[user_role]["access_level"] == "full":
            self.log(f"Acceso permitido para el rol: {user_role}")
            return True
        else:
            self.log(f"Acceso denegado para el rol: {user_role}", logging.ERROR)
            return False
    
    def authenticate_kaggle(self):
        """Autentica la conexión con Kaggle."""
        self.log("Autenticando en Kaggle...")
        api = KaggleApi()
        api.authenticate()
        self.log("Autenticación en Kaggle exitosa.")
        return api
    
    def download_dataset(self, api):
        """Descarga un dataset desde Kaggle."""
        self.log(f"Descargando dataset: {self.dataset_name}...")
        api.dataset_download_files(self.dataset_name, path=".", unzip=True)
        self.log("Descarga completada.")
    
    def load_csv(self):
        """Carga los archivos CSV descargados en un DataFrame."""
        csv_files = [file for file in os.listdir(".") if file.endswith(".csv")]
        if csv_files:
            self.log(f"Archivos CSV encontrados: {csv_files}")
            return pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        else:
            self.log("No se encontraron archivos CSV en el dataset descargado.", logging.ERROR)
            raise FileNotFoundError("No se encontraron archivos CSV en el dataset descargado.")
    
    def clean_data(self, df):
        """Limpia y preprocesa los datos."""
        self.log("Iniciando limpieza de datos...")
        
        if 'session_id' in df.columns:
            df.drop('session_id', axis=1, inplace=True)
            self.log("Columna 'session_id' eliminada correctamente.")
        
        df.dropna(inplace=True)
        self.log("Valores NaN eliminados correctamente.")
        
        label_encoders = {}
        for col in ['protocol_type', 'encryption_used', 'browser_type']:
            if col in df.columns:
                df[col] = df[col].astype(str)
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col])
                self.log(f"Columna '{col}' codificada correctamente.")
        
        self.log("Limpieza de datos completada.")
        return df
    
    def load_github_token(self):
        """Carga y decrypta el token de GITHUB"""
        if not os.path.exists(self.token_file):
            raise FileNotFoundError("Token encriptado no encontrado.")
        
        with open(self.token_file, "rb") as f:
            encrypted_token = f.read()
        
        return self.decrypt_data(encrypted_token)
    
    def upload_dataframe_to_github(self, df, github_path):
        """Carga los datasets en GITHUB"""
        token = self.load_github_token()
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        content = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        
        url = f"https://api.github.com/repos/{self.github_user}/{self.github_repo}/contents/{github_path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        
        response = requests.get(url, headers=headers)
        sha = response.json().get("sha") if response.status_code == 200 else None
        
        data = {"message": f"Subiendo {github_path}", "content": content, "branch": self.github_branch}
        if sha:
            data["sha"] = sha
        
        response = requests.put(url, json=data, headers=headers)
        if response.status_code in [200, 201]:
            self.log(f"Archivo {github_path} subido correctamente a GitHub.")
        else:
            self.log(f"Error al subir {github_path}: {response.json()}", logging.ERROR)
    
    def run_pipeline(self, user_role="admin"):
        """Ejecuta todo el pipeline de datos."""
        if not self.check_user_role(user_role):
            return "Acceso denegado."
        
        try:
            api = self.authenticate_kaggle()
            self.download_dataset(api)
            df = self.load_csv()
            
            if 'session_id' in df.columns:
                df['session_id'] = df['session_id'].apply(self.encrypt_data)
                self.log("Columna 'session_id' encriptada exitosamente.")
            
            self.upload_dataframe_to_github(df, "datos/dataset_original.csv")
            df_cleaned = self.clean_data(df)
            self.upload_dataframe_to_github(df_cleaned, "datos/dataset_procesado.csv")
        
        except Exception as e:
            self.log(f"Ocurrió un error: {e}", logging.ERROR)
        
        return "\n".join(self.logs)

if __name__ == "__main__":
    pipeline = DataPipeline(
        dataset_name="dnkumars/cybersecurity-intrusion-detection-dataset",
        github_user="samuel-salazarC",
        github_repo="Data_pipeline_and_ml",
        github_branch="master"
    )
    logs = pipeline.run_pipeline()
    print(logs)
