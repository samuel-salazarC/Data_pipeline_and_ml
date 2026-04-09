from cryptography.fernet import Fernet
import os

# Generar clave de encriptación si no existe
key_file = "secret.key"
if not os.path.exists(key_file):
    key = Fernet.generate_key()
    with open(key_file, "wb") as f:
        f.write(key)

# Cargar la clave
with open(key_file, "rb") as f:
    encryptor = Fernet(f.read())

# Guardar token encriptado
GITHUB_TOKEN = ""  # Pega tu token de GitHub aquí
encrypted_token = encryptor.encrypt(GITHUB_TOKEN.encode())

with open("github_token.enc", "wb") as f:
    f.write(encrypted_token)

print("Token de GitHub encriptado y guardado correctamente.")
