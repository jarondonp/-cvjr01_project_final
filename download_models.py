import os
import urllib.request
import ssl

# Crear directorio para los modelos si no existe
if not os.path.exists('models'):
    os.makedirs('models')

# URLs correctas para los modelos
model_url = "https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/res10_300x300_ssd_iter_140000.caffemodel"
prototxt_url = "https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/deploy.prototxt.txt"

# Rutas de destino con nombres correctos
model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
prototxt_path = "models/deploy.prototxt.txt"

# Desactivar verificación SSL para evitar problemas en algunos sistemas
ssl._create_default_https_context = ssl._create_unverified_context

# Descargar los archivos
print("Descargando el archivo del modelo...")
try:
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Modelo guardado en: {model_path}")
except Exception as e:
    print(f"Error al descargar el modelo: {e}")

print("Descargando el archivo de configuración...")
try:
    urllib.request.urlretrieve(prototxt_url, prototxt_path)
    print(f"Archivo de configuración guardado en: {prototxt_path}")
except Exception as e:
    print(f"Error al descargar el archivo de configuración: {e}")

print("Descarga completada. Ahora puedes ejecutar la aplicación Streamlit.") 