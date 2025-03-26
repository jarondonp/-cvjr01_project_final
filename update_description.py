import re

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Definir el patrón a buscar
pattern = r'# App title and description\nst\.title\("Advanced Face & Feature Detection"\)\nst\.markdown\(""".*?"""\)'

# Definir el reemplazo
replacement = '''# App title and description
st.title("Advanced Face & Feature Detection")
st.markdown("""
This comprehensive facial analysis system offers multiple capabilities:

- **Face Detection**: Accurately locate faces in images and videos using OpenCV DNN
- **Feature Recognition**: Detect eyes, smiles, and other facial features
- **Face Comparison**: Compare faces between images with detailed similarity analysis
- **Face Recognition**: Register faces and identify them in new images or real-time video
- **Multi-model Analysis**: Uses multiple embedding models (VGG-Face, Facenet, OpenFace, ArcFace) for improved accuracy

Upload images or use your camera to experience advanced computer vision technology!
""")'''

# Reemplazar usando expresiones regulares
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(new_content)

print("Descripción actualizada correctamente.") 