import re

# Diccionario de traducciones
translations = {
    # Mensajes de error y advertencia
    "DeepFace no está disponible. Por favor instale la biblioteca con 'pip install deepface'": "DeepFace is not available. Please install the library with 'pip install deepface'",
    "No se pudo extraer embedding con el modelo": "Could not extract embedding with model",
    "Error al extraer embeddings": "Error extracting embeddings",
    "Error al extraer embedding": "Error extracting embedding",
    "Error al detectar atributos faciales": "Error detecting facial attributes",
    "No se detectaron rostros en la imagen. Por favor, sube otra imagen.": "No faces detected in the image. Please upload another image.",
    "Se detectaron múltiples rostros. Se utilizará el primero detectado.": "Multiple faces detected. The first one will be used.",
    "Error al extraer características faciales. Intente con otra imagen.": "Error extracting facial features. Please try with another image.",
    
    # Títulos y encabezados
    "Sistema de Reconocimiento Facial": "Face Recognition System",
    "Registrar Nuevo Rostro": "Register New Face",
    "Rostros Registrados": "Registered Faces",
    "Reconocimiento en Imagen": "Image Recognition",
    "Reconocimiento en Tiempo Real": "Real-time Recognition",
    "Resultado del reconocimiento": "Recognition Result",
    
    # Formularios y controles
    "Nombre de la persona": "Person's name",
    "Modelo de embeddings": "Embedding model",
    "Umbral de confianza para detección": "Detection Confidence",
    "Añadir a persona existente": "Add to existing person",
    "Registrar Rostro": "Register Face",
    "Seleccionar persona": "Select person",
    "Eliminar persona seleccionada": "Delete selected person",
    "Eliminar todos los rostros": "Delete all faces",
    "Registros de": "Records of",
    "eliminados.": "deleted.",
    "Base de datos de rostros eliminada.": "Face database deleted.",
    
    # Mensajes de éxito
    "Rostro adicional de": "Additional face of",
    "registrado correctamente con": "successfully registered with",
    "modelos! (Total:": "models! (Total:",
    "imágenes)": "images)",
    "Rostro de": "Face of",
    "registrado correctamente con": "successfully registered with",
    "modelos!": "models!",
    
    # Otros textos
    "Rostro registrado": "Registered Face",
    "Imágenes registradas": "Registered images",
    "Última actualización": "Last update",
    "ADVERTENCIA: La versión actual de TensorFlow (2.19) puede tener incompatibilidades con algunos modelos. Se recomienda usar HOG si experimenta problemas.": "WARNING: The current version of TensorFlow (2.19) may have incompatibilities with some models. It is recommended to use HOG if you experience problems.",
    "La biblioteca DeepFace no está disponible. Por favor instale con 'pip install deepface' para usar embeddings.": "The DeepFace library is not available. Please install with 'pip install deepface' to use embeddings.",
    "Usando método HOG por defecto.": "Using HOG method by default.",
    "Usando modelo de embeddings:": "Using embedding model:",
    "Error al usar embeddings:": "Error using embeddings:",
    "Cambiando automáticamente a método HOG...": "Automatically switching to HOG method...",
    "Usando método HOG porque DeepFace no está disponible.": "Using HOG method because DeepFace is not available.",
    "No hay rostros registrados. Por favor, registre al menos un rostro primero.": "No faces registered. Please register at least one face first."
}

# Función para traducir textos específicos en st.header, st.title, etc.
def translate_streamlit_functions(content):
    # Patrones para buscar textos en funciones de Streamlit
    patterns = [
        r'st\.header\("([^"]+)"\)',
        r'st\.title\("([^"]+)"\)',
        r'st\.subheader\("([^"]+)"\)',
        r'st\.success\("([^"]+)"\)',
        r'st\.error\("([^"]+)"\)',
        r'st\.warning\("([^"]+)"\)',
        r'st\.info\("([^"]+)"\)',
        r'st\.write\("([^"]+)"\)',
        r'st\.text_input\("([^"]+)"\)',
        r'st\.selectbox\(\s*"([^"]+)"',
        r'st\.slider\(\s*"([^"]+)"',
        r'st\.checkbox\(\s*"([^"]+)"',
        r'st\.button\("([^"]+)"\)',
        r'st\.form_submit_button\("([^"]+)"\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            for spanish, english in translations.items():
                if spanish in match:
                    new_match = match.replace(spanish, english)
                    content = content.replace(match, new_match)
    
    return content

# Función para traducir f-strings
def translate_f_strings(content):
    # Patrones para buscar f-strings
    patterns = [
        r'st\.success\(f"([^"]+)"\)',
        r'st\.error\(f"([^"]+)"\)',
        r'st\.warning\(f"([^"]+)"\)',
        r'st\.info\(f"([^"]+)"\)',
        r'st\.write\(f"([^"]+)"\)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            translated_match = match
            for spanish, english in translations.items():
                if spanish in match:
                    translated_match = translated_match.replace(spanish, english)
            
            if translated_match != match:
                content = content.replace(f'f"{match}"', f'f"{translated_match}"')
    
    return content

# Traducir casos específicos
def translate_specific_cases(content):
    # Traducir mensajes de éxito específicos
    content = content.replace(
        'st.success(f"Rostro adicional de {person_name} registrado correctamente con {len(embeddings_all_models)} modelos! (Total: {st.session_state.face_database[person_name][\'count\']} imágenes)")',
        'st.success(f"Additional face of {person_name} successfully registered with {len(embeddings_all_models)} models! (Total: {st.session_state.face_database[person_name][\'count\']} images)")'
    )
    
    content = content.replace(
        'st.success(f"Rostro de {person_name} registrado correctamente con {len(embeddings_all_models)} modelos!")',
        'st.success(f"Face of {person_name} successfully registered with {len(embeddings_all_models)} models!")'
    )
    
    content = content.replace(
        'st.success(f"Registros de {person_to_delete} eliminados.")',
        'st.success(f"Records of {person_to_delete} deleted.")'
    )
    
    return content

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Reemplazar textos directos
for spanish, english in translations.items():
    content = content.replace(spanish, english)

# Reemplazar textos en funciones de Streamlit
content = translate_streamlit_functions(content)

# Reemplazar textos en f-strings
content = translate_f_strings(content)

# Traducir casos específicos
content = translate_specific_cases(content)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Traducción completada. Todos los textos han sido traducidos al inglés.") 