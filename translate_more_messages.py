import re

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Traducir mensajes adicionales
translations = {
    # Mensajes de comparación
    'Comparando rostros...': 'Comparing faces...',
    'Comparación completada': 'Comparison completed',
    'Comparando imágenes': 'Comparing images',
    'No se detectaron rostros en ambas imágenes': 'No faces detected in both images',
    'No se detectaron rostros en la imagen 1': 'No faces detected in image 1',
    'No se detectaron rostros en la imagen 2': 'No faces detected in image 2',
    'Comparación facial': 'Facial comparison',
    'Resultados de la comparación': 'Comparison results',
    
    # Mensajes de error
    'Error al procesar': 'Error processing',
    'Error al detectar': 'Error detecting',
    'Error al comparar': 'Error comparing',
    'Error al cargar': 'Error loading',
    'Error al guardar': 'Error saving',
    'Error al generar': 'Error generating',
    'Error al extraer': 'Error extracting',
    'Error al analizar': 'Error analyzing',
    'Error al inicializar': 'Error initializing',
    
    # Mensajes de interfaz
    'Seleccione una imagen': 'Select an image',
    'Cargue una imagen': 'Upload an image',
    'Imagen 1': 'Image 1',
    'Imagen 2': 'Image 2',
    'Rostro detectado': 'Face detected',
    'Rostros detectados': 'Faces detected',
    'Similitud': 'Similarity',
    'Porcentaje de similitud': 'Similarity percentage',
    'Umbral de similitud': 'Similarity threshold',
    'Confianza de detección': 'Detection confidence',
    
    # Botones y acciones
    'Comparar': 'Compare',
    'Analizar': 'Analyze',
    'Detectar': 'Detect',
    'Procesar': 'Process',
    'Guardar': 'Save',
    'Cargar': 'Load',
    'Cancelar': 'Cancel',
    'Continuar': 'Continue',
    'Reiniciar': 'Reset'
}

# Aplicar todas las traducciones
for spanish, english in translations.items():
    content = content.replace(spanish, english)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Mensajes adicionales traducidos correctamente.") 