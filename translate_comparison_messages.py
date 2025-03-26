import re

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Traducir el radio button de método de comparación
content = content.replace(
    'Método de comparación facial',
    'Facial Comparison Method'
)

# Traducir las opciones del radio button
content = content.replace(
    '["HOG (Rápido, efectivo)", "Embeddings (Lento, más preciso)"]',
    '["HOG (Fast, effective)", "Embeddings (Slow, more precise)"]'
)

# Traducir el texto de ayuda del radio button
content = content.replace(
    'HOG utiliza histogramas de gradientes orientados para comparación rápida. Embeddings utiliza redes neuronales profundas para mayor precisión.',
    'HOG uses histograms of oriented gradients for quick comparison. Embeddings use deep neural networks for greater precision.'
)

# Traducir las referencias a las opciones en el código
content = content.replace(
    'if comparison_method == "Embeddings (Lento, más preciso)" and DEEPFACE_AVAILABLE:',
    'if comparison_method == "Embeddings (Slow, more precise)" and DEEPFACE_AVAILABLE:'
)

content = content.replace(
    'elif comparison_method == "Embeddings (Lento, más preciso)" and not DEEPFACE_AVAILABLE:',
    'elif comparison_method == "Embeddings (Slow, more precise)" and not DEEPFACE_AVAILABLE:'
)

content = content.replace(
    'comparison_method = "HOG (Rápido, efectivo)"',
    'comparison_method = "HOG (Fast, effective)"'
)

content = content.replace(
    'if comparison_method == "Embeddings (Lento, más preciso)":',
    'if comparison_method == "Embeddings (Slow, more precise)":'
)

# Traducir el texto de ayuda del selector de modelo de embedding
content = content.replace(
    'Seleccione el modelo de red neuronal para extraer embeddings faciales',
    'Select the neural network model to extract facial embeddings'
)

# Traducir el selector de color del bounding box
content = content.replace(
    'Bounding Box Color',
    'Bounding Box Color'
)

# Traducir el mensaje de error específico que aparece en la imagen
content = content.replace(
    'Error al usar embeddings: float() argument must be a string or a real number, not \'dict\'',
    'Error using embeddings: float() argument must be a string or a real number, not \'dict\''
)

# Traducir el mensaje "Using embedding model: VGG-Face"
content = content.replace(
    'Usando modelo de embeddings: VGG-Face',
    'Using embedding model: VGG-Face'
)

# Traducir el mensaje "Automatically switching to HOG method..."
content = content.replace(
    'Cambiando automáticamente a método HOG...',
    'Automatically switching to HOG method...'
)

# Traducir el mensaje "Faces detected: 1"
content = content.replace(
    'Rostros detectados:',
    'Faces detected:'
)

# Traducir el mensaje "Using embedding model: VGG-Face"
content = content.replace(
    'Usando modelo de embedding:',
    'Using embedding model:'
)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Mensajes del modo de comparación traducidos correctamente.") 