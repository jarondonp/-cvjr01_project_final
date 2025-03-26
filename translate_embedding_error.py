import re

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Buscar y traducir específicamente el mensaje de error de embeddings
error_pattern = r'Error al usar embeddings: float\(\) argument must be a string or a real number, not \'dict\''
replacement = 'Error using embeddings: float() argument must be a string or a real number, not \'dict\''

# Usar expresión regular para encontrar y reemplazar el mensaje de error
content = re.sub(error_pattern, replacement, content)

# Buscar y traducir variantes del mensaje
content = content.replace(
    'Error al usar embeddings:',
    'Error using embeddings:'
)

content = content.replace(
    'Error al procesar embeddings:',
    'Error processing embeddings:'
)

content = content.replace(
    'Error: float() argument must be a string or a real number, not \'dict\'',
    'Error: float() argument must be a string or a real number, not \'dict\''
)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Mensaje de error de embeddings traducido correctamente.") 