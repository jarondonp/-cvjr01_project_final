import re

# Leer el archivo
with open('streamlit_app.py', 'r', encoding='utf-8') as file:
    content = file.read()

# Reemplazar el título de ajustes de detección de ojos
content = content.replace(
    'st.sidebar.markdown("**Ajustes de detección de ojos**")',
    'st.sidebar.markdown("**Eye Detection Settings**")'
)

# Reemplazar el mensaje informativo
content = content.replace(
    'st.sidebar.info("Ajusta el control deslizante para cambiar la sensibilidad de la detección de ojos. Un valor más alto detectará más ojos pero puede generar falsos positivos.")',
    'st.sidebar.info("Adjust the slider to change the sensitivity of eye detection. A higher value will detect more eyes but may generate false positives.")'
)

# Guardar el archivo modificado
with open('streamlit_app.py', 'w', encoding='utf-8') as file:
    file.write(content)

print("Textos de detección de ojos traducidos correctamente.") 