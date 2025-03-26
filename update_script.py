import re

# Leer el archivo original
with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Definir el patrón a buscar y el reemplazo
pattern = r"""            else:
                # Extraer embedding del rostro
                embedding = extract_face_embeddings\(image, bboxes\[0\], model_name=model_choice\)
                
                if embedding is not None:
                    # Guardar en la base de datos
                    if add_to_existing and person_name in st\.session_state\.face_database:
                        # Añadir a persona existente
                        if 'embeddings' not in st\.session_state\.face_database\[person_name\]:
                            # Convertir entrada antigua al nuevo formato
                            old_embedding = st\.session_state\.face_database\[person_name\]\['embedding'\]
                            old_model = "VGG-Face"  # Modelo por defecto para entradas antiguas
                            st\.session_state\.face_database\[person_name\] = {
                                'embeddings': \[old_embedding\],
                                'models': \[old_model\],
                                'timestamps': \[st\.session_state\.face_database\[person_name\]\['timestamp'\]\],
                                'count': 1
                            }
                        
                        # Añadir nuevo embedding
                        st\.session_state\.face_database\[person_name\]\['embeddings'\]\.append\(embedding\["embedding"\]\)
                        st\.session_state\.face_database\[person_name\]\['models'\]\.append\(embedding\["model"\]\)
                        st\.session_state\.face_database\[person_name\]\['timestamps'\]\.append\(time\.time\(\)\)
                        st\.session_state\.face_database\[person_name\]\['count'\] \+= 1
                        
                        st\.success\(f"Rostro adicional de {person_name} registrado correctamente! \(Total: {st\.session_state\.face_database\[person_name\]\['count'\]} imágenes\)"\)
                    else:
                        # Crear nueva entrada
                        st\.session_state\.face_database\[person_name\] = {
                            'embeddings': \[embedding\["embedding"\]\],
                            'models': \[embedding\["model"\]\],
                            'timestamps': \[time\.time\(\)\],
                            'count': 1
                        }
                        st\.success\(f"Rostro de {person_name} registrado correctamente!"\)
                
                if embedding is not None:
                    # Mostrar imagen con rostro detectado
                    processed_image, _ = process_face_detections\(image, detections, confidence_threshold\)
                    st\.image\(processed_image, channels='BGR', caption="Rostro registrado"\)
                else:
                    st\.error\("Error al extraer características faciales\. Intente con otra imagen\."\)"""

replacement = """            else:
                # Extraer embeddings del rostro con todos los modelos
                embeddings_all_models = extract_face_embeddings_all_models(image, bboxes[0])
                
                if embeddings_all_models:
                    # Guardar en la base de datos
                    if add_to_existing and person_name in st.session_state.face_database:
                        # Añadir a persona existente
                        if 'embeddings' not in st.session_state.face_database[person_name]:
                            # Convertir entrada antigua al nuevo formato
                            old_embedding = st.session_state.face_database[person_name]['embedding']
                            old_model = "VGG-Face"  # Modelo por defecto para entradas antiguas
                            st.session_state.face_database[person_name] = {
                                'embeddings': [old_embedding],
                                'models': [old_model],
                                'timestamps': [st.session_state.face_database[person_name]['timestamp']],
                                'count': 1
                            }
                        
                        # Añadir nuevos embeddings de todos los modelos
                        current_time = time.time()
                        for emb_data in embeddings_all_models:
                            st.session_state.face_database[person_name]['embeddings'].append(emb_data["embedding"])
                            st.session_state.face_database[person_name]['models'].append(emb_data["model"])
                            st.session_state.face_database[person_name]['timestamps'].append(current_time)
                        
                        # Incrementar el contador solo una vez por imagen (no por cada modelo)
                        st.session_state.face_database[person_name]['count'] += 1
                        
                        st.success(f"Rostro adicional de {person_name} registrado correctamente con {len(embeddings_all_models)} modelos! (Total: {st.session_state.face_database[person_name]['count']} imágenes)")
                    else:
                        # Crear nueva entrada
                        embeddings_list = []
                        models_list = []
                        timestamps_list = []
                        current_time = time.time()
                        
                        for emb_data in embeddings_all_models:
                            embeddings_list.append(emb_data["embedding"])
                            models_list.append(emb_data["model"])
                            timestamps_list.append(current_time)
                        
                        st.session_state.face_database[person_name] = {
                            'embeddings': embeddings_list,
                            'models': models_list,
                            'timestamps': timestamps_list,
                            'count': 1
                        }
                        st.success(f"Rostro de {person_name} registrado correctamente con {len(embeddings_all_models)} modelos!")
                
                    # Mostrar imagen con rostro detectado
                    processed_image, _ = process_face_detections(image, detections, confidence_threshold)
                    st.image(processed_image, channels='BGR', caption="Rostro registrado")
                else:
                    st.error("Error al extraer características faciales. Intente con otra imagen.")"""

# Realizar el reemplazo
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Verificar si se hizo algún cambio
if new_content == content:
    print("No se encontró el patrón para reemplazar.")
else:
    # Guardar el archivo modificado
    with open('streamlit_app.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Archivo modificado correctamente.") 