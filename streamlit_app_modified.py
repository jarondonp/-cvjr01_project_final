import streamlit as st
import cv2
import numpy as np
import time
import os
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import pickle
import json
import uuid
import tempfile
from datetime import datetime
import math
import re

# Intentar importar DeepFace
DEEPFACE_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except:
    pass

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Reconocimiento Facial",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para extraer embeddings faciales usando todos los modelos disponibles
def extract_face_embeddings_all_models(image, bbox):
    """
    Extrae embeddings faciales usando todos los modelos disponibles.
    
    Args:
        image: Imagen de entrada en formato OpenCV (BGR)
        bbox: Bounding box de la cara [x1, y1, x2, y2, conf]
        
    Returns:
        Lista de diccionarios con embeddings y nombres de modelos
    """
    if not DEEPFACE_AVAILABLE:
        st.error("DeepFace no está disponible. Por favor instale la biblioteca con 'pip install deepface'")
        return None
    
    models = ["VGG-Face", "Facenet", "OpenFace", "ArcFace"]
    results = []
    
    try:
        x1, y1, x2, y2, _ = bbox
        face_img = image[y1:y2, x1:x2]
        
        # Convertir de BGR a RGB para DeepFace
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Extraer embeddings para cada modelo
        for model_name in models:
            try:
                embedding = DeepFace.represent(
                    img_path=face_img_rgb,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend="opencv"
                )
                
                # Guardar el modelo usado junto con el embedding
                result = {
                    "embedding": embedding[0]["embedding"],
                    "model": model_name
                }
                
                results.append(result)
            except Exception as e:
                st.warning(f"No se pudo extraer embedding con el modelo {model_name}: {str(e)}")
                continue
        
        return results
    
    except Exception as e:
        st.error(f"Error al extraer embeddings: {str(e)}")
        return None

# Función para extraer embeddings faciales usando modelos pre-entrenados
def extract_face_embeddings(image, bbox, model_name="VGG-Face"):
    """
    Extrae embeddings faciales usando modelos pre-entrenados.
    
    Args:
        image: Imagen de entrada en formato OpenCV (BGR)
        bbox: Bounding box de la cara [x1, y1, x2, y2, conf]
        model_name: Nombre del modelo a usar (default: VGG-Face)
        
    Returns:
        Diccionario con embedding y nombre del modelo
    """
    if not DEEPFACE_AVAILABLE:
        st.error("DeepFace no está disponible. Por favor instale la biblioteca con 'pip install deepface'")
        return None
    
    try:
        x1, y1, x2, y2, _ = bbox
        face_img = image[y1:y2, x1:x2]
        
        # Convertir de BGR a RGB para DeepFace
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Extraer embedding
        embedding = DeepFace.represent(
            img_path=face_img_rgb,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="opencv"
        )
        
        # Guardar el modelo usado junto con el embedding
        result = {
            "embedding": embedding[0]["embedding"],
            "model": model_name
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error al extraer embedding: {str(e)}")
        return None

# Sección de registro de rostros (modificada para usar extract_face_embeddings_all_models)
def register_face_ui():
    st.header("Registro de Rostros")
    
    # Inicializar base de datos si no existe
    if 'face_database' not in st.session_state:
        st.session_state.face_database = {}
    
    # Formulario de registro
    with st.form("face_registration_form"):
        person_name = st.text_input("Nombre de la persona", key="person_name_input")
        uploaded_file = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"], key="face_upload")
        confidence_threshold = st.slider("Confianza de Detección", 0.0, 1.0, 0.5, 0.01, key="reg_confidence")
        add_to_existing = st.checkbox("Añadir a persona existente", key="add_existing")
        model_choice = st.selectbox("Modelo para extracción de características", 
                                   ["VGG-Face", "Facenet", "OpenFace", "ArcFace"], 
                                   index=0, key="model_choice")
        submit_button = st.form_submit_button("Registrar Rostro")
    
    if submit_button and uploaded_file is not None and person_name:
        # Cargar imagen
        image = load_image_from_upload(uploaded_file)
        
        # Detectar rostros
        face_net = load_face_model()
        detections = detect_face_dnn(face_net, image, conf_threshold=confidence_threshold)
        bboxes = []
        
        for detection in detections:
            x1, y1, x2, y2, conf = detection
            if conf >= confidence_threshold:
                bboxes.append([x1, y1, x2, y2, conf])
        
        if not bboxes:
            st.error("No se detectaron rostros en la imagen. Intente con otra imagen o ajuste el umbral de confianza.")
        elif len(bboxes) > 1:
            st.warning(f"Se detectaron {len(bboxes)} rostros. Se registrarán todos.")
            
            # Extraer embeddings de todos los rostros
            all_embeddings = []
            for bbox in bboxes:
                embeddings_all_models = extract_face_embeddings_all_models(image, bbox)
                if embeddings_all_models:
                    all_embeddings.append(embeddings_all_models)
            
            if all_embeddings:
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
                    
                    # Añadir nuevos embeddings
                    current_time = time.time()
                    for face_embeddings in all_embeddings:
                        for emb_data in face_embeddings:
                            st.session_state.face_database[person_name]['embeddings'].append(emb_data["embedding"])
                            st.session_state.face_database[person_name]['models'].append(emb_data["model"])
                            st.session_state.face_database[person_name]['timestamps'].append(current_time)
                    
                    # Incrementar el contador (una vez por cada rostro)
                    st.session_state.face_database[person_name]['count'] += len(all_embeddings)
                    
                    st.success(f"{len(all_embeddings)} rostros adicionales de {person_name} registrados correctamente con múltiples modelos! (Total: {st.session_state.face_database[person_name]['count']} imágenes)")
                else:
                    # Crear nueva entrada
                    embeddings_list = []
                    models_list = []
                    timestamps_list = []
                    current_time = time.time()
                    
                    for face_embeddings in all_embeddings:
                        for emb_data in face_embeddings:
                            embeddings_list.append(emb_data["embedding"])
                            models_list.append(emb_data["model"])
                            timestamps_list.append(current_time)
                    
                    st.session_state.face_database[person_name] = {
                        'embeddings': embeddings_list,
                        'models': models_list,
                        'timestamps': timestamps_list,
                        'count': len(all_embeddings)
                    }
                    st.success(f"{len(all_embeddings)} rostros de {person_name} registrados correctamente con múltiples modelos!")
                
                # Mostrar imagen con rostro detectado
                processed_image, _ = process_face_detections(image, detections, confidence_threshold)
                st.image(processed_image, channels='BGR', caption="Rostro registrado")
            else:
                st.error("Error al extraer características faciales. Intente con otra imagen.")
        else:
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
                st.error("Error al extraer características faciales. Intente con otra imagen.") 