import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os
import time
import urllib.request
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Importar DeepFace para reconocimiento facial avanzado
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# Set page config with custom title and layout
st.set_page_config(
    page_title="Advanced Face & Feature Detection",
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
        model_name: Nombre del modelo a utilizar (VGG-Face, Facenet, ArcFace, etc.)
        
    Returns:
        Vector de embeddings faciales
    """
    if not DEEPFACE_AVAILABLE:
        st.error("DeepFace no está disponible. Por favor instale la biblioteca con 'pip install deepface'")
        return None
    
    try:
        x1, y1, x2, y2, _ = bbox
        face_img = image[y1:y2, x1:x2]
        
        # Convertir de BGR a RGB para DeepFace
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Extraer embeddings
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
        st.error(f"Error al extraer embeddings: {str(e)}")
        return None

# Función para comparar rostros usando embeddings
def compare_faces_embeddings(image1, bboxes1, image2, bboxes2, model_name="VGG-Face"):
    """
    Compara rostros entre dos imágenes usando embeddings faciales.
    
    Args:
        image1, image2: Imágenes en formato OpenCV
        bboxes1, bboxes2: Listas de bounding boxes para cada imagen
        model_name: Modelo a utilizar para los embeddings
        
    Returns:
        Lista de resultados de comparación
    """
    results = []
    
    for i, bbox1 in enumerate(bboxes1):
        embedding1 = extract_face_embeddings(image1, bbox1, model_name)
        
        if embedding1 is None:
            continue
        
        face1_comparisons = []
        for j, bbox2 in enumerate(bboxes2):
            embedding2 = extract_face_embeddings(image2, bbox2, model_name)
            
            if embedding2 is None:
                continue
            
            # Calcular similitud de coseno entre embeddings
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Convertir a porcentaje (0-100)
            similarity_percentage = float(similarity * 100)
            
            face1_comparisons.append({
                "face1_index": i,
                "face2_index": j,
                "similarity": similarity_percentage
            })
        
        # Ordenar comparaciones por similitud (mayor a menor)
        face1_comparisons.sort(key=lambda x: x["similarity"], reverse=True)
        results.append(face1_comparisons)
    
    return results

# Function to extract facial features using HOG (Histogram of Oriented Gradients)
def extract_facial_features(image, bbox):
    x1, y1, x2, y2, _ = bbox
    face_roi = image[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Resize for normalization
    resized_face = cv2.resize(gray_face, (100, 100))
    
    # Apply histogram equalization to improve contrast
    equalized_face = cv2.equalizeHist(resized_face)
    
    # Calculate HOG features with compatible parameters
    # Parameters must satisfy: (winSize - blockSize) % blockStride == 0
    win_size = (100, 100)
    block_size = (20, 20)     # Must be divisible by blockStride
    block_stride = (10, 10)   # Must divide (winSize - blockSize) exactly
    cell_size = (10, 10)      # Must divide blockSize exactly
    num_bins = 9              # Number of bins for the histogram
    
    # Verify that parameters are compatible
    assert (win_size[0] - block_size[0]) % block_stride[0] == 0, "Incompatible HOG parameters (width)"
    assert (win_size[1] - block_size[1]) % block_stride[1] == 0, "Incompatible HOG parameters (height)"
    assert block_size[0] % cell_size[0] == 0, "blockSize must be divisible by cellSize (width)"
    assert block_size[1] % cell_size[1] == 0, "blockSize must be divisible by cellSize (height)"
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_features = hog.compute(equalized_face)
    
    # Extract additional features based on facial proportions
    # that can help differentiate genders
    height, width = face_roi.shape[:2]
    
    # 1. Aspect ratio - facial proportion can differ between genders
    aspect_ratio = width / height  
    
    # 2. Calculate average brightness in upper vs lower face
    # Men often have darker upper face (beard, mustache area)
    upper_half = equalized_face[:50, :]
    lower_half = equalized_face[50:, :]
    upper_brightness = np.mean(upper_half)
    lower_brightness = np.mean(lower_half)
    brightness_ratio = upper_brightness / (lower_brightness + 1e-5)  # Avoid division by zero
    
    # 3. Texture variance - male faces often have more texture variance
    texture_variance = np.var(equalized_face)
    
    # 4. Edge density - male faces often have stronger edges (facial hair, etc.)
    edges = cv2.Canny(equalized_face, 100, 200)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    
    # 5. Extract facial regions for more precise comparison
    # These regions are less affected by gender differences
    
    # Eye region (upper 40% of face) - similar across genders
    eye_region = equalized_face[:40, :]
    eye_region_hog = cv2.HOGDescriptor((eye_region.shape[1], eye_region.shape[0]), 
                                      (10, 10), (5, 5), (5, 5), 9)
    eye_hog_features = eye_region_hog.compute(eye_region)
    
    # Nose bridge region (middle vertical strip) - similar across genders
    nose_region = equalized_face[:, 40:60]
    nose_variance = np.var(nose_region)
    
    # Create a combined feature vector
    # Add facial proportion features
    gender_features = np.array([
        aspect_ratio * 10,       # Scale to be comparable with HOG features
        brightness_ratio * 5,    # Scale appropriately
        texture_variance / 1000,  # Scale down variance
        edge_density / 10        # Scale edge density
    ])
    
    # Combine all features
    # Give more weight to HOG features and eye region which are less gender-specific
    combined_features = np.concatenate([
        hog_features * 1.2,      # Increase weight of general HOG features
        eye_hog_features * 1.5,  # Give high weight to eye region (gender-neutral)
        [nose_variance / 1000],  # Add nose bridge feature
        gender_features          # Add gender-specific features
    ])
    
    return combined_features

# Function to calculate similarity between two feature vectors
def calculate_similarity(features1, features2):
    # Get the lengths of different feature sections
    # The last 4 elements are our gender features
    gender_features_len = 4
    nose_feature_len = 1
    
    # Calculate total length of HOG features (main + eye region)
    total_len = len(features1)
    hog_eye_len = len(features1) - gender_features_len - nose_feature_len
    main_hog_len = int(hog_eye_len * 0.4)  # Approximate based on our weighting
    
    # Separate features
    main_hog1 = features1[:main_hog_len]
    main_hog2 = features2[:main_hog_len]
    
    eye_hog1 = features1[main_hog_len:hog_eye_len]
    eye_hog2 = features2[main_hog_len:hog_eye_len]
    
    nose_feature1 = features1[hog_eye_len:hog_eye_len+nose_feature_len]
    nose_feature2 = features2[hog_eye_len:hog_eye_len+nose_feature_len]
    
    gender_features1 = features1[-gender_features_len:]
    gender_features2 = features2[-gender_features_len:]
    
    # Normalize and calculate similarity for each feature set
    # Main HOG features (general face)
    norm_main_hog1 = main_hog1 / (np.linalg.norm(main_hog1) + 1e-5)
    norm_main_hog2 = main_hog2 / (np.linalg.norm(main_hog2) + 1e-5)
    main_hog_similarity = np.dot(norm_main_hog1.flatten(), norm_main_hog2.flatten())
    
    # Eye region HOG features (less gender-specific)
    norm_eye_hog1 = eye_hog1 / (np.linalg.norm(eye_hog1) + 1e-5)
    norm_eye_hog2 = eye_hog2 / (np.linalg.norm(eye_hog2) + 1e-5)
    eye_hog_similarity = np.dot(norm_eye_hog1.flatten(), norm_eye_hog2.flatten())
    
    # Nose feature similarity - convert to scalar values to avoid numpy array formatting issues
    nose_val1 = float(nose_feature1[0])
    nose_val2 = float(nose_feature2[0])
    nose_similarity = 1.0 - min(1.0, abs(nose_val1 - nose_val2) / (nose_val1 + 1e-5))
    
    # Calculate gender difference score (higher means more different)
    # Extract individual gender features
    aspect_ratio1, brightness_ratio1, texture_var1, edge_density1 = gender_features1
    aspect_ratio2, brightness_ratio2, texture_var2, edge_density2 = gender_features2
    
    # Calculate differences in each gender feature
    aspect_diff = abs(aspect_ratio1 - aspect_ratio2) / 10  # Undo scaling
    brightness_diff = abs(brightness_ratio1 - brightness_ratio2) / 5  # Undo scaling
    texture_diff = abs(texture_var1 - texture_var2) * 1000  # Undo scaling
    edge_diff = abs(edge_density1 - edge_density2) * 10  # Undo scaling
    
    # Combine differences with appropriate weights
    gender_diff_score = (
        aspect_diff * 0.4 +       # Aspect ratio is a strong gender indicator
        brightness_diff * 0.3 +   # Brightness difference (facial hair)
        texture_diff * 0.2 +      # Texture variance
        edge_diff * 0.1           # Edge density
    )
    
    # Normalize gender difference to a 0-1 scale
    gender_diff_normalized = min(1.0, gender_diff_score / 0.5)
    
    # Calculate weighted similarity score
    # Give more weight to eye region and less to gender-specific features
    weighted_similarity = (
        main_hog_similarity * 0.4 +    # General face features
        eye_hog_similarity * 0.5 +     # Eye region (more weight as it's less gender-specific)
        nose_similarity * 0.1          # Nose bridge
    )
    
    # Calculate additional similarity metrics for better discrimination
    
    # 1. Check if hair color/style is similar (using upper part of face HOG)
    hair_region_similarity = main_hog_similarity * 1.2  # Boost importance of hair region
    
    # 2. Check facial structure similarity (using aspect ratio and proportions)
    structure_similarity = 1.0 - aspect_diff * 2.0  # Higher when aspect ratios are similar (reduced from 2.5)
    structure_similarity = max(0, min(1, structure_similarity))  # Clamp to 0-1
    
    # 3. Check texture similarity (important for similar-looking people)
    texture_similarity = 1.0 - texture_diff * 1.5  # Reduced from 2.0
    texture_similarity = max(0, min(1, texture_similarity))  # Clamp to 0-1
    
    # 4. Check if both faces have similar expressions (using edge patterns)
    expression_similarity = 1.0 - edge_diff * 1.5  # Reduced from 2.0
    expression_similarity = max(0, min(1, expression_similarity))  # Clamp to 0-1
    
    # 5. Calculate a more precise facial structure similarity score
    # This helps better distinguish different people
    precise_structure_score = (
        (1.0 - aspect_diff * 2.5) * 0.6 +  # Reduced from 3.5
        (1.0 - brightness_diff * 2.0) * 0.4  # Reduced from 2.5
    )
    precise_structure_score = max(0, min(1, precise_structure_score))
    
    # 6. Calculate a more detailed hair similarity score
    # Upper 20% of the face contains hairline information
    hair_similarity = hair_region_similarity * 1.8  # Increased from 1.5 to give more weight to hair similarity
    hair_similarity = max(0, min(1, hair_similarity))
    
    # 7. Calculate a critical difference score that can reduce similarity
    # This helps create a gap between different people
    critical_diff_score = (
        (aspect_diff > 0.25) * 0.3 +  # Increased threshold from 0.15 to 0.25
        (brightness_diff > 0.3) * 0.3 +  # Increased threshold from 0.2 to 0.3
        (texture_diff > 0.25) * 0.2 +  # Increased threshold from 0.15 to 0.25
        (edge_diff > 0.25) * 0.2  # Increased threshold from 0.15 to 0.25
    )
    
    # Combine all similarity metrics with appropriate weights
    enhanced_similarity = (
        weighted_similarity * 0.35 +       # Base HOG similarity
        hair_similarity * 0.3 +            # Hair similarity (increased from 0.25 to 0.3)
        precise_structure_score * 0.2 +    # Precise facial structure (reduced from 0.25 to 0.2)
        texture_similarity * 0.1 +         # Texture patterns
        expression_similarity * 0.05       # Expression
    )
    
    # Convert to percentage with a power curve to enhance differences
    # Use a milder power curve (1.1 instead of 1.3) to avoid excessive penalties
    base_similarity = ((enhanced_similarity + 1) / 2) ** 1.1 * 100
    
    # Apply a penalty factor for low similarity, but less aggressively
    if weighted_similarity < 0.2:  # Reduced from 0.25
        base_similarity = base_similarity * 0.8  # Reduced penalty from 0.6 to 0.8
    
    # Apply gender-based penalty - more balanced approach
    gender_penalty = 0.0
    
    # Apply penalty when gender differences are detected, but with higher threshold
    if gender_diff_normalized > 0.4:  # Increased from 0.3
        # Adjust penalty based on similarity of gender-neutral features (eyes)
        eye_factor = max(0, 1.0 - ((eye_hog_similarity + 1) / 2))  # 0 for perfect eye match
        adjusted_penalty = gender_diff_normalized * 0.3 * (0.5 + eye_factor)  # Reduced from 0.5 to 0.3
        gender_penalty = min(0.3, adjusted_penalty)  # Reduced cap from 0.5 to 0.3
    
    # Apply the gender-based penalty
    similarity_percentage = base_similarity * (1.0 - gender_penalty)
    
    # Apply additional scaling to reduce similarity for different people
    # But only when they are clearly different
    if gender_diff_normalized > 0.45 and base_similarity < 70:  # Increased from 0.35/80 to 0.45/70
        # Apply reduction for clearly different people
        similarity_percentage = similarity_percentage * 0.85  # Reduced from 0.75 to 0.85
    
    # Apply critical difference penalty - but less aggressively
    if critical_diff_score > 0.6:  # Increased threshold from 0.5 to 0.6
        similarity_percentage = similarity_percentage * (1.0 - critical_diff_score * 0.3)  # Reduced from 0.5 to 0.3
    
    # Boost similarity for very similar people
    if base_similarity > 65 and gender_diff_normalized < 0.3:  # Reduced from 75/0.2 to 65/0.3
        similarity_boost = min(1.5, 1.0 + (0.3 - gender_diff_normalized) * 2.0)  # Increased max boost from 1.4 to 1.5
        similarity_percentage = min(100, similarity_percentage * similarity_boost)
    
    # Apply a stronger differentiation between similar and different people
    # This creates a more pronounced gap between similar and different individuals
    if similarity_percentage > 65:  # Reduced from 70 to 65
        # Boost higher similarities even more
        similarity_percentage = 65 + (similarity_percentage - 65) * 1.3  # Increased from 1.2 to 1.3
    elif similarity_percentage < 40:  # Reduced from 50 to 40
        # Reduce lower similarities, but less aggressively
        similarity_percentage = similarity_percentage * 0.9  # Reduced from 0.8 to 0.9
    
    # Final adjustment to better distribute similarity scores
    # Compress very low scores and expand middle-high scores
    if similarity_percentage < 30:  # Reduced from 40 to 30
        # Compress very low scores less aggressively
        similarity_percentage = 20 + (similarity_percentage - 20) * 0.9  # Changed from 10/0.75 to 20/0.9
    elif 30 <= similarity_percentage < 65:  # Changed from 40/70 to 30/65
        # Expand middle range to better differentiate similar people
        similarity_percentage = 30 + (similarity_percentage - 30) * 1.2  # Reduced from 1.3 to 1.2
    
    # Apply a final scaling to create separation between different people
    # But much less aggressively
    if similarity_percentage < 50:  # Reduced from 60 to 50
        # Apply a milder non-linear reduction
        reduction_factor = 1.0 - ((50 - similarity_percentage) / 50) * 0.2  # Reduced from 0.4 to 0.2
        similarity_percentage = similarity_percentage * reduction_factor
    
    # Special case for very similar faces (like the ones in the example)
    # Check if hair color/style is very similar and facial structure is similar
    if hair_similarity > 0.8 and structure_similarity > 0.7:
        # Boost similarity significantly for faces with very similar hair and structure
        similarity_percentage = min(100, similarity_percentage * 1.4)
    
    return float(similarity_percentage)

# Function to compare all faces between two images
def compare_faces(image1, bboxes1, image2, bboxes2):
    results = []
    
    for i, bbox1 in enumerate(bboxes1):
        features1 = extract_facial_features(image1, bbox1)
        
        face1_comparisons = []
        for j, bbox2 in enumerate(bboxes2):
            features2 = extract_facial_features(image2, bbox2)
            
            similarity = calculate_similarity(features1, features2)
            face1_comparisons.append({
                "face1_index": i,
                "face2_index": j,
                "similarity": similarity
            })
        
        # Sort comparisons by similarity (highest to lowest)
        face1_comparisons.sort(key=lambda x: x["similarity"], reverse=True)
        results.append(face1_comparisons)
    
    return results

# Function to generate a detailed comparison report in English
def generate_comparison_report_english(comparison_results, bboxes1, bboxes2):
    """Generate a detailed comparison report in English."""
    report = []
    
    # Add title and basic information
    report.append("# Face Comparison Report")
    report.append("")
    
    # Add summary section
    report.append("## Summary")
    report.append(f"- Number of faces in first image: {len(bboxes1)}")
    report.append(f"- Number of faces in second image: {len(bboxes2)}")
    
    # Calculate matches above threshold
    threshold = 45.0  # Updated from 60.0 to 45.0
    matches_above_threshold = 0
    all_similarities = []
    
    for face_comparisons in comparison_results:
        for comp in face_comparisons:
            similarity = float(comp["similarity"])
            all_similarities.append(similarity)
            if similarity >= threshold:
                matches_above_threshold += 1
    
    report.append(f"- Matches above threshold ({threshold}%): {matches_above_threshold}")
    
    # Add general statistics
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        max_similarity = max(all_similarities)
        min_similarity = min(all_similarities)
        
        report.append(f"- Average similarity: {avg_similarity:.2f}%")
        report.append(f"- Maximum similarity: {max_similarity:.2f}%")
        report.append(f"- Minimum similarity: {min_similarity:.2f}%")
    
    report.append("")
    
    # Add detailed comparison results
    report.append("## Detailed Comparison Results")
    
    for i, face_comparisons in enumerate(comparison_results):
        report.append(f"### Face {i+1} from first image")
        
        if not face_comparisons:
            report.append("No comparisons available for this face.")
            continue
        
        sorted_comparisons = sorted(face_comparisons, key=lambda x: float(x["similarity"]), reverse=True)
        
        for comp in sorted_comparisons:
            face2_index = comp["face2_index"]
            similarity = float(comp["similarity"])
            
            # Determine similarity level
            if similarity >= 80:  # Updated from 80 to 80
                level = "HIGH"
            elif similarity >= 65:  # Updated from 65 to 65
                level = "MEDIUM"
            elif similarity >= 35:  # Updated from 40 to 35
                level = "LOW"
            else:
                level = "VERY LOW"
            
            report.append(f"- Comparison with Face {face2_index+1} from second image:")
            report.append(f"  - Similarity: {similarity:.2f}% ({level})")
        
        report.append("")
    
    # Add conclusion
    report.append("## Conclusion")
    
    # Find best matches for each face
    best_matches = []
    for i, face_comparisons in enumerate(comparison_results):
        if face_comparisons:
            best_match = max(face_comparisons, key=lambda x: float(x["similarity"]))
            best_match["face1_index"] = i
            best_matches.append(best_match)
    
    # Determine conclusion based on best matches
    if any(float(match["similarity"]) >= threshold for match in best_matches):
        if any(float(match["similarity"]) >= 70 for match in best_matches):  # Updated from 80 to 70
            report.append("HIGH similarity matches found between images. These individuals are likely the same person or very closely related.")
        elif any(float(match["similarity"]) >= 50 for match in best_matches):  # Updated from 65 to 50
            report.append("MEDIUM similarity matches found between images. These individuals share significant facial features but may be different people.")
        else:
            report.append("LOW similarity matches found between images. These individuals share some facial features but are likely different people.")
    else:
        report.append("No significant matches found between images. These individuals appear to be different people.")
    
    return "\n".join(report)

# Function to draw match lines between faces
def draw_face_matches(image1, bboxes1, image2, bboxes2, comparison_results, threshold=45.0):  # Updated from 60.0 to 45.0
    # Create a combined image
    # First, resize images to have the same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    
    # Calculate new dimensions to maintain aspect ratio
    new_h = max(h1, h2)
    new_w1 = int(w1 * (new_h / h1))
    new_w2 = int(w2 * (new_h / h2))
    
    # Resize images
    image1_resized = cv2.resize(image1, (new_w1, new_h))
    image2_resized = cv2.resize(image2, (new_w2, new_h))
    
    # Create a combined image
    combined_img = np.zeros((new_h, new_w1 + new_w2, 3), dtype=np.uint8)
    combined_img[:, :new_w1] = image1_resized
    combined_img[:, new_w1:] = image2_resized
    
    # Calculate scaling factors for bounding boxes
    scale_x1 = new_w1 / w1
    scale_y1 = new_h / h1
    scale_x2 = new_w2 / w2
    scale_y2 = new_h / h2
    
    # Draw bounding boxes on the first image
    for i, bbox1 in enumerate(bboxes1):
        # Scale the bounding box coordinates
        x1, y1, x2, y2, _ = bbox1
        x1_scaled = int(x1 * scale_x1)
        y1_scaled = int(y1 * scale_y1)
        x2_scaled = int(x2 * scale_x1)
        y2_scaled = int(y2 * scale_y1)
        
        # Draw the bounding box
        cv2.rectangle(combined_img, (x1_scaled, y1_scaled), 
                     (x2_scaled, y2_scaled), (0, 255, 0), 2)
        
        # Add face number
        cv2.putText(combined_img, f"Face {i+1}", (x1_scaled, y1_scaled - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw bounding boxes on the second image
    for i, bbox2 in enumerate(bboxes2):
        # Scale the bounding box coordinates
        x1, y1, x2, y2, _ = bbox2
        x1_scaled = int(x1 * scale_x2) + new_w1  # Add offset for second image
        y1_scaled = int(y1 * scale_y2)
        x2_scaled = int(x2 * scale_x2) + new_w1
        y2_scaled = int(y2 * scale_y2)
        
        # Draw the bounding box
        cv2.rectangle(combined_img, (x1_scaled, y1_scaled), 
                     (x2_scaled, y2_scaled), (0, 255, 0), 2)
        
        # Add face number
        cv2.putText(combined_img, f"Face {i+1}", (x1_scaled, y1_scaled - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw lines between matching faces
    for i, face_comparisons in enumerate(comparison_results):
        if face_comparisons:
            best_match = face_comparisons[0]
            similarity = float(best_match["similarity"])
            
            if similarity >= threshold:
                # Calculate face centers
                bbox1 = bboxes1[i]
                bbox2 = bboxes2[best_match["face2_index"]]
                
                center1_x = (bbox1[0] + bbox1[2]) // 2
                center1_y = (bbox1[1] + bbox1[3]) // 2
                
                center2_x = (bbox2[0] + bbox2[2]) // 2
                center2_y = (bbox2[1] + bbox2[3]) // 2
                
                # Scale the center points
                center1_x = int(center1_x * scale_x1)
                center1_y = int(center1_y * scale_y1)
                center2_x = int(center2_x * scale_x2) + new_w1  # Add offset for second image
                center2_y = int(center2_y * scale_y2)
                
                # Determine line color based on similarity
                if similarity >= 70:  # Updated from 75 to 70
                    color = (0, 255, 0)  # Green for high similarity
                elif similarity >= 50:  # Updated from 60 to 50
                    color = (0, 165, 255)  # Orange for medium similarity
                else:
                    color = (0, 0, 255)  # Red for low similarity
                
                # Draw the line
                cv2.line(combined_img, (center1_x, center1_y), (center2_x, center2_y), color, 2)
                
                # Add similarity text
                mid_x = (center1_x + center2_x) // 2
                mid_y = (center1_y + center2_y) // 2
                cv2.putText(combined_img, f"{similarity:.2f}%", (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return combined_img

# App title and description
st.title("Advanced Face & Feature Detection")
st.markdown("""
This app uses OpenCV's DNN models to detect faces and additional facial features in images and videos.
Upload an image or video and adjust the settings to see the detection in action!
""")

# Sidebar for navigation and controls
st.sidebar.title("Controls & Settings")

# Initialize session_state to store original image and camera state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'feature_camera_running' not in st.session_state:
    st.session_state.feature_camera_running = False

# Navigation menu
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["About", "Face Detection", "Feature Detection", "Comparison Mode", "Face Recognition"]
)

# Function to load DNN models with caching and auto-download
@st.cache_resource
def load_face_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Correct model file names
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt.txt"
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(modelFile):
        missing_files.append(modelFile)
    if not os.path.exists(configFile):
        missing_files.append(configFile)
    
    if missing_files:
        st.error("Missing model files: " + ", ".join(missing_files))
        st.error("Please manually download the following files:")
        st.code("""
        1. Download the model file:
           URL: https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/res10_300x300_ssd_iter_140000.caffemodel
           Save as: models/res10_300x300_ssd_iter_140000.caffemodel
           
        2. Download the configuration file:
           URL: https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/deploy.prototxt.txt
           Save as: models/deploy.prototxt.txt
        """)
        st.stop()
    
    # Load model
    try:
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        return net
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_feature_models():
    # Load pre-trained models for eye and smile detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return eye_cascade, smile_cascade

# Function for detecting faces in an image
def detect_face_dnn(net, frame, conf_threshold=0.5):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for processing face detections
def process_face_detections(frame, detections, conf_threshold=0.5, bbox_color=(0, 255, 0)):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    
    # Create a copy for drawing on
    result_frame = frame.copy()
    
    # Loop over all detections and draw bounding boxes around each face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * frame_w))
            y1 = max(0, int(detections[0, 0, i, 4] * frame_h))
            x2 = min(frame_w, int(detections[0, 0, i, 5] * frame_w))
            y2 = min(frame_h, int(detections[0, 0, i, 6] * frame_h))
            
            bboxes.append([x1, y1, x2, y2, confidence])
            
            # Calculate line thickness based on image size
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            
            # Draw bounding box with confidence score
            cv2.rectangle(
                result_frame, 
                (x1, y1), (x2, y2), 
                bbox_color, 
                bb_line_thickness, 
                cv2.LINE_8
            )
            
            # Display confidence score
            label = f"{confidence:.2f}"
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(result_frame, (x1, y1-label_size[1]), (x1+label_size[0], y1+base_line), (255, 255, 255), cv2.FILLED)
            cv2.putText(result_frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_frame, bboxes

# Function to detect facial features (eyes, smile) with improved profile face handling
def detect_facial_features(frame, bboxes, eye_cascade, smile_cascade, detect_eyes=True, detect_smile=True, smile_sensitivity=15, eye_sensitivity=5):
    result_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Counters for detection summary
    eye_count = 0
    smile_count = 0
    
    for bbox in bboxes:
        x1, y1, x2, y2, _ = bbox
        roi_gray = gray[y1:y2, x1:x2]
        roi_color = result_frame[y1:y2, x1:x2]
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Detect eyes if enabled
        if detect_eyes:
            # Adjust region of interest to focus on the upper part of the face
            upper_face_y1 = y1
            upper_face_y2 = y1 + int(face_height * 0.55)  # Slightly reduced to focus more on the eye area
            
            # For profile faces, we need to search the entire upper region
            # as well as the left and right sides separately
            
            # Full upper region for profile faces
            upper_face_roi_gray = gray[upper_face_y1:upper_face_y2, x1:x2]
            upper_face_roi_color = result_frame[upper_face_y1:upper_face_y2, x1:x2]
            
            # Split the upper region into two halves (left and right) to search for eyes individually
            mid_x = x1 + face_width // 2
            left_eye_roi_gray = gray[upper_face_y1:upper_face_y2, x1:mid_x]
            right_eye_roi_gray = gray[upper_face_y1:upper_face_y2, mid_x:x2]
            
            left_eye_roi_color = result_frame[upper_face_y1:upper_face_y2, x1:mid_x]
            right_eye_roi_color = result_frame[upper_face_y1:upper_face_y2, mid_x:x2]
            
            # Apply histogram equalization and contrast enhancement for all regions
            if upper_face_roi_gray.size > 0:
                upper_face_roi_gray = cv2.equalizeHist(upper_face_roi_gray)
                
                # Enhance contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                upper_face_roi_gray = clahe.apply(upper_face_roi_gray)
                
                # First try to detect eyes in the full upper region (for profile faces)
                full_eyes = eye_cascade.detectMultiScale(
                    upper_face_roi_gray, 
                    scaleFactor=1.02,  # More sensitive for profile faces
                    minNeighbors=max(1, eye_sensitivity-3),  # Even more sensitive
                    minSize=(int(face_width * 0.07), int(face_width * 0.07)),
                    maxSize=(int(face_width * 0.3), int(face_width * 0.3))
                )
                
                # If we found eyes in the full region, use those
                if len(full_eyes) > 0:
                    # Sort by size (area) and take up to 2 largest
                    full_eyes = sorted(full_eyes, key=lambda e: e[2] * e[3], reverse=True)
                    full_eyes = full_eyes[:2]  # Take at most 2 eyes
                    
                    for ex, ey, ew, eh in full_eyes:
                        eye_count += 1
                        cv2.rectangle(upper_face_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        cv2.putText(upper_face_roi_color, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # If no eyes found in full region, try left and right separately
                    if left_eye_roi_gray.size > 0:
                        left_eye_roi_gray = cv2.equalizeHist(left_eye_roi_gray)
                        left_eye_roi_gray = clahe.apply(left_eye_roi_gray)
                        
                        left_eyes = eye_cascade.detectMultiScale(
                            left_eye_roi_gray, 
                            scaleFactor=1.03,
                            minNeighbors=max(1, eye_sensitivity-2),
                            minSize=(int(face_width * 0.08), int(face_width * 0.08)),
                            maxSize=(int(face_width * 0.25), int(face_width * 0.25))
                        )
                        
                        if len(left_eyes) > 0:
                            # Sort by size and take the largest
                            left_eyes = sorted(left_eyes, key=lambda e: e[2] * e[3], reverse=True)
                            left_eye = left_eyes[0]
                            eye_count += 1
                            
                            # Draw rectangle for the left eye
                            ex, ey, ew, eh = left_eye
                            cv2.rectangle(left_eye_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                            cv2.putText(left_eye_roi_color, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    if right_eye_roi_gray.size > 0:
                        right_eye_roi_gray = cv2.equalizeHist(right_eye_roi_gray)
                        right_eye_roi_gray = clahe.apply(right_eye_roi_gray)
                        
                        right_eyes = eye_cascade.detectMultiScale(
                            right_eye_roi_gray, 
                            scaleFactor=1.03,
                            minNeighbors=max(1, eye_sensitivity-2),
                            minSize=(int(face_width * 0.08), int(face_width * 0.08)),
                            maxSize=(int(face_width * 0.25), int(face_width * 0.25))
                        )
                        
                        if len(right_eyes) > 0:
                            # Sort by size and take the largest
                            right_eyes = sorted(right_eyes, key=lambda e: e[2] * e[3], reverse=True)
                            right_eye = right_eyes[0]
                            eye_count += 1
                            
                            # Draw rectangle for the right eye
                            ex, ey, ew, eh = right_eye
                            cv2.rectangle(right_eye_roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                            cv2.putText(right_eye_roi_color, "Eye", (ex, ey-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Detect smile if enabled
        if detect_smile:
            # For profile faces, we need to adjust the region of interest
            # Try multiple regions to improve detection
            
            # Standard region (middle to bottom)
            lower_face_y1 = y1 + int(face_height * 0.5)
            lower_face_roi_gray = gray[lower_face_y1:y2, x1:x2]
            lower_face_roi_color = result_frame[lower_face_y1:y2, x1:x2]
            
            # Alternative region (lower third)
            alt_lower_face_y1 = y1 + int(face_height * 0.65)
            alt_lower_face_roi_gray = gray[alt_lower_face_y1:y2, x1:x2]
            
            # Apply histogram equalization and enhance contrast
            smile_detected = False
            
            if lower_face_roi_gray.size > 0:
                lower_face_roi_gray = cv2.equalizeHist(lower_face_roi_gray)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lower_face_roi_gray = clahe.apply(lower_face_roi_gray)
                
                # Try with standard parameters
                smiles = smile_cascade.detectMultiScale(
                    lower_face_roi_gray, 
                    scaleFactor=1.2,
                    minNeighbors=smile_sensitivity,
                    minSize=(int(face_width * 0.25), int(face_width * 0.15)),
                    maxSize=(int(face_width * 0.7), int(face_width * 0.4))
                )
                
                if len(smiles) > 0:
                    # Sort by size and take the largest
                    smiles = sorted(smiles, key=lambda s: s[2] * s[3], reverse=True)
                    sx, sy, sw, sh = smiles[0]
                    
                    # Increment smile counter
                    smile_count += 1
                    smile_detected = True
                    
                    # Draw rectangle for the smile
                    cv2.rectangle(lower_face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
                    cv2.putText(lower_face_roi_color, "Smile", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # If no smile detected in standard region, try alternative region
            if not smile_detected and alt_lower_face_roi_gray.size > 0:
                alt_lower_face_roi_gray = cv2.equalizeHist(alt_lower_face_roi_gray)
                alt_lower_face_roi_gray = clahe.apply(alt_lower_face_roi_gray)
                
                # Try with more sensitive parameters
                alt_smiles = smile_cascade.detectMultiScale(
                    alt_lower_face_roi_gray, 
                    scaleFactor=1.1,
                    minNeighbors=max(1, smile_sensitivity-5),  # More sensitive
                    minSize=(int(face_width * 0.2), int(face_width * 0.1)),
                    maxSize=(int(face_width * 0.6), int(face_width * 0.3))
                )
                
                if len(alt_smiles) > 0:
                    # Sort by size and take the largest
                    alt_smiles = sorted(alt_smiles, key=lambda s: s[2] * s[3], reverse=True)
                    sx, sy, sw, sh = alt_smiles[0]
                    
                    # Adjust coordinates for the alternative region
                    adjusted_sy = sy + (alt_lower_face_y1 - lower_face_y1)
                    
                    # Increment smile counter
                    smile_count += 1
                    
                    # Draw rectangle for the smile (in the original lower face ROI)
                    cv2.rectangle(lower_face_roi_color, (sx, adjusted_sy), (sx+sw, adjusted_sy+sh), (0, 0, 255), 2)
                    cv2.putText(lower_face_roi_color, "Smile", (sx, adjusted_sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return result_frame, eye_count, smile_count

# Función para detectar atributos faciales (edad, género, emoción)
def detect_face_attributes(image, bbox):
    """
    Detecta atributos faciales como edad, género y emoción usando DeepFace.
    
    Args:
        image: Imagen en formato OpenCV (BGR)
        bbox: Bounding box de la cara [x1, y1, x2, y2, conf]
        
    Returns:
        Diccionario con los atributos detectados
    """
    if not DEEPFACE_AVAILABLE:
        return None
    
    try:
        x1, y1, x2, y2, _ = bbox
        face_img = image[y1:y2, x1:x2]
        
        # Convertir de BGR a RGB para DeepFace
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Analizar atributos faciales
        attributes = DeepFace.analyze(
            img_path=face_img_rgb,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            detector_backend="opencv"
        )
        
        return attributes[0]
    
    except Exception as e:
        st.error(f"Error al detectar atributos faciales: {str(e)}")
        return None

# Function to apply age and gender detection (placeholder - would need additional models)
def detect_age_gender(frame, bboxes):
    # Versión mejorada que usa DeepFace si está disponible
    result_frame = frame.copy()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, _ = bbox
        
        if DEEPFACE_AVAILABLE:
            # Intentar usar DeepFace para análisis facial
            attributes = detect_face_attributes(frame, bbox)
            
            if attributes:
                # Extraer información de atributos
                age = attributes.get('age', 'Unknown')
                gender = attributes.get('gender', 'Unknown')
                emotion = attributes.get('dominant_emotion', 'Unknown').capitalize()
                gender_prob = attributes.get('gender', {}).get('Woman', 0)
                
                # Determinar color basado en confianza
                if gender == 'Woman':
                    gender_color = (255, 0, 255)  # Magenta para mujer
                else:
                    gender_color = (255, 0, 0)    # Azul para hombre
                
                # Añadir texto con información
                cv2.putText(result_frame, f"Age: {age}", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(result_frame, f"Gender: {gender}", (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, gender_color, 2)
                cv2.putText(result_frame, f"Emotion: {emotion}", (x1, y2+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                # Fallback si DeepFace falla
                cv2.putText(result_frame, "Age: Unknown", (x1, y2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(result_frame, "Gender: Unknown", (x1, y2+40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            # Usar texto placeholder si DeepFace no está disponible
            cv2.putText(result_frame, "Age: 25-35", (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(result_frame, "Gender: Unknown", (x1, y2+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return result_frame

# Function to generate download link for processed image
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to process video frames
def process_video(video_path, face_net, eye_cascade, smile_cascade, conf_threshold=0.5, detect_eyes=False, detect_smile=False, bbox_color=(0, 255, 0), smile_sensitivity=15, eye_sensitivity=5):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create temporary output file
    temp_dir = tempfile.mkdtemp()
    temp_output_path = os.path.join(temp_dir, "processed_video.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create a progress bar
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process video frames
    current_frame = 0
    processing_times = []
    
    # Total counters for statistics
    total_faces = 0
    total_eyes = 0
    total_smiles = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Start timing for performance metrics
        start_time = time.time()
        
        # Detect faces
        detections = detect_face_dnn(face_net, frame, conf_threshold)
        processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color)
        
        # Update face counter
        total_faces += len(bboxes)
        
        # Detect facial features if enabled
        if detect_eyes or detect_smile:
            processed_frame, eye_count, smile_count = detect_facial_features(
                processed_frame, 
                bboxes, 
                eye_cascade, 
                smile_cascade,
                detect_eyes,
                detect_smile,
                smile_sensitivity,
                eye_sensitivity
            )
            # Update counters
            total_eyes += eye_count
            total_smiles += smile_count
        
        # End timing
        processing_times.append(time.time() - start_time)
        
        # Write the processed frame
        out.write(processed_frame)
        
        # Update progress
        current_frame += 1
        progress_bar.progress(current_frame / frame_count)
        status_text.text(f"Processing frame {current_frame}/{frame_count}")
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate and display performance metrics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        status_text.text(f"Processing complete! Average processing time: {avg_time:.4f}s per frame")
    
    # Return detection statistics
    detection_stats = {
        "faces": total_faces // max(1, current_frame),  # Average per frame
        "eyes": total_eyes // max(1, current_frame),    # Average per frame
        "smiles": total_smiles // max(1, current_frame) # Average per frame
    }
    
    return temp_output_path, temp_dir, detection_stats

# Camera control functions
def start_camera():
    st.session_state.camera_running = True

def stop_camera():
    st.session_state.camera_running = False
    st.session_state.camera_stopped = True

def start_feature_camera():
    st.session_state.feature_camera_running = True

def stop_feature_camera():
    st.session_state.feature_camera_running = False
    st.session_state.feature_camera_stopped = True

# Main application logic
if app_mode == "About":
    st.markdown("""
    ## About This App
    
    This application uses OpenCV's Deep Neural Network (DNN) module and Haar Cascade classifiers to detect faces and facial features in images and videos.
    
    ### Features:
    - Face detection using OpenCV DNN
    - Eye and smile detection using Haar Cascades
    - Support for both image and video processing
    - Adjustable confidence threshold
    - Download options for processed media
    - Performance metrics
    
    ### How to use:
    1. Select a mode from the sidebar
    2. Upload an image or video
    3. Adjust settings as needed
    4. View and download the results
    
    ### Technologies Used:
    - Streamlit for the web interface
    - OpenCV for computer vision operations
    - Python for backend processing
    
    ### Models:
    - SSD MobileNet for face detection
    - Haar Cascades for facial features
    """)
    
    # Display a sample image or GIF
    st.image("https://opencv.org/wp-content/uploads/2019/07/detection.gif", caption="Sample face detection", use_container_width=True)

elif app_mode == "Face Detection":
    # Load the face detection model
    face_net = load_face_model()
    
    # Input type selection (Image or Video)
    input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
    
    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Adjust the threshold for face detection confidence (higher = fewer detections but more accurate)"
    )
    
    # Style options
    bbox_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
    # Convert hex color to BGR for OpenCV
    bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
    
    # Display processing metrics
    show_metrics = st.sidebar.checkbox("Show Processing Metrics", True)
    
    if input_type == "Image":
        # File uploader for images
        file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if file_buffer is not None:
            # Read the file and convert it to OpenCV format
            raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
            image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            
            # Guardar la imagen original en session_state para reprocesarla cuando cambie el umbral
            # Usar un identificador único para cada archivo para detectar cambios
            file_id = file_buffer.name + str(file_buffer.size)
            
            if 'file_id' not in st.session_state or st.session_state.file_id != file_id:
                st.session_state.file_id = file_id
                st.session_state.original_image = image.copy()
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(st.session_state.original_image, channels='BGR', use_container_width=True)
            
            # Start timing for performance metrics
            start_time = time.time()
            
            # Detect faces
            detections = detect_face_dnn(face_net, st.session_state.original_image, conf_threshold)
            processed_image, bboxes = process_face_detections(st.session_state.original_image, detections, conf_threshold, bbox_color_bgr)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Display the processed image
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, channels='BGR', use_container_width=True)
                
                # Convert OpenCV image to PIL for download
                pil_img = Image.fromarray(processed_image[:, :, ::-1])
                st.markdown(
                    get_image_download_link(pil_img, "face_detection_result.jpg", "📥 Download Processed Image"),
                    unsafe_allow_html=True
                )
            
            # Show metrics if enabled
            if show_metrics:
                st.subheader("Processing Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Processing Time", f"{processing_time:.4f} seconds")
                col2.metric("Faces Detected", len(bboxes))
                col3.metric("Confidence Threshold", f"{conf_threshold:.2f}")
                
                # Display detailed metrics in an expandable section
                with st.expander("Detailed Detection Information"):
                    if bboxes:
                        st.write("Detected faces with confidence scores:")
                        for i, bbox in enumerate(bboxes):
                            st.write(f"Face #{i+1}: Confidence = {bbox[4]:.4f}")
                    else:
                        st.write("No faces detected in the image.")
    
    else:  # Video mode
        # Video mode options
        video_source = st.radio("Select video source", ["Upload video", "Use webcam"])
        
        if video_source == "Upload video":
            # File uploader for videos
            file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
            
            if file_buffer is not None:
                # Save uploaded video to temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "input_video.mp4")
                
                with open(temp_path, "wb") as f:
                    f.write(file_buffer.read())
                
                # Display original video
                st.subheader("Original Video")
                st.video(temp_path)
                
                # Load models for feature detection (will be used in the processing)
                eye_cascade, smile_cascade = load_feature_models()
                
                # Process video button
                if st.button("Process Video"):
                    with st.spinner("Processing video... This may take a while depending on the video length."):
                        # Process the video
                        output_path, output_dir, detection_stats = process_video(
                            temp_path, 
                            face_net, 
                            eye_cascade,
                            smile_cascade,
                            conf_threshold,
                            detect_eyes=False,
                            detect_smile=False,
                            bbox_color=bbox_color_bgr,
                            eye_sensitivity=5
                        )
                        
                        # Display processed video
                        st.subheader("Processed Video")
                        st.video(output_path)
                        
                        # Mostrar estadísticas de detección
                        st.subheader("Detection Summary")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        summary_col1.metric("Avg. Faces per Frame", detection_stats["faces"])
                        
                        if detect_eyes: # type: ignore
                            summary_col2.metric("Avg. Eyes per Frame", detection_stats["eyes"])
                        else:
                            summary_col2.metric("Eyes Detected", "N/A")
                        
                        if detect_smile: # type: ignore
                            summary_col3.metric("Avg. Smiles per Frame", detection_stats["smiles"])
                        else:
                            summary_col3.metric("Smiles Detected", "N/A")
                        
                        # Provide download link
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                        
                        # Clean up temporary files
                        try:
                            os.remove(temp_path)
                            os.remove(output_path)
                            os.rmdir(temp_dir)
                            os.rmdir(output_dir)
                        except:
                            pass
        else:  # Use webcam
            st.subheader("Real-time face detection")
            st.write("Click 'Start Camera' to begin real-time face detection.")
            
            # Placeholder for webcam video
            camera_placeholder = st.empty()
            
            # Buttons to control the camera
            col1, col2 = st.columns(2)
            start_button = col1.button("Start Camera", on_click=start_camera)
            stop_button = col2.button("Stop Camera", on_click=stop_camera)
            
            # Show message when camera is stopped
            if 'camera_stopped' in st.session_state and st.session_state.camera_stopped:
                st.info("Camera stopped. Click 'Start Camera' to activate it again.")
                st.session_state.camera_stopped = False
            
            if st.session_state.camera_running:
                st.info("Camera activated. Processing real-time video...")
                # Initialize webcam
                cap = cv2.VideoCapture(0)  # 0 is typically the main webcam
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Make sure it's connected and not being used by another application.")
                    st.session_state.camera_running = False
                else:
                    # Display real-time video with face detection
                    try:
                        while st.session_state.camera_running:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Error reading frame from camera.")
                                break
                            
                            # Detect faces
                            detections = detect_face_dnn(face_net, frame, conf_threshold)
                            processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color_bgr)
                            
                            # Display the processed frame
                            camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                            
                            # Small pause to avoid overloading the CPU
                            time.sleep(0.01)
                    finally:
                        # Release the camera when stopped
                        cap.release()

elif app_mode == "Feature Detection":
    # Load all required models
    face_net = load_face_model()
    eye_cascade, smile_cascade = load_feature_models()
    
    # Feature selection checkboxes
    st.sidebar.subheader("Feature Detection Options")
    detect_eyes = st.sidebar.checkbox("Detect Eyes", True)
    
    # Add controls for eye detection sensitivity
    eye_sensitivity = 5  # Default value
    if detect_eyes:
        eye_sensitivity = st.sidebar.slider(
            "Eye Detection Sensitivity", 
            min_value=1, 
            max_value=10, 
            value=5, 
            step=1,
            help="Adjust the sensitivity of eye detection (lower value = more detections)"
        )
    
    detect_smile = st.sidebar.checkbox("Detect Smile", True)
    
    # Add controls for smile detection sensitivity
    smile_sensitivity = 15  # Default value
    if detect_smile:
        smile_sensitivity = st.sidebar.slider(
            "Smile Detection Sensitivity", 
            min_value=5, 
            max_value=30, 
            value=15, 
            step=1,
            help="Adjust the sensitivity of smile detection (lower value = more detections)"
        )
    
    detect_age_gender_option = st.sidebar.checkbox("Detect Age/Gender (Demo)", False)
    
    # Confidence threshold slider
    conf_threshold = st.sidebar.slider(
        "Face Detection Confidence", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05
    )
    
    # Style options
    bbox_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
    # Convert hex color to BGR for OpenCV
    bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
    
    # Input type selection
    input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
    
    if input_type == "Image":
        # File uploader for images
        file_buffer = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if file_buffer is not None:
            # Read the file and convert it to OpenCV format
            raw_bytes = np.asarray(bytearray(file_buffer.read()), dtype=np.uint8)
            image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            
            # Guardar la imagen original en session_state para reprocesarla cuando cambie el umbral
            # Usar un identificador único para cada archivo para detectar cambios
            file_id = file_buffer.name + str(file_buffer.size)
            
            if 'feature_file_id' not in st.session_state or st.session_state.feature_file_id != file_id:
                st.session_state.feature_file_id = file_id
                st.session_state.feature_original_image = image.copy()
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(st.session_state.feature_original_image, channels='BGR', use_container_width=True)
            
            # Start processing with face detection
            detections = detect_face_dnn(face_net, st.session_state.feature_original_image, conf_threshold)
            processed_image, bboxes = process_face_detections(st.session_state.feature_original_image, detections, conf_threshold, bbox_color_bgr)
            
            # Inicializar contadores
            eye_count = 0
            smile_count = 0
            
            # Detect facial features if any options are enabled
            if detect_eyes or detect_smile:
                processed_image, eye_count, smile_count = detect_facial_features(
                    processed_image, 
                    bboxes,
                    eye_cascade,
                    smile_cascade,
                    detect_eyes,
                    detect_smile,
                    smile_sensitivity,
                    eye_sensitivity
                )
                
            # Apply age/gender detection if enabled (demo purpose)
            if detect_age_gender_option:
                processed_image = detect_age_gender(processed_image, bboxes)
            
            # Display the processed image
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, channels='BGR', use_container_width=True)
                
                # Convert OpenCV image to PIL for download
                pil_img = Image.fromarray(processed_image[:, :, ::-1])
                st.markdown(
                    get_image_download_link(pil_img, "feature_detection_result.jpg", "📥 Download Processed Image"),
                    unsafe_allow_html=True
                )
            
            # Display detection summary
            st.subheader("Detection Summary")
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Faces Detected", len(bboxes))
            
            if detect_eyes:
                summary_col2.metric("Eyes Detected", eye_count)
            else:
                summary_col2.metric("Eyes Detected", "N/A")
            
            if detect_smile:
                summary_col3.metric("Smiles Detected", smile_count)
            else:
                summary_col3.metric("Smiles Detected", "N/A")
    
    else:  # Video mode
        st.write("Facial feature detection in video")
        
        # Video mode options
        video_source = st.radio("Select video source", ["Upload video", "Use webcam"])
        
        if video_source == "Upload video":
            st.write("Upload a video to process with facial feature detection.")
            # Similar implementation to Face Detection mode for uploaded videos
            file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
            
            if file_buffer is not None:
                # Save uploaded video to temporary file
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, "input_video.mp4")
                
                with open(temp_path, "wb") as f:
                    f.write(file_buffer.read())
                
                # Display original video
                st.subheader("Original Video")
                st.video(temp_path)
                
                # Process video button
                if st.button("Process Video"):
                    with st.spinner("Processing video... This may take a while depending on the video length."):
                        # Process the video with feature detection
                        output_path, output_dir, detection_stats = process_video(
                            temp_path, 
                            face_net, 
                            eye_cascade,
                            smile_cascade,
                            conf_threshold,
                            detect_eyes=detect_eyes,
                            detect_smile=detect_smile,
                            bbox_color=bbox_color_bgr,
                            smile_sensitivity=smile_sensitivity,
                            eye_sensitivity=eye_sensitivity
                        )
                        
                        # Display processed video
                        st.subheader("Processed Video")
                        st.video(output_path)
                        
                        # Mostrar estadísticas de detección
                        st.subheader("Detection Summary")
                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                        summary_col1.metric("Avg. Faces per Frame", detection_stats["faces"])
                        
                        if detect_eyes:
                            summary_col2.metric("Avg. Eyes per Frame", detection_stats["eyes"])
                        else:
                            summary_col2.metric("Eyes Detected", "N/A")
                        
                        if detect_smile:
                            summary_col3.metric("Avg. Smiles per Frame", detection_stats["smiles"])
                        else:
                            summary_col3.metric("Smiles Detected", "N/A")
                        
                        # Provide download link
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=video_bytes,
                            file_name="feature_detection_video.mp4",
                            mime="video/mp4"
                        )
                        
                        # Clean up temporary files
                        try:
                            os.remove(temp_path)
                            os.remove(output_path)
                            os.rmdir(temp_dir)
                            os.rmdir(output_dir)
                        except:
                            pass
        else:  # Usar cámara web
            st.subheader("Real-time facial feature detection")
            st.write("Click 'Start Camera' to begin real-time detection.")
            
            # Placeholder for webcam video
            camera_placeholder = st.empty()
            
            # Buttons to control the camera
            col1, col2 = st.columns(2)
            start_button = col1.button("Start Camera", on_click=start_feature_camera)
            stop_button = col2.button("Stop Camera", on_click=stop_feature_camera)
            
            # Show message when camera is stopped
            if 'feature_camera_stopped' in st.session_state and st.session_state.feature_camera_stopped:
                st.info("Camera stopped. Click 'Start Camera' to activate it again.")
                st.session_state.feature_camera_stopped = False
            
            if st.session_state.feature_camera_running:
                st.info("Camera activated. Processing real-time video with feature detection...")
                # Initialize webcam
                cap = cv2.VideoCapture(0)  # 0 is typically the main webcam
                
                if not cap.isOpened():
                    st.error("Could not access webcam. Make sure it's connected and not being used by another application.")
                    st.session_state.feature_camera_running = False
                else:
                    # Display real-time video with face and feature detection
                    try:
                        # Create placeholders for metrics
                        metrics_placeholder = st.empty()
                        metrics_col1, metrics_col2, metrics_col3 = metrics_placeholder.columns(3)
                        
                        # Initialize counters
                        face_count_total = 0
                        eye_count_total = 0
                        smile_count_total = 0
                        frame_count = 0
                        
                        while st.session_state.feature_camera_running:
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Error reading frame from camera.")
                                break
                            
                            # Detect faces
                            detections = detect_face_dnn(face_net, frame, conf_threshold)
                            processed_frame, bboxes = process_face_detections(frame, detections, conf_threshold, bbox_color_bgr)
                            
                            # Update face counter
                            face_count = len(bboxes)
                            face_count_total += face_count
                            
                            # Initialize counters for this frame
                            eye_count = 0
                            smile_count = 0
                            
                            # Detect facial features if enabled
                            if detect_eyes or detect_smile:
                                processed_frame, eye_count, smile_count = detect_facial_features(
                                    processed_frame, 
                                    bboxes,
                                    eye_cascade,
                                    smile_cascade,
                                    detect_eyes,
                                    detect_smile,
                                    smile_sensitivity,
                                    eye_sensitivity
                                )
                                
                                # Update total counters
                                eye_count_total += eye_count
                                smile_count_total += smile_count
                            
                            # Apply age/gender detection if enabled
                            if detect_age_gender_option:
                                processed_frame = detect_age_gender(processed_frame, bboxes)
                            
                            # Display the processed frame
                            camera_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                            
                            # Update frame counter
                            frame_count += 1
                            
                            # Update metrics every 5 frames to avoid overloading the interface
                            if frame_count % 5 == 0:
                                metrics_col1.metric("Faces Detected", face_count)
                                
                                if detect_eyes:
                                    metrics_col2.metric("Eyes Detected", eye_count)
                                else:
                                    metrics_col2.metric("Eyes Detected", "N/A")
                                
                                if detect_smile:
                                    metrics_col3.metric("Smiles Detected", smile_count)
                                else:
                                    metrics_col3.metric("Smiles Detected", "N/A")
                            
                            # Small pause to avoid overloading the CPU
                            time.sleep(0.01)
                    finally:
                        # Release the camera when stopped
                        cap.release()

elif app_mode == "Comparison Mode":
    st.subheader("Face Comparison")
    st.write("Upload two images to compare faces between them.")
    
    # Añadir explicación sobre la interpretación de resultados
    with st.expander("📌 How to interpret similarity results"):
        st.markdown("""
        ### Facial Similarity Interpretation Guide
        
        The system calculates similarity between faces based on multiple facial features and characteristics.
        
        **Similarity Ranges:**
        - **70-100%**: HIGH Similarity - Very likely to be the same person or identical twins
        - **50-70%**: MEDIUM Similarity - Possible match, requires verification
        - **30-50%**: LOW Similarity - Different people with some similar features
        - **0-30%**: VERY LOW Similarity - Completely different people
        
        **Enhanced Comparison System:**
        The system uses a sophisticated approach that:
        1. Analyzes multiple facial characteristics with advanced precision
        2. Evaluates hair style/color, facial structure, texture patterns, and expressions with improved accuracy
        3. Applies a balanced differentiation between similar and different individuals
        4. Creates a clear gap between similar and different people's scores
        5. Reduces scores for people with different facial structures
        6. Applies penalty factors for critical differences in facial features
        
        **Features Analyzed:**
        - Facial texture patterns (HOG features)
        - Eye region characteristics (highly weighted)
        - Nose bridge features
        - Hair style and color patterns (enhanced detection)
        - Precise facial proportions and structure
        - Texture and edge patterns
        - Facial expressions
        - Critical difference markers (aspect ratio, brightness patterns, texture variance)
        
        **Factors affecting similarity:**
        - Face angle and expression
        - Lighting conditions
        - Age differences
        - Image quality
        - Gender characteristics (with stronger weighting)
        - Critical facial structure differences
        
        **Important note:** This system is designed to provide highly accurate similarity scores that create a clear distinction between different individuals while still recognizing truly similar people. The algorithm now applies multiple reduction factors to ensure that different people receive appropriately low similarity scores. For official identification, always use certified systems.
        """)
    
    # Load face detection model
    face_net = load_face_model()
    
    # Side-by-side file uploaders
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First Image")
        file1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png'], key="file1")
    
    with col2:
        st.write("Second Image")
        file2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png'], key="file2")
    
    # Set confidence threshold
    conf_threshold = st.slider("Face Detection Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    # Similarity threshold for considering a match
    similarity_threshold = st.slider("Similarity Threshold (%)", min_value=35.0, max_value=95.0, value=45.0, step=5.0,
                                    help="Minimum percentage of similarity to consider two faces as a match")
    
    # Selección del método de comparación
    comparison_method = st.radio(
        "Método de comparación facial",
        ["HOG (Rápido, efectivo)", "Embeddings (Lento, más preciso)"],
        help="HOG utiliza histogramas de gradientes orientados para comparación rápida. Embeddings utiliza redes neuronales profundas para mayor precisión."
    )
    
    # Si se selecciona embeddings, mostrar opciones de modelos y advertencia
    embedding_model = "VGG-Face"
    if comparison_method == "Embeddings (Lento, más preciso)" and DEEPFACE_AVAILABLE:
        st.warning("ADVERTENCIA: La versión actual de TensorFlow (2.19) puede tener incompatibilidades con algunos modelos. Se recomienda usar HOG si experimenta problemas.")
        
        embedding_model = st.selectbox(
            "Modelo de embeddings",
            ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],  # Eliminado "DeepFace" de la lista
            help="Seleccione el modelo de red neuronal para extraer embeddings faciales"
        )
    elif comparison_method == "Embeddings (Lento, más preciso)" and not DEEPFACE_AVAILABLE:
        st.warning("La biblioteca DeepFace no está disponible. Por favor instale con 'pip install deepface' para usar embeddings.")
        st.info("Usando método HOG por defecto.")
        comparison_method = "HOG (Rápido, efectivo)"
    
    # Style options
    bbox_color = st.color_picker("Bounding Box Color", "#00FF00")
    # Convert hex color to BGR for OpenCV
    bbox_color_rgb = tuple(int(bbox_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    bbox_color_bgr = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])  # Convert RGB to BGR
    
    # Process the images when both are uploaded
    if file1 is not None and file2 is not None:
        # Read both images
        raw_bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
        image1 = cv2.imdecode(raw_bytes1, cv2.IMREAD_COLOR)
        
        raw_bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
        image2 = cv2.imdecode(raw_bytes2, cv2.IMREAD_COLOR)
        
        # Save original images in session_state
        # Use a unique identifier for each file to detect changes
        file1_id = file1.name + str(file1.size)
        file2_id = file2.name + str(file2.size)
        
        if 'file1_id' not in st.session_state or st.session_state.file1_id != file1_id:
            st.session_state.file1_id = file1_id
            st.session_state.original_image1 = image1.copy()
        
        if 'file2_id' not in st.session_state or st.session_state.file2_id != file2_id:
            st.session_state.file2_id = file2_id
            st.session_state.original_image2 = image2.copy()
        
        # Display original images
        with col1:
            st.image(st.session_state.original_image1, channels='BGR', use_container_width=True, caption="Image 1")
        
        with col2:
            st.image(st.session_state.original_image2, channels='BGR', use_container_width=True, caption="Image 2")
        
        # Detect faces in both images
        detections1 = detect_face_dnn(face_net, st.session_state.original_image1, conf_threshold)
        processed_image1, bboxes1 = process_face_detections(st.session_state.original_image1, detections1, conf_threshold, bbox_color_bgr)
        
        detections2 = detect_face_dnn(face_net, st.session_state.original_image2, conf_threshold)
        processed_image2, bboxes2 = process_face_detections(st.session_state.original_image2, detections2, conf_threshold, bbox_color_bgr)
        
        # Display processed images
        st.subheader("Detected Faces")
        proc_col1, proc_col2 = st.columns(2)
        
        with proc_col1:
            st.image(processed_image1, channels='BGR', use_container_width=True, caption="Processed Image 1")
            st.write(f"Faces detected: {len(bboxes1)}")
        
        with proc_col2:
            st.image(processed_image2, channels='BGR', use_container_width=True, caption="Processed Image 2")
            st.write(f"Faces detected: {len(bboxes2)}")
        
        # Compare faces
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            st.warning("Cannot compare: One or both images have no faces detected.")
        else:
            with st.spinner("Comparing faces..."):
                # Perform face comparison based on selected method
                if comparison_method == "Embeddings (Lento, más preciso)" and DEEPFACE_AVAILABLE:
                    try:
                        st.info(f"Usando modelo de embeddings: {embedding_model}")
                        comparison_results = compare_faces_embeddings(
                            st.session_state.original_image1, bboxes1,
                            st.session_state.original_image2, bboxes2,
                            model_name=embedding_model
                        )
                    except Exception as e:
                        st.error(f"Error al usar embeddings: {str(e)}")
                        st.info("Cambiando automáticamente a método HOG...")
                        comparison_results = compare_faces(
                            st.session_state.original_image1, bboxes1,
                            st.session_state.original_image2, bboxes2
                        )
                else:
                    # Usar método HOG tradicional
                    if comparison_method == "Embeddings (Lento, más preciso)":
                        st.warning("Usando método HOG porque DeepFace no está disponible.")
                    comparison_results = compare_faces(
                        st.session_state.original_image1, bboxes1,
                        st.session_state.original_image2, bboxes2
                    )
                
                # Generate comparison report
                report = generate_comparison_report_english(comparison_results, bboxes1, bboxes2)
                
                # Create combined image with match lines
                combined_image = draw_face_matches(
                    st.session_state.original_image1, bboxes1,
                    st.session_state.original_image2, bboxes2,
                    comparison_results,
                    threshold=similarity_threshold
                )
                
                # Show results
                st.subheader("Comparison Results")
                
                # Show combined image
                st.image(combined_image, channels='BGR', use_container_width=True, 
                        caption="Visual Comparison (red lines indicate matches above threshold)")
                
                # Show similarity statistics
                st.subheader("Similarity Statistics")
                
                # Calculate general statistics
                all_similarities = []
                for face_comparisons in comparison_results:
                    for comp in face_comparisons:
                        all_similarities.append(float(comp["similarity"]))
                
                if all_similarities:
                    avg_similarity = sum(all_similarities) / len(all_similarities)
                    max_similarity = max(all_similarities)
                    min_similarity = min(all_similarities)
                    
                    # Determinar el nivel de similitud promedio
                    if avg_similarity >= 70:  # Updated from 80 to 70
                        avg_level = "HIGH"
                        avg_color = "normal"
                    elif avg_similarity >= 50:  # Updated from 65 to 50
                        avg_level = "MEDIUM"
                        avg_color = "normal"
                    elif avg_similarity >= 30:  # Updated from 35 to 30
                        avg_level = "LOW"
                        avg_color = "inverse"
                    else:
                        avg_level = "VERY LOW"
                        avg_color = "inverse"
                    
                    # Determinar el nivel de similitud máxima
                    if max_similarity >= 70:  # Updated from 80 to 70
                        max_level = "HIGH"
                        max_color = "normal"
                    elif max_similarity >= 50:  # Updated from 65 to 50
                        max_level = "MEDIUM"
                        max_color = "normal"
                    elif max_similarity >= 30:  # Updated from 35 to 30
                        max_level = "LOW"
                        max_color = "inverse"
                    else:
                        max_level = "VERY LOW"
                        max_color = "inverse"
                    
                    # Show metrics with color coding
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Average Similarity", f"{avg_similarity:.2f}%", 
                               delta=avg_level, delta_color=avg_color)
                    col2.metric("Maximum Similarity", f"{max_similarity:.2f}%", 
                               delta=max_level, delta_color=max_color)
                    col3.metric("Minimum Similarity", f"{min_similarity:.2f}%")
                    
                    # Count matches above threshold
                    matches_above_threshold = sum(1 for s in all_similarities if s >= similarity_threshold)
                    st.metric(f"Matches above threshold ({similarity_threshold}%)", matches_above_threshold)
                    
                    # Determine if there are significant matches
                    best_matches = [face_comp[0] for face_comp in comparison_results if face_comp]
                    if any(float(match["similarity"]) >= similarity_threshold for match in best_matches):
                        if any(float(match["similarity"]) >= 70 for match in best_matches):  # Updated from 80 to 70
                            st.success("CONCLUSION: HIGH similarity matches found between images.")
                        elif any(float(match["similarity"]) >= 50 for match in best_matches):  # Updated from 65 to 50
                            st.info("CONCLUSION: MEDIUM similarity matches found between images.")
                        else:
                            st.warning("CONCLUSION: LOW similarity matches found between images.")
                    else:
                        st.error("CONCLUSION: No significant matches found between images.")
                    
                    # Añadir gráfico de distribución de similitud
                    st.subheader("Similarity Distribution")
                    
                    # Crear histograma de similitudes
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bins = [0, 30, 50, 70, 100]  # Updated from [0, 35, 65, 80, 100]
                    labels = ['Very Low', 'Low', 'Medium', 'High']
                    colors = ['darkred', 'red', 'orange', 'green']
                    
                    # Contar cuántos valores caen en cada rango
                    hist_data = [sum(1 for s in all_similarities if bins[i] <= s < bins[i+1]) for i in range(len(bins)-1)]
                    
                    # Crear gráfico de barras
                    bars = ax.bar(labels, hist_data, color=colors)
                    
                    # Añadir etiquetas
                    ax.set_xlabel('Similarity Level')
                    ax.set_ylabel('Number of Comparisons')
                    ax.set_title('Similarity Level Distribution')
                    
                    # Añadir valores sobre las barras
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{int(height)}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                
                # Show detailed report in an expandable section
                with st.expander("View Detailed Report"):
                    st.text(report)
                
                # Provide option to download the report
                st.download_button(
                    label="📥 Download Comparison Report",
                    data=report,
                    file_name="face_comparison_report.txt",
                    mime="text/plain"
                )
                
                # Provide option to download the combined image
                pil_combined_img = Image.fromarray(combined_image[:, :, ::-1])
                buf = BytesIO()
                pil_combined_img.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Comparison Image",
                    data=byte_im,
                    file_name="face_comparison.jpg",
                    mime="image/jpeg"
                )

# Add a help text for eye detection sensitivity in the Feature Detection mode
if app_mode == "Feature Detection":
    st.sidebar.markdown("**Ajustes de detección de ojos**")
    st.sidebar.info("Ajusta el control deslizante para cambiar la sensibilidad de la detección de ojos. Un valor más alto detectará más ojos pero puede generar falsos positivos.")

elif app_mode == "Face Recognition":
    st.title("Face Recognition System")
    st.markdown("""
    Este módulo permite registrar rostros y reconocerlos posteriormente en tiempo real o en imágenes.
    Utiliza embeddings faciales para una identificación precisa.
    """)
    
    # Verificar si DeepFace está disponible
    if not DEEPFACE_AVAILABLE:
        st.error("DeepFace no está disponible. Por favor instale la biblioteca con 'pip install deepface'")
        st.stop()
    
    # Cargar el modelo de detección facial
    face_net = load_face_model()
    
    # Inicializar base de datos de rostros si no existe
    if 'face_database' not in st.session_state:
        st.session_state.face_database = {}
    
    # Crear pestañas para las diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["Registrar Rostro", "Reconocimiento en Imagen", "Reconocimiento en Tiempo Real"])
    
    with tab1:
        st.header("Registrar Nuevo Rostro")
        
        # Información sobre mejores prácticas
        with st.expander("📌 Consejos para un mejor reconocimiento", expanded=False):
            st.markdown("""
            ### Mejores prácticas para el registro de rostros:
            
            1. **Calidad de imagen**: Usa imágenes claras, bien iluminadas y de alta resolución.
            2. **Múltiples ángulos**: Registra la misma persona desde diferentes ángulos para mejorar la precisión.
            3. **Expresiones faciales**: Incluye diferentes expresiones (neutral, sonriendo, etc.).
            4. **Iluminación**: Registra imágenes con diferentes condiciones de iluminación.
            5. **Accesorios**: Si la persona usa gafas u otros accesorios habitualmente, incluye imágenes con y sin ellos.
            
            Cuantas más imágenes registres de una persona, mejor será el reconocimiento.
            """)
        
        # Formulario para registrar un nuevo rostro
        with st.form("register_face_form"):
            person_name = st.text_input("Nombre de la persona")
            uploaded_file = st.file_uploader("Subir imagen con rostro", type=['jpg', 'jpeg', 'png'])
            
            # Opciones avanzadas
            with st.expander("Opciones avanzadas", expanded=False):
                model_choice = st.selectbox(
                    "Modelo de embeddings", 
                    ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                    help="Diferentes modelos pueden dar resultados distintos según las características faciales"
                )
                
                confidence_threshold = st.slider(
                    "Umbral de confianza para detección", 
                    min_value=0.3, 
                    max_value=0.9, 
                    value=0.5, 
                    step=0.05,
                    help="Un valor más alto es más restrictivo pero más preciso"
                )
                
                add_to_existing = st.checkbox(
                    "Añadir a persona existente", 
                    help="Marcar si desea añadir otra imagen para una persona ya registrada"
                )
            
            register_button = st.form_submit_button("Registrar Rostro")
        
        if register_button and uploaded_file is not None and person_name:
            # Procesar la imagen subida
            raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            
            # Detectar rostros
            detections = detect_face_dnn(face_net, image, confidence_threshold)
            _, bboxes = process_face_detections(image, detections, confidence_threshold)
            
            if not bboxes:
                st.error("No se detectaron rostros en la imagen. Por favor, sube otra imagen.")
            elif len(bboxes) > 1:
                st.warning("Se detectaron múltiples rostros. Se utilizará el primero detectado.")
                
                # Mostrar imagen con rostros detectados
                processed_image, _ = process_face_detections(image, detections, confidence_threshold)
                st.image(processed_image, channels='BGR', caption="Rostros detectados")
                
                # Extraer embeddings del primer rostro con todos los modelos
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
            else:
                # Extraer embedding del rostro
                embedding = extract_face_embeddings(image, bboxes[0], model_name=model_choice)
                
                if embedding is not None:
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
                        
                        # Añadir nuevo embedding
                        st.session_state.face_database[person_name]['embeddings'].append(embedding["embedding"])
                        st.session_state.face_database[person_name]['models'].append(embedding["model"])
                        st.session_state.face_database[person_name]['timestamps'].append(time.time())
                        st.session_state.face_database[person_name]['count'] += 1
                        
                        st.success(f"Rostro adicional de {person_name} registrado correctamente! (Total: {st.session_state.face_database[person_name]['count']} imágenes)")
                    else:
                        # Crear nueva entrada
                        st.session_state.face_database[person_name] = {
                            'embeddings': [embedding["embedding"]],
                            'models': [embedding["model"]],
                            'timestamps': [time.time()],
                            'count': 1
                        }
                        st.success(f"Rostro de {person_name} registrado correctamente!")
                
                if embedding is not None:
                    # Mostrar imagen con rostro detectado
                    processed_image, _ = process_face_detections(image, detections, confidence_threshold)
                    st.image(processed_image, channels='BGR', caption="Rostro registrado")
                else:
                    st.error("Error al extraer características faciales. Intente con otra imagen.")
        
        # Mostrar base de datos actual
        if st.session_state.face_database:
            st.subheader("Rostros Registrados")
            
            # Crear tabla con nombres y timestamps
            data = []
            for name, info in st.session_state.face_database.items():
                if 'count' in info:
                    # Nuevo formato
                    last_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['timestamps'][-1]))
                    data.append({
                        "Nombre": name, 
                        "Imágenes registradas": info['count'],
                        "Última actualización": last_timestamp
                    })
                else:
                    # Formato antiguo
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['timestamp']))
                    data.append({
                        "Nombre": name, 
                        "Imágenes registradas": 1,
                        "Última actualización": timestamp
                    })
            
            st.table(data)
            
            # Opciones para gestionar la base de datos
            col1, col2 = st.columns(2)
            
            # Eliminar persona específica
            with col1:
                person_to_delete = st.selectbox("Seleccionar persona", list(st.session_state.face_database.keys()))
                if st.button("Eliminar persona seleccionada"):
                    del st.session_state.face_database[person_to_delete]
                    st.success(f"Registros de {person_to_delete} eliminados.")
                    st.rerun()  # Reemplazado st.experimental_rerun() por st.rerun()
            
            # Eliminar todos los rostros
            with col2:
                if st.button("Eliminar todos los rostros"):
                    st.session_state.face_database = {}
                    st.success("Base de datos de rostros eliminada.")
                    st.rerun()  # Reemplazado st.experimental_rerun() por st.rerun()
    
    with tab2:
        st.header("Reconocimiento en Imagen")
        
        # Verificar si hay rostros registrados
        if not st.session_state.face_database:
            st.warning("No hay rostros registrados. Por favor, registre al menos un rostro primero.")
        else:
            # Subir imagen para reconocimiento
            uploaded_file = st.file_uploader("Subir imagen para reconocimiento", type=['jpg', 'jpeg', 'png'], key="recognition_image")
            
            # Configuración avanzada
            with st.expander("Configuración avanzada", expanded=False):
                # Configuración de umbral de similitud
                similarity_threshold = st.slider(
                    "Umbral de similitud (%)", 
                    min_value=35.0, 
                    max_value=95.0, 
                    value=45.0, 
                    step=5.0,
                    help="Porcentaje mínimo de similitud para considerar una coincidencia"
                )
                
                confidence_threshold = st.slider(
                    "Umbral de confianza para detección", 
                    min_value=0.3, 
                    max_value=0.9, 
                    value=0.5, 
                    step=0.05,
                    help="Un valor más alto es más restrictivo pero más preciso"
                )
                
                model_choice = st.selectbox(
                    "Modelo de embeddings", 
                    ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                    help="Diferentes modelos pueden dar resultados distintos según las características faciales"
                )
                
                voting_method = st.radio(
                    "Método de votación para múltiples embeddings",
                    ["Promedio", "Mejor coincidencia", "Votación ponderada"],
                    help="Cómo combinar resultados cuando hay múltiples imágenes de una persona"
                )
                
                show_all_matches = st.checkbox(
                    "Mostrar todas las coincidencias", 
                    value=False,
                    help="Mostrar las 3 mejores coincidencias para cada rostro"
                )
            
            if uploaded_file is not None:
                # Procesar la imagen subida
                raw_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                # Detectar rostros
                detections = detect_face_dnn(face_net, image, confidence_threshold)
                processed_image, bboxes = process_face_detections(image, detections, confidence_threshold)
                
                if not bboxes:
                    st.error("No se detectaron rostros en la imagen.")
                else:
                    # Mostrar imagen con rostros detectados
                    st.image(processed_image, channels='BGR', caption="Rostros detectados")
                    
                    # Reconocer cada rostro
                    result_image = image.copy()
                    
                    # Crear columnas para mostrar estadísticas
                    stats_cols = st.columns(len(bboxes) if len(bboxes) <= 3 else 3)
                    
                    for i, bbox in enumerate(bboxes):
                        # Extraer embedding del rostro
                        embedding = extract_face_embeddings(image, bbox, model_name=model_choice)
                        
                        if embedding is not None:
                            # Comparar con rostros registrados
                            matches = []
                            
                            for name, info in st.session_state.face_database.items():
                                if 'embeddings' in info:
                                    # Nuevo formato con múltiples embeddings
                                    similarities = []
                                    
                                    for idx, registered_embedding in enumerate(info['embeddings']):
                                        # Usar el mismo modelo si es posible
                                        if info['models'][idx] == model_choice:
                                            weight = 1.0  # Dar más peso a embeddings del mismo modelo
                                        else:
                                            weight = 0.8  # Peso menor para embeddings de otros modelos
                                            
                                        # Asegurarse de que los embeddings sean compatibles
                                        try:
                                            similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100 * weight
                                            similarities.append(similarity)
                                        except ValueError as e:
                                            # Si hay error de dimensiones incompatibles, omitir esta comparación
                                            st.warning(f"Omitiendo comparación con modelo incompatible: {info['models'][idx]} vs {embedding['model']}")
                                            continue
                                    
                                    # Aplicar método de votación seleccionado
                                    if voting_method == "Promedio":
                                        if similarities:  # Verificar que la lista no esté vacía
                                            final_similarity = sum(similarities) / len(similarities)
                                        else:
                                            final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                    elif voting_method == "Mejor coincidencia":
                                        if similarities:  # Verificar que la lista no esté vacía
                                            final_similarity = max(similarities)
                                        else:
                                            final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                    else:  # Votación ponderada
                                        if similarities:  # Verificar que la lista no esté vacía
                                            # Dar más peso a similitudes más altas
                                            weighted_sum = sum(s * (i+1) for i, s in enumerate(sorted(similarities)))
                                            weights_sum = sum(i+1 for i in range(len(similarities)))
                                            final_similarity = weighted_sum / weights_sum
                                        else:
                                            final_similarity = 0.0  # Valor predeterminado si no hay similitudes
                                    
                                    matches.append({"name": name, "similarity": final_similarity, "count": info['count']})
                                else:
                                    # Formato antiguo con un solo embedding
                                    registered_embedding = info['embedding']
                                    try:
                                        similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100
                                        matches.append({"name": name, "similarity": similarity, "count": 1})
                                    except ValueError as e:
                                        # Si hay error de dimensiones incompatibles, omitir esta comparación
                                        st.warning(f"Omitiendo comparación con modelo incompatible: {embedding['model']} vs {registered_embedding}")
                                        continue
                            
                            # Ordenar coincidencias por similitud
                            matches.sort(key=lambda x: x["similarity"], reverse=True)
                            
                            # Dibujar resultado en la imagen
                            x1, y1, x2, y2, _ = bbox
                            
                            if matches and matches[0]["similarity"] >= similarity_threshold:
                                # Coincidencia encontrada
                                best_match = matches[0]
                                
                                # Color basado en nivel de similitud
                                if best_match["similarity"] >= 80:
                                    color = (0, 255, 0)  # Verde para alta similitud
                                elif best_match["similarity"] >= 65:
                                    color = (0, 255, 255)  # Amarillo para media similitud
                                else:
                                    color = (0, 165, 255)  # Naranja para baja similitud
                                
                                # Dibujar rectángulo y etiqueta principal
                                label = f"{best_match['name']}: {best_match['similarity']:.1f}%"
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                                # Mostrar coincidencias adicionales si está activado
                                if show_all_matches and len(matches) > 1:
                                    for j, match in enumerate(matches[1:3]):  # Mostrar las siguientes 2 mejores coincidencias
                                        sub_label = f"#{j+2}: {match['name']}: {match['similarity']:.1f}%"
                                        cv2.putText(result_image, sub_label, (x1, y1-(j+2)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                                
                                # Mostrar estadísticas en columnas
                                col_idx = i % 3
                                with stats_cols[col_idx]:
                                    st.metric(
                                        f"Rostro {i+1}", 
                                        f"{best_match['name']}",
                                        f"{best_match['similarity']:.1f}%"
                                    )
                                    if show_all_matches and len(matches) > 1:
                                        st.write("Otras coincidencias:")
                                        for j, match in enumerate(matches[1:3]):
                                            st.write(f"- {match['name']}: {match['similarity']:.1f}%")
                            else:
                                # No hay coincidencia
                                label = "Desconocido"
                                if matches:
                                    label += f": {matches[0]['similarity']:.1f}%"
                                
                                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(result_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
                                # Mostrar estadísticas en columnas
                                col_idx = i % 3
                                with stats_cols[col_idx]:
                                    st.metric(
                                        f"Rostro {i+1}", 
                                        "Desconocido",
                                        f"{matches[0]['similarity']:.1f}%" if matches else "N/A"
                                    )
                    
                    # Mostrar resultado
                    st.subheader("Resultado del reconocimiento")
                    st.image(result_image, channels='BGR', use_container_width=True)
    
    with tab3:
        st.header("Reconocimiento en Tiempo Real")
        
        # Verificar si hay rostros registrados
        if not st.session_state.face_database:
            st.warning("No hay rostros registrados. Por favor, registre al menos un rostro primero.")
        else:
            # Configuración avanzada
            with st.expander("Configuración avanzada", expanded=False):
                # Configuración de umbral de similitud
                similarity_threshold = st.slider(
                    "Umbral de similitud (%)", 
                    min_value=35.0, 
                    max_value=95.0, 
                    value=45.0, 
                    step=5.0,
                    key="realtime_threshold",
                    help="Porcentaje mínimo de similitud para considerar una coincidencia"
                )
                
                confidence_threshold = st.slider(
                    "Umbral de confianza para detección", 
                    min_value=0.3, 
                    max_value=0.9, 
                    value=0.5, 
                    step=0.05,
                    key="realtime_confidence",
                    help="Un valor más alto es más restrictivo pero más preciso"
                )
                
                model_choice = st.selectbox(
                    "Modelo de embeddings", 
                    ["VGG-Face", "Facenet", "OpenFace", "ArcFace"],
                    key="realtime_model",
                    help="Diferentes modelos pueden dar resultados distintos según las características faciales"
                )
                
                voting_method = st.radio(
                    "Método de votación para múltiples embeddings",
                    ["Promedio", "Mejor coincidencia", "Votación ponderada"],
                    key="realtime_voting",
                    help="Cómo combinar resultados cuando hay múltiples imágenes de una persona"
                )
                
                show_confidence = st.checkbox(
                    "Mostrar porcentaje de confianza", 
                    value=True,
                    help="Mostrar el porcentaje de similitud junto al nombre"
                )
                
                stabilize_results = st.checkbox(
                    "Estabilizar resultados", 
                    value=True,
                    help="Reduce fluctuaciones en la identificación usando un promedio temporal"
                )
                
                fps_limit = st.slider(
                    "Límite de FPS", 
                    min_value=5, 
                    max_value=30, 
                    value=15, 
                    step=1,
                    help="Limitar los frames por segundo para reducir uso de CPU"
                )
            
            # Inicializar estado de la cámara
            if 'recognition_camera_running' not in st.session_state:
                st.session_state.recognition_camera_running = False
                
            # Inicializar historial de reconocimiento para estabilización
            if 'recognition_history' not in st.session_state:
                st.session_state.recognition_history = {}
            
            # Botones para controlar la cámara
            col1, col2 = st.columns(2)
            start_button = col1.button("Iniciar Cámara", key="start_recognition_camera", 
                                      on_click=lambda: setattr(st.session_state, 'recognition_camera_running', True))
            stop_button = col2.button("Detener Cámara", key="stop_recognition_camera", 
                                     on_click=lambda: setattr(st.session_state, 'recognition_camera_running', False))
            
            # Placeholder para el video
            video_placeholder = st.empty()
            
            # Placeholder para métricas
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                faces_metric = st.empty()
            with metrics_cols[1]:
                fps_metric = st.empty()
            with metrics_cols[2]:
                time_metric = st.empty()
            
            if st.session_state.recognition_camera_running:
                st.info("Cámara activada. Procesando video en tiempo real...")
                
                # Inicializar webcam
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("No se pudo acceder a la cámara. Asegúrese de que esté conectada y no esté siendo utilizada por otra aplicación.")
                    st.session_state.recognition_camera_running = False
                else:
                    try:
                        # Variables para métricas
                        frame_count = 0
                        start_time = time.time()
                        last_frame_time = start_time
                        fps_history = []
                        
                        while st.session_state.recognition_camera_running:
                            # Control de FPS
                            current_time = time.time()
                            elapsed = current_time - last_frame_time
                            if elapsed < 1.0/fps_limit:
                                time.sleep(0.01)  # Pequeña pausa para no sobrecargar la CPU
                                continue
                                
                            last_frame_time = current_time
                            
                            # Leer frame
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Error al leer frame de la cámara.")
                                break
                            
                            # Actualizar contador de frames
                            frame_count += 1
                            
                            # Calcular FPS
                            if frame_count % 5 == 0:
                                fps = 5 / (current_time - start_time)
                                fps_history.append(fps)
                                if len(fps_history) > 10:
                                    fps_history.pop(0)
                                avg_fps = sum(fps_history) / len(fps_history)
                                start_time = current_time
                                
                                # Actualizar métricas
                                fps_metric.metric("FPS", f"{avg_fps:.1f}")
                                time_metric.metric("Tiempo activo", f"{int(current_time - time.time() + st.session_state.get('camera_start_time', current_time))}s")
                            
                            # Detectar rostros
                            detections = detect_face_dnn(face_net, frame, confidence_threshold)
                            _, bboxes = process_face_detections(frame, detections, confidence_threshold)
                            
                            # Actualizar métrica de rostros
                            if frame_count % 5 == 0:
                                faces_metric.metric("Rostros detectados", len(bboxes))
                            
                            # Reconocer cada rostro
                            result_frame = frame.copy()
                            
                            for i, bbox in enumerate(bboxes):
                                face_id = f"face_{i}"
                                
                                # Extraer embedding del rostro
                                embedding = extract_face_embeddings(frame, bbox, model_name=model_choice)
                                
                                if embedding is not None:
                                    # Comparar con rostros registrados
                                    matches = []
                                    
                                    for name, info in st.session_state.face_database.items():
                                        if 'embeddings' in info:
                                            # Nuevo formato con múltiples embeddings
                                            similarities = []
                                            
                                            for idx, registered_embedding in enumerate(info['embeddings']):
                                                # Usar el mismo modelo si es posible
                                                if info['models'][idx] == model_choice:
                                                    weight = 1.0  # Dar más peso a embeddings del mismo modelo
                                                else:
                                                    weight = 0.8  # Peso menor para embeddings de otros modelos
                                                    
                                                # Asegurarse de que los embeddings sean compatibles
                                                try:
                                                    similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100 * weight
                                                    similarities.append(similarity)
                                                except ValueError as e:
                                                    # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                    continue
                                            
                                            # Aplicar método de votación seleccionado
                                            if voting_method == "Promedio":
                                                final_similarity = sum(similarities) / len(similarities)
                                            elif voting_method == "Mejor coincidencia":
                                                final_similarity = max(similarities)
                                            else:  # Votación ponderada
                                                # Dar más peso a similitudes más altas
                                                weighted_sum = sum(s * (i+1) for i, s in enumerate(sorted(similarities)))
                                                weights_sum = sum(i+1 for i in range(len(similarities)))
                                                final_similarity = weighted_sum / weights_sum
                                            
                                            matches.append({"name": name, "similarity": final_similarity})
                                        else:
                                            # Formato antiguo con un solo embedding
                                            registered_embedding = info['embedding']
                                            try:
                                                similarity = cosine_similarity([embedding["embedding"]], [registered_embedding])[0][0] * 100
                                                matches.append({"name": name, "similarity": similarity})
                                            except ValueError as e:
                                                # Si hay error de dimensiones incompatibles, omitir esta comparación
                                                st.warning(f"Omitiendo comparación con modelo incompatible: {embedding['model']} vs {registered_embedding}")
                                                continue
                                    
                                    # Ordenar coincidencias por similitud
                                    matches.sort(key=lambda x: x["similarity"], reverse=True)
                                    
                                    # Estabilizar resultados si está activado
                                    if stabilize_results and matches:
                                        best_match = matches[0]
                                        
                                        # Inicializar historial para este rostro si no existe
                                        if face_id not in st.session_state.recognition_history:
                                            st.session_state.recognition_history[face_id] = {
                                                "names": [],
                                                "similarities": []
                                            }
                                        
                                        # Añadir al historial
                                        history = st.session_state.recognition_history[face_id]
                                        history["names"].append(best_match["name"])
                                        history["similarities"].append(best_match["similarity"])
                                        
                                        # Limitar historial a los últimos 10 frames
                                        if len(history["names"]) > 10:
                                            history["names"].pop(0)
                                            history["similarities"].pop(0)
                                        
                                        # Determinar el nombre más frecuente en el historial
                                        if len(history["names"]) >= 3:  # Necesitamos al menos 3 frames para estabilizar
                                            name_counts = {}
                                            for name in history["names"]:
                                                if name not in name_counts:
                                                    name_counts[name] = 0
                                                name_counts[name] += 1
                                            
                                            # Encontrar el nombre más frecuente
                                            stable_name = max(name_counts.items(), key=lambda x: x[1])[0]
                                            
                                            # Calcular similitud promedio para ese nombre
                                            stable_similarities = [
                                                history["similarities"][i] 
                                                for i in range(len(history["names"])) 
                                                if history["names"][i] == stable_name
                                            ]
                                            stable_similarity = sum(stable_similarities) / len(stable_similarities)
                                            
                                            # Reemplazar la mejor coincidencia con el resultado estabilizado
                                            best_match = {"name": stable_name, "similarity": stable_similarity}
                                        else:
                                            best_match = matches[0]
                                    else:
                                        best_match = matches[0] if matches else None
                                    
                                    # Dibujar resultado en la imagen
                                    x1, y1, x2, y2, _ = bbox
                                    
                                    if best_match and best_match["similarity"] >= similarity_threshold:
                                        # Coincidencia encontrada
                                        # Color basado en nivel de similitud
                                        if best_match["similarity"] >= 80:
                                            color = (0, 255, 0)  # Verde para alta similitud
                                        elif best_match["similarity"] >= 65:
                                            color = (0, 255, 255)  # Amarillo para media similitud
                                        else:
                                            color = (0, 165, 255)  # Naranja para baja similitud
                                        
                                        # Dibujar rectángulo y etiqueta
                                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                                        
                                        if show_confidence:
                                            label = f"{best_match['name']}: {best_match['similarity']:.1f}%"
                                        else:
                                            label = f"{best_match['name']}"
                                            
                                        cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                    else:
                                        # No hay coincidencia
                                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        
                                        if best_match:
                                            label = f"Desconocido: {best_match['similarity']:.1f}%" if show_confidence else "Desconocido"
                                        else:
                                            label = "Desconocido"
                                            
                                        cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
                            # Mostrar resultado
                            video_placeholder.image(result_frame, channels="BGR", use_container_width=True)
                    finally:
                        # Liberar la cámara cuando se detenga
                        cap.release()
                        # Limpiar historial de reconocimiento
                        st.session_state.recognition_history = {}
            else:
                st.info("Haga clic en 'Iniciar Cámara' para comenzar el reconocimiento en tiempo real.")