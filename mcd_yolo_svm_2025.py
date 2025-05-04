# yolo_svm_2025.py
# -*- coding: utf-8 -*-
"""
Script de análisis satelital post-desastre: detección de anomalías con YOLOv4 y clasificación con SVM.
Autor: Alejandro Orozco con asistencia de ChatGPT
Fecha: 2025
Requiere: OpenCV, NumPy, Scikit-learn, scikit-image, matplotlib, glob, os, re, etc.
Este script analiza imágenes LANDSAT 8/9 (Colección 2, Niveles 1 y 2) para detectar cambios post-desastre.
"""

########################
# CONFIGURACIÓN INICIAL
########################

IMAGENES_PATH = "D:/Landsat_Analisis/"  # Ruta al disco externo con imágenes
YOLO_CONFIG = "./yolo/yolov4.cfg"        # Ruta a archivo de configuración YOLO
YOLO_WEIGHTS = "./yolo/yolov4.weights"  # Ruta a pesos pre-entrenados YOLO entrenados previamente
YOLO_CLASSES = "./yolo/obj.names"        # Nombres de clases para el modelo YOLO

# Crear carpetas necesarias si no existen
import os
for folder in ["./resultados", "./yolo"]:
    os.makedirs(folder, exist_ok=True)

#############################
# DEPENDENCIAS Y LIBRERÍAS
#############################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import greycomatrix, greycoprops
import glob
import re
from datetime import datetime

#########################################
# FUNCIONES PARA CARGA Y PROCESAMIENTO
#########################################

def cargar_imagenes_landsat(ruta_base):
    """
    Carga todas las imágenes .TIF y las agrupa por path/row identificando pre/post evento.
    """
    imagenes = glob.glob(os.path.join(ruta_base, "**/*.TIF"), recursive=True)
    imagen_dict = {}
    patron = re.compile(r'_(\d{6})_(\d{8})_')  # Extrae path/row y fecha

    for img_path in imagenes:
        nombre = os.path.basename(img_path)
        match = patron.search(nombre)
        if not match:
            continue
        path_row, fecha = match.groups()
        fecha_dt = datetime.strptime(fecha, "%Y%m%d")

        if path_row not in imagen_dict:
            imagen_dict[path_row] = []
        imagen_dict[path_row].append((fecha_dt, img_path))

    # Ordenar fechas y devolver pares (pre, post)
    pares = {}
    for key, lista in imagen_dict.items():
        if len(lista) < 2:
            continue
        lista.sort()  # Ordena por fecha
        pares[key] = (lista[0][1], lista[-1][1])
    return pares


def preprocesar_imagen(img_path):
    """
    Carga una imagen .TIF y la convierte a imagen RGB compuesta (usamos bandas 4, 3, 2 típicas).
    """
    bandas = {}
    for b in ["B4", "B3", "B2"]:
        banda_path = img_path.replace("B1.TIF", f"{b}.TIF")
        bandas[b] = cv2.imread(banda_path, cv2.IMREAD_UNCHANGED)
    rgb = cv2.merge([bandas["B2"], bandas["B3"], bandas["B4"]])
    rgb = cv2.convertScaleAbs(rgb, alpha=(255.0/np.max(rgb)))
    return rgb

#############################
# MODELO YOLOv4 - DETECCIÓN
#############################

def cargar_yolo():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    with open(YOLO_CLASSES, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def deteccion_yolo(imagen, net, output_layers):
    height, width = imagen.shape[:2]
    blob = cv2.dnn.blobFromImage(imagen, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    boxes = []
    confidences = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detecciones = [boxes[i[0]] for i in indices]
    return detecciones


def visualizar_detecciones(imagen, detecciones, nombre_out):
    img_out = imagen.copy()
    for (x, y, w, h) in detecciones:
        cv2.rectangle(img_out, (x, y), (x+w, y+h), (0,255,0), 2)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title("Detecciones YOLO")
    plt.axis('off')
    plt.savefig(f"./resultados/{nombre_out}_yolo.png")
    plt.close()

#############################################
# CLASIFICACIÓN CON SVM USANDO TEXTURAS GLCM
#############################################

def extraer_caracteristicas_glcm(region):
    """Calcula características de textura GLCM para una región de interés"""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    return np.array([greycoprops(glcm, p)[0, 0] for p in props])


def clasificar_svm(detecciones, imagen):
    X = []
    for (x, y, w, h) in detecciones:
        roi = imagen[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        feat = extraer_caracteristicas_glcm(roi)
        X.append(feat)
    X = np.array(X)
    y = [1]*len(X)  # Por ahora asumimos todas positivas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_scaled, y)
    y_pred = svm.predict(X_scaled)
    return y_pred

######################################
# VISUALIZACIÓN FINAL DE RESULTADOS
######################################

def visualizar_resultados_finales(imagen, detecciones, clasificaciones, nombre_out):
    img_out = imagen.copy()
    for idx, (x, y, w, h) in enumerate(detecciones):
        color = (0,255,0) if clasificaciones[idx] == 1 else (0,0,255)
        cv2.rectangle(img_out, (x, y), (x+w, y+h), color, 2)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title("Clasificación SVM sobre Detecciones")
    plt.axis('off')
    plt.savefig(f"./resultados/{nombre_out}_final.png")
    plt.close()

########################
# PROCESAMIENTO GLOBAL
########################

def procesar():
    pares = cargar_imagenes_landsat(IMAGENES_PATH)
    net, clases, salida_yolo = cargar_yolo()

    for key, (pre, post) in pares.items():
        print(f"Procesando zona: {key}")
        imagen_post = preprocesar_imagen(post)

        detecciones = deteccion_yolo(imagen_post, net, salida_yolo)
        visualizar_detecciones(imagen_post, detecciones, key)

        clasificaciones = clasificar_svm(detecciones, imagen_post)
        visualizar_resultados_finales(imagen_post, detecciones, clasificaciones, key)


if __name__ == "__main__":
    procesar()
