# preparacion_yolo.py
# Uso: Preparación de imágenes Landsat para detección de incendios, inundaciones y tormentas con YOLOv4
# Autor: Gabriel Alejandro Orozco con asistencia de ChatGPT
# Fecha: 2025
import os
import glob
import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.plot import reshape_as_image
import cv2

# Configuración
TIFS_PATH = "D:/Landsat_Analisis/"  # Ruta base a carpetas por evento
RGB_OUTPUT_PATH = "./rgb_output/"
YOLO_PATH = "./YOLO_LANDSAT/"

# Crear carpetas necesarias
for path in [RGB_OUTPUT_PATH, YOLO_PATH]:
    os.makedirs(path, exist_ok=True)

def obtener_grupos_bandas(carpeta_evento):
    tif_paths = glob.glob(os.path.join(carpeta_evento, "*_B[2-4].TIF"))
    imagenes_por_fecha = {}

    for path in tif_paths:
        base = os.path.basename(path)
        key = "_".join(base.split("_")[:4])  # LC08_L1TP_001002_20250311
        if key not in imagenes_por_fecha:
            imagenes_por_fecha[key] = {}
        banda = base.split("_")[-1].split(".")[0]  # B2, B3, B4
        imagenes_por_fecha[key][banda] = path

    return imagenes_por_fecha

def generar_rgb(bandas, nombre_salida):
    try:
        with rasterio.open(bandas['B4']) as red, \
             rasterio.open(bandas['B3']) as green, \
             rasterio.open(bandas['B2']) as blue:

            arr_r = red.read(1)
            arr_g = green.read(1)
            arr_b = blue.read(1)

            rgb = np.stack([arr_r, arr_g, arr_b], axis=-1)
            rgb_norm = ((rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255).astype(np.uint8)

            out_path = os.path.join(RGB_OUTPUT_PATH, nombre_salida + ".jpg")
            cv2.imwrite(out_path, cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2BGR))
            print(f"[✓] Guardado RGB: {out_path}")

    except KeyError:
        print(f"[!] Faltan bandas para: {nombre_salida}")
        return

def procesar_eventos():
    eventos = [f.path for f in os.scandir(TIFS_PATH) if f.is_dir()]
    for evento in eventos:
        grupos = obtener_grupos_bandas(evento)
        for nombre, bandas in grupos.items():
            generar_rgb(bandas, nombre)

def crear_estructura_yolo():
    folders = [
        "data/images/train", "data/images/val",
        "data/labels/train", "data/labels/val",
        "cfg", "backup"
    ]
    for f in folders:
        os.makedirs(os.path.join(YOLO_PATH, f), exist_ok=True)
    open(os.path.join(YOLO_PATH, "obj.names"), "w").write("incendio\ninundacion\ntormenta\n")
    open(os.path.join(YOLO_PATH, "obj.data"), "w").write(
        "classes = 3\n"
        "train = train.txt\n"
        "valid = val.txt\n"
        "names = obj.names\n"
        "backup = backup/\n"
    )
    print("[✓] Estructura YOLO creada.")

def generar_listas_entrenamiento():
    images = glob.glob(os.path.join(RGB_OUTPUT_PATH, "*.jpg"))
    split = int(len(images) * 0.8)
    train = images[:split]
    val = images[split:]

    with open(os.path.join(YOLO_PATH, "train.txt"), "w") as f:
        for path in train:
            f.write(os.path.abspath(path) + "\n")
    with open(os.path.join(YOLO_PATH, "val.txt"), "w") as f:
        for path in val:
            f.write(os.path.abspath(path) + "\n")
    print(f"[✓] train.txt y val.txt generados ({len(train)} train / {len(val)} val)")

if __name__ == "__main__":
    print("== Generando composiciones RGB ==")
    procesar_eventos()
    print("== Creando estructura YOLO ==")
    crear_estructura_yolo()
    generar_listas_entrenamiento()
