import cv2
import numpy as np
import time
import os
import csv
from dlclive import DLCLive, Processor

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACION DEL EXPERIMENTO
# ═══════════════════════════════════════════════════════════════════════

MODEL_PATH = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\modelo_exportado\7Abejas\DLC_8Abejas_resnet_50_iteration-1_shuffle-10000_snapshot-160.pt'

BASE_DIR   = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model'
VIDEO_PATH = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\2026-04-20 16-52-53.mkv'
CSV_PATH   = os.path.join(BASE_DIR, 'poses_completo.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_completo')

N_ABEJAS       = 7
TAMANO_RECORTE = 220
GUARDAR_CADA   = 648000  # cada 3 horas de video (3 * 3600 * 60fps)
CONF_MINIMA    = 0.3
MINUTOS        = None    # None = video completo

# PARÁMETROS CALIBRADOS
UMBRAL_FIJO      = 70
RADIO_PLATO_PCT  = 0.45
CENTRO_OFFSET_X  = -250
CENTRO_OFFSET_Y  = 60
MAX_ASPECT_RATIO = 2.0

BODYPARTS = ['Antena_1_A', 'Antena_1_B', 'Antena_2_A', 'Antena_2_B', 'Posicion']
# ═══════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detectar_centros_fijo(frame, n_abejas):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    radio_plato = int(h * RADIO_PLATO_PCT)
    centro_x = w // 2 + CENTRO_OFFSET_X
    centro_y = h // 2 + CENTRO_OFFSET_Y
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mascara, (centro_x, centro_y), radio_plato, 255, -1)
    fondo_blanco = cv2.bitwise_not(mascara)
    blur_aislado = cv2.bitwise_or(blur, fondo_blanco)

    _, thresh = cv2.threshold(blur_aislado, UMBRAL_FIJO, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh_limpio = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contornos, _ = cv2.findContours(thresh_limpio, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 800 < area < 20000:
            x, y, ancho, alto = cv2.boundingRect(c)
            aspect = max(ancho, alto) / min(ancho, alto)
            if aspect > MAX_ASPECT_RATIO:
                continue
            centros.append((x + ancho // 2, y + alto // 2))

    if len(centros) != n_abejas:
        print(f"\n[ERROR] Se detectaron {len(centros)} abejas.")
        debug_img = frame.copy()
        cv2.circle(debug_img, (centro_x, centro_y), radio_plato, (200, 200, 0), 2)
        for c in contornos:
            area = cv2.contourArea(c)
            if 800 < area < 20000:
                x, y, ancho, alto = cv2.boundingRect(c)
                aspect = max(ancho, alto) / min(ancho, alto)
                cx2 = x + ancho // 2
                cy2 = y + alto // 2
                if aspect > MAX_ASPECT_RATIO:
                    cv2.rectangle(debug_img, (x, y), (x+ancho, y+alto), (0, 165, 255), 2)
                else:
                    cv2.rectangle(debug_img, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
                    cv2.circle(debug_img, (cx2, cy2), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'debug_deteccion_error.jpg'), debug_img)
        return []

    print(f"\n[OK] Detectadas {n_abejas} abejas con éxito.")
    promedio_x = sum(c[0] for c in centros) / len(centros)
    promedio_y = sum(c[1] for c in centros) / len(centros)
    return sorted(centros, key=lambda c: np.arctan2(c[1] - promedio_y, c[0] - promedio_x))


def obtener_recorte(frame, centro, size):
    cx, cy = centro
    r = size // 2
    x1, y1 = max(0, cx-r), max(0, cy-r)
    x2, y2 = min(frame.shape[1], cx+r), min(frame.shape[0], cy+r)
    crop = frame[y1:y2, x1:x2]
    pad_b = size - crop.shape[0]
    pad_r = size - crop.shape[1]
    if pad_b > 0 or pad_r > 0:
        crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
    return crop, (x1, y1)


def dibujar_debug(frame, centros, resultados):
    img = frame.copy()
    r = TAMANO_RECORTE // 2
    h, w = frame.shape[:2]
    centro_x = w // 2 + CENTRO_OFFSET_X
    centro_y = h // 2 + CENTRO_OFFSET_Y
    cv2.circle(img, (centro_x, centro_y), int(h * RADIO_PLATO_PCT), (200, 200, 0), 2)
    for i, (cx, cy) in enumerate(centros):
        cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(img, f"Bee{i+1}", (cx - r + 5, cy - r + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if i in resultados:
            pose, (off_x, off_y) = resultados[i]
            for pt in pose:
                x_abs = float(pt[0]) + off_x
                y_abs = float(pt[1]) + off_y
                conf  = float(pt[2])
                if conf > CONF_MINIMA and not (np.isnan(x_abs) or np.isnan(y_abs)):
                    cv2.circle(img, (int(x_abs), int(y_abs)), 4, (0, 0, 255), -1)
    return img


def formatear_tiempo(segundos):
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── INICIALIZACIÓN ───────────────────────────────────────────────────
print("Abriendo video...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video        = cap.get(cv2.CAP_PROP_FPS)
duracion_video_s = total_frames / fps_video

if total_frames == 0:
    print("ERROR: no se pudo abrir el video.")
    exit()

frames_a_procesar = int(MINUTOS * 60 * fps_video) if MINUTOS else total_frames
frames_a_procesar = min(frames_a_procesar, total_frames)

print(f"Video: {total_frames} frames a {fps_video:.1f} fps")
print(f"Duración total del video: {formatear_tiempo(duracion_video_s)}")
print(f"Frames a procesar: {frames_a_procesar} ({formatear_tiempo(frames_a_procesar / fps_video)})")

ret, primer_frame = cap.read()
centros = detectar_centros_fijo(primer_frame, N_ABEJAS)
if not centros:
    exit()

img_v = dibujar_debug(primer_frame, centros, {})
cv2.imwrite(os.path.join(OUTPUT_DIR, 'centros_calibrados.jpg'), img_v)

print("\nCargando modelo DLC en RTX 3090...")
dlc_live = DLCLive(MODEL_PATH, processor=Processor(), model_type="pytorch")
crop_w, _ = obtener_recorte(primer_frame, centros[0], TAMANO_RECORTE)
dlc_live.init_inference(crop_w)

csvfile = open(CSV_PATH, 'w', newline='')
writer  = csv.writer(csvfile)
headers = ['frame', 'tiempo_seg', 'animal']
for bp in BODYPARTS:
    headers += [f'{bp}_x', f'{bp}_y', f'{bp}_conf']
writer.writerow(headers)

# ── BUCLE PRINCIPAL ──────────────────────────────────────────────────
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
count   = 0
t_start = time.time()
print(f"\nProcesando {frames_a_procesar} frames...")
print("─" * 70)

try:
    while count < frames_a_procesar:
        ret, frame = cap.read()
        if not ret:
            break

        tiempo_seg = count / fps_video
        resultados = {}

        for i, centro in enumerate(centros):
            crop, (off_x, off_y) = obtener_recorte(frame, centro, TAMANO_RECORTE)
            pose = dlc_live.get_pose(crop)

            if pose is not None and len(pose) > 0:
                pose = np.array(pose)
                if pose.ndim == 1:
                    pose = pose.reshape(-1, 3)
                resultados[i] = (pose, (off_x, off_y))
                row = [count, f"{tiempo_seg:.3f}", f"Bee{i+1}"]
                for pt in pose:
                    x_abs = float(pt[0]) + off_x
                    y_abs = float(pt[1]) + off_y
                    conf  = float(pt[2])
                    row  += [f"{x_abs:.1f}", f"{y_abs:.1f}", f"{conf:.4f}"]
                writer.writerow(row)
            else:
                row = [count, f"{tiempo_seg:.3f}", f"Bee{i+1}"]
                for _ in BODYPARTS:
                    row += ['nan', 'nan', '0.0']
                writer.writerow(row)

        csvfile.flush()

        if count % 100 == 0 and count > 0:
            elapsed  = time.time() - t_start
            fps_real = count / elapsed
            eta_s    = (frames_a_procesar - count) / fps_real
            vmph     = (fps_real / fps_video) * 60
            t_total  = total_frames / fps_real
            pct      = count / frames_a_procesar * 100
            print(
                f"F: {count}/{frames_a_procesar} ({pct:.1f}%) | "
                f"{fps_real:.1f} fps | "
                f"ETA: {formatear_tiempo(eta_s)} | "
                f"{vmph:.1f} min video/hora | "
                f"Video completo en: {formatear_tiempo(t_total)}",
                flush=True
            )

        if count % GUARDAR_CADA == 0:
            img_out = dibujar_debug(frame, centros, resultados)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{count:06d}.jpg"), img_out)

        count += 1

except KeyboardInterrupt:
    print("\n[!] Proceso detenido por el usuario.")

cap.release()
csvfile.close()
print("─" * 70)
print(f"\nListo en {formatear_tiempo(time.time() - t_start)}. Archivos en: {OUTPUT_DIR}")
