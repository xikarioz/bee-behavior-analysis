import cv2
import numpy as np
import time
import os
import h5py
import serial
from dlclive import DLCLive, Processor

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ═══════════════════════════════════════════════════════════════════════
MODEL_PATH  = '/home/francofitte/DLC_8Abejas_resnet_50_iteration-1_shuffle-10000_snapshot-160.pt'
VIDEO_PATH  = '/home/francofitte/Downloads/parte1.mkv'
OUTPUT_DIR  = '/home/francofitte/Downloads/exportado_pi'

ABEJAS_MONITOREAR  = [0, 1, 2]
N_ABEJAS           = 7
TAMANO_RECORTE     = 220
CONF_MINIMA        = 0.3
BUFFER_SIZE        = 100

DURACION_TURNO_S   = 60
UMBRAL_SLEEP_PXS   = 70.0         # px/s con dt=1/fps_video

DEBUG_HASTA_SEG    = 60
DEBUG_CADA_SEG     = 10

ARDUINO_PORT       = '/dev/ttyACM0'
ARDUINO_BAUD       = 9600
ARDUINO_SIMULADO   = True

UMBRAL_FIJO        = 70
RADIO_PLATO_PCT    = 0.45
CENTRO_OFFSET_X    = -250
CENTRO_OFFSET_Y    = 60
MAX_ASPECT_RATIO   = 2.0

BODYPARTS = ['Antena_1_A', 'Antena_1_B', 'Antena_2_A', 'Antena_2_B', 'Posicion']
COLORES   = [(0,255,0), (0,200,255), (255,100,0), (128,128,128), (128,128,128), (128,128,128), (128,128,128)]
# ═══════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ARDUINO ──────────────────────────────────────────────────────────
arduino = None
if not ARDUINO_SIMULADO:
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)
        print(f"[OK] Arduino conectado en {ARDUINO_PORT}")
    except Exception as e:
        print(f"[WARN] Arduino no disponible: {e} — modo simulado")
        ARDUINO_SIMULADO = True

def despertar_abeja(bee_idx, vel_promedio):
    print(f"\n{'='*55}")
    print(f"  [DESPERTAR] Bee{bee_idx+1} | vel_promedio={vel_promedio:.1f} px/s")
    if ARDUINO_SIMULADO:
        print(f"  → [SIMULADO] Señal Arduino: ROTAR SG90 abeja {bee_idx+1}")
    else:
        try:
            arduino.write(f"WAKE{bee_idx+1}\n".encode())
        except Exception as e:
            print(f"  → Error Arduino: {e}")
    print(f"{'='*55}\n")


# ── H5 WRITER ────────────────────────────────────────────────────────
class H5Writer:
    def __init__(self, path, bodyparts, n_abejas):
        self.f = h5py.File(path, 'w')
        self.buf = []
        self.written = 0
        self.n_bp = len(bodyparts)
        chunk = 1000
        kw = dict(compression='gzip', compression_opts=4)
        self.f.attrs['bodyparts'] = bodyparts
        self.f.attrs['n_abejas']  = n_abejas
        self.ds_frame = self.f.create_dataset('frame',      shape=(0,),               maxshape=(None,),              dtype='i4', chunks=(chunk,),             **kw)
        self.ds_time  = self.f.create_dataset('tiempo_seg', shape=(0,),               maxshape=(None,),              dtype='f4', chunks=(chunk,),             **kw)
        self.ds_bee   = self.f.create_dataset('bee',        shape=(0,),               maxshape=(None,),              dtype='i1', chunks=(chunk,),             **kw)
        self.ds_kp    = self.f.create_dataset('keypoints',  shape=(0, self.n_bp, 3), maxshape=(None, self.n_bp, 3), dtype='f4', chunks=(chunk, self.n_bp, 3), **kw)

    def write(self, frame_idx, tiempo, bee_idx, kp):
        self.buf.append((frame_idx, tiempo, bee_idx, kp))
        if len(self.buf) >= BUFFER_SIZE:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        n, nw = len(self.buf), self.written
        frames = np.array([r[0] for r in self.buf], dtype='i4')
        times  = np.array([r[1] for r in self.buf], dtype='f4')
        bees   = np.array([r[2] for r in self.buf], dtype='i1')
        kps    = np.stack([r[3] for r in self.buf]).astype('f4')
        for ds in (self.ds_frame, self.ds_time, self.ds_bee):
            ds.resize((nw + n,))
        self.ds_kp.resize((nw + n, self.n_bp, 3))
        self.ds_frame[nw:] = frames
        self.ds_time[nw:]  = times
        self.ds_bee[nw:]   = bees
        self.ds_kp[nw:]    = kps
        self.written += n
        self.buf = []
        self.f.flush()

    def close(self):
        self._flush()
        self.f.close()
        print(f"[H5] {self.written:,} filas guardadas")


# ── DETECCIÓN ────────────────────────────────────────────────────────
def detectar_centros_fijo(frame, n_abejas):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    radio_plato = int(h * RADIO_PLATO_PCT)
    centro_x = w // 2 + CENTRO_OFFSET_X
    centro_y = h // 2 + CENTRO_OFFSET_Y
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mascara, (centro_x, centro_y), radio_plato, 255, -1)
    blur_aislado = cv2.bitwise_or(blur, cv2.bitwise_not(mascara))
    _, thresh = cv2.threshold(blur_aislado, UMBRAL_FIJO, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros = []
    for c in contornos:
        area = cv2.contourArea(c)
        if 3000 < area < 5000:
            x, y, aw, ah = cv2.boundingRect(c)
            if max(aw, ah) / min(aw, ah) <= MAX_ASPECT_RATIO:
                centros.append((x + aw // 2, y + ah // 2))
    if len(centros) != n_abejas:
        print(f"[ERROR] Detectadas {len(centros)} abejas")
        return []
    pts = np.array(centros, dtype=float)
    sumas = [sum(np.linalg.norm(pts[i] - pts[j]) for j in range(len(pts)) if j != i)
             for i in range(len(pts))]
    idx_centro = int(np.argmin(sumas))
    bee7 = centros[idx_centro]
    anillo = [c for i, c in enumerate(centros) if i != idx_centro]
    cx_a = sum(c[0] for c in anillo) / len(anillo)
    cy_a = sum(c[1] for c in anillo) / len(anillo)
    def angulo_desde_12_antihorario(c):
        ang = np.arctan2(c[1] - cy_a, c[0] - cx_a)
        return (-(ang + np.pi / 2)) % (2 * np.pi)
    anillo_ordenado = sorted(anillo, key=angulo_desde_12_antihorario)
    resultado = anillo_ordenado + [bee7]
    print(f"[OK] {n_abejas} abejas | Bee7=centro en {bee7}")
    return resultado


def obtener_recorte(frame, centro, size):
    cx, cy = centro
    r = size // 2
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(frame.shape[1], cx + r), min(frame.shape[0], cy + r)
    crop = frame[y1:y2, x1:x2]
    ph = size - crop.shape[0]
    pw = size - crop.shape[1]
    if ph > 0 or pw > 0:
        crop = cv2.copyMakeBorder(crop, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    return crop, (x1, y1)


def dibujar_debug(frame, centros, pose_abs, bee_activa, vel_turno):
    img = frame.copy()
    r = TAMANO_RECORTE // 2
    for i, (cx, cy) in enumerate(centros):
        monitoreada = i in ABEJAS_MONITOREAR
        color  = COLORES[i] if i < len(COLORES) else (128, 128, 128)
        grosor = 2 if monitoreada else 1
        cv2.circle(img, (cx, cy), r, color, grosor)
        if i == bee_activa:
            label = f"Bee{i+1} [TURNO] {vel_turno:.0f}px/s"
            cv2.circle(img, (cx, cy), r + 5, color, 1)
        else:
            label = f"Bee{i+1}"
        cv2.putText(img, label, (cx - r + 5, cy - r + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, grosor)
        if i == bee_activa and pose_abs is not None:
            for pt in pose_abs:
                x_abs, y_abs, conf = float(pt[0]), float(pt[1]), float(pt[2])
                if conf > CONF_MINIMA and not (np.isnan(x_abs) or np.isnan(y_abs)):
                    cv2.circle(img, (int(x_abs), int(y_abs)), 6, color, -1)
                    cv2.circle(img, (int(x_abs), int(y_abs)), 6, (0, 0, 0), 1)
    return img


def formatear_tiempo(s):
    h, m, s = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h}h {m:02d}m {s:02d}s" if h > 0 else (f"{m}m {s:02d}s" if m > 0 else f"{s}s")


# ── INICIALIZACIÓN ───────────────────────────────────────────────────
print("Abriendo video...")
cap          = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video    = cap.get(cv2.CAP_PROP_FPS)
w_vid        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if total_frames == 0:
    print("ERROR: no se pudo abrir el video.")
    exit()

print(f"Video: {total_frames:,} frames | {fps_video:.1f} fps | {w_vid}x{h_vid}")
print(f"Duración: {formatear_tiempo(total_frames / fps_video)}")

ret, primer_frame = cap.read()
centros = detectar_centros_fijo(primer_frame, N_ABEJAS)
if not centros:
    exit()

# Imagen de centros
img_centros = primer_frame.copy()
for i, (cx, cy) in enumerate(centros):
    color  = COLORES[i] if i < len(COLORES) else (128, 128, 128)
    grosor = 2 if i in ABEJAS_MONITOREAR else 1
    cv2.circle(img_centros, (cx, cy), TAMANO_RECORTE // 2, color, grosor)
    cv2.putText(img_centros, f"Bee{i+1}", (cx - TAMANO_RECORTE//2 + 5, cy - TAMANO_RECORTE//2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, grosor)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'centros.jpg'), img_centros)
print("[OK] centros.jpg guardado")

# ── GRABACIÓN SECUENCIAL ─────────────────────────────────────────────
writer_vid = cv2.VideoWriter(
    os.path.join(OUTPUT_DIR, 'grabacion.mp4'),
    cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (w_vid, h_vid)
)

# ── H5 y DLC ─────────────────────────────────────────────────────────
nan_kp = np.full((len(BODYPARTS), 3), np.nan, dtype='f4')
nan_kp[:, 2] = 0.0
writer_h5 = H5Writer(os.path.join(OUTPUT_DIR, 'poses_pi.h5'), BODYPARTS, N_ABEJAS)

print("\nCargando modelo DLC...")
dlc_live = DLCLive(MODEL_PATH, processor=Processor(), model_type="pytorch")
crop0, _ = obtener_recorte(primer_frame, centros[0], TAMANO_RECORTE)
dlc_live.init_inference(crop0)
print("[OK] Modelo cargado.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ── ESTADO DE TURNO ──────────────────────────────────────────────────
turno_idx         = 0
bee_activa        = ABEJAS_MONITOREAR[0]
t_inicio_turno    = time.time()
velocidades_turno = []
pose_prev         = None
prev_frame_idx    = -1

# ── BUCLE DE ANÁLISIS (corre a ~3fps) ────────────────────────────────
count_analisis   = 0
t_start          = time.time()
ultimo_debug_seg = -DEBUG_CADA_SEG

print(f"\nTurnos: {DURACION_TURNO_S}s por abeja | Umbral: {UMBRAL_SLEEP_PXS} px/s")
print(f"Orden: Bee1 → Bee2 → Bee3 → Bee1 ...")
print("─" * 60)

try:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        writer_vid.write(frame)
        tiempo_seg = frame_idx / fps_video

        # ── Control de turno ────────────────────────────────────────
        tiempo_real_turno = time.time() - t_inicio_turno
        if tiempo_real_turno >= DURACION_TURNO_S:
            vels_validas = [v for v in velocidades_turno if not np.isnan(v)]
            if vels_validas:
                vel_prom = float(np.mean(vels_validas))
                print(f"\n[TURNO] Bee{bee_activa+1} terminó | vel_promedio={vel_prom:.1f} px/s", flush=True)
                if vel_prom < UMBRAL_SLEEP_PXS:
                    despertar_abeja(bee_activa, vel_prom)
                else:
                    print(f"  → Bee{bee_activa+1} activa, no se despierta\n")
            turno_idx         = (turno_idx + 1) % len(ABEJAS_MONITOREAR)
            bee_activa        = ABEJAS_MONITOREAR[turno_idx]
            t_inicio_turno    = time.time()
            velocidades_turno = []
            pose_prev         = None
            prev_frame_idx    = -1
            print(f"[TURNO] Iniciando Bee{bee_activa+1} ({DURACION_TURNO_S}s)", flush=True)

        # ── Inferencia ───────────────────────────────────────────────
        crop, (off_x, off_y) = obtener_recorte(frame, centros[bee_activa], TAMANO_RECORTE)
        pose = dlc_live.get_pose(crop)
        pose_abs = None

        if pose is not None and len(pose) > 0:
            kp = np.array(pose)[:, :3].astype('f4')
            kp_abs = kp.copy()
            kp_abs[:, 0] += off_x
            kp_abs[:, 1] += off_y
            pose_abs = kp_abs
            writer_h5.write(frame_idx, tiempo_seg, bee_activa + 1, kp_abs)

            # Velocidad con dt = 1/fps_video (escala consistente)
            if pose_prev is not None and prev_frame_idx >= 0:
                dt = (frame_idx - prev_frame_idx) / fps_video
                if dt > 0:
                    prev_kp = pose_prev[:4, :2]
                    curr_kp = kp_abs[:4, :2]
                    dists = np.sqrt(((curr_kp - prev_kp) ** 2).sum(axis=1))
                    dists_validas = dists[~np.isnan(dists)]
                    if len(dists_validas) > 0:
                        vel = float(dists_validas.mean() / dt)
                        velocidades_turno.append(vel)
            pose_prev      = kp_abs
            prev_frame_idx = frame_idx
        else:
            writer_h5.write(frame_idx, tiempo_seg, bee_activa + 1, nan_kp.copy())

        # ── Debug images ─────────────────────────────────────────────
        if tiempo_seg <= DEBUG_HASTA_SEG:
            if tiempo_seg - ultimo_debug_seg >= DEBUG_CADA_SEG:
                vel_actual = float(np.mean([v for v in velocidades_turno if not np.isnan(v)])) \
                             if velocidades_turno else 0.0
                img_debug = dibujar_debug(frame, centros, pose_abs, bee_activa, vel_actual)
                seg_str = f"{int(tiempo_seg):03d}s"
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{seg_str}.jpg"), img_debug)
                print(f"[DEBUG] debug_{seg_str}.jpg | Bee{bee_activa+1} | vel={vel_actual:.0f}px/s")
                ultimo_debug_seg = tiempo_seg

        # ── Log ──────────────────────────────────────────────────────
        if count_analisis % 10 == 0 and count_analisis > 0:
            elapsed  = time.time() - t_start
            fps_real = count_analisis / elapsed
            pct      = frame_idx / total_frames * 100
            vel_log  = velocidades_turno[-1] if velocidades_turno else 0
            print(
                f"F:{frame_idx:,} ({pct:.1f}%) {fps_real:.2f}fps | "
                f"Bee{bee_activa+1}[turno {tiempo_real_turno:.0f}/{DURACION_TURNO_S}s] "
                f"vel={vel_log:.0f}px/s",
                flush=True
            )

        count_analisis += 1
        frame_idx += 1

except KeyboardInterrupt:
    print("\n[!] Detenido.")

writer_vid.release()
writer_h5.close()
if arduino:
    arduino.close()
print("─" * 60)
print(f"Listo en {formatear_tiempo(time.time() - t_start)}")
print(f"Archivos en: {OUTPUT_DIR}")
