"""
═══════════════════════════════════════════════════════════════
  SCRIPT MAESTRO — Análisis Comportamental 7 Abejas
  Carga datos UNA sola vez. Genera 15 figuras en paper_definitivo/
═══════════════════════════════════════════════════════════════
Figuras generadas:
  01_rose_actividad_individual      Rose plot por abeja (bins 30min)
  02_polar_comparacion_6abejas      Todas superpuestas en un polar
  03_polar_4escalas                 4 paneles: rápida/media/lenta/muy lenta
  04_polar_suavizado_individual     Señal suavizada por abeja + dispersión
  05_vmd_lento_raw                  VMD M0 (>2h) por abeja, normalizado
  06_vmd_medio_raw                  VMD M1 (30min-2h) por abeja, normalizado
  07_vmd_lento_denoised             VMD M0 con sustracción Bee7
  08_vmd_medio_denoised             VMD M1 con sustracción Bee7
  09_cwt_polar                      Espectrograma polar CWT Morlet
  10_cwt_polar_denoised             CWT Morlet con sustracción Bee7
  11_hht_m0_lento                   HHT modo 0 (>120min) por abeja
  12_hht_m1_medio                   HHT modo 1 (30-120min) por abeja
  13_hht_m2_rapido                  HHT modo 2 (10-30min) por abeja
  14_hht_m3_muyrapido               HHT modo 3 (<10min) por abeja
  15_cwt_vs_hht_comparacion         CWT sin/con filtro vs HHT — 4 paneles
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import hilbert, butter, filtfilt, fftconvolve

# morlet2 fue removida en scipy >= 1.12 — reimplementación equivalente
def morlet2(M, s, w=5.0):
    """Wavelet de Morlet normalizada en energía (equivalente a scipy.signal.morlet2)."""
    x = (np.arange(0, M) - (M - 1.0) / 2) / s
    return np.sqrt(1.0 / s) * np.exp(1j * w * x) * np.exp(-0.5 * x**2) * np.pi**(-0.25)
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import matplotlib.gridspec as gridspec
import os, warnings
warnings.filterwarnings('ignore')

try:
    from vmdpy import VMD as _VMD
    TIENE_VMD = True
except ImportError:
    TIENE_VMD = False
    print("[AVISO] vmdpy no instalado — usando filtros de banda como fallback")

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

CSV_PATH   = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\poses_completo.csv'
OUTPUT_DIR = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\paper_definitivo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_MINIMA  = 0.3
BIN_SEGUNDOS = 5       # 5s — 60fps continuo lo soporta sin problema
DT           = BIN_SEGUNDOS
ALL_BEES     = ['Bee1','Bee2','Bee3','Bee4','Bee5','Bee6','Bee7']
BEES_VIVAS   = ['Bee1','Bee2','Bee3','Bee4','Bee5','Bee6']
ANTENAS      = ['Antena_1_A','Antena_1_B','Antena_2_A','Antena_2_B']

COLORES = {
    'Bee1':'#ff6b6b','Bee2':'#feca57','Bee3':'#48dbfb',
    'Bee4':'#ff9ff3','Bee5':'#54a0ff','Bee6':'#a29bfe','Bee7':'#888888',
}
COLORES_LISTA = [COLORES[b] for b in BEES_VIVAS]

END_H      = 11 + 20/60          # fin grabación: 22 Abr 11:20 AM
DURACION_S = 17*3600 + 25*60 + 13 # duración exacta: 17h 25m 13s = 62,713s
START_REAL = END_H - DURACION_S/3600  # 21 Abr 17:54:47
# En polar 24h: la zona SIN DATOS va de 11:20 hasta que empieza la grabación (17:54)
# (la grabación es CONTINUA — no hay gaps internos)
VMD_K      = 4
VMD_ALPHA  = 2000
SIGMA_FINST = 2
ALPHA_SUB   = 1.0
BETA_FLOOR  = 0.05

# CWT
W0           = 6.0
PERIODS_SPEC = np.logspace(np.log10(2), np.log10(180), 35)

# Polar HHT
PERIOD_MIN_PLOT = 2.0
PERIOD_MAX_PLOT = 360.0
R_INNER = 0.15
R_OUTER = 1.0

# Estética oscura
C_FONDO = '#080818'
C_PANEL = '#0f0f28'
C_TEXTO = '#e0e0e0'
C_GRID  = '#1a1a3a'

CMAP_SPEC = LinearSegmentedColormap.from_list('spec',
    ['#080818','#0d1b4a','#1a4080','#0077b6','#00b4d8',
     '#48cae4','#ffd166','#ef476f','#ffffff'], N=512)
CMAP_HOT = LinearSegmentedColormap.from_list('hot2',
    ['#080818','#1a004a','#6a0dad','#c77dff','#ffd166','#fff'], N=512)
CMAP_CLEAN = LinearSegmentedColormap.from_list('clean',
    ['#080818','#003333','#006666','#00b4d8','#90e0ef',
     '#caf0f8','#ffd166','#ef476f','#ffffff'], N=512)
CMAP_AMP = LinearSegmentedColormap.from_list('amp',
    ['#080818','#0d1b4a','#0077b6','#00b4d8','#90e0ef',
     '#ffd166','#ef476f','#ffffff'], N=512)

MODO_COLORES = ['#00b4d8','#a29bfe','#ffd166','#ef476f']
MODO_META = {
    0: dict(nombre='lento',     banda='>120 min',  desc='ritmo vigilia/reposo endógeno'),
    1: dict(nombre='medio',     banda='30–120 min', desc='ritmo colectivo significativo'),
    2: dict(nombre='rapido',    banda='10–30 min',  desc='ultradiano rápido'),
    3: dict(nombre='muyrapido', banda='<10 min',    desc='microritmo/vibración'),
}

period_ticks_hht = [5, 15, 30, 60, 120, 240]
theta_circ       = np.linspace(0, 2*np.pi, 300)
GAP_S = (11+20/60) / 24 * 2*np.pi          # 11:20 — fin grabación
GAP_E = START_REAL / 24 * 2*np.pi           # 17:54 — inicio grabación (dato calculado al vuelo)
# NOTA: con grabación continua no hay gaps INTERNOS. Solo hay zona sin datos
# en el reloj polar entre el fin (11:20) y el inicio (17:54) del mismo día.
log_min = np.log10(PERIOD_MIN_PLOT)
log_max = np.log10(PERIOD_MAX_PLOT)

hour_ticks    = np.arange(0, 24, 2)
hour_tick_rad = hour_ticks / 24 * 2 * np.pi
hour_tick_lbl = [f'{h:02d}h' for h in hour_ticks]

# ══════════════════════════════════════════════════════════════
# PROCESAMIENTO DE SEÑAL
# ══════════════════════════════════════════════════════════════

def construir_señal(sub, t_max, bin_s=BIN_SEGUNDOS):
    sub = sub.sort_values('tiempo_seg')
    segs = []
    for bp in ANTENAS:
        xc, yc, cc = f'{bp}_x', f'{bp}_y', f'{bp}_conf'
        if xc not in sub.columns: continue
        m  = sub[cc].values > CONF_MINIMA
        xs = sub.loc[sub.index[m], xc].values
        ys = sub.loc[sub.index[m], yc].values
        ts = sub.loc[sub.index[m], 'tiempo_seg'].values
        if len(xs) < 2: continue
        dist = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        vel  = np.where(np.diff(ts) > 0, dist/np.diff(ts), 0)
        segs.append(pd.Series(vel, index=(ts[:-1]+ts[1:])/2))
    if not segs: return None, None
    combined = pd.concat(segs, axis=1).mean(axis=1)
    bins = np.arange(0, t_max + bin_s, bin_s)
    t_c  = bins[:-1] + bin_s/2
    out  = np.full(len(t_c), np.nan)
    for i, tc in enumerate(t_c):
        mask = (combined.index >= tc-bin_s/2) & (combined.index < tc+bin_s/2)
        if mask.sum() > 0: out[i] = combined[mask].mean()
    s = pd.Series(out).interpolate('linear', limit_direction='both').fillna(0)
    return t_c, s.values

def _fallback_bands(signal, K=VMD_K):
    fs_s = 1.0 / BIN_SEGUNDOS
    period_breaks = np.logspace(np.log10(300), np.log10(7200), K-1)
    modes, prev = [], np.zeros_like(signal)
    for pb in period_breaks:
        fn = np.clip((1/pb)/(fs_s/2), 1e-6, 0.9999)
        b, a = butter(4, fn, btype='low')
        lo = filtfilt(b, a, signal)
        modes.append(lo - prev); prev = lo.copy()
    modes.append(signal - prev)
    return np.array(modes[::-1])

def vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA):
    if not TIENE_VMD: return _fallback_bands(signal, K)
    sig = signal - signal.mean()
    try:
        u, _, omega = _VMD(sig, alpha, 0, K, 0, 1, 1e-7)
        if omega.ndim == 2:
            ff = omega[:,-1] if omega.shape[0]==K else omega[-1,:]
        else: ff = np.arange(K, dtype=float)
        if len(ff) != K: ff = np.arange(K, dtype=float)
        return u[np.argsort(ff)]
    except Exception as e:
        print(f"  VMD ({e}) → fallback")
        return _fallback_bands(signal, K)

def hilbert_instantaneo(modo, dt=DT, sigma_f=SIGMA_FINST):
    analytic  = hilbert(modo)
    amp_inst  = np.abs(analytic)
    phase     = np.unwrap(np.angle(analytic))
    dphi_dt   = np.diff(phase) / dt
    dphi_sm   = gaussian_filter1d(dphi_dt, sigma=sigma_f)
    f_hz      = np.clip(dphi_sm / (2*np.pi), 1/(24*3600), 1/60)
    T_min     = 1.0 / (f_hz * 60)
    amp_mid   = (amp_inst[:-1] + amp_inst[1:]) / 2
    return amp_mid, f_hz, T_min

def cwt_morlet(signal, scales, w0=W0):
    n   = len(signal)
    out = np.zeros((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        M   = max(int(12*s), 5)
        wav = morlet2(M, s, w0)
        conv  = fftconvolve(signal, wav[::-1].conj(), mode='full')
        start = (M-1)//2
        out[i] = conv[start:start+n] / np.sqrt(s)
    return out

def normalizar(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-9: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def rose_bin(sig, clock_h, bins_h):
    n = len(bins_h) - 1
    out = np.full(n, np.nan)
    for k in range(n):
        mask = (clock_h >= bins_h[k]) & (clock_h < bins_h[k+1])
        if mask.sum() > 0: out[k] = sig[mask].mean()
    return out

# ══════════════════════════════════════════════════════════════
# HELPERS POLARES — ESTÉTICA OSCURA
# ══════════════════════════════════════════════════════════════

def setup_polar_base(ax, col_ticks=C_TEXTO, fs=7, ylim=1.2):
    ax.set_facecolor(C_PANEL)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=col_ticks, fontsize=fs)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['','50%','','100%'], color='#444', fontsize=fs-1)
    ax.spines['polar'].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.4, lw=0.4)
    ax.set_ylim(0, ylim)
    gap = np.linspace(GAP_S, GAP_E, 120)
    ax.fill_between(gap, 0, ylim, color='#04040a', alpha=0.93, zorder=0)

def draw_ref_lines(ax, ylim=1.2):
    for h_m, col_m, lbl_m in [
            (18,    '#ffd166', '18h\nInicio'),
            (0,     '#00e5ff', '00h\nMedianoche'),
            (11+20/60, '#ff6b6b', '11:20\nFin')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=1.0, ls=':', alpha=0.6, zorder=5)
        ax.text(h_m/24*2*np.pi, ylim*0.95, lbl_m,
                ha='center', va='top', color=col_m, fontsize=6, fontweight='bold')

def draw_rose(ax, bin_rad, bin_w, vals, color, is_ctrl=False):
    valid = ~np.isnan(vals)
    alpha_bar = 0.22 if is_ctrl else 0.35
    lw_line   = 0.8  if is_ctrl else 1.6
    ls_line   = '--' if is_ctrl else '-'
    for k in range(len(bin_rad)):
        if not valid[k]: continue
        ax.bar(bin_rad[k], vals[k], width=bin_w*0.80, bottom=0.02,
               color=color, alpha=alpha_bar + (0.45 if not is_ctrl else 0)*vals[k],
               edgecolor='none', zorder=2)
    if valid.sum() > 2:
        tv = bin_rad[valid]
        rv = vals[valid]
        ax.fill(np.append(tv, tv[0]), np.append(rv, rv[0]),
                color=color, alpha=0.12, zorder=1)
        ax.plot(np.append(tv, tv[0]), np.append(rv, rv[0]),
                color=color, lw=lw_line, ls=ls_line, alpha=0.9, zorder=4)
    if valid.sum() > 0 and not is_ctrl:
        pk   = np.nanargmax(vals)
        h_pk = bin_rad[pk] / (2*np.pi) * 24   # ← desde el ángulo, funciona con cualquier nº de bins
        ax.scatter(bin_rad[pk], vals[pk]+0.08, s=50,
                   color='white', zorder=10, edgecolors=color, lw=1.5)
        ax.text(bin_rad[pk], vals[pk]+0.22,
                f'{int(h_pk):02d}h', ha='center', va='center',
                color='white', fontsize=7, fontweight='bold')

def setup_polar_hht(ax, ylim=R_OUTER*1.06):
    ax.set_facecolor('#050510')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=6)
    ax.set_yticks([])
    ax.spines['polar'].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.2, lw=0.3)
    ax.set_ylim(0, ylim)
    gap = np.linspace(GAP_S, GAP_E, 120)
    ax.fill_between(gap, 0, ylim, color='#040408', alpha=0.95, zorder=0)

def periodo_a_radio(T):
    r = R_INNER + (R_OUTER-R_INNER) * (np.log10(np.clip(T, PERIOD_MIN_PLOT, PERIOD_MAX_PLOT)) - log_min) / (log_max - log_min)
    return np.clip(r, R_INNER, R_OUTER)

def draw_period_rings(ax):
    for pt in period_ticks_hht:
        if pt < PERIOD_MIN_PLOT or pt > PERIOD_MAX_PLOT: continue
        r_t = periodo_a_radio(pt)
        ax.plot(theta_circ, np.full(300, r_t),
                color='white', lw=0.3, alpha=0.18, ls='--', zorder=1)
        ax.text(0.04, r_t, f'{pt}m', color='#888', fontsize=5,
                ha='left', va='center', zorder=8)

def draw_time_refs_hht(ax, ylim=R_OUTER*1.06):
    for h_m, col_m, lbl in [(18,'#ffd166','18h'),(0,'#00e5ff','00h'),(11+20/60,'#ff6b6b','11:20')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.55, zorder=5)
        ax.text(h_m/24*2*np.pi, ylim*0.97, lbl,
                ha='center', va='top', color=col_m, fontsize=6, fontweight='bold')

def scatter_hht(ax, theta_pts, T_pts, amp_pts, color_or_cmap,
                alpha=0.55, s=2.5, use_cmap=True, vmin=None, vmax=None):
    r_pts   = periodo_a_radio(T_pts)
    in_gap  = (theta_pts >= GAP_S) & (theta_pts <= GAP_E)
    theta_f = theta_pts[~in_gap]
    r_f     = r_pts[~in_gap]
    amp_f   = amp_pts[~in_gap]
    if len(theta_f) == 0: return None
    if use_cmap:
        return ax.scatter(theta_f, r_f, c=amp_f, cmap=color_or_cmap,
                          s=s, alpha=alpha, vmin=vmin, vmax=vmax,
                          linewidths=0, rasterized=True, zorder=3)
    else:
        return ax.scatter(theta_f, r_f, color=color_or_cmap,
                          s=s, alpha=alpha, linewidths=0,
                          rasterized=True, zorder=3)

def savefig(fig, nombre, dpi=150):
    # PNG — para visualización rápida
    ruta_png = os.path.join(OUTPUT_DIR, f'{nombre}.png')
    fig.savefig(ruta_png, dpi=dpi, bbox_inches='tight', facecolor=C_FONDO)
    # PDF — vectorial para el paper (fuentes embebidas, sin pixelado)
    ruta_pdf = os.path.join(OUTPUT_DIR, f'{nombre}.pdf')
    fig.savefig(ruta_pdf, format='pdf', bbox_inches='tight', facecolor=C_FONDO)
    plt.close(fig)
    print(f"  → {nombre}.png + .pdf")

# ══════════════════════════════════════════════════════════════
# CARGA Y CÓMPUTO (UNA SOLA VEZ)
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("  CARGANDO DATOS Y COMPUTANDO (1 sola vez)")
print("=" * 60)

df    = pd.read_csv(CSV_PATH)
t_max = df['tiempo_seg'].max()
print(f"  CSV: {len(df):,} filas  |  {t_max/3600:.2f}h de video")

señales   = {}; modes_all = {}
t_ref = None; N_t = None

for bee in ALL_BEES:
    sub = df[df['animal'] == bee]
    t_c, sig = construir_señal(sub, t_max)
    if sig is None: print(f"  {bee}: sin datos"); continue
    if t_ref is None: t_ref = t_c; N_t = len(t_c)
    señales[bee] = sig
    print(f"  {bee} VMD...", end=' ', flush=True)
    modes_all[bee] = vmd_decompose(sig)
    print("OK")

t_abs_h     = END_H - (t_max - t_ref) / 3600
clock_h     = t_abs_h % 24

# ── Normalizar longitudes al mínimo entre señales y modos VMD ──────
# vmdpy a veces devuelve N-1 puntos → alinear todo antes de continuar
N_t_eff = N_t
for bee in ALL_BEES:
    if bee in modes_all:
        N_t_eff = min(N_t_eff, modes_all[bee].shape[1])
    if bee in señales:
        N_t_eff = min(N_t_eff, len(señales[bee]))

if N_t_eff < N_t:
    print(f"  [INFO] Ajustando longitud: {N_t} → {N_t_eff} bins (vmdpy off-by-one)")
    t_ref   = t_ref[:N_t_eff]
    clock_h = clock_h[:N_t_eff]
    N_t     = N_t_eff
    for bee in list(señales.keys()):
        señales[bee] = señales[bee][:N_t]
    for bee in list(modes_all.keys()):
        modes_all[bee] = modes_all[bee][:, :N_t]

clock_h_mid = (clock_h[:-1] + clock_h[1:]) / 2
theta_mid   = clock_h_mid / 24 * 2*np.pi
theta_full  = clock_h / 24 * 2*np.pi

# Bins de 30 min y 1h
n_bins30 = 48;  bins_h30 = np.linspace(0, 24, n_bins30+1)
n_bins1h = 24;  bins_h1h = np.linspace(0, 24, n_bins1h+1)
bin_ctrs30 = (bins_h30[:-1]+bins_h30[1:])/2
bin_ctrs1h = (bins_h1h[:-1]+bins_h1h[1:])/2
bin_rad30  = bin_ctrs30/24*2*np.pi;  bin_w30 = 2*np.pi/n_bins30
bin_rad1h  = bin_ctrs1h/24*2*np.pi;  bin_w1h = 2*np.pi/n_bins1h

# Referencia global de bins (usar 1h para rose plots VMD, que es el original)
bins_h = bins_h1h
bin_rad = bin_rad1h
bin_w   = bin_w1h

# Envelopes por abeja y modo (amplitud Hilbert de VMD)
print("  Calculando envelopes VMD + HHT...", flush=True)
envelopes = {}   # envelopes[bee][k] = envolvente suavizada (N,)
hht = {}         # hht[bee][k] = (amp_mid, T_min)  (N-1,)

for bee in ALL_BEES:
    if bee not in modes_all: continue
    envelopes[bee] = {}
    hht[bee] = {}
    for k in range(VMD_K):
        env = gaussian_filter1d(np.abs(hilbert(modes_all[bee][k])), sigma=5)
        envelopes[bee][k] = env
        amp_mid, _, T_min = hilbert_instantaneo(modes_all[bee][k])
        hht[bee][k] = (amp_mid, T_min)

# Rose plots por modo — bins 1h
rose_env = {}    # rose_env[bee][k] = promedio de la envolvente en cada bin 1h
for bee in ALL_BEES:
    if bee not in envelopes: continue
    rose_env[bee] = {}
    for k in range(VMD_K):
        rose_env[bee][k] = normalizar(rose_bin(envelopes[bee][k], clock_h, bins_h))

# Rose plots señal cruda (30min y 1h)
rose_raw30 = {}; rose_raw1h = {}
for bee in ALL_BEES:
    if bee not in señales: continue
    rose_raw30[bee] = normalizar(rose_bin(señales[bee], clock_h, bins_h30))
    rose_raw1h[bee] = normalizar(rose_bin(señales[bee], clock_h, bins_h1h))

# Sustracción espectral (Bee7 como referencia de ruido)
rose_env_dn = {}   # rose_env_dn[bee][k] = denoised
if 'Bee7' in rose_env:
    for bee in BEES_VIVAS:
        if bee not in rose_env: continue
        rose_env_dn[bee] = {}
        for k in range(VMD_K):
            raw_b  = rose_env[bee][k]
            raw_b7 = rose_env['Bee7'][k]
            clean  = np.maximum(raw_b - ALPHA_SUB*raw_b7, BETA_FLOOR*raw_b7)
            rose_env_dn[bee][k] = normalizar(np.where(np.isnan(raw_b), np.nan, clean))

# CWT por abeja (precalcular para usar en fig 09 y 10)
print("  Calculando CWT Morlet por abeja...", flush=True)
scales_cwt = (PERIODS_SPEC*60/BIN_SEGUNDOS)*W0/(2*np.pi)
cwt_power  = {}
for bee in ALL_BEES:
    if bee not in señales: continue
    W = cwt_morlet(señales[bee], scales_cwt)
    cwt_power[bee] = np.abs(W)**2
    print(f"    {bee}: OK", flush=True)

# CWT denoised
cwt_power_dn = {}
if 'Bee7' in cwt_power:
    pb7 = cwt_power['Bee7']
    for bee in BEES_VIVAS:
        if bee not in cwt_power: continue
        cwt_power_dn[bee] = np.maximum(cwt_power[bee] - ALPHA_SUB*pb7,
                                        BETA_FLOOR*pb7)

# Media del grupo
sig_grupo = np.mean([señales[b] for b in BEES_VIVAS if b in señales], axis=0)

# Grilla CWT para pcolormesh
r_spec   = R_INNER + (R_OUTER-R_INNER)*(np.log10(PERIODS_SPEC)-np.log10(2))/(np.log10(180)-np.log10(2))
r_spec   = np.clip(r_spec, R_INNER, R_OUTER)
dr       = np.diff(r_spec)
r_edges  = np.concatenate([[r_spec[0]-dr[0]/2], (r_spec[:-1]+r_spec[1:])/2, [r_spec[-1]+dr[-1]/2]])
dth      = np.diff(theta_full)
th_edges = np.concatenate([[theta_full[0]-dth[0]/2], (theta_full[:-1]+theta_full[1:])/2, [theta_full[-1]+dth[-1]/2]])
Th, Rr   = np.meshgrid(th_edges, r_edges)

print(f"\n  Datos listos. Generando {OUTPUT_DIR}/")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# FIG 01 — Rose plot actividad individual (6 abejas + media)
# ══════════════════════════════════════════════════════════════

print("\nFig 01 — Rose plot actividad individual...")
fig, axes = plt.subplots(2, 4, figsize=(22, 13),
                          subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
axes = axes.flatten()
fig.suptitle('Actividad Antenal Promedio por Franja Horaria — Reloj de 24h\n'
             'cada barra = promedio en esa media hora  |  21 Abr 18:00 → 22 Abr 11:20',
             color=C_TEXTO, fontsize=13, fontweight='bold', y=0.99)

media30 = np.nanmean([rose_raw30[b] for b in BEES_VIVAS if b in rose_raw30], axis=0)

for j, bee in enumerate(BEES_VIVAS):
    ax = axes[j]
    setup_polar_base(ax, COLORES[bee], fs=6)
    ax.set_title(bee, color=COLORES[bee], fontsize=12, fontweight='bold', pad=12)
    draw_rose(ax, bin_rad30, bin_w30, rose_raw30.get(bee, np.full(n_bins30, np.nan)), COLORES[bee])
    draw_ref_lines(ax)

ax = axes[6]
ax.set_visible(False)

ax = axes[7]
setup_polar_base(ax, C_TEXTO, fs=7)
ax.set_title('Media del Grupo\n(todas las abejas)', color=C_TEXTO, fontsize=10, fontweight='bold', pad=12)
for j, bee in enumerate(BEES_VIVAS):
    r = rose_raw30.get(bee, None)
    if r is None: continue
    valid = ~np.isnan(r)
    if valid.sum() > 2:
        ax.plot(np.append(bin_rad30[valid], bin_rad30[valid][0]),
                np.append(r[valid], r[valid][0]),
                color=COLORES[bee], lw=0.8, alpha=0.45, label=bee)
valid_m = ~np.isnan(media30)
ax.plot(np.append(bin_rad30[valid_m], bin_rad30[valid_m][0]),
        np.append(media30[valid_m], media30[valid_m][0]),
        color='white', lw=2.2, alpha=0.85, label='media', zorder=5)
for k in range(n_bins30):
    if not np.isnan(media30[k]):
        ax.bar(bin_rad30[k], media30[k], width=bin_w30*0.80, bottom=0.02,
               color='#00b4d8', alpha=0.25+0.45*media30[k], edgecolor='none', zorder=2)
draw_ref_lines(ax)
ax.legend(loc='lower left', bbox_to_anchor=(-0.2,-0.12), fontsize=7,
          facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.5, ncol=4)
savefig(fig, '01_rose_actividad_individual')

# ══════════════════════════════════════════════════════════════
# FIG 02 — Comparación 6 abejas en un solo polar
# ══════════════════════════════════════════════════════════════

print("Fig 02 — Polar comparación 6 abejas...")
fig, ax = plt.subplots(1, 1, figsize=(12, 12),
                        subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
setup_polar_base(ax, C_TEXTO, fs=9, ylim=1.25)
ax.set_title('Actividad Antenal por Hora del Día — 6 Abejas\n'
             'promedio en bins de 30 min  |  21 Abr 18:00 → 22 Abr 11:20',
             color=C_TEXTO, fontsize=12, fontweight='bold', pad=20)

for j, bee in enumerate(BEES_VIVAS):
    r = rose_raw30.get(bee)
    if r is None: continue
    valid = ~np.isnan(r)
    if valid.sum() < 2: continue
    tv, rv = bin_rad30[valid], r[valid]
    ax.fill(np.append(tv, tv[0]), np.append(rv, rv[0]), color=COLORES[bee], alpha=0.12)
    ax.plot(np.append(tv, tv[0]), np.append(rv, rv[0]),
            color=COLORES[bee], lw=2.0, alpha=0.9, label=bee)
    pk = np.nanargmax(r)
    ax.scatter(bin_rad30[pk], r[pk], s=80, color=COLORES[bee], zorder=10,
               edgecolors='white', lw=1.2)

valid_m30 = ~np.isnan(media30)
ax.plot(np.append(bin_rad30[valid_m30], bin_rad30[valid_m30][0]),
        np.append(media30[valid_m30], media30[valid_m30][0]),
        color='white', lw=2.5, alpha=0.7, ls='--', label='media grupo', zorder=8)

for h_m, col_m, lbl in [(18,'#ffd166','18:00\nInicio'),(0,'#00e5ff','00:00\nMedianoche'),(11+20/60,'#ff6b6b','11:20\nFin')]:
    ax.axvline(h_m/24*2*np.pi, color=col_m, lw=1.5, ls=':', alpha=0.7)
    ax.text(h_m/24*2*np.pi, 1.18, lbl, ha='center', va='center',
            color=col_m, fontsize=8, fontweight='bold')

ax.legend(loc='lower left', bbox_to_anchor=(-0.12,-0.08), fontsize=9,
          facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6, ncol=4)
savefig(fig, '02_polar_comparacion_6abejas')

# ══════════════════════════════════════════════════════════════
# FIG 03 — 4 escalas temporales (banda por cuadrante)
# ══════════════════════════════════════════════════════════════

print("Fig 03 — Polar 4 escalas temporales...")

# Definir bandas de período → filtrar señal
bandas = [
    ('<5 min',    'Rápida',      (1/300, 1/30)),
    ('5-30 min',  'Media',       (1/1800, 1/300)),
    ('30-120 min','Lenta',       (1/7200, 1/1800)),
    ('>2h',       'Muy lenta',   (1/86400, 1/7200)),
]
fs_hz = 1.0 / BIN_SEGUNDOS

fig, axes = plt.subplots(2, 2, figsize=(18, 16),
                          subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
axes = axes.flatten()
fig.suptitle('Actividad Antenal en Reloj de 24 Horas\n'
             '21 Abr 18:00 → 22 Abr 11:20  |  radio = actividad normalizada',
             color=C_TEXTO, fontsize=13, fontweight='bold', y=0.99)

for ax_i, (banda_label, banda_nombre, (flo, fhi)) in enumerate(bandas):
    ax = axes[ax_i]
    setup_polar_base(ax, C_TEXTO, fs=7, ylim=1.2)
    ax.set_title(f'{banda_nombre} ({banda_label})', color=C_TEXTO,
                 fontsize=12, fontweight='bold', pad=12)

    medias_banda = []
    for j, bee in enumerate(BEES_VIVAS):
        if bee not in señales: continue
        sig = señales[bee]
        # Filtrado de banda
        flo_c = np.clip(flo/(fs_hz/2), 1e-5, 0.9999)
        fhi_c = np.clip(fhi/(fs_hz/2), 1e-5, 0.9999)
        if flo_c < fhi_c:
            b, a = butter(3, [flo_c, fhi_c], btype='band')
            sig_f = filtfilt(b, a, sig)
        else:
            sig_f = sig
        env   = gaussian_filter1d(np.abs(hilbert(sig_f)), sigma=3)
        r_bee = normalizar(rose_bin(env, clock_h, bins_h1h))
        medias_banda.append(r_bee)
        valid = ~np.isnan(r_bee)
        if valid.sum() > 2:
            tv, rv = bin_rad1h[valid], r_bee[valid]
            ax.fill(np.append(tv, tv[0]), np.append(rv, rv[0]),
                    color=COLORES[bee], alpha=0.12)
            ax.plot(np.append(tv, tv[0]), np.append(rv, rv[0]),
                    color=COLORES[bee], lw=1.5, alpha=0.85, label=bee)

    media_b = np.nanmean(medias_banda, axis=0)
    valid_b = ~np.isnan(media_b)
    if valid_b.sum() > 2:
        ax.plot(np.append(bin_rad1h[valid_b], bin_rad1h[valid_b][0]),
                np.append(media_b[valid_b], media_b[valid_b][0]),
                color='white', lw=2.5, ls='--', alpha=0.8, label='media grupo', zorder=6)
    draw_ref_lines(ax)

axes[0].legend(loc='lower left', bbox_to_anchor=(-0.15,-0.12), fontsize=8,
               facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.5, ncol=4)
savefig(fig, '03_polar_4escalas')

# ══════════════════════════════════════════════════════════════
# FIG 04 — Individual suavizado por abeja (señal + dispersión)
# ══════════════════════════════════════════════════════════════

print("Fig 04 — Polar individual suavizado...")
fig, axes = plt.subplots(2, 4, figsize=(22, 13),
                          subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
axes = axes.flatten()
fig.suptitle('Actividad Antenal Individual — Reloj de 24h\n'
             'suavizado ~15min  |  radio = actividad relativa  |  zona oscura = sin grabación',
             color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

sig_smooth = {}
for bee in BEES_VIVAS:
    if bee not in señales: continue
    sig_smooth[bee] = gaussian_filter1d(señales[bee], sigma=15)

for j, bee in enumerate(BEES_VIVAS):
    ax = axes[j]
    setup_polar_base(ax, COLORES[bee], fs=6, ylim=1.15)
    ax.set_title(bee, color=COLORES[bee], fontsize=11, fontweight='bold', pad=10)
    if bee not in sig_smooth: continue
    sm = sig_smooth[bee]
    # Señal continua polar
    r_norm  = normalizar(sm)
    # Excluir gap
    in_gap  = (theta_full >= GAP_S) & (theta_full <= GAP_E)
    th_plot = theta_full[~in_gap]
    r_plot  = r_norm[~in_gap]
    ax.plot(th_plot, r_plot, color=COLORES[bee], lw=1.0, alpha=0.7, zorder=3)
    ax.fill_between(th_plot, 0, r_plot, color=COLORES[bee], alpha=0.18, zorder=2)
    # Pico
    if len(r_plot) > 0:
        pk_idx = np.argmax(r_plot)
        ax.scatter(th_plot[pk_idx], r_plot[pk_idx]+0.06, s=60,
                   color='white', zorder=10, edgecolors=COLORES[bee], lw=1.5)
    # Línea de inicio 18h
    ax.axvline(18/24*2*np.pi, color='#ffd166', lw=0.8, ls=':', alpha=0.5)
    ax.text(18/24*2*np.pi, 1.10, '18h\n18:00', ha='center', va='top',
            color='#ffd166', fontsize=5.5, fontweight='bold')

# Panel dispersión del grupo
ax = axes[6]
ax.set_visible(False)
ax = axes[7]
setup_polar_base(ax, C_TEXTO, fs=7, ylim=1.15)
ax.set_title('Media del Grupo + Dispersión\n(±1 desv. estándar)',
             color=C_TEXTO, fontsize=9, fontweight='bold', pad=10)

all_r = []
for bee in BEES_VIVAS:
    if bee not in sig_smooth: continue
    all_r.append(normalizar(sig_smooth[bee]))
if all_r:
    arr  = np.array(all_r)
    med  = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr,  axis=0)
    in_g = (theta_full >= GAP_S) & (theta_full <= GAP_E)
    th_g = theta_full[~in_g]; med_g = med[~in_g]; std_g = std[~in_g]
    ax.fill_between(th_g, np.clip(med_g-std_g,0,1.1), np.clip(med_g+std_g,0,1.1),
                    color='#00b4d8', alpha=0.22, zorder=2)
    ax.plot(th_g, med_g, color='white', lw=2.0, alpha=0.85, zorder=5)
    for j, bee in enumerate(BEES_VIVAS):
        if bee not in sig_smooth: continue
        rn = normalizar(sig_smooth[bee])[~in_g]
        ax.plot(th_g, rn, color=COLORES[bee], lw=0.6, alpha=0.4)
ax.axvline(18/24*2*np.pi, color='#ffd166', lw=0.8, ls=':', alpha=0.5)
savefig(fig, '04_polar_suavizado_individual')

# ══════════════════════════════════════════════════════════════
# FIG 05-06 — VMD modos raw (lento y medio)
# FIG 07-08 — VMD modos denoised (lento y medio)
# ══════════════════════════════════════════════════════════════

def figura_vmd_modo(modo_idx, variante, num):
    meta  = MODO_META[modo_idx]
    deno  = (variante == 'denoised')
    sufx  = 'denoised' if deno else 'raw'
    titulo_var = 'DENOISED (Sustracción Espectral con Bee7)' if deno else 'envolvente individual'
    print(f"Fig {num:02d} — VMD M{modo_idx} {sufx}...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 12),
                              subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
    axes = axes.flatten()
    fig.suptitle(
        f'VMD M{modo_idx} — {meta["banda"]}  |  {meta["desc"]}  |  {titulo_var}\n'
        f'Oscuridad total + IR  →  ritmos endógenos  |  Bee7 = control negativo',
        color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

    if deno:
        env_bee7 = normalizar(rose_bin(envelopes['Bee7'][modo_idx], clock_h, bins_h)) if 'Bee7' in envelopes else None

    vals_vivas = []
    for bee in BEES_VIVAS:
        if bee not in rose_env: continue
        if deno and bee in rose_env_dn:
            vals_vivas.append(rose_env_dn[bee].get(modo_idx, np.full(n_bins1h, np.nan)))
        else:
            vals_vivas.append(rose_env[bee].get(modo_idx, np.full(n_bins1h, np.nan)))
    media_g = np.nanmean(vals_vivas, axis=0)
    std_g   = np.nanstd(vals_vivas,  axis=0)
    valid_m = ~np.isnan(media_g)

    for idx, bee in enumerate(ALL_BEES):
        ax = axes[idx]
        is_b7 = (bee == 'Bee7')
        setup_polar_base(ax, COLORES[bee], fs=6)
        ax.set_title('Bee7 (ctrl−)' if is_b7 else bee,
                     color=COLORES[bee], fontsize=11, fontweight='bold', pad=10)
        if bee not in rose_env: continue
        if is_b7:
            r7 = rose_env[bee].get(modo_idx, None)
            if r7 is not None:
                draw_rose(ax, bin_rad, bin_w, r7, COLORES[bee], is_ctrl=True)
        else:
            if deno and bee in rose_env_dn:
                r = rose_env_dn[bee].get(modo_idx, np.full(n_bins1h, np.nan))
            else:
                r = rose_env[bee].get(modo_idx, np.full(n_bins1h, np.nan))
            draw_rose(ax, bin_rad, bin_w, r, COLORES[bee])
        draw_ref_lines(ax)

    # Panel promedio
    ax = axes[7]
    setup_polar_base(ax, C_TEXTO, fs=7)
    ax.set_title('Promedio grupo\n± 1 desv. estándar' + ('\n(denoised)' if deno else ''),
                 color=C_TEXTO, fontsize=9, fontweight='bold', pad=10)
    if valid_m.sum() > 1:
        hi = np.clip(media_g+std_g, 0, 1.15)
        lo = np.clip(media_g-std_g, 0, 1.15)
        tm = bin_rad[valid_m]
        ax.fill_between(np.append(tm, tm[0]),
                        np.append(lo[valid_m], lo[valid_m][0]),
                        np.append(hi[valid_m], hi[valid_m][0]),
                        color='#00b4d8', alpha=0.22, zorder=2)
        for k in range(n_bins1h):
            if np.isnan(media_g[k]): continue
            ax.bar(bin_rad[k], media_g[k], width=bin_w*0.72, bottom=0.02,
                   color='#00b4d8', alpha=0.3+0.45*media_g[k], edgecolor='none', zorder=2)
        ax.plot(np.append(tm, tm[0]),
                np.append(media_g[valid_m], media_g[valid_m][0]),
                color='white', lw=2.2, alpha=0.9, zorder=5)
        pk = np.nanargmax(media_g)
        ax.text(bin_rad[pk], media_g[pk]+0.22,
                f'pico\n{int(bin_ctrs1h[pk]):02d}h',
                ha='center', va='center', color='white',
                fontsize=8, fontweight='bold')
    if deno and 'Bee7' in rose_env:
        r7 = rose_env['Bee7'].get(modo_idx, None)
        if r7 is not None:
            valid7 = ~np.isnan(r7)
            if valid7.sum() > 2:
                ax.plot(np.append(bin_rad[valid7], bin_rad[valid7][0]),
                        np.append(r7[valid7], r7[valid7][0]),
                        color='#888888', lw=1.0, ls='--', alpha=0.6,
                        zorder=3, label='Bee7 ruido')
                ax.legend(loc='lower right', bbox_to_anchor=(1.3,-0.1), fontsize=7,
                          facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.5)
    draw_ref_lines(ax)
    if deno:
        fig.text(0.01, 0.01,
                 f'Sustracción espectral: α={ALPHA_SUB}, β={BETA_FLOOR}  |  '
                 'Bee7 actúa como referencia de ruido eléctrico + vibración + falsos positivos del tracker',
                 color='#555577', fontsize=7, style='italic')
    savefig(fig, f'{num:02d}_vmd_m{modo_idx}_{meta["nombre"]}_{sufx}')

figura_vmd_modo(0, 'raw',      5)
figura_vmd_modo(1, 'raw',      6)
figura_vmd_modo(0, 'denoised', 7)
figura_vmd_modo(1, 'denoised', 8)

# ══════════════════════════════════════════════════════════════
# FIG 09 — CWT Polar (8 paneles: 6 vivas + Bee7 + promedio)
# FIG 10 — CWT Polar Denoised
# ══════════════════════════════════════════════════════════════

def figura_cwt_polar(power_dict, num, titulo_extra=''):
    print(f"Fig {num:02d} — CWT Polar {titulo_extra}...")
    fig, axes = plt.subplots(2, 4, figsize=(22, 12),
                              subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
    axes = axes.flatten()
    fig.suptitle(
        f'Espectrograma Polar (CWT Morlet) — Potencia por Período y Hora del Día\n'
        f'Radio = período (2min interior → 180min exterior)  |  Color = potencia  |  {titulo_extra}',
        color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

    bee_order = BEES_VIVAS + ['Bee7']
    for idx, bee in enumerate(bee_order):
        ax = axes[idx]
        ax.set_facecolor('#050510')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(hour_tick_rad)
        ax.set_xticklabels(hour_tick_lbl, color=COLORES[bee], fontsize=5.5)
        ax.set_yticks([])
        ax.set_ylim(0, R_OUTER*1.05)
        ax.grid(color=C_GRID, alpha=0.2, lw=0.3)
        ax.spines['polar'].set_color(C_GRID)

        is_b7 = (bee == 'Bee7')
        ttl = f'{bee}' + (' (ctrl−)' if is_b7 else '')
        ax.set_title(ttl, color=COLORES[bee], fontsize=10, fontweight='bold', pad=8)

        if bee in power_dict:
            pwr = power_dict[bee]
            log_p = np.log10(pwr + 1e-12)
            vmin, vmax = np.percentile(log_p, 5), np.percentile(log_p, 98)
            cmap_use = CMAP_HOT if is_b7 else CMAP_SPEC
            ax.pcolormesh(Th, Rr, log_p, cmap=cmap_use,
                          vmin=vmin, vmax=vmax, shading='flat', rasterized=True)

        gap = np.linspace(GAP_S, GAP_E, 100)
        ax.fill_between(gap, 0, R_OUTER*1.05, color='#040408', alpha=0.93, zorder=3)
        for h_m, col_m in [(18,'#ffd166'),(0,'#00e5ff'),(11+20/60,'#ff6b6b')]:
            ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.5, zorder=5)
        for pt in [5, 15, 30, 60, 120]:
            r_t = R_INNER+(R_OUTER-R_INNER)*(np.log10(pt)-np.log10(2))/(np.log10(180)-np.log10(2))
            r_t = np.clip(r_t, R_INNER, R_OUTER)
            ax.plot(theta_circ, np.full(300, r_t), color='white', lw=0.25, alpha=0.15, ls='--')
            ax.text(0.05, r_t, f'{pt}m', color='#888', fontsize=4.5, va='center')

    # Panel promedio
    ax = axes[7]
    ax.set_facecolor('#050510')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=5.5)
    ax.set_yticks([])
    ax.set_ylim(0, R_OUTER*1.05)
    ax.grid(color=C_GRID, alpha=0.2, lw=0.3)
    ax.set_title('Promedio\n(6 vivas)', color=C_TEXTO, fontsize=10, fontweight='bold', pad=8)
    if len(power_dict) > 0:
        vivas_pow = [power_dict[b] for b in BEES_VIVAS if b in power_dict]
        if vivas_pow:
            avg_pwr = np.mean(vivas_pow, axis=0)
            log_pa = np.log10(avg_pwr + 1e-12)
            vmin_a, vmax_a = np.percentile(log_pa, 5), np.percentile(log_pa, 98)
            ax.pcolormesh(Th, Rr, log_pa, cmap=CMAP_SPEC,
                          vmin=vmin_a, vmax=vmax_a, shading='flat', rasterized=True)
    gap = np.linspace(GAP_S, GAP_E, 100)
    ax.fill_between(gap, 0, R_OUTER*1.05, color='#040408', alpha=0.93, zorder=3)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    savefig(fig, f'{num:02d}_cwt_polar{"_denoised" if titulo_extra else ""}')

figura_cwt_polar(cwt_power,    9,  '')
figura_cwt_polar(cwt_power_dn, 10, 'DENOISED (Sustracción Espectral Bee7)')

# ══════════════════════════════════════════════════════════════
# FIG 11-14 — HHT por modo (scatter polar)
# ══════════════════════════════════════════════════════════════

for modo_idx in range(VMD_K):
    meta = MODO_META[modo_idx]
    num  = 11 + modo_idx
    print(f"Fig {num:02d} — HHT M{modo_idx} {meta['nombre']}...")

    fig, axes = plt.subplots(2, 4, figsize=(22, 12),
                              subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
    axes = axes.flatten()
    fig.suptitle(
        f'Espectro de Hilbert  —  VMD M{modo_idx}  ({meta["banda"]}  |  {meta["desc"]})\n'
        f'Frecuencia y amplitud INSTANTÁNEAS  |  Sin ventanas  |  '
        f'Radio = período  |  Color = amplitud  |  Bee7 = control negativo',
        color=C_TEXTO, fontsize=11, fontweight='bold', y=0.99)

    all_amps = np.concatenate([hht[b][modo_idx][0] for b in BEES_VIVAS if b in hht])
    vmin_g   = np.percentile(all_amps, 2)
    vmax_g   = np.percentile(all_amps, 98)

    for idx, bee in enumerate(ALL_BEES):
        ax  = axes[idx]
        is_b7 = (bee == 'Bee7')
        setup_polar_hht(ax)
        draw_period_rings(ax)
        draw_time_refs_hht(ax)
        ax.set_title('Bee7  (ctrl −)' if is_b7 else bee,
                     color=COLORES[bee], fontsize=10, fontweight='bold', pad=10)
        if bee not in hht: continue
        amp_mid, T_mid = hht[bee][modo_idx]
        if is_b7:
            scatter_hht(ax, theta_mid, T_mid, amp_mid, '#666666',
                        alpha=0.4, s=1.5, use_cmap=False)
        else:
            scatter_hht(ax, theta_mid, T_mid, amp_mid, CMAP_AMP,
                        alpha=0.6, s=2.5, vmin=vmin_g, vmax=vmax_g)
        med_T = np.median(T_mid)
        r_med = periodo_a_radio(med_T)
        ax.plot(theta_circ, np.full(300, r_med),
                color=COLORES[bee], lw=0.7, ls='-', alpha=0.3, zorder=4)
        ax.text(np.pi/2, r_med+0.06, f'med={med_T:.0f}m',
                color=COLORES[bee], fontsize=6, ha='center', va='bottom')

    # Panel grupo
    ax = axes[7]
    setup_polar_hht(ax)
    draw_period_rings(ax)
    draw_time_refs_hht(ax)
    ax.set_title('Grupo (6 vivas)\npor abeja', color=C_TEXTO, fontsize=10, fontweight='bold', pad=10)
    ax.spines['polar'].set_color(MODO_COLORES[modo_idx])
    for bee in BEES_VIVAS:
        if bee not in hht: continue
        amp_mid, T_mid = hht[bee][modo_idx]
        scatter_hht(ax, theta_mid, T_mid, amp_mid, COLORES[bee],
                    alpha=0.35, s=1.5, use_cmap=False)

    sm = ScalarMappable(cmap=CMAP_AMP, norm=Normalize(vmin=vmin_g, vmax=vmax_g))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label('Amplitud instantánea\n(px/s)', color=C_TEXTO, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=C_TEXTO, labelcolor=C_TEXTO, labelsize=7)

    fig.text(0.01, 0.01,
             f'HHT: A(t) = |H[VMD_k](t)|,  f(t) = (1/2π)·dφ/dt  suavizado σ={SIGMA_FINST} bins. '
             f'Sin ventanas: resolución temporal máxima (1 bin = {DT}s).',
             color='#555577', fontsize=7, style='italic')
    plt.tight_layout(rect=[0, 0.04, 0.91, 0.95])
    savefig(fig, f'{num:02d}_hht_m{modo_idx}_{meta["nombre"]}')

# ══════════════════════════════════════════════════════════════
# FIG 15 — CWT vs HHT comparación (4 paneles)
# ══════════════════════════════════════════════════════════════

print("Fig 15 — CWT vs HHT comparación...")

W_cwt_grupo = cwt_morlet(sig_grupo, scales_cwt)
pwr_grupo   = np.abs(W_cwt_grupo)**2
log_pg      = np.log10(pwr_grupo + 1e-12)
vmin_pg, vmax_pg = np.percentile(log_pg, 2), np.percentile(log_pg, 98)

fig, axes = plt.subplots(1, 4, figsize=(26, 8),
                          subplot_kw=dict(projection='polar'), facecolor=C_FONDO)
fig.suptitle(
    'Comparación: CWT (ventana Morlet) vs Espectro de Hilbert (HHT) — Promedio 6 abejas\n'
    'CWT: resolución tiempo-frecuencia limitada por Heisenberg  |  '
    'HHT: frecuencia y amplitud instantáneas sin ventana',
    color=C_TEXTO, fontsize=12, fontweight='bold', y=1.01)

for ax_i, (titulo, cmap_u, extra) in enumerate([
        ('CWT\n(sin suavizar)',         CMAP_SPEC, 'cwt_raw'),
        ('CWT + filtro\n(size=20,~10min)', CMAP_SPEC, 'cwt_sm'),
        ('HHT (todos modos)\nPromedio grupo', None, 'hht_col'),
        ('HHT — amplitud\ninstantánea (color)', None, 'hht_amp'),
    ]):
    ax = axes[ax_i]
    ax.set_facecolor('#050510')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=6)
    ax.set_yticks([])
    ax.set_ylim(0, R_OUTER*1.05)
    ax.grid(color=C_GRID, alpha=0.2, lw=0.3)
    ax.set_title(titulo, color='#00b4d8' if ax_i==0 else
                 '#ffd166' if ax_i==1 else
                 'white' if ax_i==2 else '#ef476f',
                 fontsize=10, fontweight='bold', pad=10)

    if extra == 'cwt_raw':
        ax.pcolormesh(Th, Rr, log_pg, cmap=cmap_u,
                      vmin=vmin_pg, vmax=vmax_pg, shading='flat', rasterized=True)
    elif extra == 'cwt_sm':
        pwr_sm  = uniform_filter1d(pwr_grupo, size=20, axis=1)
        log_psm = np.log10(pwr_sm + 1e-12)
        ax.pcolormesh(Th, Rr, log_psm, cmap=cmap_u,
                      vmin=vmin_pg, vmax=vmax_pg, shading='flat', rasterized=True)
        ax.text(np.pi, R_OUTER*0.5, '← resolución\ntemporal perdida',
                ha='center', color='#ffd166', fontsize=7, alpha=0.8)
    elif extra == 'hht_col':
        setup_polar_hht(ax)
        draw_period_rings(ax)
        for k in range(VMD_K):
            amps_k = np.mean([hht[b][k][0] for b in BEES_VIVAS if b in hht], axis=0)
            Ts_k   = np.mean([hht[b][k][1] for b in BEES_VIVAS if b in hht], axis=0)
            scatter_hht(ax, theta_mid, Ts_k, amps_k, MODO_COLORES[k],
                        alpha=0.5, s=2.0, use_cmap=False)
    elif extra == 'hht_amp':
        setup_polar_hht(ax)
        draw_period_rings(ax)
        all_ag = np.concatenate([
            np.mean([hht[b][k][0] for b in BEES_VIVAS if b in hht], axis=0)
            for k in range(VMD_K)])
        vmn2, vmx2 = np.percentile(all_ag, 2), np.percentile(all_ag, 98)
        for k in range(VMD_K):
            amps_k = np.mean([hht[b][k][0] for b in BEES_VIVAS if b in hht], axis=0)
            Ts_k   = np.mean([hht[b][k][1] for b in BEES_VIVAS if b in hht], axis=0)
            scatter_hht(ax, theta_mid, Ts_k, amps_k, CMAP_AMP,
                        alpha=0.6, s=2.5, vmin=vmn2, vmax=vmx2)

    gap = np.linspace(GAP_S, GAP_E, 100)
    ax.fill_between(gap, 0, R_OUTER*1.05, color='#040408', alpha=0.93, zorder=3)
    for h_m, col_m in [(18,'#ffd166'),(0,'#00e5ff'),(11+20/60,'#ff6b6b')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.5, zorder=5)

sm3 = ScalarMappable(cmap=CMAP_AMP, norm=Normalize(vmin=np.percentile(all_ag,2), vmax=np.percentile(all_ag,98)))
sm3.set_array([])
cax3 = fig.add_axes([0.93, 0.15, 0.012, 0.65])
cb3  = fig.colorbar(sm3, cax=cax3)
cb3.set_label('Amplitud instantánea', color=C_TEXTO, fontsize=8)
cb3.ax.yaxis.set_tick_params(color=C_TEXTO, labelcolor=C_TEXTO, labelsize=7)

fig.text(0.01, 0.01,
    f'CWT size=20: promedia 20 bins×{DT}s = {20*DT//60}min → pierde eventos sub-{20*DT//60}min. '
    f'HHT: resolución temporal = 1 bin ({DT}s) — limitada solo por el muestreo.',
    color='#555577', fontsize=8, style='italic')
plt.tight_layout(rect=[0, 0.04, 0.92, 0.97])
savefig(fig, '15_cwt_vs_hht_comparacion')

# ══════════════════════════════════════════════════════════════
# FIG 16 — Heatmap circular de posición de antenas (7 abejas)
# ══════════════════════════════════════════════════════════════

print("Fig 16 — Heatmap circular de posición de antenas...")

from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap as LSC

TAMANO_RECORTE = 220
RADIO_HM       = TAMANO_RECORTE // 2   # 110 px
GRID_SIZE      = 220
SIGMA_HM       = 4

CMAP_HM = LSC.from_list('bee_heat', [
    '#000000','#0d0d3a','#1a237e','#0077b6','#00b4d8',
    '#90e0ef','#ffd166','#ef476f','#ffffff'], N=512)

heatmaps_pos = {}
for bee in ALL_BEES:
    sub = df[df['animal'] == bee]
    # Centro de la abeja
    if 'Posicion_x' in sub.columns:
        mc = sub['Posicion_conf'] > CONF_MINIMA if 'Posicion_conf' in sub.columns else pd.Series([True]*len(sub), index=sub.index)
        cx = sub.loc[mc, 'Posicion_x'].mean() if mc.sum() > 0 else sub['Antena_1_A_x'].mean()
        cy = sub.loc[mc, 'Posicion_y'].mean() if mc.sum() > 0 else sub['Antena_1_A_y'].mean()
    else:
        cx = sub['Antena_1_A_x'].mean()
        cy = sub['Antena_1_A_y'].mean()

    xs_rel, ys_rel = [], []
    for bp in ANTENAS:
        xc2, yc2, cc2 = f'{bp}_x', f'{bp}_y', f'{bp}_conf'
        if xc2 not in sub.columns: continue
        m = sub[cc2] > CONF_MINIMA
        xs_rel.append(sub.loc[m, xc2].values - cx)
        ys_rel.append(sub.loc[m, yc2].values - cy)

    if not xs_rel:
        heatmaps_pos[bee] = np.zeros((GRID_SIZE, GRID_SIZE))
        continue

    xr = np.concatenate(xs_rel)
    yr = np.concatenate(ys_rel)
    dist_r = np.sqrt(xr**2 + yr**2)
    mask_r = dist_r <= RADIO_HM
    xr, yr = xr[mask_r], yr[mask_r]

    bins_hm = np.linspace(-RADIO_HM, RADIO_HM, GRID_SIZE + 1)
    H, _, _ = np.histogram2d(xr, yr, bins=[bins_hm, bins_hm])
    H = gaussian_filter(H, sigma=SIGMA_HM)

    cx_g = np.linspace(-RADIO_HM, RADIO_HM, GRID_SIZE)
    CX, CY = np.meshgrid(cx_g, cx_g)
    circ_m = (CX**2 + CY**2) > RADIO_HM**2
    H[circ_m.T] = np.nan
    heatmaps_pos[bee] = H

# Layout 4+3
fig16 = plt.figure(figsize=(20, 10), facecolor=C_FONDO)
fig16.suptitle('Mapa de Calor de Antenas — 7 Abejas — 17 horas de video\n'
               'Densidad acumulada de posición de antenas dentro del ROI circular',
               color='white', fontsize=14, fontweight='bold', y=0.97)

positions16 = []
for i in range(4):
    positions16.append([0.04 + i*0.245, 0.55, 0.22, 0.40])  # fila superior
for i in range(3):
    positions16.append([0.165 + i*0.245, 0.10, 0.22, 0.40])  # fila inferior

COLORES_BEE_HM = [COLORES[b] for b in ALL_BEES]

for idx, bee in enumerate(ALL_BEES):
    H = heatmaps_pos[bee]
    col_bee = COLORES[bee]
    ax = fig16.add_axes(positions16[idx], aspect='equal')
    ax.set_facecolor('#000000')

    extent = [-RADIO_HM, RADIO_HM, -RADIO_HM, RADIO_HM]
    vmax_hm = np.nanpercentile(H, 98) if not np.all(np.isnan(H)) else 1
    ax.imshow(H.T, origin='lower', extent=extent, cmap=CMAP_HM,
              interpolation='bilinear', vmin=0, vmax=vmax_hm)

    circ_patch = Circle((0,0), RADIO_HM, fill=False, edgecolor=col_bee, linewidth=2.5, zorder=5)
    ax.add_patch(circ_patch)
    ax.axhline(0, color=col_bee, lw=0.5, alpha=0.4, zorder=4)
    ax.axvline(0, color=col_bee, lw=0.5, alpha=0.4, zorder=4)
    ax.plot(0, 0, '+', color='white', ms=6, mew=1.5, zorder=6)

    lbl = f'{bee}' + (' (ctrl−)' if bee == 'Bee7' else '')
    ax.text(0, RADIO_HM*1.08, lbl, ha='center', va='bottom',
            color=col_bee, fontsize=12, fontweight='bold')

    # Escala 50px
    ax.plot([-RADIO_HM*0.7, -RADIO_HM*0.7+50], [-RADIO_HM*0.88]*2,
            color='white', lw=1.5, alpha=0.7)
    ax.text(-RADIO_HM*0.7+25, -RADIO_HM*0.82, '50px',
            ha='center', color='white', fontsize=6, alpha=0.7)

    # % zona activa
    valid_hm = H[~np.isnan(H)]
    if len(valid_hm) > 0:
        pct = (valid_hm > valid_hm.max()*0.1).mean() * 100
        ax.text(RADIO_HM*0.98, -RADIO_HM*0.92, f'{pct:.0f}% zona activa',
                ha='right', color='white', fontsize=6, alpha=0.7)

    ax.set_xlim(-RADIO_HM*1.15, RADIO_HM*1.15)
    ax.set_ylim(-RADIO_HM*1.15, RADIO_HM*1.15)
    ax.axis('off')

# Colorbar horizontal compacta al pie — centrada, no tapa ningún panel
sm_hm = plt.cm.ScalarMappable(cmap=CMAP_HM)
sm_hm.set_array([])
cax_hm = fig16.add_axes([0.28, 0.038, 0.44, 0.016])   # [left, bottom, width, height]
cbar_hm = fig16.colorbar(sm_hm, cax=cax_hm, orientation='horizontal')
cbar_hm.set_ticks([0, 0.5, 1.0])
cbar_hm.set_ticklabels(['baja', 'media', 'alta'], color='white', fontsize=7)
cbar_hm.ax.xaxis.set_tick_params(color='white', labelsize=7)
cbar_hm.set_label('Densidad acumulada de posición de antenas', color='white', fontsize=8, labelpad=2)

fig16.text(0.5, 0.008,
           f'ROI circular: radio={RADIO_HM}px  |  confianza>{CONF_MINIMA}  |  '
           f'antenas: {", ".join(ANTENAS)}  |  suavizado σ={SIGMA_HM}',
           ha='center', color='#888888', fontsize=7.5)

savefig(fig16, '16_heatmap_circular_posicion')

# ══════════════════════════════════════════════════════════════
# FIG 17 — Resumen global: actividad + por abeja + FFT
#   • Zero-padding (×8) → resolución fina sin inventar energía
#   • Eje X en hora real del día (21 Abr 18:00 → 22 Abr 11:20)
# ══════════════════════════════════════════════════════════════

print("Fig 17 — Resumen global FFT (zero-padded, hora real)...")

from scipy.fft import fft, fftfreq

BIN_FFT    = 10      # segundos por bin para la serie global
PAD_FACTOR = 8       # multiplicador de ceros al final
# START_REAL ya definido en CONFIG (17:54:47 del 21 Abr)
END_REAL   = END_H   # 22 Abr 11:20h
# Hora real = 18:00 + t_seg/3600  (puede superar 24 → medianoche)

def seg_a_hora_real(t_seg):
    """Convierte segundos desde el inicio de grabación a hora real (puede >24h)."""
    return START_REAL + t_seg / 3600.0

def hora_real_label(h_float):
    """h_float puede ser >24 (ej. 25.5 = 01:30 del día siguiente)."""
    h_mod = h_float % 24
    return f'{int(h_mod):02d}:{int((h_mod % 1)*60):02d}'

def hacer_serie_bins_fft(t_vals, v_vals, bin_s, t_total):
    bins = np.arange(0, t_total + bin_s, bin_s)
    t_c  = bins[:-1] + bin_s / 2
    out  = np.full(len(t_c), np.nan)
    for i, tc in enumerate(t_c):
        m = (t_vals >= tc - bin_s/2) & (t_vals < tc + bin_s/2)
        if m.sum() > 0: out[i] = v_vals[m].mean()
    return t_c, out

def hacer_fft_padded(serie_1d, bin_s, pad_factor=PAD_FACTOR):
    """
    FFT con zero-padding.
    • Rellena NaNs por interpolación lineal.
    • Agrega N*(pad_factor-1) ceros al final → finer frequency grid.
    • Normaliza por N_original (no N_padded) → amplitudes correctas.
    El zero-padding NO añade información: solo interpola el espectro
    continuo subyacente, eliminando el aliasing de la grilla discreta.
    """
    s = pd.Series(serie_1d).interpolate('linear', limit_direction='both').fillna(0).values
    N_orig  = len(s)
    N_pad   = N_orig * pad_factor
    s_pad   = np.zeros(N_pad)
    s_pad[:N_orig] = s             # ceros al final — no al principio
    yf  = fft(s_pad)
    xf  = fftfreq(N_pad, bin_s)
    m   = xf > 0
    # Normalizar por N_orig: amplitud = 2/N_orig × |FFT|
    amp = 2.0 / N_orig * np.abs(yf[m])
    return xf[m], amp

def anotar_picos(ax, freq, amp, n=5, color='#feca57'):
    top = np.argsort(amp)[::-1][:n]
    for i in top:
        p_h  = 1 / (freq[i] * 3600)
        lbl  = f'{p_h*60:.0f}m' if p_h < 1 else f'{p_h:.1f}h'
        ax.annotate(lbl, (p_h, amp[i]), xytext=(0, 6),
                    textcoords='offset points', color=color,
                    fontsize=7.5, ha='center', fontweight='bold')
        ax.plot(p_h, amp[i], 'o', color=color, ms=4.5, zorder=5)

C_P17 = '#12122a'

def estilo17(ax, titulo='', xlab='', ylab=''):
    ax.set_facecolor(C_P17)
    ax.tick_params(colors=C_TEXTO, labelsize=7)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(titulo, color=C_TEXTO, fontsize=9, pad=5, fontweight='bold')
    ax.grid(True, color=C_GRID, alpha=0.5, linewidth=0.4)
    if xlab: ax.set_xlabel(xlab, color=C_TEXTO, fontsize=8)
    if ylab: ax.set_ylabel(ylab, color=C_TEXTO, fontsize=8)

# ── Serie global de velocidad ──────────────────────────────────
print("  Calculando serie global...", flush=True)
all_series = []
for bee in ALL_BEES:
    sub = df[df['animal'] == bee].sort_values('tiempo_seg')
    segs = []
    for bp in ANTENAS:
        xc2, yc2, cc2 = f'{bp}_x', f'{bp}_y', f'{bp}_conf'
        if xc2 not in sub.columns: continue
        m = sub[cc2] > CONF_MINIMA
        xs2 = sub.loc[m, xc2].values
        ys2 = sub.loc[m, yc2].values
        ts2 = sub.loc[m, 'tiempo_seg'].values
        if len(xs2) < 2: continue
        dist2 = np.sqrt(np.diff(xs2)**2 + np.diff(ys2)**2)
        vel2  = np.where(np.diff(ts2) > 0, dist2/np.diff(ts2), 0)
        segs.append(pd.Series(vel2, index=(ts2[:-1]+ts2[1:])/2))
    if segs:
        all_series.append(pd.concat(segs, axis=1).mean(axis=1))

global_vel  = pd.concat(all_series).sort_index()
global_mean = global_vel.groupby(global_vel.index).mean()
t_g, s_g    = hacer_serie_bins_fft(global_mean.index.values, global_mean.values, BIN_FFT, t_max)

# Hora real en el eje X (t_g en segundos → hora real)
t_real = seg_a_hora_real(t_g)   # ej. 18.0 … 35.33 (cruza medianoche)

# FFT con zero-padding
freq_g, amp_g = hacer_fft_padded(s_g, BIN_FFT)
per_g = 1 / (freq_g * 3600)    # períodos en horas

# ── Ticks de hora real para eje X ─────────────────────────────
# Cada 2 horas de 18:00 a 11:20 (siguiente día)
tick_vals_h  = np.arange(START_REAL, START_REAL + t_max/3600 + 1, 2)
tick_labels  = [hora_real_label(h) for h in tick_vals_h]
# Añadir el tick exacto de fin (11:20) y el de medianoche
tick_especiales = {
    START_REAL:           ('21 Abr\n18:00', '#ffd166'),
    24.0:                 ('00:00\nMedianoche', '#00e5ff'),
    START_REAL + t_max/3600: ('22 Abr\n11:20', '#ff6b6b'),
}

# ── Figura ────────────────────────────────────────────────────
fig17, axes17 = plt.subplots(3, 1, figsize=(20, 14), facecolor=C_FONDO,
                              gridspec_kw={'height_ratios': [1.2, 1.2, 1.6]})
fig17.suptitle('Análisis Global — 7 Abejas — 17h de video\n'
               '21 Abr 18:00 → 22 Abr 11:20  |  Oscuridad total + IR',
               color=C_TEXTO, fontsize=14, fontweight='bold', y=0.99)

# ── Panel A: Serie temporal global ────────────────────────────
ax = axes17[0]
estilo17(ax, 'Actividad Global de Antenas — todos los bins de 10s',
         '', 'Velocidad (px/s)')
tiene = ~np.isnan(s_g)
ax.fill_between(t_real, 0, np.where(tiene, s_g, 0), alpha=0.15, color='#48dbfb')
ax.plot(t_real, s_g, color='#ff6b6b', lw=0.6, alpha=0.9)

# Medianoche
ax.axvline(24.0, color='#00e5ff', lw=1.2, ls='--', alpha=0.7)
ax.text(24.0, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1,
        '00:00', ha='center', va='bottom', color='#00e5ff',
        fontsize=8, fontweight='bold')

ax.set_xlim(t_real[0], t_real[-1])
ax.set_xticks(tick_vals_h)
ax.set_xticklabels(tick_labels, color=C_TEXTO, fontsize=7)

# Marcadores inicio / fin
for h_esp, (lbl_esp, col_esp) in tick_especiales.items():
    ax.axvline(h_esp, color=col_esp, lw=1.2, ls=':', alpha=0.8)
    ax.text(h_esp, ax.get_ylim()[1] if ax.get_ylim()[1]>0 else 1,
            lbl_esp, ha='center', va='bottom',
            color=col_esp, fontsize=7, fontweight='bold')

# ── Panel B: Actividad por abeja por hora ────────────────────
ax = axes17[1]
estilo17(ax, 'Actividad Promedio por Abeja — cada Hora de Video',
         '', 'Actividad (px/s)')
n_horas = int(t_max / 3600) + 1
horas_seg = np.arange(n_horas) * 3600          # inicio de cada hora en seg
horas_real = seg_a_hora_real(horas_seg)         # hora real correspondiente

for bee in ALL_BEES:
    if bee not in señales: continue
    vals_h = []
    for h in range(n_horas):
        m_h = (t_ref >= h*3600) & (t_ref < (h+1)*3600)
        vals_h.append(señales[bee][m_h].mean() if m_h.sum() > 0 else np.nan)
    lw_b = 0.8 if bee == 'Bee7' else 1.5
    ls_b = '--' if bee == 'Bee7' else '-'
    ax.plot(horas_real, vals_h, 'o-', color=COLORES[bee], label=bee,
            lw=lw_b, ls=ls_b, ms=4, alpha=0.85)

ax.axvline(24.0, color='#00e5ff', lw=1.2, ls='--', alpha=0.6)
ax.set_xlim(horas_real[0] - 0.3, horas_real[-1] + 0.3)
ax.set_xticks(tick_vals_h)
ax.set_xticklabels(tick_labels, color=C_TEXTO, fontsize=7)
for h_esp, (_, col_esp) in tick_especiales.items():
    ax.axvline(h_esp, color=col_esp, lw=1.0, ls=':', alpha=0.7)
ax.legend(fontsize=8, facecolor=C_P17, labelcolor=C_TEXTO,
          framealpha=0.5, ncol=7, loc='upper right')
ax.set_xlabel('Hora del día', color=C_TEXTO, fontsize=8)

# ── Panel C: Espectro FFT con zero-padding ────────────────────
ax = axes17[2]
estilo17(ax,
         f'Espectro FFT Global — Zero-padding ×{PAD_FACTOR}  '
         f'(N={len(s_g)} bins×{BIN_FFT}s → {len(s_g)*PAD_FACTOR} puntos FFT)\n'
         f'Períodos entre 10 min y 17h  |  Amplitud correcta: normalizado por N_original',
         'Período (horas)', 'Amplitud (px/s)')

mask_fft = (per_g >= 10/60) & (per_g <= 20)
ax.plot(per_g[mask_fft], amp_g[mask_fft], color='#ff6b6b', lw=0.9, alpha=0.9)
anotar_picos(ax, freq_g[mask_fft], amp_g[mask_fft], n=6)
ax.set_xlim(10/60, min(18, per_g[mask_fft].max()))
ax.set_ylim(bottom=0)

# Líneas verticales de períodos clave
for p_ref, col_ref, lbl_ref in [
        (1.0,  '#ffd166', '1h'),
        (2.0,  '#a29bfe', '2h'),
        (17.33,'#ff6b6b', 'duración\ngrabación')]:
    if p_ref <= 18:
        ax.axvline(p_ref, color=col_ref, lw=0.8, ls=':', alpha=0.5)
        ax.text(p_ref, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 1,
                lbl_ref, ha='center', va='top',
                color=col_ref, fontsize=6.5, alpha=0.8)

# Zonas de banda
for p_lo, p_hi, lbl, col in [
        (10/60, 0.5, 'ultradiano\nrápido',  '#00b4d8'),
        (0.5,   2.0, 'ultradiano\nmedio',   '#a29bfe'),
        (2.0,   6.0, 'ultradiano\nlento',   '#ffd166')]:
    p_hi_c = min(p_hi, 18)
    if p_lo < p_hi_c:
        ax.axvspan(p_lo, p_hi_c, color=col, alpha=0.07, zorder=0)
        ax.text((p_lo + p_hi_c)/2, amp_g[mask_fft].max()*0.7,
                lbl, ha='center', color=col, fontsize=7, alpha=0.65)

# Nota metodológica
fig17.text(0.01, 0.005,
    f'Zero-padding: señal de {len(s_g)} bins × {BIN_FFT}s completada con {len(s_g)*(PAD_FACTOR-1)} ceros → '
    f'{len(s_g)*PAD_FACTOR} puntos FFT. '
    f'Resolución frecuencial: Δf = 1/(N×{BIN_FFT}s) = {1/(len(s_g)*BIN_FFT)*3600*60:.2f} ciclos/h. '
    f'Amplitudes normalizadas por N_original.',
    color='#555577', fontsize=6.5, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
savefig(fig17, '17_resumen_global_fft')

# ══════════════════════════════════════════════════════════════
# FIG 18 — Actividad HHT total: Σ envolventes VMD en polar
#
# Radio r(t) = Σₖ |H[VMDₖ(t)]|   (suma de amplitudes instantáneas)
# Ángulo θ(t) = hora del día
# Color = modo dominante en cada instante (argmax Aₖ)
#
# Justificación: la suma de amplitudes HHT es la envolvente total
# de toda la energía oscilatoria. A diferencia de la señal cruda,
# no tiene componente de portadora — es puramente la envolvente.
# El modo dominante muestra qué escala temporal mueve a la abeja.
# ══════════════════════════════════════════════════════════════

print("\nFig 18 — Actividad HHT total (Σ envolventes) + modo dominante...")

# ── Calcular A_total y modo dominante por abeja ──────────────
hht_total   = {}   # A_total(t) = Σₖ Aₖ(t)
hht_dommode = {}   # k_dom(t)   = argmax Aₖ(t)
hht_rms     = {}   # A_rms(t)   = √(Σₖ Aₖ²(t))  — alternativa energética

for bee in ALL_BEES:
    if bee not in modes_all: continue
    amps = []
    for k in range(VMD_K):
        a = np.abs(hilbert(modes_all[bee][k]))  # amplitud instantánea cruda
        amps.append(a)
    amps = np.array(amps)                        # shape (K, N)
    hht_total[bee]   = amps.sum(axis=0)          # Σ amplitudes
    hht_rms[bee]     = np.sqrt((amps**2).sum(axis=0))  # RMS energético
    hht_dommode[bee] = np.argmax(amps, axis=0)   # modo dominante en cada t

# ── Configuración de la figura ────────────────────────────────
fig18 = plt.figure(figsize=(24, 14), facecolor=C_FONDO)
fig18.suptitle(
    'Actividad HHT Total — Envolvente Σ de modos VMD en reloj polar\n'
    r'$r(t) = \sum_k |H[\mathrm{VMD}_k(t)]|$   |   '
    'Color = modo oscilatorio dominante en cada instante   |   '
    'Bee7 = control negativo',
    color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

gs18 = gridspec.GridSpec(2, 4, figure=fig18,
                          hspace=0.30, wspace=0.28,
                          left=0.03, right=0.88, top=0.93, bottom=0.06)

# Layout correcto:
# Fila 0: Bee1(0,0)  Bee2(0,1)  Bee3(0,2)  Grupo(0,3)  ← grupo arriba derecha
# Fila 1: Bee4(1,0)  Bee5(1,1)  Bee6(1,2)  Bee7(1,3)   ← Bee7 abajo derecha

bee_positions18 = {
    'Bee1': (0, 0), 'Bee2': (0, 1), 'Bee3': (0, 2),
    'Bee4': (1, 0), 'Bee5': (1, 1), 'Bee6': (1, 2),
    'Bee7': (1, 3),
}

# Escala de amplitud global (percentil 98 de las 6 vivas) para ylim consistente
amp_max_global = np.percentile(
    np.concatenate([hht_total[b] for b in BEES_VIVAS if b in hht_total]), 98)

ylim18 = amp_max_global * 1.15

# Ticks circulares de referencia (en px/s)
amp_ticks = np.linspace(0, amp_max_global, 5)[1:]  # excluir 0

def setup_polar_hht18(ax, ylim=ylim18):
    ax.set_facecolor('#050510')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=6)
    ax.set_yticks(amp_ticks)
    ax.set_yticklabels([f'{v:.0f}' for v in amp_ticks], color='#444', fontsize=5)
    ax.set_ylim(0, ylim)
    ax.spines['polar'].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.25, lw=0.35)
    # Zona sin datos
    gap = np.linspace(GAP_S, GAP_E, 120)
    ax.fill_between(gap, 0, ylim, color='#04040a', alpha=0.93, zorder=0)

for bee, (row, col) in bee_positions18.items():
    ax = fig18.add_subplot(gs18[row, col], projection='polar')
    is_b7 = (bee == 'Bee7')

    setup_polar_hht18(ax)
    ax.set_title('Bee7  (ctrl −)' if is_b7 else bee,
                 color=COLORES[bee], fontsize=10, fontweight='bold', pad=10)

    if bee not in hht_total: continue

    A_tot = hht_total[bee]
    k_dom = hht_dommode[bee]
    theta = theta_full                 # N puntos = mismo largo que A_tot

    # Filtrar zona sin datos
    in_gap = (theta >= GAP_S) & (theta <= GAP_E)

    if is_b7:
        # Bee7: línea gris continua, sin color por modo
        th_p = theta[~in_gap];  r_p = A_tot[~in_gap]
        ax.plot(th_p, r_p, color='#555555', lw=0.6, alpha=0.5, zorder=3)
        ax.fill_between(th_p, 0, r_p, color='#333333', alpha=0.18, zorder=2)
    else:
        # Colorear por modo dominante
        th_p = theta[~in_gap]
        r_p  = A_tot[~in_gap]
        kd_p = k_dom[~in_gap]

        ax.fill_between(th_p, 0, r_p, color=COLORES[bee], alpha=0.10, zorder=1)

        colors_seg = np.array([matplotlib.colors.to_rgba(MODO_COLORES[k], alpha=0.75)
                                for k in kd_p])
        ax.scatter(th_p, r_p, c=colors_seg, s=1.8,
                   linewidths=0, rasterized=True, zorder=4)
        ax.plot(th_p, r_p, color=COLORES[bee], lw=0.7, alpha=0.45, zorder=3)

        pk_idx  = np.argmax(r_p)
        h_pk    = th_p[pk_idx] / (2*np.pi) * 24
        ax.scatter(th_p[pk_idx], r_p[pk_idx], s=60,
                   color='white', zorder=10, edgecolors=COLORES[bee], lw=1.5)
        ax.text(th_p[pk_idx], r_p[pk_idx] + ylim18*0.08,
                f'pico\n{hora_real_label(START_REAL + h_pk)[:5]}',
                ha='center', va='bottom', color='white',
                fontsize=6.5, fontweight='bold', zorder=11)

    # Referencia inicio/fin
    for h_m, col_m in [(18,'#ffd166'),(0,'#00e5ff'),(11+20/60,'#ff6b6b')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.5, zorder=5)

# ── Panel grupo: (0, 3) — no se superpone con ninguna abeja ──
ax_g = fig18.add_subplot(gs18[0, 3], projection='polar')
setup_polar_hht18(ax_g, ylim=ylim18)
ax_g.set_title('Grupo (6 vivas)\nΣ envolventes VMD',
               color=C_TEXTO, fontsize=10, fontweight='bold', pad=10)

# Promedio grupal del A_total (denoised con Bee7)
A_b7 = hht_total.get('Bee7', np.zeros(N_t))
A_group_stack = []
for bee in BEES_VIVAS:
    if bee not in hht_total: continue
    A_clean = np.maximum(hht_total[bee] - A_b7, 0.05 * A_b7)
    A_group_stack.append(A_clean)
    # Traza individual tenue
    in_gap = (theta_full >= GAP_S) & (theta_full <= GAP_E)
    ax_g.plot(theta_full[~in_gap], A_clean[~in_gap],
              color=COLORES[bee], lw=0.6, alpha=0.35, zorder=2)

if A_group_stack:
    A_mean  = np.mean(A_group_stack, axis=0)
    A_std   = np.std(A_group_stack,  axis=0)
    in_gap  = (theta_full >= GAP_S) & (theta_full <= GAP_E)
    th_g    = theta_full[~in_gap]
    r_g     = A_mean[~in_gap]
    std_g   = A_std[~in_gap]

    ax_g.fill_between(th_g,
                      np.clip(r_g - std_g, 0, None),
                      r_g + std_g,
                      color='#00b4d8', alpha=0.18, zorder=2)
    ax_g.plot(th_g, r_g, color='white', lw=1.8, alpha=0.9, zorder=5)

    # Pico grupal
    pk_g = np.argmax(r_g)
    h_pk_g = th_g[pk_g] / (2*np.pi) * 24
    ax_g.scatter(th_g[pk_g], r_g[pk_g], s=80, color='white',
                 zorder=10, edgecolors='#00b4d8', lw=2)
    ax_g.text(th_g[pk_g], r_g[pk_g] + ylim18*0.1,
              f'pico\n{hora_real_label(START_REAL + h_pk_g)[:5]}',
              ha='center', va='bottom', color='white',
              fontsize=7, fontweight='bold', zorder=11)

for h_m, col_m in [(18,'#ffd166'),(0,'#00e5ff'),(11+20/60,'#ff6b6b')]:
    ax_g.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.5, zorder=5)

# ── Leyenda de modos ─────────────────────────────────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0],[0], color=MODO_COLORES[k], lw=3, alpha=0.85,
           label=f'M{k} {MODO_META[k]["banda"]}')
    for k in range(VMD_K)
]
legend_handles.append(Line2D([0],[0], color='white', lw=2, alpha=0.7, label='Media grupo'))
fig18.legend(handles=legend_handles,
             loc='lower center', ncol=5,
             fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXTO,
             framealpha=0.7, bbox_to_anchor=(0.44, 0.01),
             title='Color = modo VMD dominante en cada instante  |  '
                   'Radio = px/s  |  Sustracción Bee7 en panel grupo',
             title_fontsize=7)

# ── Nota metodológica ─────────────────────────────────────────
fig18.text(0.01, 0.005,
    r'$r(t) = \sum_{k=0}^{3} |H[\mathrm{VMD}_k(t)]|$   '
    r'donde H = Transformada de Hilbert analítica.   '
    r'Modo dominante: $k^*(t) = \mathrm{argmax}_k\, A_k(t)$.   '
    'Panel grupo: sustracción espectral Bee7 (piso=5%).',
    color='#555577', fontsize=7, style='italic')

savefig(fig18, '18_hht_actividad_total_polar')

# ══════════════════════════════════════════════════════════════
# FIG 19 — ACTOGRAMA (raster plot cronobiológico)
# ══════════════════════════════════════════════════════════════

print("\nFig 19 — Actograma (raster plot)...")

BIN_ACT = 300   # 5 minutos

bins_act  = np.arange(0, t_max + BIN_ACT, BIN_ACT)
t_act_seg = bins_act[:-1] + BIN_ACT / 2
t_act_h   = seg_a_hora_real(t_act_seg)
n_tbins   = len(t_act_seg)

act_matrix = np.full((len(ALL_BEES), n_tbins), np.nan)
for i, bee in enumerate(ALL_BEES):
    if bee not in señales: continue
    sig = señales[bee]
    for j, tc in enumerate(t_act_seg):
        m = (t_ref >= tc - BIN_ACT/2) & (t_ref < tc + BIN_ACT/2)
        if m.sum() > 0:
            act_matrix[i, j] = sig[m].mean()

act_norm = np.zeros_like(act_matrix)
for i in range(len(ALL_BEES)):
    row = act_matrix[i]; valid = ~np.isnan(row)
    if valid.sum() > 0:
        mn, mx = row[valid].min(), row[valid].max()
        act_norm[i] = np.where(valid, (row - mn)/(mx - mn + 1e-9), np.nan)

CMAP_RASTER = LinearSegmentedColormap.from_list('raster',
    ['#080818','#0d1b4a','#0077b6','#00b4d8','#ffd166','#ef476f','#ffffff'], N=512)

fig19, axes19 = plt.subplots(2, 1, figsize=(20, 9), facecolor=C_FONDO,
                              gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})
fig19.suptitle(f'Actograma — Actividad Antenal por Abeja  |  '
               f'21 Abr {int(START_REAL):02d}:{int((START_REAL%1)*60):02d} → 22 Abr 11:20\n'
               'Bins de 5 min  |  Color = actividad normalizada por abeja  |  '
               '60fps CONTINUO — sin gaps internos  |  Bee7 = control negativo',
               color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

ax = axes19[0]
ax.set_facecolor(C_FONDO)
im19 = ax.imshow(act_norm, aspect='auto', cmap=CMAP_RASTER, vmin=0, vmax=1,
                 interpolation='nearest',
                 extent=[t_act_h[0], t_act_h[-1], -0.5, len(ALL_BEES)-0.5])
ax.axvline(24.0, color='#00e5ff', lw=1.5, ls='--', alpha=0.8, zorder=5)
ax.text(24.0, len(ALL_BEES)-0.4, '00:00\nMedianoche', ha='center', va='bottom',
        color='#00e5ff', fontsize=8, fontweight='bold')
for h_m, col_m, lbl_m in [(START_REAL, '#ffd166', f'21 Abr\n{int(START_REAL):02d}:54'),
                            (END_H+24,  '#ff6b6b', '22 Abr\n11:20')]:
    ax.axvline(h_m, color=col_m, lw=1.5, ls=':', alpha=0.8)
    ax.text(h_m, len(ALL_BEES)-0.4, lbl_m, ha='center', va='bottom',
            color=col_m, fontsize=7, fontweight='bold')
ax.set_yticks(range(len(ALL_BEES)))
ax.set_yticklabels(ALL_BEES, fontsize=9)
for i, bee in enumerate(ALL_BEES):
    ax.get_yticklabels()[i].set_color(COLORES[bee])
ax.tick_params(colors=C_TEXTO, labelsize=8)
ax.set_xlim(t_act_h[0], t_act_h[-1])
ax.tick_params(labelbottom=False)
ax.spines[:].set_visible(False)
cax19 = fig19.add_axes([0.92, 0.35, 0.012, 0.55])
cb19 = fig19.colorbar(im19, cax=cax19)
cb19.set_label('Actividad\nnormalizada', color=C_TEXTO, fontsize=8)
cb19.ax.yaxis.set_tick_params(color=C_TEXTO, labelcolor=C_TEXTO, labelsize=7)

ax2 = axes19[1]
ax2.set_facecolor(C_P17)
grupo_mean = np.nanmean(act_matrix[:6], axis=0)
gn = (grupo_mean - np.nanmin(grupo_mean))/(np.nanmax(grupo_mean)-np.nanmin(grupo_mean)+1e-9)
ax2.fill_between(t_act_h, 0, gn, color='#00b4d8', alpha=0.4)
ax2.plot(t_act_h, gn, color='white', lw=1.0, alpha=0.8)
ax2.axvline(24.0, color='#00e5ff', lw=1.2, ls='--', alpha=0.7)
ax2.set_xlim(t_act_h[0], t_act_h[-1]); ax2.set_ylim(0, 1.1)
ax2.set_ylabel('Media\ngrupal', color=C_TEXTO, fontsize=8)
ax2.set_xlabel('Hora del día', color=C_TEXTO, fontsize=9)
ax2.set_xticks(tick_vals_h)
ax2.set_xticklabels(tick_labels, color=C_TEXTO, fontsize=8)
ax2.tick_params(colors=C_TEXTO); ax2.spines[:].set_visible(False)
ax2.grid(color=C_GRID, alpha=0.4, lw=0.4)
fig19.text(0.01, 0.005,
    f'Grabación CONTINUA 60fps  |  Bin={BIN_ACT//60}min  |  '
    f'Inicio: 21 Abr {int(START_REAL):02d}:{int((START_REAL%1)*60):02d}:47  →  Fin: 22 Abr 11:20  |  '
    f'Duración total: {int(DURACION_S//3600)}h {int((DURACION_S%3600)//60)}m {int(DURACION_S%60)}s',
    color='#555577', fontsize=7, style='italic')
savefig(fig19, '19_actograma_raster')

# ══════════════════════════════════════════════════════════════
# FIG 20 — MATRIZ PLV (Phase Locking Value) entre pares
# ══════════════════════════════════════════════════════════════

print("Fig 20 — Matriz PLV de coherencia de fase entre pares...")

def calcular_plv_matrix(bees, modo_idx):
    n = len(bees); plv_mat = np.full((n, n), np.nan); fases = {}
    for bee in bees:
        if bee not in modes_all: continue
        fases[bee] = np.unwrap(np.angle(hilbert(modes_all[bee][modo_idx])))
    for i, bi in enumerate(bees):
        for j, bj in enumerate(bees):
            if bi not in fases or bj not in fases:
                plv_mat[i, j] = np.nan; continue
            plv_mat[i, j] = np.abs(np.mean(np.exp(1j*(fases[bi]-fases[bj]))))
    return plv_mat

CMAP_PLV = LinearSegmentedColormap.from_list('plv',
    ['#080818','#0d1b4a','#0077b6','#00b4d8','#90e0ef','#ffd166','#ef476f','#ffffff'], N=512)

fig20, axes20 = plt.subplots(1, 4, figsize=(22, 6), facecolor=C_FONDO)
fig20.suptitle('Matriz de Coherencia de Fase (PLV) entre Pares de Abejas\n'
               'PLV(i,j) = |⟨exp(i·(φᵢ−φⱼ))⟩| ∈ [0,1]  |  '
               '1 = sincronización perfecta  |  0 = independencia total',
               color=C_TEXTO, fontsize=12, fontweight='bold', y=1.01)
bee_lbl = [b.replace('Bee','B') for b in ALL_BEES]

for k in range(VMD_K):
    ax = axes20[k]; ax.set_facecolor(C_PANEL)
    plv = calcular_plv_matrix(ALL_BEES, k)
    im = ax.imshow(plv, cmap=CMAP_PLV, vmin=0, vmax=1, interpolation='nearest', aspect='equal')
    for i in range(len(ALL_BEES)):
        for j in range(len(ALL_BEES)):
            if not np.isnan(plv[i, j]):
                ax.text(j, i, f'{plv[i,j]:.2f}', ha='center', va='center',
                        color='black' if plv[i,j]>0.6 else 'white', fontsize=7, fontweight='bold')
    ax.set_xticks(range(len(ALL_BEES))); ax.set_xticklabels(bee_lbl, color=C_TEXTO, fontsize=8)
    ax.set_yticks(range(len(ALL_BEES)))
    ax.set_yticklabels(bee_lbl if k==0 else [], color=C_TEXTO, fontsize=8)
    for lbl in ax.get_xticklabels()+ax.get_yticklabels():
        bf = 'Bee'+lbl.get_text().replace('B','')
        if bf in COLORES: lbl.set_color(COLORES[bf])
    ax.set_title(f'M{k}  {MODO_META[k]["banda"]}\n{MODO_META[k]["desc"]}',
                 color=MODO_COLORES[k], fontsize=9, fontweight='bold', pad=8)
    ax.axhline(5.5, color='#ff6b6b', lw=1.5, ls='--', alpha=0.7)
    ax.axvline(5.5, color='#ff6b6b', lw=1.5, ls='--', alpha=0.7)
    mask_v = np.ones((7,7), dtype=bool); np.fill_diagonal(mask_v, False)
    mask_v[6,:]=False; mask_v[:,6]=False
    ax.text(0.02, 0.02, f'PLV̄ vivas = {np.nanmean(plv[mask_v]):.3f}',
            transform=ax.transAxes, color='#ffd166', fontsize=8, fontweight='bold', va='bottom')

plt.colorbar(im, ax=axes20[-1], shrink=0.85, pad=0.02,
             label='PLV').ax.yaxis.set_tick_params(color=C_TEXTO, labelcolor=C_TEXTO)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
savefig(fig20, '20_matriz_plv_coherencia')

# ══════════════════════════════════════════════════════════════
# FIG 21 — KURAMOTO R(t): PARÁMETRO DE ORDEN TEMPORAL
# ══════════════════════════════════════════════════════════════

print("Fig 21 — Kuramoto R(t) parámetro de orden temporal...")

UMBRAL_QUIMERA = 0.5

fig21, axes21 = plt.subplots(VMD_K, 1, figsize=(20, 14), facecolor=C_FONDO, sharex=True)
fig21.suptitle('Parámetro de Orden de Kuramoto R(t) — Dinámica de Sincronización Colectiva\n'
               r'$R(t) = \left|\frac{1}{N}\sum_{k=1}^{N} e^{i\phi_k(t)}\right|$  '
               f'|  R=1: sincronización total  |  R=0: caos  |  '
               f'Umbral quimera: R < {UMBRAL_QUIMERA}',
               color=C_TEXTO, fontsize=11, fontweight='bold', y=0.99)

t_h_plot = seg_a_hora_real(t_ref)

for k in range(VMD_K):
    ax = axes21[k]
    ax.set_facecolor(C_P17); ax.spines[:].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.4, lw=0.4)
    fases_k = [np.unwrap(np.angle(hilbert(modes_all[bee][k])))
               for bee in BEES_VIVAS if bee in modes_all]
    if not fases_k: continue
    R_t = np.abs(np.mean(np.exp(1j*np.array(fases_k)), axis=0))
    R_sm = gaussian_filter1d(R_t, sigma=10)
    is_q = R_sm < UMBRAL_QUIMERA
    ax.fill_between(t_h_plot, 0, 1, where=is_q, color='#4a0080', alpha=0.25,
                    label=f'Estado quimera (R<{UMBRAL_QUIMERA})', zorder=1)
    ax.plot(t_h_plot, R_t, color='#333355', lw=0.4, alpha=0.5, zorder=2)
    ax.plot(t_h_plot, R_sm, color=MODO_COLORES[k], lw=1.8, alpha=0.9, zorder=3)
    R_mean = R_sm.mean()
    ax.axhline(R_mean, color='white', lw=1.2, ls='--', alpha=0.7, zorder=4)
    ax.text(t_h_plot[-1]*0.99, R_mean+0.02, f'R̄={R_mean:.3f}',
            color='white', fontsize=8, ha='right', fontweight='bold')
    ax.axhline(UMBRAL_QUIMERA, color='#ff6b6b', lw=0.9, ls=':', alpha=0.6)
    pct_q = is_q.mean()*100
    ax.text(0.01, 0.92, f'M{k}  {MODO_META[k]["banda"]}  |  {pct_q:.0f}% tiempo en quimera',
            transform=ax.transAxes, color=MODO_COLORES[k], fontsize=9, fontweight='bold', va='top')
    ax.set_ylim(0, 1.05); ax.set_ylabel('R(t)', color=C_TEXTO, fontsize=9)
    ax.tick_params(colors=C_TEXTO, labelsize=8)
    ax.axvline(24.0, color='#00e5ff', lw=1.0, ls=':', alpha=0.6)
    if k == 0:
        ax.legend(loc='lower right', fontsize=8, facecolor=C_PANEL,
                  labelcolor=C_TEXTO, framealpha=0.6)

axes21[-1].set_xlabel('Hora del día', color=C_TEXTO, fontsize=9)
axes21[-1].set_xticks(tick_vals_h)
axes21[-1].set_xticklabels(tick_labels, color=C_TEXTO, fontsize=8)
fig21.text(0.01, 0.005,
    r'Fase: φₖ(t) = arg[H[VMDₖ(t)]].  R(t) suavizado σ=10 bins.  '
    'Estado quimera: coexistencia de subredes sincronizadas y desincronizadas.',
    color='#555577', fontsize=7, style='italic')
savefig(fig21, '21_kuramoto_R_temporal')

# ══════════════════════════════════════════════════════════════
# FIG 22 — SCATTER AMPLITUD vs PERÍODO HHT
# ══════════════════════════════════════════════════════════════

print("Fig 22 — Scatter amplitud vs período HHT...")

CMAP_TOD = matplotlib.colormaps.get_cmap('twilight_shifted')
norm_tod = Normalize(vmin=clock_h_mid.min(), vmax=clock_h_mid.max())

fig22, axes22 = plt.subplots(2, 4, figsize=(22, 11), facecolor=C_FONDO)
fig22.suptitle('Estados Oscilatorios: Amplitud vs Período Instantáneo (HHT)\n'
               'Cada punto = 1 bin  |  Color = hora del día  |  '
               'Agrupamientos → estados oscilatorios preferidos del colectivo',
               color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

pt_ticks = [p for p in [5,10,15,30,60,120,240,360]
            if PERIOD_MIN_PLOT <= p <= PERIOD_MAX_PLOT]

for k in range(VMD_K):
    ax_top = axes22[0, k]
    ax_top.set_facecolor('#050510'); ax_top.spines[:].set_visible(False)
    ax_top.grid(color=C_GRID, alpha=0.3, lw=0.3)
    for bee in BEES_VIVAS:
        if bee not in hht: continue
        amp_k, T_k = hht[bee][k]
        mask_r = (T_k >= PERIOD_MIN_PLOT) & (T_k <= PERIOD_MAX_PLOT)
        ax_top.scatter(np.log10(T_k[mask_r]), amp_k[mask_r],
                       c=clock_h_mid[mask_r], cmap=CMAP_TOD, norm=norm_tod,
                       s=0.8, alpha=0.2, linewidths=0, rasterized=True)
    ax_top.set_title(f'M{k} — {MODO_META[k]["banda"]}\nGrupo (6 vivas)',
                     color=MODO_COLORES[k], fontsize=9, fontweight='bold', pad=6)
    ax_top.set_xticks([np.log10(p) for p in pt_ticks])
    ax_top.set_xticklabels([], fontsize=7)
    ax_top.set_ylabel('Amplitud inst. (px/s)', color=C_TEXTO, fontsize=8)
    ax_top.tick_params(colors=C_TEXTO, labelsize=7)

    ax_bot = axes22[1, k]
    ax_bot.set_facecolor('#050510'); ax_bot.spines[:].set_visible(False)
    ax_bot.grid(color=C_GRID, alpha=0.3, lw=0.3)
    REP = 'Bee3'
    if REP in hht:
        amp_k, T_k = hht[REP][k]
        mask_r = (T_k >= PERIOD_MIN_PLOT) & (T_k <= PERIOD_MAX_PLOT)
        ax_bot.hexbin(np.log10(T_k[mask_r]), amp_k[mask_r],
                      gridsize=30, cmap='inferno', mincnt=1, alpha=0.7,
                      linewidths=0, rasterized=True)
        ax_bot.scatter(np.log10(T_k[mask_r]), amp_k[mask_r],
                       c=clock_h_mid[mask_r], cmap=CMAP_TOD, norm=norm_tod,
                       s=1.5, alpha=0.45, linewidths=0, rasterized=True, zorder=3)
    ax_bot.set_title(f'{REP} (líder) — densidad 2D',
                     color=COLORES[REP], fontsize=9, fontweight='bold', pad=6)
    ax_bot.set_xlabel('log₁₀(Período [min])', color=C_TEXTO, fontsize=8)
    ax_bot.set_ylabel('Amplitud inst. (px/s)', color=C_TEXTO, fontsize=8)
    ax_bot.set_xticks([np.log10(p) for p in pt_ticks])
    ax_bot.set_xticklabels([f'{p}m' for p in pt_ticks], color=C_TEXTO, fontsize=7)
    ax_bot.tick_params(colors=C_TEXTO, labelsize=7)

sm_tod = ScalarMappable(cmap=CMAP_TOD, norm=norm_tod)
sm_tod.set_array([])
cax22 = fig22.add_axes([0.92, 0.15, 0.012, 0.70])
cb22 = fig22.colorbar(sm_tod, cax=cax22)
cb22.set_label('Hora del día', color=C_TEXTO, fontsize=9)
cb22.ax.yaxis.set_tick_params(color=C_TEXTO, labelcolor=C_TEXTO, labelsize=7)
cb22_ticks = np.linspace(clock_h_mid.min(), clock_h_mid.max(), 6)
cb22.set_ticks(cb22_ticks)
cb22.set_ticklabels([hora_real_label(START_REAL+h)[:5] for h in cb22_ticks], color=C_TEXTO)

fig22.text(0.01, 0.005,
    'Fila sup.: grupo completo (α=0.2).  Fila inf.: Bee3 (líder) con hexbin densidad 2D.  '
    'Agrupamientos en A(t)−T(t) → estados oscilatorios recurrentes.',
    color='#555577', fontsize=7, style='italic')
plt.tight_layout(rect=[0, 0.03, 0.91, 0.95])
savefig(fig22, '22_scatter_amplitud_periodo_hht')

# ══════════════════════════════════════════════════════════════
# FIG 23 — ENVOLVENTES HHT POR MODO VMD (dominio temporal)
#
# Mismo estilo que Fig 17 pero mostrando |H[VMDₖ(t)]| en vez
# de la señal cruda. Una fila por modo. X = hora real del día.
# Permite ver directamente cuándo y cuánto oscila cada escala.
# ══════════════════════════════════════════════════════════════

print("\nFig 23 — Envolventes HHT por modo VMD (serie temporal)...")

# Suavizado de la envolvente para visualización (σ en bins)
SIGMA_ENV_VIZ = 6   # ~30s a 5s/bin → suaviza ruido de alta freq sin distorsionar ritmos

# Hora real para cada bin
t_real_full = seg_a_hora_real(t_ref)   # misma longitud que señales (N_t)

# Ticks de hora para eje X
tick_vals_h2  = np.arange(np.floor(t_real_full[0]), t_real_full[-1]+1, 2)
tick_labels2  = [hora_real_label(h) for h in tick_vals_h2]

# Número de paneles: 1 señal cruda + 4 modos + espacio
n_panels = 1 + VMD_K
height_ratios = [1.2] + [1.0]*VMD_K

fig23, axes23 = plt.subplots(n_panels, 1, figsize=(22, 16),
                              facecolor=C_FONDO,
                              gridspec_kw={'height_ratios': height_ratios,
                                           'hspace': 0.10})
fig23.suptitle(
    'Envolventes Instantáneas Hilbert por Modo VMD — Dominio Temporal\n'
    r'$A_k(t) = |H[\mathrm{VMD}_k(t)]|$  — suavizado para visualización  |  '
    f'21 Abr {int(START_REAL):02d}:{int((START_REAL%1)*60):02d} → 22 Abr 11:20  |  '
    '60 fps continuo',
    color=C_TEXTO, fontsize=12, fontweight='bold', y=0.99)

C_MED = '#ffffff'   # línea media grupo
C_B7  = '#666666'   # Bee7

# ── Panel 0: señal cruda (referencia) ────────────────────────
ax0 = axes23[0]
ax0.set_facecolor(C_P17)
ax0.spines[:].set_visible(False)
ax0.grid(color=C_GRID, alpha=0.4, lw=0.4)

# Media grupal de señal cruda suavizada
sig_grupo_sm = gaussian_filter1d(sig_grupo[:len(t_real_full)], sigma=SIGMA_ENV_VIZ)
sig_b7_sm    = gaussian_filter1d(señales.get('Bee7', np.zeros(N_t))[:len(t_real_full)],
                                  sigma=SIGMA_ENV_VIZ)

for bee in BEES_VIVAS:
    if bee not in señales: continue
    s_sm = gaussian_filter1d(señales[bee][:len(t_real_full)], sigma=SIGMA_ENV_VIZ)
    ax0.plot(t_real_full, s_sm, color=COLORES[bee], lw=0.6, alpha=0.35)

ax0.plot(t_real_full, sig_b7_sm, color=C_B7, lw=0.7, alpha=0.5, ls='--', label='Bee7 (ruido)')
ax0.fill_between(t_real_full, 0, sig_grupo_sm, color='#00b4d8', alpha=0.12)
ax0.plot(t_real_full, sig_grupo_sm, color=C_MED, lw=1.6, alpha=0.9, label='Media grupo')

ax0.set_ylabel('Velocidad\n(px/s)', color=C_TEXTO, fontsize=8)
ax0.set_title('Señal cruda (referencia) — actividad antenal promedio',
              color='#aaaaaa', fontsize=8, pad=4, loc='left')
ax0.tick_params(colors=C_TEXTO, labelsize=7, labelbottom=False)
ax0.set_xlim(t_real_full[0], t_real_full[-1])
ax0.axvline(24.0, color='#00e5ff', lw=1.2, ls='--', alpha=0.6)
ax0.legend(loc='upper right', fontsize=7, facecolor=C_PANEL,
           labelcolor=C_TEXTO, framealpha=0.6, ncol=2)

# ── Paneles 1–4: envolvente por modo ─────────────────────────
for k in range(VMD_K):
    ax = axes23[k + 1]
    ax.set_facecolor(C_P17)
    ax.spines[:].set_visible(False)
    ax.grid(color=C_GRID, alpha=0.4, lw=0.4)

    # Calcular envolventes individuales
    env_vivas = []
    for bee in BEES_VIVAS:
        if bee not in modes_all: continue
        env_raw = np.abs(hilbert(modes_all[bee][k]))
        env_sm  = gaussian_filter1d(env_raw[:len(t_real_full)], sigma=SIGMA_ENV_VIZ)
        ax.plot(t_real_full, env_sm,
                color=COLORES[bee], lw=0.7, alpha=0.40, zorder=2)
        env_vivas.append(env_sm)

    # Bee7 como referencia de ruido
    if 'Bee7' in modes_all:
        env_b7 = gaussian_filter1d(
            np.abs(hilbert(modes_all['Bee7'][k]))[:len(t_real_full)],
            sigma=SIGMA_ENV_VIZ)
        ax.plot(t_real_full, env_b7, color=C_B7, lw=0.8,
                alpha=0.6, ls='--', zorder=3, label='Bee7 (ruido)')

    # Media y dispersión grupal
    if env_vivas:
        env_arr  = np.array(env_vivas)
        env_mean = env_arr.mean(axis=0)
        env_std  = env_arr.std(axis=0)

        ax.fill_between(t_real_full,
                        np.clip(env_mean - env_std, 0, None),
                        env_mean + env_std,
                        color=MODO_COLORES[k], alpha=0.18, zorder=1)
        ax.plot(t_real_full, env_mean,
                color=MODO_COLORES[k], lw=2.0, alpha=0.95, zorder=5,
                label='Media grupo')

        # Denoised: media − Bee7
        if 'Bee7' in modes_all:
            env_dn = np.maximum(env_mean - env_b7, BETA_FLOOR * env_b7)
            ax.plot(t_real_full, env_dn,
                    color='white', lw=1.0, alpha=0.55, ls=':', zorder=4,
                    label='Media denoised')

        # Pico de la media
        pk_i = np.argmax(env_mean)
        h_pk = hora_real_label(t_real_full[pk_i])
        ax.scatter(t_real_full[pk_i], env_mean[pk_i],
                   s=60, color='white', zorder=10,
                   edgecolors=MODO_COLORES[k], lw=1.5)
        ax.text(t_real_full[pk_i], env_mean[pk_i] * 1.05,
                f'pico\n{h_pk[:5]}',
                ha='center', va='bottom', color='white',
                fontsize=7, fontweight='bold', zorder=11)

    # Medianoche
    ax.axvline(24.0, color='#00e5ff', lw=1.2, ls='--', alpha=0.6)

    ax.set_ylabel('Amplitud\n(px/s)', color=C_TEXTO, fontsize=8)
    ax.set_title(f'M{k}  {MODO_META[k]["banda"]}  —  {MODO_META[k]["desc"]}',
                 color=MODO_COLORES[k], fontsize=9, fontweight='bold',
                 pad=4, loc='left')
    ax.tick_params(colors=C_TEXTO, labelsize=7,
                   labelbottom=(k == VMD_K - 1))
    ax.set_xlim(t_real_full[0], t_real_full[-1])

    if k == 0:
        # Leyenda solo en el primer modo
        ax.legend(loc='upper right', fontsize=7, facecolor=C_PANEL,
                  labelcolor=C_TEXTO, framealpha=0.6, ncol=3)

# Eje X en el panel inferior
ax_last = axes23[-1]
ax_last.set_xlabel('Hora del día', color=C_TEXTO, fontsize=9)
ax_last.set_xticks(tick_vals_h2)
ax_last.set_xticklabels(tick_labels2, color=C_TEXTO, fontsize=8)

# Marcadores inicio/fin en todos los paneles
for ax in axes23:
    ax.axvline(START_REAL, color='#ffd166', lw=1.0, ls=':', alpha=0.6)
    ax.axvline(END_H + 24, color='#ff6b6b', lw=1.0, ls=':', alpha=0.6)

# Etiqueta inicio/fin en panel superior
y_top = axes23[0].get_ylim()[1] if axes23[0].get_ylim()[1] > 0 else 1
for h_m, col_m, lbl_m in [
        (START_REAL, '#ffd166', f'21 Abr\n{int(START_REAL):02d}:54'),
        (24.0,       '#00e5ff', '00:00\nMed.'),
        (END_H+24,   '#ff6b6b', '22 Abr\n11:20')]:
    axes23[0].text(h_m, axes23[0].get_ylim()[1] if axes23[0].get_ylim()[1] > 0 else 1,
                   lbl_m, ha='center', va='bottom',
                   color=col_m, fontsize=7, fontweight='bold')

# Leyenda de abejas en la derecha
from matplotlib.lines import Line2D as L2D
bee_handles = [L2D([0],[0], color=COLORES[b], lw=2, label=b) for b in BEES_VIVAS]
bee_handles.append(L2D([0],[0], color=C_B7, lw=1.5, ls='--', label='Bee7'))
fig23.legend(handles=bee_handles, loc='center right',
             fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXTO,
             framealpha=0.7, ncol=1,
             bbox_to_anchor=(0.99, 0.5),
             title='Abejas', title_fontsize=8)

fig23.text(0.01, 0.005,
    r'$A_k(t) = |H[\mathrm{VMD}_k(t)]|$ — amplitud instantánea Hilbert, '
    f'suavizado σ={SIGMA_ENV_VIZ} bins × {BIN_SEGUNDOS}s = {SIGMA_ENV_VIZ*BIN_SEGUNDOS}s.  '
    'Banda = ±1σ entre abejas vivas.  Línea blanca punteada = media denoised (sustracción Bee7).',
    color='#555577', fontsize=7, style='italic')

plt.tight_layout(rect=[0, 0.02, 0.96, 0.97])
savefig(fig23, '23_hht_envolventes_temporal')

# ══════════════════════════════════════════════════════════════
# FIG 24 — ENVOLVENTES HHT INDIVIDUALES: GRILLA MODOS × ABEJAS
#
# Filas: señal cruda + 4 modos VMD  (5 filas)
# Cols:  Bee1..Bee6 + Bee7          (7 columnas)
# Cada panel = envolvente individual de esa abeja en ese modo
# ══════════════════════════════════════════════════════════════

print("\nFig 24 — Grilla individual HHT: modos × abejas...")

N_ROWS24 = 1 + VMD_K   # señal cruda + 4 modos
N_COLS24 = len(ALL_BEES)  # 7 abejas

fig24, axes24 = plt.subplots(
    N_ROWS24, N_COLS24,
    figsize=(28, 16),
    facecolor=C_FONDO,
    gridspec_kw={'hspace': 0.12, 'wspace': 0.06})

fig24.suptitle(
    'Envolventes Instantáneas Hilbert — Individual por Abeja y Modo VMD\n'
    r'$A_k(t) = |H[\mathrm{VMD}_k(t)]|$  |  '
    f'21 Abr {int(START_REAL):02d}:{int((START_REAL%1)*60):02d} → 22 Abr 11:20  |  60 fps continuo',
    color=C_TEXTO, fontsize=12, fontweight='bold', y=0.995)

# Precalcular ylim por fila para escala consistente entre abejas
ylims_row = []

# Fila 0: señal cruda — ylim global de las 6 vivas
ymax_raw = np.percentile(
    np.concatenate([señales[b][:len(t_real_full)] for b in BEES_VIVAS if b in señales]), 98)
ylims_row.append(ymax_raw * 1.15)

# Filas 1-4: envolvente por modo
for k in range(VMD_K):
    vals = np.concatenate([
        gaussian_filter1d(np.abs(hilbert(modes_all[b][k]))[:len(t_real_full)], sigma=SIGMA_ENV_VIZ)
        for b in BEES_VIVAS if b in modes_all])
    ylims_row.append(np.percentile(vals, 98) * 1.2)

# Etiquetas de fila (eje Y)
row_labels = ['Señal\ncruda'] + [f'M{k}\n{MODO_META[k]["banda"]}' for k in range(VMD_K)]
row_colors = ['#aaaaaa'] + MODO_COLORES

for row in range(N_ROWS24):
    for col, bee in enumerate(ALL_BEES):
        ax = axes24[row, col]
        ax.set_facecolor('#06060f')
        ax.spines[:].set_visible(False)
        ax.set_ylim(0, ylims_row[row])
        ax.set_xlim(t_real_full[0], t_real_full[-1])

        is_b7 = (bee == 'Bee7')
        col_bee = COLORES[bee]
        lw_bee  = 0.5 if is_b7 else 0.7

        # ── Datos a plotear ──────────────────────────────────
        if row == 0:
            # Señal cruda
            if bee in señales:
                sig_sm = gaussian_filter1d(
                    señales[bee][:len(t_real_full)], sigma=SIGMA_ENV_VIZ)
                ax.fill_between(t_real_full, 0, sig_sm,
                                color=col_bee, alpha=0.20, zorder=1)
                ax.plot(t_real_full, sig_sm,
                        color=col_bee, lw=lw_bee, alpha=0.85, zorder=2)
        else:
            k = row - 1
            if bee in modes_all:
                env = gaussian_filter1d(
                    np.abs(hilbert(modes_all[bee][k]))[:len(t_real_full)],
                    sigma=SIGMA_ENV_VIZ)
                # Denoised solo para abejas vivas
                if not is_b7 and 'Bee7' in modes_all:
                    env_b7 = gaussian_filter1d(
                        np.abs(hilbert(modes_all['Bee7'][k]))[:len(t_real_full)],
                        sigma=SIGMA_ENV_VIZ)
                    env_dn = np.maximum(env - env_b7, BETA_FLOOR * env_b7)
                    # Fill entre raw y denoised (zona sustracción)
                    ax.fill_between(t_real_full, env_dn, env,
                                    color='#ff6b6b', alpha=0.10, zorder=1,
                                    label='ruido Bee7')
                    ax.fill_between(t_real_full, 0, env_dn,
                                    color=MODO_COLORES[k], alpha=0.22, zorder=2)
                    ax.plot(t_real_full, env_dn,
                            color='white', lw=0.5, alpha=0.45, ls=':', zorder=4)
                else:
                    ax.fill_between(t_real_full, 0, env,
                                    color=col_bee if is_b7 else MODO_COLORES[k],
                                    alpha=0.18, zorder=1)
                ax.plot(t_real_full, env,
                        color=col_bee, lw=lw_bee, alpha=0.85, zorder=3)

                # Pico
                pk_i = np.argmax(env)
                ax.scatter(t_real_full[pk_i], env[pk_i],
                           s=18, color='white', zorder=10,
                           edgecolors=col_bee, lw=0.8)

        # Medianoche
        ax.axvline(24.0, color='#00e5ff', lw=0.6, ls='--', alpha=0.5)

        # ── Títulos y etiquetas ──────────────────────────────
        if row == 0:
            # Nombre de abeja en la primera fila
            ttl = f'Bee7\n(ctrl−)' if is_b7 else bee
            ax.set_title(ttl, color=col_bee, fontsize=9,
                         fontweight='bold', pad=5)

        if col == 0:
            # Etiqueta de modo en la primera columna
            ax.set_ylabel(row_labels[row], color=row_colors[row],
                          fontsize=8, fontweight='bold', labelpad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(labelleft=False)

        # Ticks X solo en la última fila
        if row < N_ROWS24 - 1:
            ax.tick_params(labelbottom=False)
        else:
            # Ticks cada 4 horas para no saturar
            t4 = np.arange(np.ceil(t_real_full[0]/4)*4, t_real_full[-1]+1, 4)
            ax.set_xticks(t4)
            ax.set_xticklabels([hora_real_label(h)[:5] for h in t4],
                               color=C_TEXTO, fontsize=6, rotation=45)

        ax.tick_params(colors=C_TEXTO, labelsize=6)
        ax.grid(color=C_GRID, alpha=0.25, lw=0.25)

        # Banda Bee7 en las vivas (fila 0 = cruda, filas 1-4 = modos)
        if is_b7 and row > 0:
            # Bee7 con fondo más oscuro para distinguirla
            ax.set_facecolor('#0a060f')
            for sp in ['left']:
                ax.spines[sp].set_visible(True)
                ax.spines[sp].set_color('#ff6b6b')
                ax.spines[sp].set_linewidth(1.2)

# Línea separadora visual entre vivas y Bee7
for row in range(N_ROWS24):
    axes24[row, 5].spines['right'].set_visible(True)
    axes24[row, 5].spines['right'].set_color('#444466')
    axes24[row, 5].spines['right'].set_linewidth(1.5)
    axes24[row, 5].spines['right'].set_linestyle('--')

# Etiqueta X en el centro del eje inferior
fig24.text(0.44, 0.01, 'Hora del día', color=C_TEXTO,
           fontsize=9, ha='center')

# Nota metodológica
fig24.text(0.01, 0.005,
    r'$A_k(t) = |H[\mathrm{VMD}_k(t)]|$ — '
    f'suavizado σ={SIGMA_ENV_VIZ}×{BIN_SEGUNDOS}s={SIGMA_ENV_VIZ*BIN_SEGUNDOS}s.  '
    'Fill rojo = componente eliminada por sustracción Bee7.  '
    'Fill de modo = señal denoised.  Línea blanca punteada = denoised.  '
    '00:00 = línea cian punteada.',
    color='#555577', fontsize=6.5, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.975])
savefig(fig24, '24_hht_envolventes_individual_grid')

print(f"""
{'='*60}
  24 figuras generadas en:
   {OUTPUT_DIR}
{'='*60}
  01  Rose actividad individual (bins 30min)
  02  Polar comparacion 6 abejas superpuestas
  03  Polar 4 escalas temporales
  04  Individual suavizado + dispersion grupo
  05  VMD M0 lento - raw
  06  VMD M1 medio - raw
  07  VMD M0 lento - denoised (Bee7)
  08  VMD M1 medio - denoised (Bee7)
  09  CWT espectrograma polar
  10  CWT espectrograma polar denoised
  11  HHT modo 0 - >120min
  12  HHT modo 1 - 30-120min
  13  HHT modo 2 - 10-30min
  14  HHT modo 3 - <10min
  15  CWT vs HHT comparacion
  16  Heatmap circular posicion antenas
  17  Resumen global: actividad + FFT (zero-padded)
  18  Actividad HHT total: suma envolventes VMD polar
  19  Actograma raster plot cronobiologico
  20  Matriz PLV coherencia de fase entre pares
  21  Kuramoto R(t) parametro de orden temporal
  22  Scatter amplitud vs periodo HHT
  23  Envolventes HHT por modo VMD (temporal, grupo)
  24  Grilla individual HHT: modos x abejas
{'='*60}
Dataset: 60fps CONTINUO  |  {int(DURACION_S//3600)}h {int((DURACION_S%3600)//60)}m {int(DURACION_S%60)}s
Inicio: 21 Abr {int(START_REAL):02d}:{int((START_REAL%1)*60):02d}:47  ->  Fin: 22 Abr 11:20
{'='*60}
""")
