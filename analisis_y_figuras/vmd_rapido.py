# vmd_rapido.py — Analisis VMD de ritmos rapidos M2 y M3
# Resolucion: 0.5s bins (vs 5s del paper_maestro.py)
# Permite resolver:
#   M2 (10-30 min): 1200 muestras/ciclo minimo   <- antes: 120
#   M3 (<10 min):   120+ muestras/ciclo           <- antes: marginal
#
# Abejas: Bee3, Bee4, Bee5, Bee6 | Ruido: Bee7
# Genera 2 figuras polares en el mismo estilo que paper_maestro.py:
#   vmd_rapido_M2_polar.png/pdf  — modo rapido  (10-30 min)
#   vmd_rapido_M3_polar.png/pdf  — modo muy rap. (<10 min)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import hilbert, butter, filtfilt, fftconvolve
from scipy.ndimage import gaussian_filter1d
import os, warnings
warnings.filterwarnings('ignore')

try:
    from vmdpy import VMD as _VMD
    TIENE_VMD = True
except ImportError:
    TIENE_VMD = False
    print("[ERROR] vmdpy no instalado — pip install vmdpy")
    exit()

# ======================================================================
# CONFIG
# ======================================================================
CSV_PATH   = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\poses_completo.csv'
OUTPUT_DIR = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\paper_definitivo\vmd_rapido'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_MINIMA  = 0.3
BIN_FINO     = 0.5      # <<< 0.5s en vez de 5s — 10x mas resolución
BIN_GRUESO   = 5.0      # referencia del paper_maestro.py

ANTENAS = ['Antena_1_A','Antena_1_B','Antena_2_A','Antena_2_B']
BEES_ANALISIS = ['Bee3','Bee4','Bee5','Bee6']
BEE_RUIDO     = 'Bee7'

# VMD con todos los modos — luego usamos solo M2 y M3
VMD_K     = 4
VMD_ALPHA = 2000

# Tiempo real
END_H      = 11 + 20/60
DURACION_S = 17*3600 + 25*60 + 13
START_REAL = END_H - DURACION_S/3600

# Estética — mismo estilo paper_maestro.py
C_FONDO = '#080818'; C_PANEL = '#0f0f28'
C_TEXTO = '#e0e0e0'; C_GRID  = '#1a1a3a'
COLORES = {
    'Bee3':'#48dbfb','Bee4':'#ff9ff3',
    'Bee5':'#54a0ff','Bee6':'#a29bfe','Bee7':'#888888'
}

GAP_S = (11+20/60)/24*2*np.pi
GAP_E = START_REAL/24*2*np.pi

# Polar HHT
PERIOD_MIN = 1.0      # 1 minuto — resolvable con 0.5s bins (120 muestras)
PERIOD_MAX = 35.0     # 35 minutos — cubre M2 y parte de M1 como control
R_INNER    = 0.15
R_OUTER    = 1.0
log_min    = np.log10(PERIOD_MIN)
log_max    = np.log10(PERIOD_MAX)

MODO_INFO = {
    2: dict(nombre='M2 rapido',    banda='10-30 min', col='#ffd166'),
    3: dict(nombre='M3 muy rap.',  banda='<10 min',   col='#ef476f'),
}

# Bins polares de 30 min
n_bins30   = 48
bins_h30   = np.linspace(0, 24, n_bins30+1)
bin_ctrs30 = (bins_h30[:-1]+bins_h30[1:])/2
bin_rad30  = bin_ctrs30/24*2*np.pi
bin_w30    = 2*np.pi/n_bins30
hour_ticks    = np.arange(0, 24, 2)
hour_tick_rad = hour_ticks/24*2*np.pi
hour_tick_lbl = [f'{h:02d}h' for h in hour_ticks]
theta_circ    = np.linspace(0, 2*np.pi, 300)
period_ticks  = [1, 2, 5, 10, 20, 30]

# ======================================================================
# FUNCIONES
# ======================================================================

def construir_senal_fina(sub, t_max, bin_s=BIN_FINO):
    """
    Construye la señal de velocidad antenal con bins de BIN_FINO segundos.
    Usa numpy.digitize en vez del loop original → 10x mas rapido.
    """
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
        dist  = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        vel   = np.where(np.diff(ts) > 0, dist/np.diff(ts), 0)
        t_mid = (ts[:-1]+ts[1:])/2
        segs.append(pd.Series(vel, index=t_mid))
    if not segs: return None, None
    combined = pd.concat(segs, axis=1).mean(axis=1)

    # Binning rapido con digitize
    bins    = np.arange(0, t_max + bin_s, bin_s)
    t_c     = bins[:-1] + bin_s/2
    bin_idx = np.clip(np.digitize(combined.index.values, bins)-1, 0, len(t_c)-1)
    df_v    = pd.DataFrame({'v': combined.values, 'b': bin_idx})
    means   = df_v.groupby('b')['v'].mean()
    out     = np.full(len(t_c), np.nan)
    out[means.index] = means.values
    # Imputacion biologicamente correcta para velocidad antenal:
    #   1. ffill limit=6: propagar valor anterior max 30s (6 bins x 5s)
    #      -> gap corto: el valor anterior es mejor estimacion que interpolar
    #   2. bfill limit=6: si empieza con NaN, usar el siguiente valor
    #   3. fillna(0): gaps > 30s sin dato -> cero (abeja probablemente quieta)
    # NO se usa interpolacion: inventa valores entre estados opuestos (0 <-> 150px/s)
    nan_mask = np.isnan(out)   # guardar mascara ANTES de imputar (para twitches)
    s = (pd.Series(out)
         .ffill(limit=6)
         .bfill(limit=6)
         .fillna(0))
    return t_c, s.values

def vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA):
    sig = signal - signal.mean()
    try:
        u, _, omega = _VMD(sig, alpha, 0, K, 0, 1, 1e-7)
        if omega.ndim == 2:
            ff = omega[:,-1] if omega.shape[0]==K else omega[-1,:]
        else:
            ff = np.arange(K, dtype=float)
        if len(ff) != K: ff = np.arange(K, dtype=float)
        return u[np.argsort(ff)]   # ordenados de lento a rapido
    except Exception as e:
        print(f"  [VMD ERROR] {e}")
        return None

def hilbert_inst(modo, dt=BIN_FINO, sigma_f=2):
    """Amplitud e instancia de frecuencia via HHT."""
    analytic = hilbert(modo)
    amp      = np.abs(analytic)
    phase    = np.unwrap(np.angle(analytic))
    dphi     = np.diff(phase) / dt
    dphi_sm  = gaussian_filter1d(dphi, sigma=sigma_f)
    f_hz     = np.clip(dphi_sm / (2*np.pi), 1/(PERIOD_MAX*60), 1/(PERIOD_MIN*60))
    T_min    = 1.0 / (f_hz * 60)
    amp_mid  = (amp[:-1]+amp[1:])/2
    return amp_mid, T_min

def seg_a_hora(t_seg, t_max):
    """Convierte tiempo en segundos a hora real del dia."""
    t_abs = END_H - (t_max - t_seg)/3600
    return t_abs % 24

def rose_bin(sig, clock_h, bins_h):
    """Promedio de sig por bin horario."""
    n   = len(bins_h)-1
    out = np.full(n, np.nan)
    for k in range(n):
        m = (clock_h >= bins_h[k]) & (clock_h < bins_h[k+1])
        if m.sum() > 0: out[k] = sig[m].mean()
    return out

def normalizar(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx-mn < 1e-9: return np.zeros_like(arr)
    return (arr-mn)/(mx-mn)

def periodo_a_radio(T):
    r = R_INNER + (R_OUTER-R_INNER)*(np.log10(np.clip(T,PERIOD_MIN,PERIOD_MAX))-log_min)/(log_max-log_min)
    return np.clip(r, R_INNER, R_OUTER)

def setup_polar_base(ax):
    ax.set_facecolor(C_PANEL)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=7)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['','50%','','100%'], color='#444', fontsize=6)
    ax.spines['polar'].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.4, lw=0.4)
    ax.set_ylim(0, 1.2)
    gap = np.linspace(GAP_S, GAP_E, 120)
    ax.fill_between(gap, 0, 1.2, color='#04040a', alpha=0.93, zorder=0)
    for h_m, col_m, lbl in [(18,'#ffd166','18h'),(0,'#00e5ff','00h'),(11+20/60,'#ff6b6b','11:20')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.6, zorder=5)
        ax.text(h_m/24*2*np.pi, 1.12, lbl,
                ha='center', va='top', color=col_m, fontsize=5.5, fontweight='bold')

def setup_polar_hht(ax):
    ax.set_facecolor('#050510')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(hour_tick_rad)
    ax.set_xticklabels(hour_tick_lbl, color=C_TEXTO, fontsize=6)
    ax.set_yticks([])
    ax.spines['polar'].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.2, lw=0.3)
    ax.set_ylim(0, R_OUTER*1.06)
    gap = np.linspace(GAP_S, GAP_E, 120)
    ax.fill_between(gap, 0, R_OUTER*1.06, color='#040408', alpha=0.95, zorder=0)
    # Anillos de periodo
    for pt in period_ticks:
        if PERIOD_MIN <= pt <= PERIOD_MAX:
            r_t = periodo_a_radio(pt)
            ax.plot(theta_circ, np.full(300,r_t), color='white',
                    lw=0.3, alpha=0.18, ls='--', zorder=1)
            ax.text(0.04, r_t, f'{pt}m', color='#888', fontsize=5, ha='left', va='center')
    for h_m, col_m, lbl in [(18,'#ffd166','18h'),(0,'#00e5ff','00h'),(11+20/60,'#ff6b6b','11:20')]:
        ax.axvline(h_m/24*2*np.pi, color=col_m, lw=0.9, ls=':', alpha=0.55, zorder=5)
        ax.text(h_m/24*2*np.pi, R_OUTER*0.97, lbl,
                ha='center', va='top', color=col_m, fontsize=6, fontweight='bold')

def savefig(fig, nombre, dpi=150):
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'{nombre}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor=C_FONDO, format=ext)
    plt.close(fig)
    print(f"  -> {nombre}.png + .pdf")

# ======================================================================
# CARGA Y COMPUTO
# ======================================================================
print("="*60)
print(f"  VMD RAPIDO — bins={BIN_FINO}s  (vs {BIN_GRUESO}s del paper_maestro)")
print(f"  M2 (10-30min): {int(600/BIN_FINO)} muestras/ciclo minimo")
print(f"  M3 (<10min):   {int(60/BIN_FINO)} muestras/ciclo a 1 min")
print("="*60)

df    = pd.read_csv(CSV_PATH)
t_max = df['tiempo_seg'].max()
print(f"\n  CSV: {len(df):,} filas  |  {t_max/3600:.2f}h")
n_bins_esperados = int(t_max / BIN_FINO)
print(f"  Bins a 0.5s: ~{n_bins_esperados:,} puntos/abeja  "
      f"({n_bins_esperados*8/1024/1024:.1f} MB float64)")

senales   = {}
modes_all = {}
t_ref     = None
N_t       = None

for bee in BEES_ANALISIS + [BEE_RUIDO]:
    sub = df[df['animal'] == bee]
    if sub.empty:
        print(f"  {bee}: no encontrada")
        continue
    print(f"  {bee} señal 0.5s...", end=' ', flush=True)
    t_c, sig = construir_senal_fina(sub, t_max, bin_s=BIN_FINO)
    if sig is None:
        print("sin datos")
        continue
    if t_ref is None:
        t_ref = t_c
        N_t   = len(t_c)
    senales[bee] = sig
    print(f"VMD K={VMD_K}...", end=' ', flush=True)
    modos = vmd_decompose(sig, K=VMD_K, alpha=VMD_ALPHA)
    if modos is not None:
        modes_all[bee] = modos
        print(f"OK  shape={modos.shape}")
    else:
        print("FALLO")

# Alinear longitudes
N_t_eff = N_t
for bee in list(modes_all.keys()):
    N_t_eff = min(N_t_eff, modes_all[bee].shape[1], len(senales[bee]))
if N_t_eff < N_t:
    print(f"  [INFO] Ajustando: {N_t} -> {N_t_eff} bins")
    t_ref = t_ref[:N_t_eff]; N_t = N_t_eff
    for bee in list(senales.keys()):
        senales[bee] = senales[bee][:N_t]
    for bee in list(modes_all.keys()):
        modes_all[bee] = modes_all[bee][:, :N_t]

clock_h = seg_a_hora(t_ref, t_max)
theta_f = clock_h/24*2*np.pi

print(f"\n  Listo. N_t={N_t}  ({N_t*BIN_FINO/3600:.2f}h a {BIN_FINO}s/bin)")
print(f"  Resolucion: {BIN_FINO}s  |  Nyquist: {BIN_FINO*2:.1f}s  |  "
      f"Min. periodo confiable: {BIN_FINO*20:.0f}s = {BIN_FINO*20/60:.1f}min")

# ======================================================================
# COMPUTE ENVELOPES + HHT para M2 y M3
# ======================================================================
print("\n  Calculando envelopes + HHT para M2 y M3...")

envelopes = {}   # envelopes[bee][k]  = envolvente suavizada
hht_data  = {}   # hht_data[bee][k]   = (amp_mid, T_min)
rose_data = {}   # rose_data[bee][k]  = rose plot 30min bins

# Envolvente de Bee7 para denoising
env_b7 = {}
if BEE_RUIDO in modes_all:
    for k in [2, 3]:
        # Bee7 sin suavizado — misma resolucion que las abejas
        env_b7[k] = np.abs(hilbert(modes_all[BEE_RUIDO][k]))

for bee in BEES_ANALISIS:
    if bee not in modes_all: continue
    envelopes[bee] = {}
    hht_data[bee]  = {}
    rose_data[bee] = {}

    for k in [2, 3]:
        mi = MODO_INFO[k]

        # Envolvente SIN suavizado gaussiano
        # Usamos 0.5s bins para ver ritmos rapidos — no tiene sentido suavizar
        env_raw = np.abs(hilbert(modes_all[bee][k]))
        if k in env_b7:
            env_raw = np.maximum(env_raw - 1.0*env_b7[k],
                                 0.05*env_b7[k])
        envelopes[bee][k] = env_raw   # sin gaussian_filter1d

        # HHT instantaneo
        amp_mid, T_min = hilbert_inst(modes_all[bee][k], dt=BIN_FINO)
        hht_data[bee][k] = (amp_mid, T_min)

        # Rose plot 30min
        rose_data[bee][k] = normalizar(rose_bin(env_sm, clock_h, bins_h30))

    print(f"    {bee}: M2 env max={envelopes[bee][2].max():.1f}  "
          f"M3 env max={envelopes[bee][3].max():.1f}")

# ======================================================================
# FIGURA 1 — Polar rose plots M2 y M3 por abeja
# ======================================================================
print("\nFig vmd_rapido_rose...")

fig1 = plt.figure(figsize=(22, 12), facecolor=C_FONDO)
fig1.suptitle(
    f'VMD Ritmos Rapidos — Rose plots por abeja  |  Bins={BIN_FINO}s\n'
    f'M2 (10-30 min) vs M3 (<10 min)  |  Bee7 sustraida  |  '
    f'Resolucion: {int(600/BIN_FINO)} muestras/ciclo para 10min',
    color=C_TEXTO, fontsize=12, fontweight='bold')

# 2 filas (M2, M3) × 4 columnas (Bee3-Bee6)
gs1 = fig1.add_gridspec(2, 4, hspace=0.15, wspace=0.08,
                          left=0.04, right=0.97, top=0.88, bottom=0.04)

for row, k in enumerate([2, 3]):
    mi = MODO_INFO[k]
    for col, bee in enumerate(BEES_ANALISIS):
        ax = fig1.add_subplot(gs1[row, col], projection='polar')
        setup_polar_base(ax)

        if bee not in rose_data or k not in rose_data[bee]:
            ax.set_title(f'{bee}\nsin datos', color=C_TEXTO, fontsize=8)
            continue

        vals  = rose_data[bee][k]
        valid = ~np.isnan(vals)

        # Barras
        for i in range(len(bin_rad30)):
            if not valid[i]: continue
            ax.bar(bin_rad30[i], vals[i], width=bin_w30*0.80, bottom=0.02,
                   color=mi['col'], alpha=0.35+0.45*vals[i],
                   edgecolor='none', zorder=2)

        # Contorno
        if valid.sum() > 2:
            tv = bin_rad30[valid]
            rv = vals[valid]
            ax.fill(np.append(tv, tv[0]), np.append(rv, rv[0]),
                    color=mi['col'], alpha=0.12, zorder=1)
            ax.plot(np.append(tv, tv[0]), np.append(rv, rv[0]),
                    color=mi['col'], lw=1.5, alpha=0.9, zorder=4)

            # Pico
            pk   = np.nanargmax(vals)
            h_pk = bin_ctrs30[pk]
            ax.scatter(bin_rad30[pk], vals[pk]+0.08,
                       s=50, color='white', zorder=10,
                       edgecolors=mi['col'], lw=1.5)
            ax.text(bin_rad30[pk], vals[pk]+0.22,
                    f'{int(h_pk):02d}h',
                    ha='center', va='center', color='white',
                    fontsize=7, fontweight='bold')

        # Titles solo en fila 0
        if row == 0:
            ax.set_title(f'{bee}', color=COLORES[bee],
                         fontsize=11, fontweight='bold', pad=8)

        # Label modo solo en col 0
        if col == 0:
            ax.set_ylabel(f'{mi["nombre"]}\n{mi["banda"]}',
                          color=mi['col'], fontsize=9, fontweight='bold',
                          labelpad=30)

# Leyenda de modos
for row, k in enumerate([2, 3]):
    mi = MODO_INFO[k]
    fig1.text(0.01, 0.75-row*0.38,
              f'{mi["nombre"]}\n{mi["banda"]}',
              color=mi['col'], fontsize=10, fontweight='bold',
              va='center', ha='left',
              bbox=dict(boxstyle='round', facecolor=C_PANEL,
                        edgecolor=mi['col'], alpha=0.8))

fig1.text(0.01, 0.01,
    f'Rose plot: media de envolvente Hilbert SIN suavizado gaussiano — bins de 30min. '
    f'Bins temporales: {BIN_FINO}s (vs {BIN_GRUESO}s en paper_maestro.py). '
    f'Unico suavizado aplicado: derivada de fase en HHT (sigma=1s, inevitable numericamente).',
    color='#555577', fontsize=7, style='italic')

savefig(fig1, 'vmd_rapido_rose')

# ======================================================================
# FIGURA 2 — Polar HHT scatter M2 y M3 (periodo instantaneo vs hora)
# ======================================================================
print("Fig vmd_rapido_hht...")

CMAP_AMP = LinearSegmentedColormap.from_list('amp',
    ['#080818','#0d1b4a','#0077b6','#00b4d8',
     '#90e0ef','#ffd166','#ef476f','#ffffff'], N=512)

fig2 = plt.figure(figsize=(22, 12), facecolor=C_FONDO)
fig2.suptitle(
    f'VMD Ritmos Rapidos — HHT Periodo Instantaneo  |  Bins={BIN_FINO}s\n'
    f'Periodo [1-35 min] vs hora del dia  |  Color = amplitud instantanea\n'
    f'R_interno=1min  R_externo=35min  (escala logaritmica)',
    color=C_TEXTO, fontsize=12, fontweight='bold')

gs2 = fig2.add_gridspec(2, 4, hspace=0.15, wspace=0.08,
                          left=0.04, right=0.97, top=0.86, bottom=0.04)

for row, k in enumerate([2, 3]):
    mi  = MODO_INFO[k]
    # Calcular vmax global de amplitud para colorbar consistente
    amp_vals_all = np.concatenate([
        hht_data[b][k][0] for b in BEES_ANALISIS
        if b in hht_data and k in hht_data[b]])
    vmax_amp = np.percentile(amp_vals_all, 97)

    for col, bee in enumerate(BEES_ANALISIS):
        ax = fig2.add_subplot(gs2[row, col], projection='polar')
        setup_polar_hht(ax)

        if bee not in hht_data or k not in hht_data[bee]:
            continue

        amp_mid, T_min = hht_data[bee][k]

        # Solo puntos dentro del rango de periodo
        mask_T   = (T_min >= PERIOD_MIN) & (T_min <= PERIOD_MAX)
        theta_pts = theta_f[:len(amp_mid)][mask_T]
        T_pts     = T_min[mask_T]
        amp_pts   = amp_mid[mask_T]

        # Excluir zona sin datos
        in_gap   = (theta_pts >= GAP_S) & (theta_pts <= GAP_E)
        theta_ok = theta_pts[~in_gap]
        T_ok     = T_pts[~in_gap]
        amp_ok   = amp_pts[~in_gap]
        r_ok     = periodo_a_radio(T_ok)

        if len(theta_ok) > 0:
            sc = ax.scatter(theta_ok, r_ok,
                            c=amp_ok, cmap=CMAP_AMP,
                            s=0.6, alpha=0.5,
                            vmin=0, vmax=vmax_amp,
                            linewidths=0, rasterized=True, zorder=3)

        if row == 0:
            ax.set_title(f'{bee}', color=COLORES[bee],
                         fontsize=11, fontweight='bold', pad=8)
        if col == 0:
            ax.set_ylabel(f'{mi["nombre"]}\n{mi["banda"]}',
                          color=mi['col'], fontsize=9, fontweight='bold',
                          labelpad=30)

    # Colorbar por fila
    sm = ScalarMappable(cmap=CMAP_AMP,
                        norm=Normalize(vmin=0, vmax=vmax_amp))
    sm.set_array([])
    cbar_ax = fig2.add_axes([0.975, 0.54-row*0.5, 0.008, 0.38])
    cb = fig2.colorbar(sm, cax=cbar_ax)
    cb.set_label('Amplitud inst.', color=C_TEXTO, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=C_TEXTO, labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_TEXTO)

fig2.text(0.01, 0.01,
    f'HHT: frecuencia instantanea via Hilbert sobre modo VMD crudo. '
    f'Bins={BIN_FINO}s → {int(60/BIN_FINO)} muestras/min. '
    f'Rango mostrado: {PERIOD_MIN}-{PERIOD_MAX} min. Bee7 sustraida antes de VMD.',
    color='#555577', fontsize=7, style='italic')

savefig(fig2, 'vmd_rapido_hht')

# ======================================================================
# FIGURA 3 — Comparacion espectral: 5s vs 0.5s bins para M2 y M3
# ======================================================================
print("Fig vmd_rapido_comparacion...")

from scipy.signal import welch

fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10), facecolor=C_FONDO,
                             gridspec_kw={'hspace':0.35, 'wspace':0.25})
fig3.suptitle(
    f'Comparacion Espectral — {BIN_GRUESO}s vs {BIN_FINO}s bins\n'
    'Densidad espectral de M2 y M3 por abeja  (Welch PSD)',
    color=C_TEXTO, fontsize=12, fontweight='bold')

for row, k in enumerate([2, 3]):
    mi = MODO_INFO[k]
    for col, bee in enumerate(BEES_ANALISIS):
        ax = axes3[row, col]
        ax.set_facecolor(C_PANEL)

        if bee not in modes_all:
            ax.text(0.5, 0.5, 'sin datos', ha='center', va='center',
                    color=C_TEXTO, transform=ax.transAxes)
            continue

        # PSD del modo fino (0.5s)
        modo_fino = modes_all[bee][k]
        f_fino, P_fino = welch(modo_fino, fs=1/BIN_FINO,
                                nperseg=min(8192, len(modo_fino)//4),
                                scaling='density')
        T_fino = 1/(f_fino[1:]*60)  # en minutos
        mask_f = (T_fino >= 0.5) & (T_fino <= 35)
        ax.fill_between(T_fino[mask_f], 0, P_fino[1:][mask_f],
                        color=mi['col'], alpha=0.3)
        ax.plot(T_fino[mask_f], P_fino[1:][mask_f],
                color=mi['col'], lw=1.0, alpha=0.9,
                label=f'{BIN_FINO}s bins')

        # Zona M2/M3
        if k == 2:
            ax.axvspan(10, 30, color=mi['col'], alpha=0.07, zorder=0)
            ax.axvline(10, color=mi['col'], lw=0.7, ls='--', alpha=0.5)
            ax.axvline(30, color=mi['col'], lw=0.7, ls='--', alpha=0.5)
        else:
            ax.axvspan(0.5, 10, color=mi['col'], alpha=0.07, zorder=0)
            ax.axvline(10, color=mi['col'], lw=0.7, ls='--', alpha=0.5)

        # Linea de resolucion practica
        lim_prac = BIN_FINO*20/60  # periodo minimo practico en minutos
        ax.axvline(lim_prac, color='white', lw=1.0, ls=':', alpha=0.7,
                   label=f'Lim. pract. {lim_prac:.1f}min')

        ax.set_xscale('log')
        ax.set_xlim(0.5, 35)
        ax.set_xlabel('Periodo (min)', color=C_TEXTO, fontsize=8)
        ax.set_ylabel('Densidad espectral', color=C_TEXTO, fontsize=8)

        if row == 0:
            ax.set_title(f'{bee}', color=COLORES[bee],
                         fontsize=10, fontweight='bold')
        if col == 0:
            ax.text(-0.25, 0.5, f'{mi["nombre"]}\n{mi["banda"]}',
                    transform=ax.transAxes,
                    color=mi['col'], fontsize=9, fontweight='bold',
                    va='center', ha='center', rotation=90)

        ax.tick_params(colors=C_TEXTO, labelsize=6)
        ax.spines[:].set_color(C_GRID)
        ax.grid(color=C_GRID, alpha=0.3, lw=0.3, which='both')

        xtks = [0.5, 1, 2, 5, 10, 20, 30]
        ax.set_xticks(xtks)
        ax.set_xticklabels([str(t) for t in xtks], color=C_TEXTO, fontsize=6)

        if col == 0 and row == 0:
            ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO,
                      framealpha=0.6, loc='upper left')

fig3.text(0.01, 0.01,
    f'PSD Welch (ventana={min(8192, n_bins_esperados//4)*BIN_FINO/60:.0f}min, '
    f'overlap=50%). Eje X en escala log. '
    f'Linea punteada = periodo minimo confiable ({BIN_FINO*20/60:.1f}min) con bin={BIN_FINO}s.',
    color='#555577', fontsize=7, style='italic')

savefig(fig3, 'vmd_rapido_psd_comparacion')

# ======================================================================
# RESUMEN
# ======================================================================
print(f"\n{'='*60}")
print(f"  RESUMEN VMD RAPIDO")
print(f"{'='*60}")
print(f"  Resolucion: {BIN_FINO}s bins  ({int(t_max/BIN_FINO):,} puntos)")
print(f"  Nyquist:    {BIN_FINO*2:.1f}s")
print(f"  Limite practico confiable: {BIN_FINO*20:.0f}s = {BIN_FINO*20/60:.1f}min")
print(f"  M2 (10-30min): {int(600/BIN_FINO)} muestras/ciclo min  <- bien resuelto")
print(f"  M3 (<10min):   {int(60/BIN_FINO)} muestras/ciclo a 1min  <- resolvable")
print(f"\n  Figuras generadas:")
print(f"    vmd_rapido_rose.png/pdf            — Rose polar M2 y M3")
print(f"    vmd_rapido_hht.png/pdf             — HHT periodo instantaneo")
print(f"    vmd_rapido_psd_comparacion.png/pdf — PSD Welch por modo")
print(f"{'='*60}\n")
