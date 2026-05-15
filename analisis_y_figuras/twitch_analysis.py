# twitch_analysis.py — Comparacion de twitches en alta vs baja actividad
# Detecta eventos rapidos (twitches) en M2/M3 y los compara segun
# el estado de actividad lenta (M0) en ese momento.
#
# Hipotesis: si los twitches M2/M3 son distintos en alta vs baja actividad,
# entonces M2/M3 tiene contenido biologico, no es solo ruido.
#
# Genera 4 figuras:
#   twitch_01_threshold_distribucion   histograma M0 + threshold automatico
#   twitch_02_deteccion_temporal       serie temporal con twitches marcados
#   twitch_03_morfologia_comparada     forma de onda promedio twitch alta/baja
#   twitch_04_estadisticas             tasa, amplitud, duracion por estado

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
import os, warnings
warnings.filterwarnings('ignore')

try:
    from vmdpy import VMD as _VMD
    TIENE_VMD = True
except ImportError:
    print("[ERROR] pip install vmdpy"); exit()

# ======================================================================
# CONFIG
# ======================================================================
CSV_PATH   = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\poses_completo.csv'
OUTPUT_DIR = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\paper_definitivo\twitch_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_MINIMA   = 0.3
BIN_FINO      = 0.5    # 0.5s para ver twitches
BIN_GRUESO    = 5.0    # referencia
ANTENAS       = ['Antena_1_A','Antena_1_B','Antena_2_A','Antena_2_B']
BEES_ANALISIS = ['Bee3','Bee4','Bee5','Bee6']
BEE_RUIDO     = 'Bee7'

VMD_K     = 4
VMD_ALPHA = 2000

END_H      = 11 + 20/60
DURACION_S = 17*3600 + 25*60 + 13
START_REAL = END_H - DURACION_S/3600

# Parametros de deteccion de twitches
VENTANA_ESTADO_MIN   = 10.0   # minutos para estimar estado de actividad local
VENTANA_TWITCH_S     = 30.0   # segundos: ventana de contexto alrededor de cada twitch
TWITCH_UMBRAL_SIGMA  = 2.0    # twitches = picos > media + N*std de M2/M3
TWITCH_DISTANCIA_S   = 2.0    # separacion minima entre twitches (segundos)
PERCENTIL_ALTA       = 66.0   # percentil de M0 por encima del cual = alta actividad
PERCENTIL_BAJA       = 33.0   # percentil por debajo del cual = baja actividad
N_TWITCH_MAX         = 500    # maximo twitches a analizar por abeja/estado

# Estetica
C_FONDO = '#080818'; C_PANEL = '#0f0f28'
C_TEXTO = '#e0e0e0'; C_GRID  = '#1a1a3a'
COL_ALTA = '#ffd166'   # alta actividad
COL_BAJA = '#48dbfb'   # baja actividad
COLORES  = {'Bee3':'#48dbfb','Bee4':'#ff9ff3',
             'Bee5':'#54a0ff','Bee6':'#a29bfe','Bee7':'#888888'}

# ======================================================================
# FUNCIONES
# ======================================================================
def construir_senal(sub, t_max, bin_s=BIN_FINO):
    sub = sub.sort_values('tiempo_seg')
    segs = []
    for bp in ANTENAS:
        xc,yc,cc = f'{bp}_x',f'{bp}_y',f'{bp}_conf'
        if xc not in sub.columns: continue
        m  = sub[cc].values > CONF_MINIMA
        xs = sub.loc[sub.index[m], xc].values
        ys = sub.loc[sub.index[m], yc].values
        ts = sub.loc[sub.index[m],'tiempo_seg'].values
        if len(xs)<2: continue
        dist = np.sqrt(np.diff(xs)**2+np.diff(ys)**2)
        vel  = np.where(np.diff(ts)>0, dist/np.diff(ts), 0)
        segs.append(pd.Series(vel, index=(ts[:-1]+ts[1:])/2))
    if not segs: return None, None
    combined = pd.concat(segs,axis=1).mean(axis=1)
    bins    = np.arange(0, t_max+bin_s, bin_s)
    t_c     = bins[:-1]+bin_s/2
    idx     = np.clip(np.digitize(combined.index.values, bins)-1, 0, len(t_c)-1)
    df_v    = pd.DataFrame({'v':combined.values,'b':idx})
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
    return t_c, s.values, nan_mask

def vmd_decompose(signal):
    sig = signal - signal.mean()
    try:
        u,_,omega = _VMD(sig, VMD_ALPHA, 0, VMD_K, 0, 1, 1e-7)
        if omega.ndim==2:
            ff = omega[:,-1] if omega.shape[0]==VMD_K else omega[-1,:]
        else: ff=np.arange(VMD_K,dtype=float)
        if len(ff)!=VMD_K: ff=np.arange(VMD_K,dtype=float)
        return u[np.argsort(ff)]
    except Exception as e:
        print(f"  VMD error: {e}"); return None

def detectar_twitches(modo_rapido, bin_s=BIN_FINO,
                      umbral_sigma=TWITCH_UMBRAL_SIGMA,
                      distancia_s=TWITCH_DISTANCIA_S):
    """
    Detecta eventos (twitches) en un modo rapido.
    Un twitch = pico de amplitud instantanea > media + N*std,
    con distancia minima entre picos.
    Retorna indices de los picos.
    """
    amp = np.abs(hilbert(modo_rapido))
    mu  = np.median(amp)
    sd  = np.std(amp)
    umbral  = mu + umbral_sigma * sd
    dist_bins = max(1, int(distancia_s / bin_s))
    picos, props = find_peaks(amp,
                              height=umbral,
                              distance=dist_bins,
                              prominence=sd*0.5)
    return picos, amp, umbral

def extraer_forma_onda(modo, picos, ventana_s=VENTANA_TWITCH_S, bin_s=BIN_FINO):
    """
    Extrae la forma de onda de cada twitch (ventana centrada en el pico).
    Retorna array (n_twitches, n_bins_ventana).
    """
    w = int(ventana_s / bin_s)
    if w % 2 == 0: w += 1
    mitad = w // 2
    formas = []
    for p in picos:
        if p < mitad or p + mitad >= len(modo):
            continue
        seg = modo[p-mitad : p+mitad+1]
        # Normalizar por amplitud maxima
        mx = np.abs(seg).max()
        if mx > 1e-9:
            formas.append(seg / mx)
    return np.array(formas) if formas else np.zeros((0, w))

def savefig(fig, nombre, dpi=150):
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(OUTPUT_DIR,f'{nombre}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor=C_FONDO,format=ext)
    plt.close(fig)
    print(f"  -> {nombre}.png + .pdf")

def setup_ax(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXTO, labelsize=7)
    ax.spines[:].set_color(C_GRID)

# ======================================================================
# CARGA Y COMPUTO
# ======================================================================
print("="*60)
print("  TWITCH ANALYSIS — Alta vs Baja Actividad")
print(f"  Bins={BIN_FINO}s  |  Umbral twitches: media+{TWITCH_UMBRAL_SIGMA}*std")
print(f"  Estado ALTA: percentil >{PERCENTIL_ALTA}% de M0")
print(f"  Estado BAJA: percentil <{PERCENTIL_BAJA}% de M0")
print("="*60)

df    = pd.read_csv(CSV_PATH)
t_max = df['tiempo_seg'].max()
print(f"\n  CSV: {len(df):,} filas  |  {t_max/3600:.2f}h")

senales = {}; modes_all = {}; nan_masks = {}; t_ref = None; N_t = None

for bee in BEES_ANALISIS + [BEE_RUIDO]:
    sub = df[df['animal']==bee]
    if sub.empty: continue
    print(f"  {bee}...", end=' ', flush=True)
    t_c, sig, nan_mask_bee = construir_senal(sub, t_max)
    if sig is None: print("sin datos"); continue
    if t_ref is None: t_ref=t_c; N_t=len(t_c)
    senales[bee]=sig
    nan_masks[bee]=nan_mask_bee
    modos = vmd_decompose(sig)
    if modos is not None: modes_all[bee]=modos; print("OK")
    else: print("VMD fallo")

# Alinear
N_t_eff=N_t
for bee in list(modes_all.keys()):
    N_t_eff=min(N_t_eff,modes_all[bee].shape[1],len(senales[bee]))
if N_t_eff<N_t:
    t_ref=t_ref[:N_t_eff]; N_t=N_t_eff
    for bee in list(senales.keys()): senales[bee]=senales[bee][:N_t]
    for bee in list(modes_all.keys()): modes_all[bee]=modes_all[bee][:,:N_t]

clock_h = (END_H-(t_max-t_ref)/3600)%24
t_h     = START_REAL + np.arange(N_t)*BIN_FINO/3600  # hora real

win_estado = int(VENTANA_ESTADO_MIN*60/BIN_FINO)

print(f"\n  N_t={N_t}  ({N_t*BIN_FINO/3600:.2f}h a {BIN_FINO}s/bin)")

# ======================================================================
# ANALISIS PRINCIPAL POR ABEJA
# ======================================================================
resultados = {}   # resultados[bee] = dict con todo

for bee in BEES_ANALISIS:
    if bee not in modes_all: continue
    print(f"\n  Analizando {bee}...")

    m0 = modes_all[bee][0]   # modo lento  — define estado
    m2 = modes_all[bee][2]   # modo rapido — twitches
    m3 = modes_all[bee][3]   # modo muy rapido

    # Envolvente M0 suavizada — estado de actividad lento
    env_m0 = uniform_filter1d(np.abs(hilbert(m0)), size=win_estado)

    # Thresholds de estado
    th_alta = np.percentile(env_m0, PERCENTIL_ALTA)
    th_baja = np.percentile(env_m0, PERCENTIL_BAJA)
    mask_alta = env_m0 >= th_alta
    mask_baja = env_m0 <= th_baja
    mask_media= (~mask_alta) & (~mask_baja)

    print(f"    Alta actividad:  {mask_alta.mean()*100:.1f}% del tiempo  "
          f"(M0 env >= {th_alta:.1f})")
    print(f"    Baja actividad:  {mask_baja.mean()*100:.1f}% del tiempo  "
          f"(M0 env <= {th_baja:.1f})")

    # Deteccion de twitches en M2
    picos_m2, amp_m2, umbral_m2 = detectar_twitches(m2)
    # Deteccion de twitches en M3
    picos_m3, amp_m3, umbral_m3 = detectar_twitches(m3)

    # Clasificar cada twitch segun el estado en que ocurre
    def clasificar(picos, mask_a, mask_b):
        alta  = picos[mask_a[picos]]
        baja  = picos[mask_b[picos]]
        media = picos[(~mask_a[picos]) & (~mask_b[picos])]
        return alta, baja, media

    p2_alta, p2_baja, p2_med = clasificar(picos_m2, mask_alta, mask_baja)
    p3_alta, p3_baja, p3_med = clasificar(picos_m3, mask_alta, mask_baja)

    dur_alta_h = mask_alta.sum()*BIN_FINO/3600
    dur_baja_h = mask_baja.sum()*BIN_FINO/3600

    # Tasa de twitches (por hora)
    tasa_m2_alta = len(p2_alta)/max(dur_alta_h,0.01)
    tasa_m2_baja = len(p2_baja)/max(dur_baja_h,0.01)
    tasa_m3_alta = len(p3_alta)/max(dur_alta_h,0.01)
    tasa_m3_baja = len(p3_baja)/max(dur_baja_h,0.01)

    print(f"    Twitches M2: {len(p2_alta)} alta ({tasa_m2_alta:.0f}/h)  "
          f"| {len(p2_baja)} baja ({tasa_m2_baja:.0f}/h)  "
          f"| ratio={tasa_m2_baja/max(tasa_m2_alta,0.01):.2f}x")
    print(f"    Twitches M3: {len(p3_alta)} alta ({tasa_m3_alta:.0f}/h)  "
          f"| {len(p3_baja)} baja ({tasa_m3_baja:.0f}/h)  "
          f"| ratio={tasa_m3_baja/max(tasa_m3_alta,0.01):.2f}x")

    # Amplitud de twitches
    amp_m2_alta = amp_m2[p2_alta[:N_TWITCH_MAX]] if len(p2_alta) else np.array([])
    amp_m2_baja = amp_m2[p2_baja[:N_TWITCH_MAX]] if len(p2_baja) else np.array([])
    amp_m3_alta = amp_m3[p3_alta[:N_TWITCH_MAX]] if len(p3_alta) else np.array([])
    amp_m3_baja = amp_m3[p3_baja[:N_TWITCH_MAX]] if len(p3_baja) else np.array([])

    # Forma de onda promedio
    formas_m2_alta = extraer_forma_onda(m2, p2_alta[:N_TWITCH_MAX])
    formas_m2_baja = extraer_forma_onda(m2, p2_baja[:N_TWITCH_MAX])
    formas_m3_alta = extraer_forma_onda(m3, p3_alta[:N_TWITCH_MAX])
    formas_m3_baja = extraer_forma_onda(m3, p3_baja[:N_TWITCH_MAX])

    resultados[bee] = dict(
        env_m0=env_m0, m0=m0, m2=m2, m3=m3,
        amp_m2=amp_m2, amp_m3=amp_m3,
        umbral_m2=umbral_m2, umbral_m3=umbral_m3,
        th_alta=th_alta, th_baja=th_baja,
        mask_alta=mask_alta, mask_baja=mask_baja, mask_media=mask_media,
        p2_alta=p2_alta, p2_baja=p2_baja,
        p3_alta=p3_alta, p3_baja=p3_baja,
        tasa_m2_alta=tasa_m2_alta, tasa_m2_baja=tasa_m2_baja,
        tasa_m3_alta=tasa_m3_alta, tasa_m3_baja=tasa_m3_baja,
        amp_m2_alta=amp_m2_alta, amp_m2_baja=amp_m2_baja,
        amp_m3_alta=amp_m3_alta, amp_m3_baja=amp_m3_baja,
        formas_m2_alta=formas_m2_alta, formas_m2_baja=formas_m2_baja,
        formas_m3_alta=formas_m3_alta, formas_m3_baja=formas_m3_baja,
        dur_alta_h=dur_alta_h, dur_baja_h=dur_baja_h,
    )

# ======================================================================
# FIG 1 — Distribucion de M0 y threshold
# ======================================================================
print("\nFig twitch_01_threshold_distribucion...")
fig1, axes1 = plt.subplots(2, 4, figsize=(22, 10), facecolor=C_FONDO,
                            gridspec_kw={'hspace':0.35,'wspace':0.25})
fig1.suptitle(
    'Distribucion de Actividad M0 — Definicion de Estados Alta/Baja\n'
    f'Alta: percentil >{PERCENTIL_ALTA}%  |  Baja: percentil <{PERCENTIL_BAJA}%  |  '
    f'Ventana estado: {VENTANA_ESTADO_MIN:.0f} min',
    color=C_TEXTO, fontsize=12, fontweight='bold')

for col, bee in enumerate(BEES_ANALISIS):
    if bee not in resultados: continue
    r = resultados[bee]

    # Fila 0: histograma con thresholds
    ax = axes1[0, col]; setup_ax(ax)
    counts, edges, _ = ax.hist(r['env_m0'], bins=80,
                                color=COLORES[bee], alpha=0.6, edgecolor='none')
    ax.axvline(r['th_alta'], color=COL_ALTA, lw=2.0, ls='-',
               label=f'Alta >{PERCENTIL_ALTA}p: {r["th_alta"]:.1f}')
    ax.axvline(r['th_baja'], color=COL_BAJA, lw=2.0, ls='-',
               label=f'Baja <{PERCENTIL_BAJA}p: {r["th_baja"]:.1f}')
    ax.fill_betweenx([0, counts.max()], 0, r['th_baja'],
                     color=COL_BAJA, alpha=0.12)
    ax.fill_betweenx([0, counts.max()], r['th_alta'],
                     r['env_m0'].max()*1.05, color=COL_ALTA, alpha=0.12)
    ax.set_xlabel('Amplitud envolvente M0', color=C_TEXTO, fontsize=8)
    ax.set_ylabel('Frecuencia', color=C_TEXTO, fontsize=8)
    ax.set_title(f'{bee}', color=COLORES[bee], fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3, axis='y')

    # Fila 1: serie temporal M0 con estados sombreados (primeras 2h)
    ax2 = axes1[1, col]; setup_ax(ax2)
    n_show = min(N_t, int(2*3600/BIN_FINO))   # mostrar 2h
    t_show = t_h[:n_show]
    ax2.fill_between(t_show, 0, r['env_m0'][:n_show],
                     color=COLORES[bee], alpha=0.2)
    ax2.plot(t_show, r['env_m0'][:n_show],
             color=COLORES[bee], lw=0.6, alpha=0.9)
    ax2.axhline(r['th_alta'], color=COL_ALTA, lw=1.5, ls='--', alpha=0.8)
    ax2.axhline(r['th_baja'], color=COL_BAJA, lw=1.5, ls='--', alpha=0.8)
    ax2.fill_between(t_show[:n_show], 0,
                     r['env_m0'][:n_show]*r['mask_alta'][:n_show],
                     color=COL_ALTA, alpha=0.3, label='Alta')
    ax2.fill_between(t_show[:n_show], 0,
                     r['env_m0'][:n_show]*r['mask_baja'][:n_show],
                     color=COL_BAJA, alpha=0.3, label='Baja')
    ax2.set_xlabel('Hora del dia', color=C_TEXTO, fontsize=8)
    ax2.set_ylabel('Envolvente M0', color=C_TEXTO, fontsize=8)
    ax2.set_title(f'Serie temporal — primeras 2h', color=C_TEXTO, fontsize=8)
    ax2.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
    ax2.grid(color=C_GRID, alpha=0.3, lw=0.3)

savefig(fig1, 'twitch_01_threshold_distribucion')

# ======================================================================
# FIG 2 — Deteccion temporal de twitches (ventana de 30 min)
# ======================================================================
print("Fig twitch_02_deteccion_temporal...")
fig2 = plt.figure(figsize=(26, 16), facecolor=C_FONDO)
fig2.suptitle(
    'Deteccion de Twitches M2 y M3 por Estado de Actividad\n'
    f'Ventana mostrada: 30 min  |  '
    f'Rojo=twitch en alta actividad  Cyan=twitch en baja actividad',
    color=C_TEXTO, fontsize=12, fontweight='bold')

gs2 = GridSpec(4, 4, figure=fig2, hspace=0.45, wspace=0.25,
               left=0.05, right=0.97, top=0.90, bottom=0.05)

# Buscar una ventana de 30 min con transicion alta->baja para mostrar
N_SHOW_30MIN = int(30*60/BIN_FINO)

for col, bee in enumerate(BEES_ANALISIS):
    if bee not in resultados: continue
    r = resultados[bee]

    # Encontrar segmento interesante: donde haya twitches de ambos estados
    # Buscar zona con transicion
    cambios = np.diff(r['mask_baja'].astype(int))
    transiciones = np.where(np.abs(cambios)>0)[0]
    if len(transiciones) > 0:
        inicio = max(0, transiciones[len(transiciones)//2] - N_SHOW_30MIN//2)
    else:
        inicio = N_t//4
    fin = min(N_t, inicio + N_SHOW_30MIN)

    t_seg    = t_h[inicio:fin]
    m0_seg   = r['env_m0'][inicio:fin]
    m2_seg   = r['m2'][inicio:fin]
    m3_seg   = r['m3'][inicio:fin]
    amp2_seg = r['amp_m2'][inicio:fin]
    amp3_seg = r['amp_m3'][inicio:fin]

    for row, (modo_seg, amp_seg, picos_g, picos_alta, picos_baja,
              umbral, modo_nombre, modo_col) in enumerate([
        (m2_seg, amp2_seg,
         r['p2_alta'][(r['p2_alta']>=inicio)&(r['p2_alta']<fin)]-inicio,
         r['p2_alta'][(r['p2_alta']>=inicio)&(r['p2_alta']<fin)]-inicio,
         r['p2_baja'][(r['p2_baja']>=inicio)&(r['p2_baja']<fin)]-inicio,
         r['umbral_m2'], 'M2 rapido', '#ffd166'),
        (m3_seg, amp3_seg,
         r['p3_alta'][(r['p3_alta']>=inicio)&(r['p3_alta']<fin)]-inicio,
         r['p3_alta'][(r['p3_alta']>=inicio)&(r['p3_alta']<fin)]-inicio,
         r['p3_baja'][(r['p3_baja']>=inicio)&(r['p3_baja']<fin)]-inicio,
         r['umbral_m3'], 'M3 muy rap.', '#ef476f')]):

        ax = fig2.add_subplot(gs2[row*2, col])
        setup_ax(ax)

        # Estado de fondo
        ax.fill_between(t_seg, 0, m0_seg.max(),
                        where=r['mask_alta'][inicio:fin],
                        color=COL_ALTA, alpha=0.12, label='Alta act.')
        ax.fill_between(t_seg, 0, m0_seg.max(),
                        where=r['mask_baja'][inicio:fin],
                        color=COL_BAJA, alpha=0.12, label='Baja act.')

        # Modo VMD
        ax.plot(t_seg, modo_seg, color=modo_col, lw=0.4, alpha=0.7)
        ax.axhline(umbral, color='white', lw=0.8, ls=':', alpha=0.5,
                   label=f'Umbral twitch')
        ax.axhline(-umbral, color='white', lw=0.8, ls=':', alpha=0.5)

        # Twitches
        if len(picos_alta):
            ax.scatter(t_seg[picos_alta], modo_seg[picos_alta],
                       color=COL_ALTA, s=20, zorder=10,
                       edgecolors='white', lw=0.5, label=f'Twitch alta ({len(picos_alta)})')
        if len(picos_baja):
            ax.scatter(t_seg[picos_baja], modo_seg[picos_baja],
                       color=COL_BAJA, s=20, zorder=10,
                       edgecolors='white', lw=0.5, label=f'Twitch baja ({len(picos_baja)})')

        ax.set_ylabel(modo_nombre, color=modo_col, fontsize=8, fontweight='bold')
        ax.set_xlabel('Hora del dia', color=C_TEXTO, fontsize=7)
        if row == 0:
            ax.set_title(f'{bee}', color=COLORES[bee],
                         fontsize=10, fontweight='bold')
        ax.legend(fontsize=5, facecolor=C_PANEL, labelcolor=C_TEXTO,
                  framealpha=0.6, loc='upper right', ncol=2)
        ax.grid(color=C_GRID, alpha=0.3, lw=0.3)

        # Envolvente M0 en subplot inferior
        ax_m0 = fig2.add_subplot(gs2[row*2+1, col])
        setup_ax(ax_m0)
        ax_m0.fill_between(t_seg, 0, m0_seg, color='white', alpha=0.15)
        ax_m0.plot(t_seg, m0_seg, color='white', lw=0.8, alpha=0.8)
        ax_m0.axhline(r['th_alta'], color=COL_ALTA, lw=1.0, ls='--', alpha=0.7)
        ax_m0.axhline(r['th_baja'], color=COL_BAJA, lw=1.0, ls='--', alpha=0.7)
        ax_m0.set_ylabel('Envolvente M0\n(estado)', color='white', fontsize=7)
        ax_m0.set_xlabel('Hora del dia', color=C_TEXTO, fontsize=7)
        ax_m0.grid(color=C_GRID, alpha=0.3, lw=0.3)

savefig(fig2, 'twitch_02_deteccion_temporal')

# ======================================================================
# FIG 3 — Morfologia comparada de twitches
# ======================================================================
print("Fig twitch_03_morfologia_comparada...")
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10), facecolor=C_FONDO,
                            gridspec_kw={'hspace':0.40,'wspace':0.25})
fig3.suptitle(
    'Morfologia de Twitches — Alta vs Baja Actividad\n'
    'Forma de onda promedio ± std  |  Normalizada por amplitud maxima  |  '
    f'Ventana: {VENTANA_TWITCH_S:.0f}s centrada en el pico',
    color=C_TEXTO, fontsize=12, fontweight='bold')

t_waveform = (np.arange(int(VENTANA_TWITCH_S/BIN_FINO)+1) - int(VENTANA_TWITCH_S/BIN_FINO)//2)*BIN_FINO

for col, bee in enumerate(BEES_ANALISIS):
    if bee not in resultados: continue
    r = resultados[bee]

    for row, (formas_a, formas_b, modo_col, modo_nom, modo_idx) in enumerate([
        (r['formas_m2_alta'], r['formas_m2_baja'], '#ffd166', 'M2', 2),
        (r['formas_m3_alta'], r['formas_m3_baja'], '#ef476f', 'M3', 3)]):

        ax = axes3[row, col]; setup_ax(ax)

        for formas, col_est, lbl in [
            (formas_a, COL_ALTA, f'Alta (n={len(formas_a)})'),
            (formas_b, COL_BAJA, f'Baja (n={len(formas_b)})')]:
            if len(formas) == 0: continue
            n_w = min(formas.shape[1], len(t_waveform))
            mu  = formas[:, :n_w].mean(axis=0)
            sd  = formas[:, :n_w].std(axis=0)
            t_w = t_waveform[:n_w]
            ax.fill_between(t_w, mu-sd, mu+sd,
                            color=col_est, alpha=0.25)
            ax.plot(t_w, mu, color=col_est, lw=2.0,
                    label=lbl, alpha=0.9)

        ax.axvline(0, color='white', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='white', lw=0.4, ls='-', alpha=0.2)
        ax.set_xlabel('Tiempo desde pico (s)', color=C_TEXTO, fontsize=8)
        ax.set_ylabel('Amplitud norm.', color=C_TEXTO, fontsize=8)

        if row == 0:
            ax.set_title(f'{bee}', color=COLORES[bee],
                         fontsize=11, fontweight='bold')
        if col == 0:
            ax.text(-0.28, 0.5, modo_nom, transform=ax.transAxes,
                    color=modo_col, fontsize=11, fontweight='bold',
                    va='center', rotation=90)

        ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
        ax.grid(color=C_GRID, alpha=0.3, lw=0.3)

        # Estadistica: son distintas las formas?
        if len(formas_a) > 5 and len(formas_b) > 5:
            n_w = min(formas_a.shape[1], formas_b.shape[1], len(t_waveform))
            diff_norm = np.abs(formas_a[:,:n_w].mean(0) - formas_b[:,:n_w].mean(0)).mean()
            ax.text(0.98, 0.04,
                    f'Dif. media norm.: {diff_norm:.3f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    color='#aaaaaa', fontsize=7)

savefig(fig3, 'twitch_03_morfologia_comparada')

# ======================================================================
# FIG 4 — Estadisticas: tasa y amplitud alta vs baja
# ======================================================================
print("Fig twitch_04_estadisticas...")
fig4, axes4 = plt.subplots(2, 4, figsize=(22, 10), facecolor=C_FONDO,
                            gridspec_kw={'hspace':0.40,'wspace':0.30})
fig4.suptitle(
    'Estadisticas de Twitches — Alta vs Baja Actividad  |  M2 y M3\n'
    'Tasa (eventos/hora) y amplitud instantanea por estado',
    color=C_TEXTO, fontsize=12, fontweight='bold')

for col, bee in enumerate(BEES_ANALISIS):
    if bee not in resultados: continue
    r = resultados[bee]

    # Fila 0: tasa de twitches
    ax = axes4[0, col]; setup_ax(ax)
    labels_modo = ['M2\nrapido', 'M3\nmuy rap.']
    tasa_alta = [r['tasa_m2_alta'], r['tasa_m3_alta']]
    tasa_baja = [r['tasa_m2_baja'], r['tasa_m3_baja']]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x-w/2, tasa_alta, width=w, color=COL_ALTA,
                alpha=0.85, label='Alta actividad', edgecolor='none')
    b2 = ax.bar(x+w/2, tasa_baja, width=w, color=COL_BAJA,
                alpha=0.85, label='Baja actividad', edgecolor='none')
    for bar, v in zip(list(b1)+list(b2), tasa_alta+tasa_baja):
        ax.text(bar.get_x()+bar.get_width()/2, v+max(tasa_alta+tasa_baja)*0.01,
                f'{v:.0f}', ha='center', va='bottom',
                color='white', fontsize=8, fontweight='bold')
    # Ratio
    for i,(ta,tb) in enumerate(zip(tasa_alta,tasa_baja)):
        ratio = tb/max(ta,0.01)
        col_r = COL_BAJA if ratio>1.2 else COL_ALTA if ratio<0.8 else '#aaaaaa'
        ax.text(x[i], max(ta,tb)*1.08,
                f'{ratio:.1f}x', ha='center', color=col_r,
                fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels_modo, color=C_TEXTO, fontsize=9)
    ax.set_ylabel('Tasa (twitches/hora)', color=C_TEXTO, fontsize=8)
    ax.set_title(f'{bee}', color=COLORES[bee], fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3, axis='y')

    # Fila 1: distribucion de amplitudes (boxplot)
    ax2 = axes4[1, col]; setup_ax(ax2)
    data_box = [r['amp_m2_alta'], r['amp_m2_baja'],
                r['amp_m3_alta'], r['amp_m3_baja']]
    cols_box  = [COL_ALTA, COL_BAJA, COL_ALTA, COL_BAJA]
    labels_box = ['M2\nAlta', 'M2\nBaja', 'M3\nAlta', 'M3\nBaja']
    bp = ax2.boxplot([d for d in data_box if len(d)>0],
                     patch_artist=True,
                     medianprops=dict(color='white', lw=2),
                     whiskerprops=dict(color=C_TEXTO),
                     capprops=dict(color=C_TEXTO),
                     flierprops=dict(marker='.', color=C_TEXTO, markersize=2))
    valid_idx = [i for i,d in enumerate(data_box) if len(d)>0]
    for patch, i in zip(bp['boxes'], valid_idx):
        patch.set_facecolor(cols_box[i]); patch.set_alpha(0.7)
    ax2.set_xticks(range(1, len(valid_idx)+1))
    ax2.set_xticklabels([labels_box[i] for i in valid_idx],
                        color=C_TEXTO, fontsize=8)
    ax2.set_ylabel('Amplitud instantanea twitch', color=C_TEXTO, fontsize=8)
    ax2.grid(color=C_GRID, alpha=0.3, lw=0.3, axis='y')

# Anotacion global
fig4.text(0.01, 0.01,
    f'Umbral twitch: media + {TWITCH_UMBRAL_SIGMA}*std de amplitud Hilbert del modo. '
    f'Ratio >1 = mas twitches en baja actividad (posible firma de sueno). '
    f'Ratio = tasa_baja / tasa_alta.',
    color='#555577', fontsize=7, style='italic')

savefig(fig4, 'twitch_04_estadisticas')

# ======================================================================
# RESUMEN
# ======================================================================
print(f"\n{'='*60}")
print("  RESUMEN TWITCH ANALYSIS")
print("="*60)
print(f"  {'Abeja':6} {'M2 ratio':>10} {'M3 ratio':>10}  Interpretacion")
print(f"  {'─'*50}")
for bee in BEES_ANALISIS:
    if bee not in resultados: continue
    r = resultados[bee]
    r2 = r['tasa_m2_baja']/max(r['tasa_m2_alta'],0.01)
    r3 = r['tasa_m3_baja']/max(r['tasa_m3_alta'],0.01)
    interp = ('+ twitches en BAJA act.' if r2>1.2 and r3>1.2
              else '+ twitches en ALTA act.' if r2<0.8 and r3<0.8
              else 'sin diferencia clara')
    print(f"  {bee:6} {r2:>10.2f}x {r3:>10.2f}x  {interp}")
print()
print("  Ratio > 1.0 = mas twitches cuando la abeja esta inactiva")
print("  Ratio < 1.0 = mas twitches cuando la abeja esta activa")
print("  Si ratio >> 1 consistentemente: twitches son firma de sueno")
print(f"{'='*60}\n")

# ======================================================================
# FIG 5 — AUTOCORRELACION DE ESTADOS
# ======================================================================
# Para cada abeja:
#   - Autocorrelacion de la envolvente M0 (senal continua)
#   - Autocorrelacion del estado binario alta/baja
#   - Tiempo caracteristico de persistencia (decay 1/e)
#   - Picos secundarios = ciclicidad en las transiciones de estado
#
# Ademas: cross-correlacion entre abejas para ver si
# el estado de una predice el estado de otra con algun lag.
# ======================================================================
print("Fig twitch_05_autocorrelacion...")

from scipy.signal import correlate, correlation_lags

LAG_MAX_MIN = 120   # maximo lag a mostrar en minutos
LAG_MAX_BINS = int(LAG_MAX_MIN * 60 / BIN_FINO)

def autocorr_norm(signal, max_lag):
    """
    Autocorrelacion normalizada (coeficiente de Pearson) hasta max_lag.
    Retorna (lags_s, acorr).
    """
    s = signal - signal.mean()
    n = len(s)
    # Correlacion completa via FFT
    full = correlate(s, s, mode='full')
    lags = correlation_lags(n, n, mode='full')
    # Solo lags positivos hasta max_lag
    mask = (lags >= 0) & (lags <= max_lag)
    ac   = full[mask] / full[full.size//2]   # normalizar por lag=0
    lg   = lags[mask] * BIN_FINO             # convertir a segundos
    return lg, ac

def tiempo_decaimiento(lags_s, acorr, nivel=1/np.e):
    """Tiempo al que la autocorrelacion cae a 1/e."""
    idx = np.where(acorr <= nivel)[0]
    if len(idx) == 0:
        return lags_s[-1]   # no decae en el rango
    return lags_s[idx[0]]

def encontrar_picos_periodicos(lags_s, acorr, min_lag_min=5):
    """Picos secundarios en la autocorrelacion (posible periodicidad)."""
    min_dist = int(min_lag_min*60 / (lags_s[1]-lags_s[0])) if len(lags_s)>1 else 10
    # Solo buscar en lags > 5 min
    mask = lags_s > min_lag_min*60
    if not mask.any():
        return np.array([]), np.array([])
    ac_sub = acorr[mask]
    ls_sub = lags_s[mask]
    pks, props = find_peaks(ac_sub, height=0.05, distance=min_dist,
                             prominence=0.03)
    if len(pks):
        orden = np.argsort(ac_sub[pks])[::-1][:4]
        return ls_sub[pks[orden]]/60, ac_sub[pks[orden]]   # en minutos
    return np.array([]), np.array([])

# Calcular autocorrelaciones
acorr_data = {}   # acorr_data[bee] = dict
for bee in BEES_ANALISIS:
    if bee not in resultados: continue
    r = resultados[bee]

    # Autocorrelacion M0 envolvente (continua)
    lags_m0, ac_m0 = autocorr_norm(r['env_m0'], LAG_MAX_BINS)
    tau_m0 = tiempo_decaimiento(lags_m0, ac_m0)
    pks_min_m0, pks_amp_m0 = encontrar_picos_periodicos(lags_m0, ac_m0)

    # Autocorrelacion estado ALTA (binario)
    sig_alta = r['mask_alta'].astype(float)
    lags_alta, ac_alta = autocorr_norm(sig_alta, LAG_MAX_BINS)
    tau_alta = tiempo_decaimiento(lags_alta, ac_alta)

    # Autocorrelacion estado BAJA (binario)
    sig_baja = r['mask_baja'].astype(float)
    lags_baja, ac_baja = autocorr_norm(sig_baja, LAG_MAX_BINS)
    tau_baja = tiempo_decaimiento(lags_baja, ac_baja)

    # Autocorrelacion de la tasa de twitches M2
    # Construir serie de tasa de twitches (1 si hay twitch, 0 si no)
    twitch_m2_serie = np.zeros(N_t)
    if len(r['p2_alta']) > 0:
        twitch_m2_serie[r['p2_alta']] = 1
    if len(r['p2_baja']) > 0:
        twitch_m2_serie[r['p2_baja']] = 1
    # Suavizar con ventana de 5 min para obtener tasa local
    tasa_local_m2 = uniform_filter1d(twitch_m2_serie.astype(float),
                                      size=int(5*60/BIN_FINO))
    lags_tw, ac_tw = autocorr_norm(tasa_local_m2, LAG_MAX_BINS)
    pks_min_tw, pks_amp_tw = encontrar_picos_periodicos(lags_tw, ac_tw)

    acorr_data[bee] = dict(
        lags_m0=lags_m0/60, ac_m0=ac_m0,        # lags en minutos
        lags_alta=lags_alta/60, ac_alta=ac_alta,
        lags_baja=lags_baja/60, ac_baja=ac_baja,
        lags_tw=lags_tw/60, ac_tw=ac_tw,
        tau_m0_min=tau_m0/60, tau_alta_min=tau_alta/60, tau_baja_min=tau_baja/60,
        pks_min_m0=pks_min_m0, pks_amp_m0=pks_amp_m0,
        pks_min_tw=pks_min_tw, pks_amp_tw=pks_amp_tw,
    )
    picos_str_print = str([round(p,0) for p in pks_min_m0[:3]])
    print(f"  {bee}: tau_M0={tau_m0/60:.1f}min  "
          f"tau_alta={tau_alta/60:.1f}min  "
          f"tau_baja={tau_baja/60:.1f}min  "
          f"picos_M0={picos_str_print}")
# ---- Cross-correlacion entre abejas --------------------------------
print("  Calculando cross-correlacion entre abejas...")
pares = [(BEES_ANALISIS[i], BEES_ANALISIS[j])
         for i in range(len(BEES_ANALISIS))
         for j in range(i+1, len(BEES_ANALISIS))]

xcorr_data = {}   # xcorr_data[(b1,b2)] = (lags_min, xcorr)
LAG_X_MIN  = 60   # max lag cross-corr en minutos
LAG_X_BINS = int(LAG_X_MIN*60/BIN_FINO)

PARES_COLS = {
    ('Bee3','Bee4'):'#ff6b6b', ('Bee3','Bee5'):'#ffd166',
    ('Bee3','Bee6'):'#06d6a0', ('Bee4','Bee5'):'#48dbfb',
    ('Bee4','Bee6'):'#a29bfe', ('Bee5','Bee6'):'#ff9ff3',
}

for b1, b2 in pares:
    if b1 not in resultados or b2 not in resultados: continue
    s1 = resultados[b1]['env_m0'] - resultados[b1]['env_m0'].mean()
    s2 = resultados[b2]['env_m0'] - resultados[b2]['env_m0'].mean()
    n  = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    full = correlate(s1, s2, mode='full')
    lags = correlation_lags(n, n, mode='full')
    norm = np.sqrt(np.sum(s1**2)*np.sum(s2**2))
    mask = (lags >= -LAG_X_BINS) & (lags <= LAG_X_BINS)
    xc   = full[mask] / norm
    lg   = lags[mask] * BIN_FINO / 60   # minutos
    xcorr_data[(b1,b2)] = (lg, xc)
    lag_max_idx = np.argmax(np.abs(xc))
    print(f"  Cross-corr {b1}-{b2}: max={xc[lag_max_idx]:.3f} "
          f"en lag={lg[lag_max_idx]:.1f}min")

# ---- Figura --------------------------------------------------------
fig5 = plt.figure(figsize=(26, 18), facecolor=C_FONDO)
fig5.suptitle(
    'Autocorrelacion y Cross-Correlacion de Estados  |  Bee3-Bee4-Bee5-Bee6\n'
    'Tiempo de persistencia de cada estado y ciclicidad en las transiciones',
    color=C_TEXTO, fontsize=13, fontweight='bold')

gs5 = GridSpec(3, 4, figure=fig5,
               hspace=0.42, wspace=0.28,
               left=0.06, right=0.97, top=0.90, bottom=0.06)

# Fila 0: Autocorrelacion M0 por abeja
for col, bee in enumerate(BEES_ANALISIS):
    if bee not in acorr_data: continue
    d  = acorr_data[bee]
    ax = fig5.add_subplot(gs5[0, col]); setup_ax(ax)

    # M0 continuo
    ax.plot(d['lags_m0'], d['ac_m0'],
            color=COLORES[bee], lw=1.2, alpha=0.9, label='M0 (continuo)')
    # Estado alta
    ax.plot(d['lags_alta'], d['ac_alta'],
            color=COL_ALTA, lw=0.9, alpha=0.7, ls='--', label='Estado alta')
    # Estado baja
    ax.plot(d['lags_baja'], d['ac_baja'],
            color=COL_BAJA, lw=0.9, alpha=0.7, ls='--', label='Estado baja')

    # Nivel 1/e
    ax.axhline(1/np.e, color='white', lw=0.8, ls=':', alpha=0.5,
               label=f'1/e ({1/np.e:.2f})')
    ax.axhline(0, color='white', lw=0.4, alpha=0.2)

    # Marcar tau de M0
    ax.axvline(d['tau_m0_min'], color=COLORES[bee], lw=1.0, ls='--', alpha=0.6)
    ax.text(d['tau_m0_min']+0.5, 0.92,
            f'tau={d["tau_m0_min"]:.1f}m',
            color=COLORES[bee], fontsize=7, fontweight='bold')

    # Picos periodicos
    for p, a in zip(d['pks_min_m0'], d['pks_amp_m0']):
        ax.scatter(p, a, color='white', s=40, zorder=10,
                   edgecolors=COLORES[bee], lw=1.5)
        ax.annotate(f'{p:.0f}m',
                    xy=(p, a), xytext=(p+1, a+0.04),
                    color='white', fontsize=7, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.6))

    ax.set_xlim(0, LAG_MAX_MIN)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel('Lag (min)', color=C_TEXTO, fontsize=8)
    ax.set_ylabel('Autocorr. (Pearson)', color=C_TEXTO, fontsize=8)
    ax.set_title(f'{bee}\nAutocorr. M0 — persistencia de estado',
                 color=COLORES[bee], fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO,
              framealpha=0.6, loc='upper right')
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3)

    # Banda de confianza 95%
    ci = 1.96/np.sqrt(N_t)
    ax.fill_between(d['lags_m0'], -ci, ci,
                    color='white', alpha=0.07, label='IC 95%')

# Fila 1: Autocorrelacion de tasa de twitches
for col, bee in enumerate(BEES_ANALISIS):
    if bee not in acorr_data: continue
    d  = acorr_data[bee]
    ax = fig5.add_subplot(gs5[1, col]); setup_ax(ax)

    ax.fill_between(d['lags_tw'], 0, d['ac_tw'],
                    color='#ef476f', alpha=0.25)
    ax.plot(d['lags_tw'], d['ac_tw'],
            color='#ef476f', lw=1.2, alpha=0.9, label='Tasa twitches M2')
    ax.axhline(1/np.e, color='white', lw=0.8, ls=':', alpha=0.5)
    ax.axhline(0, color='white', lw=0.4, alpha=0.2)

    # Picos de periodicidad
    for p, a in zip(d['pks_min_tw'], d['pks_amp_tw']):
        ax.scatter(p, a, color='white', s=40, zorder=10,
                   edgecolors='#ef476f', lw=1.5)
        ax.annotate(f'{p:.0f}m',
                    xy=(p, a), xytext=(p+1, a+0.03),
                    color='white', fontsize=7, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='white', lw=0.6))

    # IC 95%
    ci_t = 1.96/np.sqrt(N_t)
    ax.fill_between(d['lags_tw'], -ci_t, ci_t,
                    color='white', alpha=0.07)

    ax.set_xlim(0, LAG_MAX_MIN)
    ax.set_xlabel('Lag (min)', color=C_TEXTO, fontsize=8)
    ax.set_ylabel('Autocorr. tasa twitches', color=C_TEXTO, fontsize=8)
    ax.set_title(f'{bee}\nAutocorr. tasa twitches M2\n'
                 f'(picos = periodicidad en eventos)',
                 color='#ef476f', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3)

# Fila 2: Cross-correlacion entre pares de abejas (M0)
ax_xc = fig5.add_subplot(gs5[2, :2]); setup_ax(ax_xc)

for (b1,b2),(lg,xc) in xcorr_data.items():
    col_par = PARES_COLS.get((b1,b2), 'white')
    ax_xc.plot(lg, xc, color=col_par, lw=1.0, alpha=0.85,
               label=f'{b1}-{b2}')

ax_xc.axhline(0, color='white', lw=0.4, alpha=0.2)
ax_xc.axvline(0, color='white', lw=0.8, ls='--', alpha=0.4,
              label='Lag=0 (simultaneo)')

# IC 95%
ci_x = 1.96/np.sqrt(N_t)
ax_xc.fill_between(lg, -ci_x, ci_x, color='white', alpha=0.07, label='IC 95%')

ax_xc.set_xlim(-LAG_X_MIN, LAG_X_MIN)
ax_xc.set_xlabel('Lag (min)  [negativo = A predice B]', color=C_TEXTO, fontsize=9)
ax_xc.set_ylabel('Cross-corr. M0 (Pearson)', color=C_TEXTO, fontsize=9)
ax_xc.set_title(
    'Cross-correlacion entre pares de abejas — Envolvente M0\n'
    'Pico en lag=0: sincronia instantanea  |  '
    'Pico en lag>0: una abeja lidera a la otra',
    color=C_TEXTO, fontsize=10, fontweight='bold')
ax_xc.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXTO,
             framealpha=0.7, ncol=2)
ax_xc.grid(color=C_GRID, alpha=0.3, lw=0.3)

# Tabla resumen de tiempos caracteristicos
ax_tab = fig5.add_subplot(gs5[2, 2:]); setup_ax(ax_tab)
ax_tab.axis('off')
tabla = ['TIEMPOS DE PERSISTENCIA DE ESTADO (tau @ 1/e)\n',
         f"{'Abeja':6}  {'tau M0':>10}  {'tau Alta':>10}  {'tau Baja':>10}  {'Picos M0':>20}"]
tabla.append('─'*62)
for bee in BEES_ANALISIS:
    if bee not in acorr_data: continue
    d = acorr_data[bee]
    picos_str = ', '.join([f'{p:.0f}min' for p in d['pks_min_m0'][:3]]) or 'ninguno'
    tabla.append(
        f"{bee:6}  {d['tau_m0_min']:>8.1f}m  "
        f"{d['tau_alta_min']:>8.1f}m  "
        f"{d['tau_baja_min']:>8.1f}m  "
        f"{picos_str}")
tabla.append('')
tabla.append('tau = tiempo al que la autocorrelacion cae a 1/e')
tabla.append('Picos M0 = periodos de ciclicidad detectados')
tabla.append('')
tabla.append('Cross-correlacion — lag del maximo |r|:')
for (b1,b2),(lg,xc) in xcorr_data.items():
    idx = np.argmax(np.abs(xc))
    tabla.append(f'  {b1}-{b2}: r={xc[idx]:.3f} en lag={lg[idx]:.1f}min')

ax_tab.text(0.03, 0.97, '\n'.join(tabla),
            transform=ax_tab.transAxes, va='top', ha='left',
            fontsize=8, color=C_TEXTO, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#0a0a1e',
                      edgecolor='#2a2a4a', alpha=0.9))
ax_tab.set_title('Resumen cuantitativo', color=C_TEXTO, fontsize=9, fontweight='bold')

fig5.text(0.01, 0.01,
    f'Autocorr: Pearson normalizada. IC 95% = ±1.96/sqrt(N). '
    f'Lag max = {LAG_MAX_MIN}min. '
    f'Picos secundarios en autocorr. = ciclicidad en las transiciones de estado. '
    f'Cross-corr lag=0 = sincronia entre abejas.',
    color='#555577', fontsize=7, style='italic')

savefig(fig5, 'twitch_05_autocorrelacion')

# Agregar al resumen
print(f"\n{'='*60}")
print("  RESUMEN AUTOCORRELACION")
print("="*60)
print(f"  {'Abeja':6}  {'tau M0':>8}  {'tau Alta':>9}  {'tau Baja':>9}  Periodicidad")
print(f"  {'─'*60}")
for bee in BEES_ANALISIS:
    if bee not in acorr_data: continue
    d = acorr_data[bee]
    picos = ', '.join([f'{p:.0f}m' for p in d['pks_min_m0'][:2]]) or 'no detectada'
    print(f"  {bee:6}  {d['tau_m0_min']:>7.1f}m  "
          f"{d['tau_alta_min']:>8.1f}m  "
          f"{d['tau_baja_min']:>8.1f}m  {picos}")
print()
print("  Cross-correlacion (lag del maximo):")
for (b1,b2),(lg,xc) in xcorr_data.items():
    idx = np.argmax(np.abs(xc))
    sinc = 'SINCRONICAS' if abs(lg[idx])<2 else f'{b1} lidera {abs(lg[idx]):.0f}min' if lg[idx]<0 else f'{b2} lidera {abs(lg[idx]):.0f}min'
    print(f"    {b1}-{b2}: r={xc[idx]:.3f}  lag={lg[idx]:.1f}min  -> {sinc}")
print(f"{'='*60}\n")
