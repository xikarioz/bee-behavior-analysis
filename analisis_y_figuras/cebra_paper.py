# cebra_paper.py — Pipeline CEBRA completo para abejas en cepo
# Script unificado y reproducible para paper cientifico
#
# Figuras generadas (25-34):
#   25  CEBRA-Time embedding (3D + estados + latente temporal)
#   26  Grid embeddings por hipotesis (hora/modo/angulo/sueno)
#   27  Ranking InfoNCE — jerarquia de hipotesis
#   28  Divergencia inter-abeja (sliding window)
#   29  Estados comportamentales (k-means)
#   30  Series temporales de latentes
#   31  Toro: geometria + PSDs Welch
#   32  Lomb-Scargle + CWT Morlet sobre latentes
#   33  TDA — Homologia Persistente
#   34  Kuramoto topologico + Wasserstein + Persistence Landscape

# ─────────────── IMPORTS ───────────────────────────────────────────
import os, warnings, random
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.signal import hilbert, butter, filtfilt, welch, find_peaks
from scipy.signal import detrend as sp_detrend, fftconvolve
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans

try:
    import cebra
    from cebra import CEBRA
except ImportError:
    print('[ERROR] pip install cebra'); exit()

try:
    from vmdpy import VMD as _VMD
    TIENE_VMD = True
except ImportError:
    TIENE_VMD = False
    print('[AVISO] vmdpy no instalado - usando filtros de banda')

try:
    from ripser import ripser
    from persim import PersistenceImager, wasserstein, plot_diagrams
    TIENE_TDA = True
except ImportError:
    TIENE_TDA = False
    print('[AVISO] pip install ripser persim')

try:
    from astropy.timeseries import LombScargle
    TIENE_LS = True
except ImportError:
    TIENE_LS = False
    print('[AVISO] pip install astropy')

try:
    import torch
    TIENE_TORCH = True
except ImportError:
    TIENE_TORCH = False

# ─────────────── SEMILLAS — REPRODUCIBILIDAD TOTAL ─────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TIENE_TORCH:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# ─────────────── CONFIG ────────────────────────────────────────────
CSV_PATH   = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\poses_completo.csv'
OUTPUT_DIR = r'C:\Users\franco\Downloads\francoabeja\papafrita\abejas_model\paper_definitivo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_MINIMA   = 0.3
BIN_SEGUNDOS  = 5
VMD_K         = 4
VMD_ALPHA     = 2000
ALPHA_SUB     = 1.0
BETA_FLOOR    = 0.05
SIGMA_ENV_VIZ = 6
END_H         = 11 + 20/60
DURACION_S    = 17*3600 + 25*60 + 13
START_REAL    = END_H - DURACION_S/3600

BEES_CEBRA  = ['Bee3','Bee4','Bee5','Bee6']
ANTENAS     = ['Antena_1_A','Antena_1_B','Antena_2_A','Antena_2_B']
COL_OCELO   = 'Posicion'

CEBRA_DIM   = 3
MODOS_VALIDOS = [0, 1]   # Solo M0 (>120min) y M1 (30-120min): resolubles con bin=5s
CEBRA_ITER  = 10000
CEBRA_BATCH = 512
CEBRA_LR    = 3e-4
CEBRA_OFFS  = 72
N_ESTADOS   = 3
WIN_DIV     = 360
SUB_TDA     = 800    # reducido para evitar explosion de simplices en ripser
W0_CWT      = 6.0

C_FONDO = '#080818'; C_PANEL = '#0f0f28'
C_TEXTO = '#e0e0e0'; C_GRID  = '#1a1a3a'
COLORES = {'Bee3':'#48dbfb','Bee4':'#ff9ff3',
           'Bee5':'#54a0ff','Bee6':'#a29bfe','Bee7':'#888888'}
COLS_LAT    = ['#ef476f','#ffd166','#06d6a0']
ESTADO_COLS = ['#ef476f','#ffd166','#06d6a0']
ESTADO_NOMS = ['Activo','Transicion','Reposo']
COL_ALTA = '#ffd166'   # alta actividad
COL_BAJA = '#48dbfb'   # baja actividad
VMD_BANDAS  = [
    ('>120 min',   120, 9999, '#00b4d8', 'M0'),
    ('30-120 min',  30,  120, '#a29bfe', 'M1'),
    ('10-30 min',   10,   30, '#ffd166', 'M2'),
    ('<10 min',      2,   10, '#ef476f', 'M3'),
]
MODELOS_META = {
    'time'  : dict(nombre='CEBRA-Time',         color='#aaaaaa', label_desc='sin supervision'),
    'hora'  : dict(nombre='Hora del dia',        color='#00b4d8', label_desc='(sin,cos) 24h'),
    'modo'  : dict(nombre='Modo VMD dominante',  color='#a29bfe', label_desc='argmax norm.'),
    'angulo': dict(nombre='Angulo antenal',       color='#ffd166', label_desc='escapo medio'),
    'sueno' : dict(nombre='Estado activo/reposo',color='#06d6a0', label_desc='umbral p25'),
}

# ─────────────── FUNCIONES DE SENAL ───────────────────────────────
def construir_senal(sub, t_max, bin_s=BIN_SEGUNDOS):
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
    bins = np.arange(0, t_max+bin_s, bin_s)
    t_c  = bins[:-1]+bin_s/2
    out  = np.full(len(t_c), np.nan)
    for i,tc in enumerate(t_c):
        mask = (combined.index>=tc-bin_s/2)&(combined.index<tc+bin_s/2)
        if mask.sum()>0: out[i]=combined[mask].mean()
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

def _fallback_bands(signal, K=VMD_K):
    fs_s = 1.0/BIN_SEGUNDOS
    pb   = np.logspace(np.log10(300),np.log10(7200),K-1)
    modes,prev = [],np.zeros_like(signal)
    for p in pb:
        fn = np.clip((1/p)/(fs_s/2),1e-6,0.9999)
        b,a = butter(4,fn,btype='low')
        lo  = filtfilt(b,a,signal)
        modes.append(lo-prev); prev=lo.copy()
    modes.append(signal-prev)
    return np.array(modes[::-1])

def vmd_decompose(signal, K=VMD_K, alpha=VMD_ALPHA):
    if not TIENE_VMD: return _fallback_bands(signal,K)
    sig = signal-signal.mean()
    try:
        u,_,omega = _VMD(sig,alpha,0,K,0,1,1e-7)
        ff = omega[:,-1] if omega.ndim==2 and omega.shape[0]==K else \
             omega[-1,:] if omega.ndim==2 else np.arange(K,dtype=float)
        if len(ff)!=K: ff=np.arange(K,dtype=float)
        return u[np.argsort(ff)]
    except Exception as e:
        print(f'  VMD ({e}) -> fallback')
        return _fallback_bands(signal,K)

def seg_a_hora_real(t): return START_REAL + t/3600.0
def hora_real_label(h):
    hm=h%24; return f'{int(hm):02d}:{int((hm%1)*60):02d}'

def savefig(fig, nombre, dpi=150, outdir=None):
    d = outdir if outdir else OUTPUT_DIR
    for ext in ['png','pdf']:
        fig.savefig(os.path.join(d,f'{nombre}.{ext}'),
                    dpi=dpi, bbox_inches='tight',
                    facecolor=C_FONDO, format=ext)
    plt.close(fig)
    print(f'  -> {os.path.basename(d)}/{nombre}.png + .pdf')

def setup_ax3d(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXTO,labelsize=5)
    for sp in [ax.xaxis,ax.yaxis,ax.zaxis]:
        sp.pane.fill=False; sp.pane.set_edgecolor(C_GRID)

def setup_ax2d(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXTO,labelsize=6)
    ax.spines[:].set_color(C_GRID)

def draw_vmd_bands(ax, ymax=1.0):
    for (_,lo,hi,col,tag) in VMD_BANDAS:
        hi_p=min(hi,360); lo_p=max(lo,2)
        ax.axvspan(lo_p,hi_p,color=col,alpha=0.08,zorder=0)
        xmid=np.sqrt(lo_p*hi_p)
        ax.text(xmid,ymax*0.93,tag,ha='center',va='top',
                color=col,fontsize=7,fontweight='bold',alpha=0.9)

# ─────────────── FEATURES ANGULARES ──────────────────────────────
def calcular_angulos_antena(sub, t_ref_arr, N_t):
    sub = sub.sort_values('tiempo_seg')
    def get_xy(col):
        xc,yc,cc = f'{col}_x',f'{col}_y',f'{col}_conf'
        if xc not in sub.columns: return np.zeros(N_t),np.zeros(N_t)
        m = sub[cc].values>CONF_MINIMA
        if m.sum()<5: return np.zeros(N_t),np.zeros(N_t)
        ts_v = sub.loc[sub.index[m],'tiempo_seg'].values
        return (np.interp(t_ref_arr,ts_v,sub.loc[sub.index[m],xc].values),
                np.interp(t_ref_arr,ts_v,sub.loc[sub.index[m],yc].values))
    ox,oy   = get_xy(COL_OCELO)
    a1x,a1y = get_xy('Antena_1_A'); p1x,p1y = get_xy('Antena_1_B')
    a2x,a2y = get_xy('Antena_2_A'); p2x,p2y = get_xy('Antena_2_B')
    def ang_and_bend(ax,ay,px,py):
        ex,ey = ax-ox,ay-oy; fx,fy = px-ax,py-ay
        dot = ex*fx+ey*fy
        mod = np.sqrt(ex**2+ey**2)*np.sqrt(fx**2+fy**2)+1e-9
        return np.arctan2(ey,ex),np.arctan2(fy,fx),np.arccos(np.clip(dot/mod,-1,1))
    sce1,fla1,bn1 = ang_and_bend(a1x,a1y,p1x,p1y)
    sce2,fla2,bn2 = ang_and_bend(a2x,a2y,p2x,p2y)
    return {'ang_scape_1':sce1,'ang_flag_1':fla1,'bend_1':bn1,
            'ang_scape_2':sce2,'ang_flag_2':fla2,'bend_2':bn2}

# ─────────────── CARGA Y COMPUTO ─────────────────────────────────
print('='*60)
print('  Cargando CSV + VMD + angulos')
print('='*60)
df    = pd.read_csv(CSV_PATH)
t_max = df['tiempo_seg'].max()
print(f'  CSV: {len(df):,} filas  |  {t_max/3600:.2f}h')
senales={};modes_all={};angulos_all={}
t_ref=None;N_t=None
for bee in BEES_CEBRA+['Bee7']:
    sub=df[df['animal']==bee]
    if sub.empty: continue
    t_c,sig=construir_senal(sub,t_max)
    if sig is None: continue
    if t_ref is None: t_ref=t_c;N_t=len(t_c)
    senales[bee]=sig
    print(f'  {bee} VMD...', end=' ',flush=True)
    modes_all[bee]=vmd_decompose(sig)
    angulos_all[bee]=calcular_angulos_antena(sub,t_ref,N_t)
    print('OK')
N_t_eff=N_t
for bee in list(modes_all.keys()):
    N_t_eff=min(N_t_eff,modes_all[bee].shape[1],len(senales[bee]))
if N_t_eff<N_t:
    t_ref=t_ref[:N_t_eff];N_t=N_t_eff
    for bee in list(senales.keys()): senales[bee]=senales[bee][:N_t]
    for bee in list(modes_all.keys()): modes_all[bee]=modes_all[bee][:,:N_t]
    for bee in list(angulos_all.keys()): angulos_all[bee]={k:v[:N_t] for k,v in angulos_all[bee].items()}
t_abs_h=END_H-(t_max-t_ref)/3600
clock_h=t_abs_h%24
t_real_full=seg_a_hora_real(t_ref)
hora_norm=clock_h/24.0
t_sec=np.arange(N_t)*BIN_SEGUNDOS
t_real_cebra=t_real_full[:N_t]
print(f'  Listo. N_t={N_t}  ({N_t*BIN_SEGUNDOS/3600:.2f}h)')

# ─────────────── FEATURES + LABELS ──────────────────────────────
def zscore(arr):
    mu=arr.mean(axis=0,keepdims=True)
    sd=np.where(arr.std(axis=0,keepdims=True)<1e-9,1.0,arr.std(axis=0,keepdims=True))
    return (arr-mu)/sd

def construir_X():
    feats=[];names=[];env_b7={}
    if 'Bee7' in modes_all:
        for k in MODOS_VALIDOS:
            env_b7[k]=gaussian_filter1d(np.abs(hilbert(modes_all['Bee7'][k][:N_t])),sigma=SIGMA_ENV_VIZ)
    for bee in BEES_CEBRA:
        if bee not in modes_all: continue
        for k in MODOS_VALIDOS:
            env=gaussian_filter1d(np.abs(hilbert(modes_all[bee][k][:N_t])),sigma=SIGMA_ENV_VIZ)
            if k in env_b7: env=np.maximum(env-ALPHA_SUB*env_b7[k],BETA_FLOOR*env_b7[k])
            feats.append(env);names.append(f'{bee}_M{k}')
        if bee in angulos_all:
            for an,av in angulos_all[bee].items():
                feats.append(av[:N_t]);names.append(f'{bee}_{an}')
    X_raw=np.stack(feats,axis=1).astype(np.float64)
    return zscore(X_raw),names,X_raw

print('  Construyendo X (VMD + angulos)...')
X,feat_names,X_raw=construir_X()
n_vmd=len(BEES_CEBRA)*len(MODOS_VALIDOS);n_ang=len(BEES_CEBRA)*6
print(f'  X shape: {X.shape}  ({n_vmd} VMD + {n_ang} angulares)')

def mk_hora():
    ang=clock_h[:N_t]/24.0*2*np.pi
    return np.stack([np.sin(ang),np.cos(ang)],axis=1).astype(np.float64)

def mk_modo():
    envs=[]
    for k in MODOS_VALIDOS:
        e_k=np.mean([gaussian_filter1d(np.abs(hilbert(modes_all[b][k][:N_t])),sigma=SIGMA_ENV_VIZ)
                     for b in BEES_CEBRA if b in modes_all],axis=0)
        mn,mx=e_k.min(),e_k.max()
        envs.append((e_k-mn)/(mx-mn+1e-9))
    lbl=np.argmax(np.stack(envs,axis=1),axis=1).astype(np.int32)
    print(f'    modo valores unicos: {np.unique(lbl)}'); return lbl

def mk_angulo():
    vals=[]
    for bee in BEES_CEBRA:
        if bee not in angulos_all: continue
        vals.append((angulos_all[bee]['ang_scape_1'][:N_t]+angulos_all[bee]['ang_scape_2'][:N_t])/2)
    return np.mean(vals,axis=0).reshape(-1,1).astype(np.float64)

def mk_sueno():
    act=np.mean([senales[b][:N_t] for b in BEES_CEBRA if b in senales],axis=0)
    act_sm=gaussian_filter1d(act,sigma=SIGMA_ENV_VIZ*6)
    return (act_sm<=np.percentile(act_sm,25)).astype(np.int32)

label_hora=mk_hora();label_modo=mk_modo()
label_ang=mk_angulo();label_sueno=mk_sueno()

# ─────────────── ENTRENAMIENTO CEBRA ─────────────────────────────
def entrenar(nombre, label=None):
    print(f'  Entrenando {nombre}...',flush=True)
    kw=dict(model_architecture='offset10-model',batch_size=CEBRA_BATCH,
            learning_rate=CEBRA_LR,max_iterations=CEBRA_ITER,
            time_offsets=CEBRA_OFFS,output_dimension=CEBRA_DIM,
            device='cuda_if_available',verbose=True)
    if label is not None: kw['conditional']='time_delta'
    m=CEBRA(**kw)
    if TIENE_TORCH: torch.manual_seed(SEED)
    np.random.seed(SEED)
    m.fit(X) if label is None else m.fit(X,label)
    emb=m.transform(X)
    m.save(os.path.join(OUTPUT_DIR,f'cebra_{nombre}_model.pt'))
    loss=np.array(m.state_dict_['loss']).min()
    print(f'    InfoNCE={loss:.4f}  emb={emb.shape}')
    return m,emb,loss

print('='*60+' MODELOS M0-M4 '+'='*20)
modelos={}
mod_time,emb_time,lt=entrenar('time'); modelos['time']=dict(emb=emb_time,loss=lt)
mod_hora,emb_hora,lh=entrenar('hora',label_hora); modelos['hora']=dict(emb=emb_hora,loss=lh)
mod_modo,emb_modo,lm=entrenar('modo',label_modo); modelos['modo']=dict(emb=emb_modo,loss=lm)
mod_ang,emb_ang,la=entrenar('angulo',label_ang);   modelos['angulo']=dict(emb=emb_ang,loss=la)
mod_sue,emb_sue,ls=entrenar('sueno',label_sueno); modelos['sueno']=dict(emb=emb_sue,loss=ls)

print('='*60+' MODELOS POR ABEJA '+'='*15)
emb_por_abeja={}
for bee in BEES_CEBRA:
    idx=[i for i,n in enumerate(feat_names) if n.startswith(bee)]
    X_bee=zscore(X_raw[:,idx])
    if TIENE_TORCH: torch.manual_seed(SEED)
    np.random.seed(SEED)
    m_bee=CEBRA(model_architecture='offset10-model',batch_size=CEBRA_BATCH,
               learning_rate=CEBRA_LR,max_iterations=CEBRA_ITER,
               time_offsets=CEBRA_OFFS,output_dimension=CEBRA_DIM,
               device='cuda_if_available',verbose=True)
    m_bee.fit(X_bee)
    emb_por_abeja[bee]=m_bee.transform(X_bee)
    m_bee.save(os.path.join(OUTPUT_DIR,f'cebra_{bee}_model.pt'))
    # GUARDAR embedding individual (necesario para Kuramoto/Wasserstein)
    np.save(os.path.join(OUTPUT_DIR,f'cebra_emb_{bee}.npy'),emb_por_abeja[bee])
    print(f'  {bee}: InfoNCE={np.array(m_bee.state_dict_["loss"]).min():.4f}')

# Guardar embeddings globales
np.save(os.path.join(OUTPUT_DIR,'cebra_time_embedding.npy'),emb_time)
np.save(os.path.join(OUTPUT_DIR,'cebra_labels_hora.npy'),label_hora)
np.save(os.path.join(OUTPUT_DIR,'cebra_X_features.npy'),X)

# ─────────────── DIVERGENCIA + CLUSTERING ────────────────────────
pares=[(BEES_CEBRA[i],BEES_CEBRA[j])
       for i in range(len(BEES_CEBRA)) for j in range(i+1,len(BEES_CEBRA))]
div_por_par={}
for b1,b2 in pares:
    dist=np.linalg.norm(emb_por_abeja[b1]-emb_por_abeja[b2],axis=1)
    div_por_par[(b1,b2)]=np.convolve(dist,np.ones(WIN_DIV)/WIN_DIV,mode='same')

km=KMeans(n_clusters=N_ESTADOS,n_init=20,random_state=SEED)
estados=km.fit_predict(emb_time)
act_grupo=np.mean([gaussian_filter1d(senales[b][:N_t],sigma=SIGMA_ENV_VIZ)
                   for b in BEES_CEBRA if b in senales],axis=0)
orden=np.argsort([-act_grupo[estados==k].mean() for k in range(N_ESTADOS)])
mapa={k:i for i,k in enumerate(orden)}
estados_ord=np.array([mapa[e] for e in estados])
act_media=act_grupo
print(f'  Estados: {[(ESTADO_NOMS[i],(estados_ord==i).mean()*100) for i in range(N_ESTADOS)]}')

# ══════════════════════════════════════════════════════════════════
# FIGURAS 25-30
# ══════════════════════════════════════════════════════════════════

# ── Fig 25: CEBRA-Time ───────────────────────────────────────────
print('Fig 25...')
fig25=plt.figure(figsize=(22,7),facecolor=C_FONDO)
fig25.suptitle('CEBRA-Time — Embedding no supervisado  |  Bee3-Bee4-Bee5-Bee6\n'
               f'{n_vmd} VMD-env denoised + {n_ang} angulares  |  Bee7 sustraida  |  {CEBRA_ITER} iter',
               color=C_TEXTO,fontsize=11,fontweight='bold')
ax=fig25.add_subplot(1,4,1,projection='3d'); setup_ax3d(ax)
sc=ax.scatter(emb_time[:,0],emb_time[:,1],emb_time[:,2],c=hora_norm,cmap='hsv',
              s=1.2,alpha=0.5,linewidths=0,rasterized=True)
ax.set_title('3D — hora del dia',color=C_TEXTO,fontsize=9)
cb=plt.colorbar(sc,ax=ax,shrink=0.5,pad=0.12)
cb.set_label('Hora/24h',color=C_TEXTO,fontsize=6)
cb.ax.yaxis.set_tick_params(color=C_TEXTO,labelsize=5)
for sp in [ax.xaxis,ax.yaxis,ax.zaxis]: sp.pane.fill=False
ax2=fig25.add_subplot(1,4,2,projection='3d'); setup_ax3d(ax2)
for k in range(N_ESTADOS):
    mk=estados_ord==k
    ax2.scatter(emb_time[mk,0],emb_time[mk,1],emb_time[mk,2],
                c=ESTADO_COLS[k],s=1.2,alpha=0.5,linewidths=0,rasterized=True,label=ESTADO_NOMS[k])
ax2.set_title('3D — estado',color=C_TEXTO,fontsize=9)
ax2.legend(fontsize=6,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6,markerscale=4)
for sp in [ax2.xaxis,ax2.yaxis,ax2.zaxis]: sp.pane.fill=False
ax3=fig25.add_subplot(1,4,3); setup_ax2d(ax3)
ax3.scatter(emb_time[:,0],emb_time[:,1],c=hora_norm,cmap='hsv',s=0.8,alpha=0.4,linewidths=0,rasterized=True)
ax3.set_xlabel('Dim 0',color=C_TEXTO,fontsize=8); ax3.set_ylabel('Dim 1',color=C_TEXTO,fontsize=8)
ax3.set_title('Dim 0 vs 1 (hora)',color=C_TEXTO,fontsize=9)
ax4=fig25.add_subplot(1,4,4); setup_ax2d(ax4)
for i in range(len(t_real_cebra)-1):
    ax4.plot(t_real_cebra[i:i+2],emb_time[i:i+2,0],color=ESTADO_COLS[estados_ord[i]],lw=0.4,alpha=0.7)
ax4.axvline(24.0,color='#00e5ff',lw=0.8,ls='--',alpha=0.5)
ax4.set_xlabel('Hora',color=C_TEXTO,fontsize=8); ax4.set_ylabel('Latente 0',color=C_TEXTO,fontsize=8)
ax4.set_title('Latente 0 temporal',color=C_TEXTO,fontsize=9)
t4=np.arange(np.ceil(t_real_cebra[0]/4)*4,t_real_cebra[-1]+1,4)
ax4.set_xticks(t4); ax4.set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=5,rotation=45)
ax4.legend(handles=[Line2D([0],[0],color=ESTADO_COLS[k],lw=2,label=ESTADO_NOMS[k]) for k in range(N_ESTADOS)],
           fontsize=6,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
plt.tight_layout(rect=[0,0.04,1,0.93]); savefig(fig25,'25_cebra_time_embedding')

# ── Fig 26: Grid embeddings ──────────────────────────────────────
print('Fig 26...')
fig26,axs=plt.subplots(2,4,figsize=(28,12),facecolor=C_FONDO,
                       gridspec_kw={'wspace':0.12,'hspace':0.30})
fig26.suptitle('CEBRA-Behavior — Embeddings por hipotesis  |  Bee3-Bee4-Bee5-Bee6',
               color=C_TEXTO,fontsize=12,fontweight='bold')
configs=[('hora',hora_norm,'hsv','Hora del dia'),
         ('modo',label_modo.astype(float)/3,'plasma','Modo VMD dominante'),
         ('angulo',(label_ang[:,0]+np.pi)/(2*np.pi),'twilight','Angulo escapo'),
         ('sueno',label_sueno.astype(float),'RdYlGn_r','Estado reposo')]
for row,(key,cvals,cmap,ttl) in enumerate(configs):
    meta=MODELOS_META[key]; emb_b=modelos[key]['emb']; loss_v=modelos[key]['loss']
    for ci,(d0,d1) in enumerate([(0,1),(1,2)]):
        ax_g=axs[row//2,(row%2)*2+ci]; setup_ax2d(ax_g)
        sc_g=ax_g.scatter(emb_b[:,d0],emb_b[:,d1],c=cvals,cmap=cmap,
                          s=0.8,alpha=0.45,linewidths=0,rasterized=True,vmin=0,vmax=1)
        ax_g.set_title(f'{meta["nombre"]}\nDim {d0} vs {d1}',color=meta['color'],fontsize=9,fontweight='bold')
        ax_g.set_xlabel(f'Dim {d0}',color=C_TEXTO,fontsize=7); ax_g.set_ylabel(f'Dim {d1}',color=C_TEXTO,fontsize=7)
        plt.colorbar(sc_g,ax=ax_g,shrink=0.7).ax.yaxis.set_tick_params(color=C_TEXTO,labelsize=5)
        ax_g.text(0.02,0.97,f'InfoNCE={loss_v:.4f}',transform=ax_g.transAxes,
                  color=meta['color'],fontsize=7,va='top',fontweight='bold')
savefig(fig26,'26_cebra_embeddings_por_label')

# ── Fig 27: Ranking InfoNCE ──────────────────────────────────────
print('Fig 27...')
fig27,ax27=plt.subplots(figsize=(10,5),facecolor=C_FONDO)
ax27.set_facecolor(C_PANEL)
keys_ord=sorted(MODELOS_META.keys(),key=lambda k:modelos[k]['loss'])
losses=[modelos[k]['loss'] for k in keys_ord]
cols27=[MODELOS_META[k]['color'] for k in keys_ord]
nombres27=[MODELOS_META[k]['nombre'] for k in keys_ord]
bars27=ax27.barh(range(len(keys_ord)),losses,color=cols27,edgecolor='none',height=0.6,alpha=0.85)
ax27.axvline(modelos['time']['loss'],color='white',lw=1.5,ls='--',alpha=0.7,label='CEBRA-Time baseline')
for i,(bar,loss,nom) in enumerate(zip(bars27,losses,nombres27)):
    ax27.text(loss+0.002,i,f'{loss:.4f}',va='center',color='white',fontsize=9,fontweight='bold')
ax27.set_yticks(range(len(keys_ord))); ax27.set_yticklabels(nombres27,color=C_TEXTO,fontsize=10)
ax27.set_xlabel('InfoNCE Loss (menor = mejor)',color=C_TEXTO,fontsize=10)
ax27.set_title('Ranking de hipotesis — Que variable organiza mejor los datos?\n'
               'Cualquier label con InfoNCE < baseline esta realmente en los datos',
               color=C_TEXTO,fontsize=11,fontweight='bold')
ax27.tick_params(colors=C_TEXTO,labelsize=8); ax27.spines[:].set_color(C_GRID)
ax27.legend(fontsize=9,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
ax27.invert_yaxis()
ganador=keys_ord[0]
fig27.text(0.5,0.02,f'Ganador: {MODELOS_META[ganador]["nombre"]} (InfoNCE={modelos[ganador]["loss"]:.4f})',
           color=MODELOS_META[ganador]['color'],fontsize=9,ha='center',fontweight='bold')
plt.tight_layout(rect=[0,0.05,1,1]); savefig(fig27,'27_cebra_ranking_infonce')

# ── Fig 28: Divergencia ─────────────────────────────────────────
print('Fig 28...')
PARES_COLS={('Bee3','Bee4'):'#ff6b6b',('Bee3','Bee5'):'#ffd166',
            ('Bee3','Bee6'):'#06d6a0',('Bee4','Bee5'):'#48dbfb',
            ('Bee4','Bee6'):'#a29bfe',('Bee5','Bee6'):'#ff9ff3'}
fig28,axes28=plt.subplots(2,1,figsize=(22,10),facecolor=C_FONDO,
                          gridspec_kw={'height_ratios':[2,1],'hspace':0.12})
fig28.suptitle('Divergencia Inter-Abeja — Distancia en Espacio Latente Individual\n'
               f'Ventana deslizante {WIN_DIV*BIN_SEGUNDOS//60}min  |  Pico = desincronizacion',
               color=C_TEXTO,fontsize=11,fontweight='bold')
ax_d=axes28[0]; ax_d.set_facecolor(C_PANEL)
for (b1,b2),dist in div_por_par.items():
    ax_d.plot(t_real_cebra,dist,color=PARES_COLS.get((b1,b2),'white'),lw=0.8,alpha=0.8,label=f'{b1}-{b2}')
ax_d.axvline(24.0,color='#00e5ff',lw=0.8,ls='--',alpha=0.5)
ax_d.set_ylabel('Distancia latente',color=C_TEXTO,fontsize=9)
ax_d.tick_params(colors=C_TEXTO,labelsize=7,labelbottom=False)
ax_d.spines[:].set_color(C_GRID); ax_d.grid(color=C_GRID,alpha=0.3,lw=0.3)
ax_d.legend(fontsize=8,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6,ncol=3)
ax_a=axes28[1]; ax_a.set_facecolor(C_PANEL)
ax_a.fill_between(t_real_cebra,0,act_media,color='#00b4d8',alpha=0.25)
ax_a.plot(t_real_cebra,act_media,color='#00b4d8',lw=1.0)
ax_a.axvline(24.0,color='#00e5ff',lw=0.8,ls='--',alpha=0.5)
ax_a.set_ylabel('Actividad media',color=C_TEXTO,fontsize=8)
ax_a.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=9)
ax_a.tick_params(colors=C_TEXTO,labelsize=7); ax_a.spines[:].set_color(C_GRID)
ax_a.set_xticks(t4); ax_a.set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=6,rotation=45)
savefig(fig28,'28_cebra_divergencia_interabeja')

# ── Fig 29: Estados ─────────────────────────────────────────────
print('Fig 29...')
fig29=plt.figure(figsize=(22,8),facecolor=C_FONDO)
fig29.suptitle(f'Estados de Comportamiento — K-means k={N_ESTADOS} en CEBRA-Time\n'
               'Bee3-Bee4-Bee5-Bee6  |  Deteccion no supervisada',
               color=C_TEXTO,fontsize=11,fontweight='bold')
ax29a=fig29.add_subplot(1,3,1,projection='3d'); setup_ax3d(ax29a)
for k in range(N_ESTADOS):
    mk=estados_ord==k
    ax29a.scatter(emb_time[mk,0],emb_time[mk,1],emb_time[mk,2],
                  c=ESTADO_COLS[k],s=1.5,alpha=0.55,linewidths=0,rasterized=True,label=ESTADO_NOMS[k])
ax29a.set_title('Embedding 3D por estado',color=C_TEXTO,fontsize=9)
ax29a.legend(fontsize=7,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6,markerscale=4)
for sp in [ax29a.xaxis,ax29a.yaxis,ax29a.zaxis]: sp.pane.fill=False
ax29b=fig29.add_subplot(1,3,2); setup_ax2d(ax29b)
bins_h=np.linspace(0,24,49)
for k in range(N_ESTADOS):
    h_v=clock_h[estados_ord==k]
    cnt,_=np.histogram(h_v,bins=bins_h)
    ax29b.bar((bins_h[:-1]+bins_h[1:])/2,cnt/cnt.sum()*100,
              width=24/48,color=ESTADO_COLS[k],alpha=0.6,label=ESTADO_NOMS[k])
ax29b.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
ax29b.set_ylabel('% del estado',color=C_TEXTO,fontsize=8)
ax29b.set_title('Distribucion horaria',color=C_TEXTO,fontsize=9)
ax29b.legend(fontsize=8,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
ax29b.grid(color=C_GRID,alpha=0.3,lw=0.3)
ax29c=fig29.add_subplot(1,3,3); setup_ax2d(ax29c)
act_raw=np.mean([senales[b][:N_t] for b in BEES_CEBRA if b in senales],axis=0)
bp=ax29c.boxplot([act_raw[estados_ord==k] for k in range(N_ESTADOS)],
                 patch_artist=True,medianprops=dict(color='white',lw=2),
                 whiskerprops=dict(color=C_TEXTO),capprops=dict(color=C_TEXTO),
                 flierprops=dict(marker='.',color=C_TEXTO,markersize=2))
for patch,col in zip(bp['boxes'],ESTADO_COLS): patch.set_facecolor(col);patch.set_alpha(0.7)
ax29c.set_xticklabels(ESTADO_NOMS,color=C_TEXTO,fontsize=9)
ax29c.set_ylabel('Velocidad antenal (px/s)',color=C_TEXTO,fontsize=8)
ax29c.set_title('Actividad por estado',color=C_TEXTO,fontsize=9)
ax29c.grid(color=C_GRID,alpha=0.3,lw=0.3,axis='y')
plt.tight_layout(rect=[0,0.04,1,0.93]); savefig(fig29,'29_cebra_estados_comportamiento')

# ── Fig 30: Latentes temporales ──────────────────────────────────
print('Fig 30...')
fig30,axes30=plt.subplots(CEBRA_DIM+1,1,figsize=(24,14),facecolor=C_FONDO,
    gridspec_kw={'height_ratios':[0.8]+[1]*CEBRA_DIM,'hspace':0.08})
fig30.suptitle('Latentes CEBRA-Time en el Tiempo — coloreados por estado\n'
               'Cada dimension = componente independiente de la dinamica antenal',
               color=C_TEXTO,fontsize=11,fontweight='bold')
ax=axes30[0]; ax.set_facecolor(C_PANEL)
ax.fill_between(t_real_cebra,0,act_media,color='#00b4d8',alpha=0.2)
ax.plot(t_real_cebra,act_media,color='#00b4d8',lw=0.8)
for k in range(N_ESTADOS):
    ax.fill_between(t_real_cebra,0,np.where(estados_ord==k,act_media,0),color=ESTADO_COLS[k],alpha=0.35)
ax.set_ylabel('Actividad\nmedia',color=C_TEXTO,fontsize=7)
ax.tick_params(colors=C_TEXTO,labelsize=6,labelbottom=False)
ax.spines[:].set_color(C_GRID); ax.grid(color=C_GRID,alpha=0.3,lw=0.3)
ax.axvline(24.0,color='#00e5ff',lw=0.8,ls='--',alpha=0.5)
for dim in range(CEBRA_DIM):
    ax=axes30[dim+1]; ax.set_facecolor(C_PANEL)
    for i in range(len(t_real_cebra)-1):
        ax.plot(t_real_cebra[i:i+2],emb_time[i:i+2,dim],
                color=ESTADO_COLS[estados_ord[i]],lw=0.5,alpha=0.8)
    ax.axvline(24.0,color='#00e5ff',lw=0.8,ls='--',alpha=0.5)
    ax.set_ylabel(f'Latente {dim}',color=C_TEXTO,fontsize=8)
    ax.tick_params(colors=C_TEXTO,labelsize=6,labelbottom=(dim==CEBRA_DIM-1))
    ax.spines[:].set_color(C_GRID); ax.grid(color=C_GRID,alpha=0.3,lw=0.3)
    ax.set_xlim(t_real_cebra[0],t_real_cebra[-1])
axes30[-1].set_xticks(t4)
axes30[-1].set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=6,rotation=45)
axes30[-1].set_xlabel('Hora del dia',color=C_TEXTO,fontsize=9)
fig30.legend(handles=[Line2D([0],[0],color=ESTADO_COLS[k],lw=2,label=ESTADO_NOMS[k]) for k in range(N_ESTADOS)],
             loc='upper right',fontsize=9,facecolor=C_PANEL,labelcolor=C_TEXTO,
             framealpha=0.7,bbox_to_anchor=(0.99,0.97))
plt.tight_layout(rect=[0,0.02,0.97,0.94]); savefig(fig30,'30_cebra_latentes_temporales')

# ══════════════════════════════════════════════════════════════════
# ANALISIS DE PERIODOS + TDA
# ══════════════════════════════════════════════════════════════════

# Coordenadas toroidales
theta_deg = np.degrees(np.arctan2(emb_time[:,1],emb_time[:,0]))
r_xy      = np.sqrt(emb_time[:,0]**2+emb_time[:,1]**2)
R_torus   = np.median(r_xy)
dr        = r_xy-R_torus
phi_deg   = np.degrees(np.arctan2(emb_time[:,2],dr))
theta_rad = np.radians(theta_deg)
phi_rad   = np.radians(phi_deg)
theta_norm_t=(theta_deg-theta_deg.mean())/theta_deg.std()
clock_norm_t=(clock_h-clock_h.mean())/clock_h.std()
corr_circ=np.corrcoef(theta_norm_t,clock_norm_t)[0,1]
print(f'  r(theta, hora) = {corr_circ:.3f}')

# PSD Welch
NPERSEG=min(4096,N_t//4); NOVERLAP=NPERSEG//2
def psd_welch(sig):
    f,P=welch(sp_detrend(sig),fs=1/BIN_SEGUNDOS,nperseg=NPERSEG,noverlap=NOVERLAP,scaling='density')
    mask=f>0; T=1.0/(f[mask]*60)
    return T[::-1],P[mask][::-1]
def peaks_rango(T,P,lo=5,hi=240,n=5):
    mask=(T>=lo)&(T<=hi)
    if not mask.any(): return np.array([]),np.array([])
    Ts,Ps=T[mask],P[mask]
    pks,_=find_peaks(Ps,height=np.percentile(Ps,70),distance=8,prominence=np.percentile(Ps,70)*0.2)
    if not len(pks): return np.array([]),np.array([])
    o=np.argsort(Ps[pks])[::-1][:n]
    return Ts[pks[o]],Ps[pks[o]]

senales_psd={'Latente 0':emb_time[:,0],'Latente 1':emb_time[:,1],
             'Latente 2':emb_time[:,2],'Angulo tubo phi (deg)':phi_deg}
cols_psd=['#ef476f','#ffd166','#06d6a0','#a29bfe']
psds={};pks_psd={}
for nm,sig in senales_psd.items():
    T,P=psd_welch(sig); psds[nm]=(T,P)
    pks_T,pks_P=peaks_rango(T,P); pks_psd[nm]=(pks_T,pks_P)
    print(f'  PSD {nm}: picos={[f"{t:.0f}min" for t in pks_T[:3]]}')

# ── Fig 31: Toro + PSDs ─────────────────────────────────────────
print('Fig 31...')
fig31=plt.figure(figsize=(26,16),facecolor=C_FONDO)
fig31.suptitle('Estructura Toroidal CEBRA-Time — Extraccion de Periodos\n'
               'El toro indica dos ritmos periodicos independientes superpuestos',
               color=C_TEXTO,fontsize=13,fontweight='bold')
gs31=fig31.add_gridspec(3,4,hspace=0.38,wspace=0.30,left=0.06,right=0.97,top=0.90,bottom=0.06)
ax00=fig31.add_subplot(gs31[0,0]); ax00.set_facecolor(C_PANEL)
sc00=ax00.scatter(emb_time[:,0],emb_time[:,1],c=clock_h,cmap='hsv',s=0.5,alpha=0.4,
                  linewidths=0,rasterized=True,vmin=0,vmax=24)
circ_t=np.linspace(0,2*np.pi,300)
ax00.plot(R_torus*np.cos(circ_t),R_torus*np.sin(circ_t),color='white',lw=1.0,ls='--',alpha=0.5)
ax00.set_xlabel('Dim 0',color=C_TEXTO,fontsize=8); ax00.set_ylabel('Dim 1',color=C_TEXTO,fontsize=8)
ax00.set_title('Vista XY — anillo principal\n(angulo θ = componente circadiana)',color='#00b4d8',fontsize=8,fontweight='bold')
ax00.tick_params(colors=C_TEXTO,labelsize=6); ax00.spines[:].set_color(C_GRID); ax00.set_aspect('equal')
cb=plt.colorbar(sc00,ax=ax00,shrink=0.7); cb.set_label('Hora del dia',color=C_TEXTO,fontsize=6)
cb.ax.yaxis.set_tick_params(color=C_TEXTO,labelsize=5)
ax01=fig31.add_subplot(gs31[0,1]); ax01.set_facecolor(C_PANEL)
sc01=ax01.scatter(dr,emb_time[:,2],c=phi_deg,cmap='twilight',s=0.5,alpha=0.4,
                  linewidths=0,rasterized=True,vmin=-180,vmax=180)
ax01.set_xlabel('r_xy - R (desviacion radial)',color=C_TEXTO,fontsize=8)
ax01.set_ylabel('Dim 2 (eje Z)',color=C_TEXTO,fontsize=8)
ax01.set_title('Vista tubo — (dr, Z)\n(angulo φ = componente ultradiana)',color='#a29bfe',fontsize=8,fontweight='bold')
ax01.tick_params(colors=C_TEXTO,labelsize=6); ax01.spines[:].set_color(C_GRID)
cb2=plt.colorbar(sc01,ax=ax01,shrink=0.7); cb2.set_label('Angulo phi (°)',color=C_TEXTO,fontsize=6)
cb2.ax.yaxis.set_tick_params(color=C_TEXTO,labelsize=5)
ax02=fig31.add_subplot(gs31[0,2]); ax02.set_facecolor(C_PANEL)
ax02.scatter(t_real_cebra,theta_deg,c=clock_h,cmap='hsv',s=0.3,alpha=0.35,linewidths=0,rasterized=True,vmin=0,vmax=24)
ax02.set_xlabel('Hora real',color=C_TEXTO,fontsize=8); ax02.set_ylabel('theta (°)',color=C_TEXTO,fontsize=8)
ax02.set_title(f'Angulo principal theta en el tiempo\nr = {corr_circ:.2f} con hora del dia',color='#00b4d8',fontsize=8,fontweight='bold')
ax02.tick_params(colors=C_TEXTO,labelsize=6); ax02.spines[:].set_color(C_GRID); ax02.grid(color=C_GRID,alpha=0.3,lw=0.3)
ax03=fig31.add_subplot(gs31[0,3]); ax03.set_facecolor(C_PANEL)
phi_sm=gaussian_filter1d(phi_deg,sigma=6)
ax03.plot(t_real_cebra,phi_sm,color='#a29bfe',lw=0.7,alpha=0.8)
ax03.fill_between(t_real_cebra,phi_sm,0,color='#a29bfe',alpha=0.15)
ax03.set_xlabel('Hora real',color=C_TEXTO,fontsize=8); ax03.set_ylabel('phi suavizado (°)',color=C_TEXTO,fontsize=8)
ax03.set_title('Angulo del tubo phi en el tiempo\n(oscilaciones = ritmos ultradianos)',color='#a29bfe',fontsize=8,fontweight='bold')
ax03.tick_params(colors=C_TEXTO,labelsize=6); ax03.spines[:].set_color(C_GRID); ax03.grid(color=C_GRID,alpha=0.3,lw=0.3)
nombres_psd_l=list(senales_psd.keys())
for idx,(nm,col) in enumerate(zip(nombres_psd_l,cols_psd)):
    row=1+idx//2; ci=idx%2
    ax=fig31.add_subplot(gs31[row,ci*2:ci*2+2]); ax.set_facecolor(C_PANEL)
    T,P=psds[nm]; mask=(T>=2)&(T<=360)
    for (_,lo,hi,bc,tag) in VMD_BANDAS:
        ax.axvspan(max(lo,2),min(hi,360),color=bc,alpha=0.07,zorder=0)
        ax.text(np.sqrt(max(lo,2)*min(hi,360)),np.nanmax(P[mask])*0.95,tag,
                ha='center',va='top',color=bc,fontsize=5.5,fontweight='bold',alpha=0.8)
    ax.fill_between(T[mask],0,P[mask],color=col,alpha=0.25,zorder=1)
    ax.plot(T[mask],P[mask],color=col,lw=0.9,alpha=0.9,zorder=2)
    for pt,pp in zip(*pks_psd[nm]):
        if 2<=pt<=360:
            ax.scatter(pt,pp,color='white',s=40,zorder=10,edgecolors=col,lw=1.2)
            ax.annotate(f'{pt:.0f} min',xy=(pt,pp),xytext=(pt,pp*1.15),
                        ha='center',va='bottom',color='white',fontsize=7,fontweight='bold',
                        arrowprops=dict(arrowstyle='->',color='white',lw=0.7,shrinkB=3))
    ax.set_xscale('log'); ax.set_xlim(2,360)
    ax.set_xlabel('Periodo (min)',color=C_TEXTO,fontsize=8)
    ax.set_ylabel('Densidad espectral',color=C_TEXTO,fontsize=8)
    ax.set_title(f'PSD Welch — {nm}',color=col,fontsize=10,fontweight='bold')
    ax.tick_params(colors=C_TEXTO,labelsize=7); ax.spines[:].set_color(C_GRID)
    ax.grid(color=C_GRID,alpha=0.3,lw=0.3,which='both')
    pt_ticks=[5,10,20,30,60,90,120,180,240,360]
    ax.set_xticks([p for p in pt_ticks if 2<=p<=360])
    ax.set_xticklabels([str(p) for p in pt_ticks if 2<=p<=360],color=C_TEXTO,fontsize=6)
fig31.text(0.01,0.01,f'Welch PSD (ventana={NPERSEG*BIN_SEGUNDOS/60:.0f}min, overlap=50%). Escala X log. Bandas = modos VMD.',
           color='#555577',fontsize=6.5,style='italic')
savefig(fig31,'31_cebra_periodos_toro')

# ── Fig 32: Lomb-Scargle + CWT ──────────────────────────────────
print('Fig 32...')
if TIENE_LS:
    FREQ_MIN=1/(360*60); FREQ_MAX=1/(2*60)
    ls_res={}
    for dim in range(3):
        sig_dt=sp_detrend(emb_time[:,dim])
        ls=LombScargle(t_sec,sig_dt)
        freq,power=ls.autopower(minimum_frequency=FREQ_MIN,maximum_frequency=FREQ_MAX,samples_per_peak=20)
        periods_min=1/(freq*60)
        fap=ls.false_alarm_level([0.10,0.05,0.01])
        pks,_=find_peaks(power,height=fap[1],distance=15,prominence=fap[1]*0.2)
        if len(pks): pks=pks[np.argsort(power[pks])[::-1]]
        ls_res[dim]=dict(periods=periods_min,power=power,fap=fap,
                         peak_T=periods_min[pks][:6] if len(pks) else np.array([]),
                         peak_P=power[pks][:6] if len(pks) else np.array([]))
        print(f'  LS Lat{dim}: picos={[f"{t:.0f}m" for t in (periods_min[pks][:3] if len(pks) else [])]}')
    periods_ref=ls_res[0]['periods']
    power_comb=sum(ls_res[d]['power']/ls_res[d]['power'].max() for d in range(3))/3
    pks_c,_=find_peaks(power_comb,height=np.percentile(power_comb,85),distance=15)
    pks_c=pks_c[np.argsort(power_comb[pks_c])[::-1]]

    # CWT Morlet
    PERIODS_CWT=np.logspace(np.log10(2),np.log10(360),50)
    def morlet2(M,s,w=W0_CWT):
        x=(np.arange(0,M)-(M-1)/2)/s
        return np.sqrt(1/s)*np.exp(1j*w*x)*np.exp(-0.5*x**2)*np.pi**(-0.25)
    def cwt_morlet(sig):
        n=len(sig); out=np.zeros((len(PERIODS_CWT),n))
        for i,p in enumerate(PERIODS_CWT):
            s=p*60/BIN_SEGUNDOS*W0_CWT/(2*np.pi)
            M=max(int(12*s),5); wav=morlet2(M,s)
            conv=fftconvolve(sig,wav[::-1].conj(),mode='full')
            st=(M-1)//2; out[i]=np.abs(conv[st:st+n])**2/s
        return out
    cwt_all={d:cwt_morlet(sp_detrend(emb_time[:,d])) for d in range(3)}

    CMAP_CWT=matplotlib.colors.LinearSegmentedColormap.from_list('cwt',
        ['#080818','#0d1b4a','#1a4080','#0077b6','#00b4d8','#48cae4','#ffd166','#ef476f','#ffffff'],N=512)

    fig32=plt.figure(figsize=(28,14),facecolor=C_FONDO)
    fig32.suptitle('Analisis de Periodos sobre Latentes CEBRA-Time\n'
                   'Lomb-Scargle (superior) + CWT Morlet (inferior) — Bee3-Bee4-Bee5-Bee6',
                   color=C_TEXTO,fontsize=13,fontweight='bold')
    gs32=GridSpec(2,4,figure=fig32,hspace=0.35,wspace=0.22,left=0.05,right=0.97,top=0.90,bottom=0.07)
    for dim in range(3):
        ax=fig32.add_subplot(gs32[0,dim]); ax.set_facecolor(C_PANEL)
        r=ls_res[dim]; mask=(r['periods']>=2)&(r['periods']<=360)
        T_p=r['periods'][mask]; P_p=r['power'][mask]; ymax=P_p.max()*1.15
        draw_vmd_bands(ax,ymax=ymax)
        for fv,fl,fls in zip(r['fap'],['p=10%','p=5%','p=1%'],[':','--','-.']):
            ax.axhline(fv,color='white',lw=0.8,ls=fls,alpha=0.5,label=fl)
        ax.fill_between(T_p,0,P_p,color=COLS_LAT[dim],alpha=0.25)
        ax.plot(T_p,P_p,color=COLS_LAT[dim],lw=0.9,alpha=0.9)
        for pt,pp in zip(r['peak_T'],r['peak_P']):
            if 2<=pt<=360:
                ax.scatter(pt,pp,color='white',s=50,zorder=10,edgecolors=COLS_LAT[dim],lw=1.5)
                ax.annotate(f'{pt:.0f} min',xy=(pt,pp),xytext=(pt,pp+ymax*0.08),
                            ha='center',va='bottom',color='white',fontsize=7.5,fontweight='bold',
                            arrowprops=dict(arrowstyle='->',color='white',lw=0.7,shrinkB=2))
        ax.set_xscale('log'); ax.set_xlim(2,360); ax.set_ylim(0,ymax)
        ax.set_xlabel('Periodo (min)',color=C_TEXTO,fontsize=8)
        ax.set_ylabel('Potencia LS',color=C_TEXTO,fontsize=8)
        ax.set_title(f'Lomb-Scargle — Latente {dim}',color=COLS_LAT[dim],fontsize=10,fontweight='bold')
        ax.tick_params(colors=C_TEXTO,labelsize=6); ax.spines[:].set_color(C_GRID)
        ax.grid(color=C_GRID,alpha=0.3,lw=0.3,which='both')
        ax.set_xticks([5,10,20,30,60,120,180,360])
        ax.set_xticklabels(['5','10','20','30','60','120','180','360'],color=C_TEXTO,fontsize=6)
        if dim==0: ax.legend(fontsize=6,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6,loc='upper left')
    ax_c=fig32.add_subplot(gs32[0,3]); ax_c.set_facecolor(C_PANEL)
    mask_c=(periods_ref>=2)&(periods_ref<=360)
    ymc=power_comb[mask_c].max()*1.15
    draw_vmd_bands(ax_c,ymax=ymc)
    ax_c.fill_between(periods_ref[mask_c],0,power_comb[mask_c],color='white',alpha=0.18)
    ax_c.plot(periods_ref[mask_c],power_comb[mask_c],color='white',lw=1.0)
    for pt in periods_ref[pks_c[:5]]:
        if 2<=pt<=360: ax_c.axvline(pt,color='#ffd166',lw=1.0,ls='--',alpha=0.7)
    ax_c.set_xscale('log'); ax_c.set_xlim(2,360); ax_c.set_ylim(0,ymc)
    ax_c.set_xlabel('Periodo (min)',color=C_TEXTO,fontsize=8)
    ax_c.set_ylabel('Potencia combinada',color=C_TEXTO,fontsize=8)
    ax_c.set_title('LS combinado\n(media 3 latentes)',color='white',fontsize=10,fontweight='bold')
    ax_c.tick_params(colors=C_TEXTO,labelsize=6); ax_c.spines[:].set_color(C_GRID)
    for dim in range(3):
        ax=fig32.add_subplot(gs32[1,dim]); ax.set_facecolor(C_PANEL)
        cwt_n=cwt_all[dim]/(PERIODS_CWT[:,np.newaxis]+1e-9)
        ax.pcolormesh(t_real_cebra,PERIODS_CWT,cwt_n,cmap=CMAP_CWT,
                      vmin=0,vmax=np.percentile(cwt_n,99),shading='auto',rasterized=True)
        for (_,lo,hi,col,tag) in VMD_BANDAS:
            if 2<=lo<=360: ax.axhline(lo,color=col,lw=0.6,ls='--',alpha=0.5)
        ax.set_yscale('log'); ax.set_ylim(2,360)
        ax.set_ylabel('Periodo (min)',color=C_TEXTO,fontsize=8)
        ax.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
        ax.set_title(f'CWT Morlet — Latente {dim}',color=COLS_LAT[dim],fontsize=10,fontweight='bold')
        ax.tick_params(colors=C_TEXTO,labelsize=6); ax.spines[:].set_color(C_GRID)
        ax.set_yticks([5,10,20,30,60,120,240])
        ax.set_yticklabels(['5','10','20','30','60','120','240'],color=C_TEXTO,fontsize=6)
    ax_cc=fig32.add_subplot(gs32[1,3]); ax_cc.set_facecolor(C_PANEL)
    cwt_s=sum(cwt_all[d]/(cwt_all[d].max()+1e-9) for d in range(3))/3
    cwt_sn=cwt_s/(PERIODS_CWT[:,np.newaxis]+1e-9)
    ax_cc.pcolormesh(t_real_cebra,PERIODS_CWT,cwt_sn,cmap=CMAP_CWT,
                     vmin=0,vmax=np.percentile(cwt_sn,99),shading='auto',rasterized=True)
    for (_,lo,hi,col,tag) in VMD_BANDAS:
        if 2<=lo<=360: ax_cc.axhline(lo,color=col,lw=0.6,ls='--',alpha=0.5)
    ax_cc.set_yscale('log'); ax_cc.set_ylim(2,360)
    ax_cc.set_ylabel('Periodo (min)',color=C_TEXTO,fontsize=8)
    ax_cc.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
    ax_cc.set_title('CWT combinado\n(media 3 latentes)',color='white',fontsize=10,fontweight='bold')
    ax_cc.tick_params(colors=C_TEXTO,labelsize=6); ax_cc.spines[:].set_color(C_GRID)
    ax_cc.set_yticks([5,10,20,30,60,120,240])
    ax_cc.set_yticklabels(['5','10','20','30','60','120','240'],color=C_TEXTO,fontsize=6)
    fig32.text(0.01,0.01,'LS: lineas = FAP (p=10%,5%,1%). CWT: potencia normalizada por periodo.',
               color='#555577',fontsize=6.5,style='italic')
    savefig(fig32,'32_lomb_scargle_cwt_latentes')
else:
    print('  [SKIP] Fig 32: pip install astropy')

# ── Fig 33: TDA ─────────────────────────────────────────────────
print('Fig 33...')
if TIENE_TDA:
    STEP=max(1,N_t//SUB_TDA)
    emb_sub=emb_time[::STEP]
    print(f'  TDA: {len(emb_sub)} pts...',end=' ',flush=True)
    try:
        result=ripser(emb_sub,maxdim=2,thresh=1.2)
        diagrams=result['dgms']
        print('OK')
    except Exception as e_tda:
        print(f'ERROR: {e_tda}')
        print('  [TDA] Reintentando con SUB_TDA=400 y thresh=0.8...')
        emb_sub2 = emb_time[::max(1,N_t//400)]
        try:
            result=ripser(emb_sub2,maxdim=2,thresh=0.8)
            diagrams=result['dgms']
            print('  OK con parametros reducidos')
        except Exception as e2:
            print(f'  TDA fallo: {e2} — omitiendo Fig 33')
            diagrams=None
    h1=diagrams[1]; h1f=h1[h1[:,1]<np.inf]
    h2=diagrams[2]; h2f=h2[h2[:,1]<np.inf]
    pers_h1=h1f[:,1]-h1f[:,0] if len(h1f) else np.array([])
    pers_h2=h2f[:,1]-h2f[:,0] if len(h2f) else np.array([])
    ord_h1=np.argsort(pers_h1)[::-1] if len(pers_h1) else []
    h1_top=h1f[ord_h1[:5]] if len(ord_h1) else np.array([[]])
    pers_top=pers_h1[ord_h1[:5]] if len(ord_h1) else np.array([])
    b0=(diagrams[0][:,1]==np.inf).sum()
    b1=int((pers_h1>np.percentile(pers_h1,50)).sum()) if len(pers_h1) else 0
    b2=int((pers_h2>np.percentile(pers_h2,50)).sum()) if len(pers_h2) else 0
    topo='TORO' if (b0==1 and b1==2 and b2==1) else 'ANILLO' if (b0==1 and b1==1) else 'OTRA'
    print(f'  TDA: beta0={b0} beta1={b1} beta2={b2} -> {topo}')

    fig33=plt.figure(figsize=(22,12),facecolor=C_FONDO)
    fig33.suptitle(f'TDA — Homologia Persistente del Embedding CEBRA\n'
                   f'Topologia: {topo}  |  beta0={b0}  beta1={b1}  beta2={b2}  (toro ideal: 1,2,1)',
                   color=C_TEXTO,fontsize=12,fontweight='bold')
    gs33=GridSpec(2,3,figure=fig33,hspace=0.38,wspace=0.28,left=0.06,right=0.97,top=0.88,bottom=0.08)
    COLS_HOM={0:'#ef476f',1:'#ffd166',2:'#06d6a0'}
    LABS_HOM={0:'H0 (componentes)',1:'H1 (loops)',2:'H2 (cavidades)'}
    ax_pd=fig33.add_subplot(gs33[0,0]); ax_pd.set_facecolor(C_PANEL)
    for hdim,diag in enumerate(diagrams[:3]):
        if not len(diag): continue
        fin=diag[diag[:,1]<np.inf]; inf_p=diag[diag[:,1]==np.inf]
        if len(fin): ax_pd.scatter(fin[:,0],fin[:,1],color=COLS_HOM[hdim],s=15,alpha=0.7,label=LABS_HOM[hdim],zorder=3)
        if len(inf_p):
            yM=max([d[d[:,1]<np.inf][:,1].max() for d in diagrams[:3] if len(d[d[:,1]<np.inf])]+[1])
            ax_pd.scatter(inf_p[:,0],np.full(len(inf_p),yM*1.05),color=COLS_HOM[hdim],s=25,marker='^',alpha=0.9,zorder=4)
    lx=ax_pd.get_xlim(); ax_pd.plot([0,lx[1]],[0,lx[1]],color='white',lw=0.8,ls='--',alpha=0.4)
    ax_pd.set_xlabel('Nacimiento',color=C_TEXTO,fontsize=9); ax_pd.set_ylabel('Muerte',color=C_TEXTO,fontsize=9)
    ax_pd.set_title('Diagrama de Persistencia\n(lejos diagonal = mas robusto)',color=C_TEXTO,fontsize=9,fontweight='bold')
    ax_pd.tick_params(colors=C_TEXTO,labelsize=7); ax_pd.spines[:].set_color(C_GRID)
    ax_pd.legend(fontsize=7,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
    ax_pd.grid(color=C_GRID,alpha=0.3,lw=0.3)
    ax_bc=fig33.add_subplot(gs33[0,1]); ax_bc.set_facecolor(C_PANEL)
    if len(h1f):
        s_i=np.argsort(pers_h1)[::-1]; h1_s=h1f[s_i]; p_s=pers_h1[s_i]
        for i in range(min(15,len(h1_s))):
            col_b='#ffd166' if i<2 else '#444466'; lw_b=2.5 if i<2 else 0.8
            ax_bc.plot([h1_s[i,0],h1_s[i,1]],[i,i],color=col_b,lw=lw_b,alpha=0.95 if i<2 else 0.5,solid_capstyle='round')
            if i<2: ax_bc.text(h1_s[i,1]+0.01,i,f'pers={p_s[i]:.3f}',va='center',color='#ffd166',fontsize=7,fontweight='bold')
    ax_bc.set_xlabel('Filtracion (radio Vietoris-Rips)',color=C_TEXTO,fontsize=8)
    ax_bc.set_ylabel('Generador H1',color=C_TEXTO,fontsize=8)
    ax_bc.set_title('Barcode H1 — Loops del toro\n(amarillo = 2 mas persistentes)',color='#ffd166',fontsize=9,fontweight='bold')
    ax_bc.tick_params(colors=C_TEXTO,labelsize=7); ax_bc.spines[:].set_color(C_GRID)
    ax_bc.grid(color=C_GRID,alpha=0.2,lw=0.3,axis='x')
    ax_top=fig33.add_subplot(gs33[0,2]); ax_top.set_facecolor(C_PANEL); ax_top.axis('off')
    betti_t=[1,2,1]; betti_a=[b0,b1,b2]; betti_n=['beta0\n(comp.)','beta1\n(loops)','beta2\n(cav.)']
    for i,(xp,nom,bt,ba) in enumerate(zip([0.2,0.5,0.8],betti_n,betti_t,betti_a)):
        col_b='#06d6a0' if bt==ba else '#ef476f'
        ax_top.text(xp,0.78,str(bt),ha='center',fontsize=28,color='#aaaaaa',fontweight='bold',transform=ax_top.transAxes)
        ax_top.text(xp,0.55,str(ba),ha='center',fontsize=28,color=col_b,fontweight='bold',transform=ax_top.transAxes)
        ax_top.text(xp,0.40,'V' if bt==ba else 'X',ha='center',fontsize=20,color=col_b,transform=ax_top.transAxes)
        ax_top.text(xp,0.28,nom,ha='center',fontsize=8,color=C_TEXTO,transform=ax_top.transAxes)
    ax_top.text(0.5,0.95,'Numeros de Betti',ha='center',fontsize=10,color=C_TEXTO,fontweight='bold',transform=ax_top.transAxes)
    ax_top.text(0.5,0.86,'Esperado (toro)',ha='center',fontsize=8,color='#aaaaaa',transform=ax_top.transAxes)
    ax_top.text(0.5,0.63,'Detectado',ha='center',fontsize=8,color=C_TEXTO,transform=ax_top.transAxes)
    ax_top.text(0.5,0.13,f'Topologia: {topo}',ha='center',fontsize=12,
                color='#06d6a0' if topo=='TORO' else '#ffd166',fontweight='bold',transform=ax_top.transAxes)
    ax_top.set_title('Confirmacion topologica',color=C_TEXTO,fontsize=9,fontweight='bold')
    ax_p1=fig33.add_subplot(gs33[1,0]); ax_p1.set_facecolor(C_PANEL)
    if len(pers_top):
        b33=ax_p1.bar(range(len(pers_top)),pers_top,
                      color=['#ffd166' if i<2 else '#444466' for i in range(len(pers_top))],
                      edgecolor='none',alpha=0.85)
        for i,(b,p) in enumerate(zip(b33,pers_top)):
            ax_p1.text(b.get_x()+b.get_width()/2,p+0.002,f'{p:.3f}',ha='center',va='bottom',color='white',fontsize=8,fontweight='bold')
        ax_p1.set_xticks(range(len(pers_top))); ax_p1.set_xticklabels([f'G{i+1}' for i in range(len(pers_top))],color=C_TEXTO,fontsize=8)
        if len(pers_top)>=3:
            ax_p1.axhline(pers_top[2],color='#ef476f',lw=1.0,ls='--',alpha=0.7)
    ax_p1.set_xlabel('Generador H1',color=C_TEXTO,fontsize=9); ax_p1.set_ylabel('Persistencia',color=C_TEXTO,fontsize=9)
    ax_p1.set_title('Persistencia loops H1\n(G1,G2 = ciclos del toro)',color='#ffd166',fontsize=9,fontweight='bold')
    ax_p1.tick_params(colors=C_TEXTO,labelsize=7); ax_p1.spines[:].set_color(C_GRID)
    ax_p1.grid(color=C_GRID,alpha=0.3,lw=0.3,axis='y')
    ax_tab=fig33.add_subplot(gs33[1,1:]); ax_tab.set_facecolor(C_PANEL); ax_tab.axis('off')
    todos_det=[]
    if TIENE_LS:
        for d in range(3):
            for pt in ls_res[d]['peak_T']:
                if 2<=pt<=360:
                    band='M0>120m' if pt>120 else 'M1 30-120m' if pt>30 else 'M2 10-30m' if pt>10 else 'M3<10m'
                    todos_det.append((pt,band,f'Lat.{d}'))
    grupos=[]
    for pt,band,fuente in sorted(todos_det):
        merged=False
        for g in grupos:
            if abs(pt-g['p'])/max(pt,g['p'])<0.20: g['f'].append(fuente);g['ps'].append(pt);merged=True;break
        if not merged: grupos.append({'p':pt,'band':band,'f':[fuente],'ps':[pt]})
    grupos.sort(key=lambda g:-len(g['f']))
    lines_tab=['PERIODOS DETECTADOS (Lomb-Scargle, p<0.05):\n',
               f'{"Periodo":>10}  {"Banda VMD":>12}  Consistencia']
    lines_tab.append('─'*44)
    for g in grupos[:8]:
        pmean=np.mean(g['ps']); nf=len(g['f']); stars='*'*nf
        lines_tab.append(f'{pmean:>8.0f} min  {g["band"]:>12}  {stars} ({nf}/3 lat.)')
    lines_tab.append(f'\nTOPOLOGIA (TDA): {topo}  beta0={b0} beta1={b1} beta2={b2}')
    if len(pers_top)>=2: lines_tab.append(f'Ciclo 1: pers={pers_top[0]:.3f}   Ciclo 2: pers={pers_top[1]:.3f}')
    ax_tab.text(0.03,0.95,'\n'.join(lines_tab),transform=ax_tab.transAxes,va='top',ha='left',
                fontsize=8.5,color=C_TEXTO,fontfamily='monospace',
                bbox=dict(boxstyle='round',facecolor='#0a0a1e',edgecolor='#2a2a4a',alpha=0.9))
    ax_tab.set_title('Resumen integrado (* = aparece en mas latentes)',color=C_TEXTO,fontsize=9,fontweight='bold')
    fig33.text(0.01,0.01,'TDA: Vietoris-Rips sobre submuestra de embeddings. Gap de persistencia G2-G3 = evidencia toroidal.',
               color='#555577',fontsize=6.5,style='italic')
    savefig(fig33,'33_tda_homologia_persistente')
else:
    print('  [SKIP] Fig 33: pip install ripser persim')

# ══════════════════════════════════════════════════════════════════
# FIG 34: KURAMOTO TOPOLOGICO + WASSERSTEIN + PERSISTENCE LANDSCAPE
# (version corregida de spyder2cebra.py)
# ══════════════════════════════════════════════════════════════════
print('Fig 34...')
if TIENE_TDA:
    # Fases topologicas (angulo en plano XY de cada embedding individual)
    fases={}
    for b in BEES_CEBRA:
        fases[b]=np.arctan2(emb_por_abeja[b][:,1],emb_por_abeja[b][:,0])

    # Kuramoto R(t)
    sum_cx=np.zeros(N_t,dtype=complex)
    for b in BEES_CEBRA: sum_cx+=np.exp(1j*fases[b])
    R_t=np.abs(sum_cx)/len(BEES_CEBRA)
    R_sm=gaussian_filter1d(R_t,sigma=SIGMA_ENV_VIZ*4)
    R_mean=np.mean(R_t)

    # Baseline shuffleado (R esperado por azar)
    np.random.seed(SEED)
    R_shuf=[]
    for _ in range(100):
        sc=sum(np.exp(1j*np.random.permutation(fases[b])) for b in BEES_CEBRA)
        R_shuf.append(np.mean(np.abs(sc)/len(BEES_CEBRA)))
    R_shuf_mean=np.mean(R_shuf); R_shuf_std=np.std(R_shuf)
    print(f'  Kuramoto: R={R_mean:.3f}  shuf={R_shuf_mean:.3f}±{R_shuf_std:.3f}')

    # Wasserstein — TODOS los pares (correcto: 6 pares)
    STEP_W=max(1,N_t//SUB_TDA)
    diags_bee={}
    for b in BEES_CEBRA:
        sub_b=emb_por_abeja[b][::STEP_W]
        diags_bee[b]=ripser(sub_b,maxdim=1)['dgms'][1]

    wass_all={}
    for b1,b2 in pares:
        d=wasserstein(diags_bee[b1],diags_bee[b2])
        wass_all[f'{b1}-{b2}']=d
        print(f'  Wasserstein {b1}-{b2}: {d:.4f}')

    # Persistence Landscape — fit correcto
    all_diags=[diags_bee[b] for b in BEES_CEBRA]
    # Filtrar diagramas vacios
    all_diags_nz=[d for d in all_diags if len(d)>0]
    landscapes={}
    if all_diags_nz:
        pimgr=PersistenceImager(pixel_size=0.05)
        pimgr.fit(all_diags_nz)   # FIT primero con todos
        for b in BEES_CEBRA:
            if len(diags_bee[b])>0:
                landscapes[b]=pimgr.transform(diags_bee[b])

    # Filtrado topologico: reconstruccion desde fase
    # Usamos sin(phi) + cos(phi) para capturar la oscilacion del tubo
    phi_sin=np.sin(phi_rad); phi_cos=np.cos(phi_rad)
    topo_filtered=gaussian_filter1d(phi_sin,sigma=SIGMA_ENV_VIZ*4)

    # ── Figura ───────────────────────────────────────────────────
    fig34=plt.figure(figsize=(24,14),facecolor=C_FONDO)
    fig34.suptitle('Analisis Topologico Avanzado — Embedding CEBRA por Abeja\n'
                   'Kuramoto de fase | Wasserstein (6 pares) | Persistence Landscape | Filtrado topologico',
                   color=C_TEXTO,fontsize=12,fontweight='bold')
    gs34=GridSpec(2,3,figure=fig34,hspace=0.38,wspace=0.30,left=0.06,right=0.97,top=0.90,bottom=0.08)

    # Panel 0,0: Kuramoto R(t) con baseline
    ax34a=fig34.add_subplot(gs34[0,0]); ax34a.set_facecolor(C_PANEL)
    ax34a.fill_between(t_real_cebra,R_shuf_mean-2*R_shuf_std,R_shuf_mean+2*R_shuf_std,
                       color='#444466',alpha=0.4,label='Azar ±2σ')
    ax34a.axhline(R_shuf_mean,color='#444466',lw=1.0,ls='--',alpha=0.8)
    ax34a.plot(t_real_cebra,R_t,color='#00e5ff',lw=0.4,alpha=0.3)
    ax34a.plot(t_real_cebra,R_sm,color='#00e5ff',lw=1.5,alpha=0.9,label=f'R(t) suavizado')
    ax34a.axhline(R_mean,color='white',lw=1.0,ls=':',alpha=0.7,label=f'R medio={R_mean:.2f}')
    ax34a.axvline(24.0,color='white',lw=0.8,ls='--',alpha=0.5)
    ax34a.set_ylim(0,1); ax34a.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
    ax34a.set_ylabel('R(t) — Parametro de orden',color=C_TEXTO,fontsize=8)
    ax34a.set_title('Sincronizacion de Fase Topologica (Kuramoto)\nR>azar = sincronizacion real entre abejas',
                    color='#00e5ff',fontsize=9,fontweight='bold')
    ax34a.tick_params(colors=C_TEXTO,labelsize=7); ax34a.spines[:].set_color(C_GRID)
    ax34a.grid(color=C_GRID,alpha=0.3,lw=0.3)
    ax34a.set_xticks(t4); ax34a.set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=5,rotation=45)
    ax34a.legend(fontsize=7,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
    signif='R > azar: sincronizacion significativa' if R_mean>R_shuf_mean+2*R_shuf_std else 'R ~= azar: sincronizacion marginal'
    ax34a.text(0.5,0.03,signif,transform=ax34a.transAxes,ha='center',
               color='#06d6a0' if R_mean>R_shuf_mean+2*R_shuf_std else '#ffd166',fontsize=7,fontweight='bold')

    # Panel 0,1: Wasserstein todos los pares
    ax34b=fig34.add_subplot(gs34[0,1]); ax34b.set_facecolor(C_PANEL)
    w_labels=list(wass_all.keys()); w_vals=list(wass_all.values())
    w_cols=[PARES_COLS.get(tuple(k.split('-')),PARES_COLS.get(tuple(reversed(k.split('-'))),'white')) for k in w_labels]
    bars_w=ax34b.bar(range(len(w_labels)),w_vals,color=w_cols,edgecolor='none',alpha=0.85)
    for b,v in zip(bars_w,w_vals):
        ax34b.text(b.get_x()+b.get_width()/2,v+max(w_vals)*0.01,f'{v:.2f}',
                   ha='center',va='bottom',color='white',fontsize=8,fontweight='bold')
    ax34b.set_xticks(range(len(w_labels))); ax34b.set_xticklabels(w_labels,color=C_TEXTO,fontsize=8,rotation=30,ha='right')
    ax34b.set_ylabel('Distancia de Wasserstein',color=C_TEXTO,fontsize=8)
    ax34b.set_title('Divergencia Topologica — Wasserstein H1\n(todos los pares, distancia entre diagramas de persistencia)',
                    color='#ef476f',fontsize=9,fontweight='bold')
    ax34b.tick_params(colors=C_TEXTO,labelsize=7); ax34b.spines[:].set_color(C_GRID)
    ax34b.grid(color=C_GRID,alpha=0.3,lw=0.3,axis='y')
    ax34b.text(0.5,0.97,'Mayor distancia = topologias mas distintas entre el par',
               transform=ax34b.transAxes,ha='center',va='top',color='#aaaaaa',fontsize=7,style='italic')

    # Panel 0,2: Persistence Landscape Bee3 (ejemplo)
    ax34c=fig34.add_subplot(gs34[0,2]); ax34c.set_facecolor(C_PANEL)
    if 'Bee3' in landscapes and landscapes['Bee3'] is not None:
        ls_img=landscapes['Bee3']
        if hasattr(ls_img,'toarray'): ls_img=ls_img.toarray()
        ls_arr=np.array(ls_img)
        if ls_arr.ndim==3: ls_arr=ls_arr[0]
        ax34c.imshow(ls_arr,cmap='magma',aspect='auto',origin='lower')
        ax34c.set_title('Persistence Landscape H1 — Bee3\n(topologia del espacio de estados)',color='#a29bfe',fontsize=9,fontweight='bold')
    else:
        ax34c.text(0.5,0.5,'Sin datos H1 suficientes',ha='center',va='center',
                   color=C_TEXTO,fontsize=10,transform=ax34c.transAxes)
        ax34c.set_title('Persistence Landscape H1 — Bee3',color='#a29bfe',fontsize=9,fontweight='bold')
    ax34c.tick_params(colors=C_TEXTO,labelsize=7); ax34c.spines[:].set_color(C_GRID)
    ax34c.set_xlabel('Birth (px)',color=C_TEXTO,fontsize=7); ax34c.set_ylabel('Death (px)',color=C_TEXTO,fontsize=7)

    # Panel 1,0: Fases individuales
    ax34d=fig34.add_subplot(gs34[1,0]); ax34d.set_facecolor(C_PANEL)
    for b in BEES_CEBRA:
        fase_sm=gaussian_filter1d(np.unwrap(fases[b]),sigma=SIGMA_ENV_VIZ*2)
        ax34d.plot(t_real_cebra,fase_sm,color=COLORES[b],lw=0.8,alpha=0.8,label=b)
    ax34d.axvline(24.0,color='white',lw=0.8,ls='--',alpha=0.5)
    ax34d.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
    ax34d.set_ylabel('Fase topologica unwrapped (rad)',color=C_TEXTO,fontsize=8)
    ax34d.set_title('Fases individuales por abeja\n(divergencia = lineas que se separan)',
                    color=C_TEXTO,fontsize=9,fontweight='bold')
    ax34d.tick_params(colors=C_TEXTO,labelsize=7); ax34d.spines[:].set_color(C_GRID)
    ax34d.grid(color=C_GRID,alpha=0.3,lw=0.3)
    ax34d.set_xticks(t4); ax34d.set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=5,rotation=45)
    ax34d.legend(fontsize=8,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)

    # Panel 1,1: Filtrado topologico
    ax34e=fig34.add_subplot(gs34[1,1]); ax34e.set_facecolor(C_PANEL)
    ax34e.plot(t_real_cebra,sp_detrend(emb_time[:N_t,0]),color='white',alpha=0.25,lw=0.4,label='Latente 0 original')
    ax34e.plot(t_real_cebra,topo_filtered,color='#ffd166',lw=1.5,alpha=0.9,label='Filtrado topologico (sin phi)')
    ax34e.axvline(24.0,color='white',lw=0.8,ls='--',alpha=0.5)
    ax34e.set_xlabel('Hora del dia',color=C_TEXTO,fontsize=8)
    ax34e.set_ylabel('Amplitud',color=C_TEXTO,fontsize=8)
    ax34e.set_title('Filtrado Topologico\n(componente ultradiana del tubo del toro)',
                    color='#ffd166',fontsize=9,fontweight='bold')
    ax34e.tick_params(colors=C_TEXTO,labelsize=7); ax34e.spines[:].set_color(C_GRID)
    ax34e.grid(color=C_GRID,alpha=0.3,lw=0.3)
    ax34e.legend(fontsize=8,facecolor=C_PANEL,labelcolor=C_TEXTO,framealpha=0.6)
    ax34e.set_xticks(t4); ax34e.set_xticklabels([hora_real_label(h)[:5] for h in t4],color=C_TEXTO,fontsize=5,rotation=45)

    # Panel 1,2: Tabla Wasserstein
    ax34f=fig34.add_subplot(gs34[1,2]); ax34f.set_facecolor(C_PANEL); ax34f.axis('off')
    wass_sorted=sorted(wass_all.items(),key=lambda x:x[1])
    tab_lines=['WASSERSTEIN H1 — RANKING:\n',
               f'{"Par":>12}   {"Distancia":>10}   Interpretacion']
    tab_lines.append('─'*44)
    wm=np.mean(list(wass_all.values())); ws=np.std(list(wass_all.values()))
    for par,val in wass_sorted:
        interp='similar' if val<wm-0.5*ws else 'divergente' if val>wm+0.5*ws else 'intermedio'
        tab_lines.append(f'{par:>12}   {val:>10.4f}   {interp}')
    tab_lines.append(f'\nKURAMOTO: R_medio={R_mean:.3f}  Azar={R_shuf_mean:.3f}±{R_shuf_std:.3f}')
    ax34f.text(0.03,0.95,'\n'.join(tab_lines),transform=ax34f.transAxes,va='top',ha='left',
               fontsize=8.5,color=C_TEXTO,fontfamily='monospace',
               bbox=dict(boxstyle='round',facecolor='#0a0a1e',edgecolor='#2a2a4a',alpha=0.9))
    ax34f.set_title('Resumen topologico',color=C_TEXTO,fontsize=9,fontweight='bold')
    fig34.text(0.01,0.01,'Kuramoto: baseline shuffleado 100 permutaciones. Wasserstein: Vietoris-Rips H1 submuestra. PL: fit global.',
               color='#555577',fontsize=6.5,style='italic')
    savefig(fig34,'34_kuramoto_wasserstein_landscape')
else:
    print('  [SKIP] Fig 34: pip install ripser persim')

# ══════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('  PIPELINE CEBRA COMPLETADO')
print('='*60)
print(f'  Output: {OUTPUT_DIR}')
print('  Figuras generadas:')
for n,d in [('25','CEBRA-Time embedding'),('26','Embeddings por hipotesis'),
            ('27','Ranking InfoNCE'),('28','Divergencia inter-abeja'),
            ('29','Estados comportamentales'),('30','Latentes temporales'),
            ('31','Toro + PSDs Welch'),('32','Lomb-Scargle + CWT'),
            ('33','TDA homologia persistente'),('34','Kuramoto + Wasserstein + Landscape')]:
    print(f'    Fig {n}: {d}')
print('\n  Ranking InfoNCE:')
for rank,(k,val) in enumerate(sorted(modelos.items(),key=lambda x:x[1]['loss']),1):
    print(f'    {rank}. {MODELOS_META[k]["nombre"]:25s}  InfoNCE={val["loss"]:.4f}')
print('\n  Seed usado:', SEED, '— resultados 100% reproducibles')
print('='*60)

# ══════════════════════════════════════════════════════════════════
# FIG 35 — COMPARACION: 2 modos (M0+M1) vs 4 modos (M0-M3)
# Demuestra por que M2 y M3 no son resolubles con bins de 5s
# ══════════════════════════════════════════════════════════════════
print("\nFig 35 - Comparacion 2 modos vs 4 modos...")

# ─── Construir X con los 4 modos para comparar ───────────────────
def construir_X_4modos():
    feats=[]; names=[]; env_b7={}
    if 'Bee7' in modes_all:
        for k in range(VMD_K):
            env_b7[k]=gaussian_filter1d(np.abs(hilbert(modes_all['Bee7'][k][:N_t])),sigma=SIGMA_ENV_VIZ)
    for bee in BEES_CEBRA:
        if bee not in modes_all: continue
        for k in range(VMD_K):
            env=gaussian_filter1d(np.abs(hilbert(modes_all[bee][k][:N_t])),sigma=SIGMA_ENV_VIZ)
            if k in env_b7: env=np.maximum(env-ALPHA_SUB*env_b7[k],BETA_FLOOR*env_b7[k])
            feats.append(env); names.append(f'{bee}_M{k}')
        if bee in angulos_all:
            for an,av in angulos_all[bee].items():
                feats.append(av[:N_t]); names.append(f'{bee}_{an}')
    X_raw4=np.stack(feats,axis=1).astype(np.float64)
    return zscore(X_raw4), names

print("  Construyendo X con 4 modos (todos)...")
X4, feat_names4 = construir_X_4modos()
print(f"  X4 shape: {X4.shape}  (16 VMD + 24 angulares)")
print(f"  X2 shape: {X.shape}   ({n_vmd} VMD + {n_ang} angulares)  <- el que usa el paper")

# ─── Entrenar modelos comparativos con 4 modos ───────────────────
print("\n  Entrenando modelos con 4 modos para comparar...")
modelos4 = {}
for key, lbl in [('time',None), ('hora',label_hora),
                  ('modo',label_modo), ('angulo',label_ang), ('sueno',label_sueno)]:
    kw=dict(model_architecture='offset10-model', batch_size=CEBRA_BATCH,
            learning_rate=CEBRA_LR, max_iterations=CEBRA_ITER,
            time_offsets=CEBRA_OFFS, output_dimension=CEBRA_DIM,
            device='cuda_if_available', verbose=False)
    if lbl is not None: kw['conditional']='time_delta'
    if TIENE_TORCH: torch.manual_seed(SEED)
    np.random.seed(SEED)
    m4=CEBRA(**kw)
    m4.fit(X4) if lbl is None else m4.fit(X4, lbl)
    emb4=m4.transform(X4)
    loss4=np.array(m4.state_dict_['loss']).min()
    modelos4[key]=dict(emb=emb4, loss=loss4)
    print(f"    {MODELOS_META[key]['nombre']:25s}  InfoNCE(4modos)={loss4:.4f}  "
          f"InfoNCE(2modos)={modelos[key]['loss']:.4f}  "
          f"diff={modelos[key]['loss']-loss4:+.4f}")

# ─── PSD de M2 y M3 para mostrar que son ruido ──────────────────
print("  Calculando PSD de M2 y M3...")
psds_modos = {}
modo_cols_all = ['#00b4d8','#a29bfe','#ffd166','#ef476f']
for k in range(VMD_K):
    sig_k = np.mean([
        gaussian_filter1d(np.abs(hilbert(modes_all[b][k][:N_t])), sigma=SIGMA_ENV_VIZ)
        for b in BEES_CEBRA if b in modes_all], axis=0)
    T,P = psd_welch(sig_k)
    psds_modos[k] = (T, P)

# ─── Figura 35 ────────────────────────────────────────────────────
fig35 = plt.figure(figsize=(26, 16), facecolor=C_FONDO)
fig35.suptitle(
    'Justificacion del Uso de Solo M0+M1 en CEBRA\n'
    f'Bins de {BIN_SEGUNDOS}s → Nyquist = {BIN_SEGUNDOS*2}s | '
    f'Resolucion practica confiable: periodos > {BIN_SEGUNDOS*20}s = {BIN_SEGUNDOS*20//60} min',
    color=C_TEXTO, fontsize=12, fontweight='bold')

gs35 = GridSpec(3, 4, figure=fig35,
                hspace=0.42, wspace=0.28,
                left=0.05, right=0.97, top=0.90, bottom=0.06)

# ── Fila 0: PSD de cada modo VMD ─────────────────────────────────
for k in range(VMD_K):
    ax = fig35.add_subplot(gs35[0, k])
    ax.set_facecolor(C_PANEL)
    T, P = psds_modos[k]
    mask = (T >= 2) & (T <= 360)
    col = modo_cols_all[k]

    # Zona de resolucion confiable
    ax.axvspan(2, BIN_SEGUNDOS*20/60, color='#ef476f', alpha=0.15, zorder=0,
               label=f'No confiable\n(<{BIN_SEGUNDOS*20//60} min)')
    ax.axvline(BIN_SEGUNDOS*20/60, color='#ef476f', lw=1.5, ls='--', alpha=0.8)
    ax.axvline(BIN_SEGUNDOS*2/60, color='#ef476f', lw=0.8, ls=':', alpha=0.6,
               label=f'Nyquist ({BIN_SEGUNDOS*2}s)')

    ax.fill_between(T[mask], 0, P[mask], color=col, alpha=0.25)
    ax.plot(T[mask], P[mask], color=col, lw=1.0, alpha=0.9)

    # Bandas VMD
    for (_, lo, hi, bc, tag) in VMD_BANDAS:
        ax.axvspan(max(lo,2), min(hi,360), color=bc, alpha=0.06, zorder=0)

    resolvable = 'RESOLVABLE' if k in [0,1] else 'NO CONFIABLE'
    col_r = '#06d6a0' if k in [0,1] else '#ef476f'
    ax.set_title(f'M{k} — {["lento >120min","medio 30-120min","rapido 10-30min","muy rap. <10min"][k]}\n'
                 f'→ {resolvable} con bin={BIN_SEGUNDOS}s',
                 color=col_r, fontsize=9, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(2, 360)
    ax.set_xlabel('Periodo (min)', color=C_TEXTO, fontsize=8)
    ax.set_ylabel('Densidad espectral', color=C_TEXTO, fontsize=8)
    ax.tick_params(colors=C_TEXTO, labelsize=6)
    ax.spines[:].set_color(C_GRID)
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3, which='both')
    ticks = [5, 10, 20, 30, 60, 120, 180, 360]
    ax.set_xticks([t for t in ticks if 2<=t<=360])
    ax.set_xticklabels([str(t) for t in ticks if 2<=t<=360], color=C_TEXTO, fontsize=6)
    if k == 0:
        ax.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)

# ── Fila 1: Comparacion InfoNCE 2 vs 4 modos ─────────────────────
ax_cmp = fig35.add_subplot(gs35[1, :2])
ax_cmp.set_facecolor(C_PANEL)

keys_plot = list(MODELOS_META.keys())
x_pos = np.arange(len(keys_plot))
w = 0.35

bars2 = ax_cmp.bar(x_pos - w/2,
                   [modelos[k]['loss'] for k in keys_plot],
                   width=w, color=[MODELOS_META[k]['color'] for k in keys_plot],
                   alpha=0.9, label='2 modos (M0+M1) — recomendado', edgecolor='none')
bars4 = ax_cmp.bar(x_pos + w/2,
                   [modelos4[k]['loss'] for k in keys_plot],
                   width=w, color=[MODELOS_META[k]['color'] for k in keys_plot],
                   alpha=0.4, label='4 modos (M0-M3) — con ruido', edgecolor='white',
                   linewidth=0.5)

# Valores encima de cada barra
for pos, k in zip(x_pos, keys_plot):
    v2 = modelos[k]['loss'];  v4 = modelos4[k]['loss']
    ax_cmp.text(pos-w/2, v2+0.005, f'{v2:.4f}', ha='center', va='bottom',
                color='white', fontsize=6.5, fontweight='bold')
    ax_cmp.text(pos+w/2, v4+0.005, f'{v4:.4f}', ha='center', va='bottom',
                color='#aaaaaa', fontsize=6.5)
    # Diferencia
    diff = v2 - v4
    col_d = '#06d6a0' if diff < 0 else '#ef476f'
    ax_cmp.text(pos, min(v2,v4)-0.015, f'{diff:+.4f}', ha='center', va='top',
                color=col_d, fontsize=6, fontweight='bold')

ax_cmp.set_xticks(x_pos)
ax_cmp.set_xticklabels([MODELOS_META[k]['nombre'] for k in keys_plot],
                        color=C_TEXTO, fontsize=8, rotation=20, ha='right')
ax_cmp.set_ylabel('InfoNCE Loss (menor = mejor)', color=C_TEXTO, fontsize=9)
ax_cmp.set_title('Comparacion InfoNCE: 2 modos vs 4 modos\n'
                 'Diferencia bajo 0 = 2 modos gana (menos ruido)',
                 color=C_TEXTO, fontsize=10, fontweight='bold')
ax_cmp.tick_params(colors=C_TEXTO, labelsize=7)
ax_cmp.spines[:].set_color(C_GRID)
ax_cmp.grid(color=C_GRID, alpha=0.3, lw=0.3, axis='y')
ax_cmp.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.7)

# ── Comparacion embeddings (toro 2D) ─────────────────────────────
for col_i, (label, emb_use, title, col) in enumerate([
    ('2 modos (M0+M1)', emb_time,              '2 modos — embedding', '#06d6a0'),
    ('4 modos (M0-M3)', modelos4['time']['emb'], '4 modos — embedding', '#ef476f')], 2):
    ax_t = fig35.add_subplot(gs35[1, col_i])
    ax_t.set_facecolor(C_PANEL)
    sc_t = ax_t.scatter(emb_use[:, 0], emb_use[:, 1],
                        c=hora_norm, cmap='hsv',
                        s=0.6, alpha=0.4, linewidths=0, rasterized=True)
    ax_t.set_xlabel('Dim 0', color=C_TEXTO, fontsize=8)
    ax_t.set_ylabel('Dim 1', color=C_TEXTO, fontsize=8)
    ax_t.set_title(title + f'\nDim 0 vs 1 — coloreado por hora',
                   color=col, fontsize=9, fontweight='bold')
    ax_t.tick_params(colors=C_TEXTO, labelsize=6)
    ax_t.spines[:].set_color(C_GRID)
    ax_t.set_aspect('equal')
    plt.colorbar(sc_t, ax=ax_t, shrink=0.7).ax.yaxis.set_tick_params(
        color=C_TEXTO, labelsize=5)

# ── Fila 2: Tabla resumen + argumento Nyquist ─────────────────────
ax_arg = fig35.add_subplot(gs35[2, :2])
ax_arg.set_facecolor(C_PANEL)
ax_arg.axis('off')

nyquist_s = BIN_SEGUNDOS * 2
practica_s = BIN_SEGUNDOS * 20
nyquist_min = nyquist_s / 60
practica_min = practica_s / 60

resumen = (
    f'ARGUMENTO DE MUESTREO — Justificacion de MODOS_VALIDOS = [0, 1]\n'
    f'{"─"*54}\n'
    f'  Bin de muestreo:       {BIN_SEGUNDOS} segundos\n'
    f'  Frecuencia Nyquist:    {1/nyquist_s:.3f} Hz  (periodo minimo teorico: {nyquist_min:.0f}s)\n'
    f'  Resolucion practica:   ~20 muestras/ciclo → periodo min. confiable: {practica_min:.0f} min\n'
    f'\n'
    f'  Modo  Periodo tipico   Muestras/ciclo   Decision\n'
    f'  {"─"*50}\n'
    f'  M0    >120 min         >1440            INCLUIR en CEBRA ✓\n'
    f'  M1    30-120 min       360-1440         INCLUIR en CEBRA ✓\n'
    f'  M2    10-30 min        120-360          EXCLUIR — marginal con bin={BIN_SEGUNDOS}s ✗\n'
    f'  M3    <10 min          <120             EXCLUIR — no confiable con bin={BIN_SEGUNDOS}s ✗\n'
    f'\n'
    f'  Para resolver M2 y M3: rebin a 60s (analisis separado, no CEBRA)'
)
ax_arg.text(0.03, 0.95, resumen, transform=ax_arg.transAxes,
            va='top', ha='left', fontsize=8.5, color=C_TEXTO,
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#0a0a1e',
                      edgecolor='#2a2a4a', alpha=0.9))
ax_arg.set_title('Argumento de muestreo', color=C_TEXTO, fontsize=9, fontweight='bold')

# ── Delta InfoNCE por modelo ──────────────────────────────────────
ax_delta = fig35.add_subplot(gs35[2, 2:])
ax_delta.set_facecolor(C_PANEL)
deltas = [modelos[k]['loss'] - modelos4[k]['loss'] for k in keys_plot]
cols_d = ['#06d6a0' if d < 0 else '#ef476f' for d in deltas]
bars_d = ax_delta.bar(range(len(keys_plot)), deltas, color=cols_d,
                      edgecolor='none', alpha=0.85)
ax_delta.axhline(0, color='white', lw=1.0, ls='-', alpha=0.5)
for i, (b, d) in enumerate(zip(bars_d, deltas)):
    ax_delta.text(b.get_x()+b.get_width()/2,
                  d + (0.003 if d >= 0 else -0.003),
                  f'{d:+.4f}', ha='center',
                  va='bottom' if d >= 0 else 'top',
                  color='white', fontsize=8, fontweight='bold')
ax_delta.set_xticks(range(len(keys_plot)))
ax_delta.set_xticklabels([MODELOS_META[k]['nombre'] for k in keys_plot],
                          color=C_TEXTO, fontsize=8, rotation=20, ha='right')
ax_delta.set_ylabel('Delta InfoNCE\n(2 modos − 4 modos)', color=C_TEXTO, fontsize=9)
ax_delta.set_title('Impacto de excluir M2/M3\n'
                   'Verde = 2 modos gana (menos ruido) | Rojo = 4 modos gana',
                   color=C_TEXTO, fontsize=10, fontweight='bold')
ax_delta.tick_params(colors=C_TEXTO, labelsize=7)
ax_delta.spines[:].set_color(C_GRID)
ax_delta.grid(color=C_GRID, alpha=0.3, lw=0.3, axis='y')
ax_delta.text(0.5, 0.98,
    'Si verde domina: M2/M3 eran ruido. Si rojo domina: M2/M3 aportaban.',
    transform=ax_delta.transAxes, ha='center', va='top',
    color='#aaaaaa', fontsize=7, style='italic')

fig35.text(0.01, 0.01,
    f'PSD: Welch sobre envolvente media de cada modo (Bee3-Bee6, denoised Bee7). '
    f'Linea roja = limite practico de resolucion ({practica_min:.0f}min con bin={BIN_SEGUNDOS}s). '
    f'Delta = InfoNCE(2modos) - InfoNCE(4modos): negativo = 2 modos es mejor.',
    color='#555577', fontsize=6.5, style='italic')

savefig(fig35, '35_comparacion_2modos_vs_4modos')

print(f"\n{'='*60}")
print(f"  RESULTADO COMPARACION 2 vs 4 MODOS")
print(f"{'='*60}")
mejora = [(k, modelos[k]['loss']-modelos4[k]['loss']) for k in keys_plot]
mejora.sort(key=lambda x: x[1])
for k, d in mejora:
    signo = "2modos GANA" if d < 0 else "4modos gana"
    print(f"  {MODELOS_META[k]['nombre']:25s}  delta={d:+.4f}  [{signo}]")
print(f"{'='*60}")

# ══════════════════════════════════════════════════════════════════
# ANALISIS DE TWITCHES CON CEBRA — Opciones A, B y C
OUTPUT_DIR_TWITCHES = os.path.join(OUTPUT_DIR, 'cebra_twitches')
os.makedirs(OUTPUT_DIR_TWITCHES, exist_ok=True)
# ══════════════════════════════════════════════════════════════════
#
# A) Tasa de twitches como label → ¿el embedding lento predice twitches?
# B) Twitch-triggered trajectories → ¿los twitches de sueño vs vigilia
#    viven en zonas distintas del embedding?
# C) Features multi-escala (M0/M1 + tasa twitches) → embedding unificado
#
# Nuevas figuras:
#   36_twitch_infonce_ranking        — InfoNCE con label twitch (A)
#   37_twitch_triggered_embedding    — trayectorias en embedding (B)
#   38_twitch_multiscala_embedding   — embedding multi-escala (C)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CEBRA TWITCHES — Opciones A, B, C")
print("="*60)

from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── Deteccion de twitches en M2/M3 a 5s bins ─────────────────────
def detectar_twitches_5s(modo, sigma_umbral=2.0, dist_bins=2):
    """Detecta picos en amplitud Hilbert de un modo VMD (bins 5s)."""
    amp = np.abs(hilbert(modo))
    mu  = np.median(amp)
    sd  = np.std(amp)
    picos, _ = find_peaks(amp,
                           height=mu + sigma_umbral*sd,
                           distance=dist_bins,
                           prominence=sd*0.3)
    return picos, amp

# Construir serie de tasa de twitches por abeja
print("  Detectando twitches en M2/M3 (5s bins)...")
twitch_por_abeja  = {}   # twitch_por_abeja[bee] = serie binaria (N_t,)
env_m0_por_abeja  = {}   # envolvente M0 para clasificar estado

WIN_ESTADO_BINS = int(10*60/BIN_SEGUNDOS)   # ventana 10 min para estado

for bee in BEES_CEBRA:
    if bee not in modes_all: continue
    serie_bee = np.zeros(N_t)
    for k in [2, 3]:
        picos, amp = detectar_twitches_5s(modes_all[bee][k][:N_t])
        serie_bee[picos] += amp[picos]
    twitch_por_abeja[bee] = serie_bee
    env_m0_por_abeja[bee] = uniform_filter1d(
        np.abs(hilbert(modes_all[bee][0][:N_t])),
        size=WIN_ESTADO_BINS)
    n_tw = (serie_bee > 0).sum()
    print(f"    {bee}: {n_tw} twitches  "
          f"({n_tw/(N_t*BIN_SEGUNDOS/3600):.0f}/h)")

# Tasa de grupo (suma de todas las abejas, suavizada)
twitch_grupo = sum(twitch_por_abeja.values())
WIN_RATE_BINS = int(5*60/BIN_SEGUNDOS)   # suavizado 5 min
twitch_rate_sm = gaussian_filter1d(twitch_grupo, sigma=WIN_RATE_BINS)

# Clasificacion de cada bin: alta/baja actividad segun M0
env_m0_grupo = np.mean(list(env_m0_por_abeja.values()), axis=0)
th_alta = np.percentile(env_m0_grupo, 66)
th_baja = np.percentile(env_m0_grupo, 33)
mask_alta_tw = env_m0_grupo >= th_alta
mask_baja_tw = env_m0_grupo <= th_baja

n_alta = mask_alta_tw.sum(); n_baja = mask_baja_tw.sum()
tw_alta = (twitch_grupo[mask_alta_tw]>0).sum() / max(n_alta*BIN_SEGUNDOS/3600, 0.01)
tw_baja = (twitch_grupo[mask_baja_tw]>0).sum() / max(n_baja*BIN_SEGUNDOS/3600, 0.01)
print(f"\n  Tasa twitches en alta actividad: {tw_alta:.0f}/h")
print(f"  Tasa twitches en baja actividad: {tw_baja:.0f}/h")
print(f"  Ratio baja/alta: {tw_baja/max(tw_alta,0.01):.2f}x")

# ══════════════════════════════════════════════════════════════════
# OPCION A — Tasa de twitches como label CEBRA-Behavior
# ══════════════════════════════════════════════════════════════════
print("\n  [A] Entrenando CEBRA con label twitch rate...")
label_twitch = twitch_rate_sm.reshape(-1,1).astype(np.float64)
label_twitch = ((label_twitch - label_twitch.mean()) /
                (label_twitch.std() + 1e-9))   # z-score

if TIENE_TORCH: torch.manual_seed(SEED)
np.random.seed(SEED)
mod_tw = CEBRA(model_architecture='offset10-model',
               batch_size=CEBRA_BATCH, learning_rate=CEBRA_LR,
               conditional='time_delta',
               max_iterations=CEBRA_ITER, time_offsets=CEBRA_OFFS,
               output_dimension=CEBRA_DIM,
               device='cuda_if_available', verbose=True)
mod_tw.fit(X, label_twitch)
emb_twitch_label = mod_tw.transform(X)
loss_tw = np.array(mod_tw.state_dict_['loss']).min()
modelos['twitch'] = dict(emb=emb_twitch_label, loss=loss_tw)
MODELOS_META['twitch'] = dict(nombre='Tasa twitches',
                                color='#ef476f',
                                label_desc='tasa M2/M3 suavizada')
mod_tw.save(os.path.join(OUTPUT_DIR, 'cebra_twitch_model.pt'))
print(f"  [A] InfoNCE={loss_tw:.4f}  "
      f"({'gana' if loss_tw < modelos['time']['loss'] else 'pierde'} "
      f"vs baseline {modelos['time']['loss']:.4f})")

# ══════════════════════════════════════════════════════════════════
# OPCION C — Features multi-escala (M0/M1 + tasa twitches por abeja)
# ══════════════════════════════════════════════════════════════════
print("\n  [C] Construyendo X multi-escala (M0/M1 + twitch rate)...")
tw_feats = []
tw_names = []
for bee in BEES_CEBRA:
    if bee not in twitch_por_abeja: continue
    rate = gaussian_filter1d(twitch_por_abeja[bee], sigma=WIN_RATE_BINS)
    tw_feats.append(rate)
    tw_names.append(f'{bee}_twitch_rate')

if tw_feats:
    X_tw_extra = zscore(np.stack(tw_feats, axis=1))   # (N_t, 4)
    X_multi    = np.concatenate([X, X_tw_extra], axis=1)
    print(f"  [C] X_multi shape: {X_multi.shape}  "
          f"({X.shape[1]} M0/M1+ang + {X_tw_extra.shape[1]} twitch_rate)")

    if TIENE_TORCH: torch.manual_seed(SEED)
    np.random.seed(SEED)
    mod_multi = CEBRA(model_architecture='offset10-model',
                      batch_size=CEBRA_BATCH, learning_rate=CEBRA_LR,
                      max_iterations=CEBRA_ITER, time_offsets=CEBRA_OFFS,
                      output_dimension=CEBRA_DIM,
                      device='cuda_if_available', verbose=True)
    mod_multi.fit(X_multi)
    emb_multi = mod_multi.transform(X_multi)
    loss_multi = np.array(mod_multi.state_dict_['loss']).min()
    mod_multi.save(os.path.join(OUTPUT_DIR, 'cebra_multiscala_model.pt'))
    print(f"  [C] InfoNCE={loss_multi:.4f}  "
          f"({'mejor' if loss_multi < modelos['time']['loss'] else 'peor'} "
          f"que M0/M1 solo: {modelos['time']['loss']:.4f})")
else:
    X_multi = X; emb_multi = emb_time; loss_multi = modelos['time']['loss']
    print("  [C] Sin features de twitches disponibles")

# ══════════════════════════════════════════════════════════════════
# OPCION B — Twitch-triggered trajectories en embedding
# ══════════════════════════════════════════════════════════════════
print("\n  [B] Extrayendo trayectorias twitch-triggered...")

WIN_TRIG_BINS = int(5*60/BIN_SEGUNDOS)   # +/- 5 min alrededor del twitch
t_trig = (np.arange(2*WIN_TRIG_BINS+1) - WIN_TRIG_BINS) * BIN_SEGUNDOS/60  # min

traj_alta = []   # trayectorias durante alta actividad
traj_baja = []   # trayectorias durante baja actividad

for bee in BEES_CEBRA:
    if bee not in twitch_por_abeja: continue
    # Picos de twitches de esta abeja
    picos_bee = np.where(twitch_por_abeja[bee] > 0)[0]
    for p in picos_bee:
        if p < WIN_TRIG_BINS or p + WIN_TRIG_BINS >= N_t:
            continue
        ventana = emb_time[p-WIN_TRIG_BINS : p+WIN_TRIG_BINS+1]  # (2W+1, 3)
        if mask_alta_tw[p]:
            traj_alta.append(ventana)
        elif mask_baja_tw[p]:
            traj_baja.append(ventana)

traj_alta = np.array(traj_alta) if traj_alta else np.zeros((0, 2*WIN_TRIG_BINS+1, 3))
traj_baja = np.array(traj_baja) if traj_baja else np.zeros((0, 2*WIN_TRIG_BINS+1, 3))

print(f"  [B] Trayectorias alta actividad: {len(traj_alta)}")
print(f"  [B] Trayectorias baja actividad: {len(traj_baja)}")

# ══════════════════════════════════════════════════════════════════
# FIG 36 — InfoNCE ranking actualizado con opcion A
# ══════════════════════════════════════════════════════════════════
print("\nFig 36 - InfoNCE ranking con twitches...")
fig36, ax36 = plt.subplots(figsize=(12, 6), facecolor=C_FONDO)
ax36.set_facecolor(C_PANEL)

keys_ord36 = sorted(MODELOS_META.keys(),
                    key=lambda k: modelos[k]['loss'] if k in modelos else 99)
losses36  = [modelos[k]['loss'] if k in modelos else 99 for k in keys_ord36]
cols36    = [MODELOS_META[k]['color'] for k in keys_ord36]
noms36    = [MODELOS_META[k]['nombre'] for k in keys_ord36]

bars36 = ax36.barh(range(len(keys_ord36)), losses36,
                   color=cols36, edgecolor='none', height=0.6, alpha=0.85)
ax36.axvline(modelos['time']['loss'], color='white',
             lw=1.5, ls='--', alpha=0.7, label='CEBRA-Time baseline')

for i, (bar, loss, nom) in enumerate(zip(bars36, losses36, noms36)):
    ax36.text(loss+0.003, i, f'{loss:.4f}',
              va='center', color='white', fontsize=9, fontweight='bold')

ax36.set_yticks(range(len(keys_ord36)))
ax36.set_yticklabels(noms36, color=C_TEXTO, fontsize=10)
ax36.set_xlabel('InfoNCE Loss (menor = mejor)', color=C_TEXTO, fontsize=10)
ax36.set_title(
    'Ranking InfoNCE actualizado — incluyendo tasa de twitches (Opcion A)\n'
    'Si twitches < baseline: los ritmos lentos predicen cuando ocurren los twitches',
    color=C_TEXTO, fontsize=11, fontweight='bold')
ax36.tick_params(colors=C_TEXTO, labelsize=8)
ax36.spines[:].set_color(C_GRID)
ax36.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.6)
ax36.invert_yaxis()

ganador36 = keys_ord36[0]
tw_result = ('TWITCHES ACOPLADOS al ciclo sueno/vigilia'
             if loss_tw < modelos['time']['loss']
             else 'twitches NO predichos por ritmos lentos')
fig36.text(0.5, 0.01,
    f'Opcion A (twitch rate): InfoNCE={loss_tw:.4f} | ' + tw_result,
    color=MODELOS_META['twitch']['color'],
    fontsize=9, ha='center', fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 1])
savefig(fig36, '36_twitch_infonce_ranking', outdir=OUTPUT_DIR_TWITCHES)

# ══════════════════════════════════════════════════════════════════
# FIG 37 — Twitch-triggered trajectories en embedding (Opcion B)
# ══════════════════════════════════════════════════════════════════
print("Fig 37 - Twitch-triggered embedding...")
fig37 = plt.figure(figsize=(24, 10), facecolor=C_FONDO)
fig37.suptitle(
    'Twitch-Triggered Trajectories en Embedding CEBRA-Time (Opcion B)\n'
    'Trayectoria del embedding ±5min alrededor de cada twitch | Alta vs Baja actividad',
    color=C_TEXTO, fontsize=12, fontweight='bold')
gs37 = GridSpec(2, 3, figure=fig37, hspace=0.38, wspace=0.28,
                left=0.06, right=0.97, top=0.88, bottom=0.06)

# Paneles 2D: dim0 vs dim1, dim1 vs dim2
for col_i, (d0, d1) in enumerate([(0,1),(1,2)]):
    ax = fig37.add_subplot(gs37[:,col_i])
    ax.set_facecolor(C_PANEL)

    if len(traj_alta) > 0:
        mu_a = traj_alta[:,:,d0].mean(0)
        sd_a = traj_alta[:,:,d0].std(0)
        mu_b_y = traj_alta[:,:,d1].mean(0)
        ax.fill_between(mu_a, mu_b_y - sd_a, mu_b_y + sd_a,
                        color=COL_ALTA, alpha=0.2)
        ax.plot(mu_a, mu_b_y, color=COL_ALTA, lw=2.5,
                label=f'Alta act. (n={len(traj_alta)})', alpha=0.9)
        ax.scatter(mu_a[WIN_TRIG_BINS], mu_b_y[WIN_TRIG_BINS],
                   color=COL_ALTA, s=120, zorder=10,
                   edgecolors='white', lw=2)

    if len(traj_baja) > 0:
        mu_a2 = traj_baja[:,:,d0].mean(0)
        sd_a2 = traj_baja[:,:,d0].std(0)
        mu_b_y2= traj_baja[:,:,d1].mean(0)
        ax.fill_between(mu_a2, mu_b_y2 - sd_a2, mu_b_y2 + sd_a2,
                        color=COL_BAJA, alpha=0.2)
        ax.plot(mu_a2, mu_b_y2, color=COL_BAJA, lw=2.5,
                label=f'Baja act. (n={len(traj_baja)})', alpha=0.9)
        ax.scatter(mu_a2[WIN_TRIG_BINS], mu_b_y2[WIN_TRIG_BINS],
                   color=COL_BAJA, s=120, zorder=10,
                   edgecolors='white', lw=2)

    ax.set_xlabel(f'Dim {d0}', color=C_TEXTO, fontsize=9)
    ax.set_ylabel(f'Dim {d1}', color=C_TEXTO, fontsize=9)
    ax.set_title(f'Dim {d0} vs {d1}\nTrayectoria media ± std',
                 color=C_TEXTO, fontsize=10, fontweight='bold')
    ax.tick_params(colors=C_TEXTO, labelsize=7)
    ax.spines[:].set_color(C_GRID)
    ax.legend(fontsize=9, facecolor=C_PANEL, labelcolor=C_TEXTO, framealpha=0.7)
    ax.grid(color=C_GRID, alpha=0.3, lw=0.3)
    ax.text(0.5, 0.97, '● = momento del twitch',
            transform=ax.transAxes, ha='center', va='top',
            color='white', fontsize=7)

# Panel temporal: cada dimension vs tiempo alrededor del twitch
ax_t = fig37.add_subplot(gs37[:,2])
ax_t.set_facecolor(C_PANEL)
dim_cols = COLS_LAT
for dim in range(CEBRA_DIM):
    for traj, col_est, lbl in [
        (traj_alta, COL_ALTA, f'Alta (dim{dim})'),
        (traj_baja, COL_BAJA, f'Baja (dim{dim})')]:
        if len(traj) == 0: continue
        mu  = traj[:,:,dim].mean(0)
        sd  = traj[:,:,dim].std(0)
        ax_t.fill_between(t_trig, mu-sd, mu+sd,
                          color=col_est, alpha=0.08)
        ax_t.plot(t_trig, mu,
                  color=col_est, lw=1.5 if dim==0 else 0.8,
                  ls=['-','--',':'][dim], alpha=0.85,
                  label=lbl if dim==0 else None)

ax_t.axvline(0, color='white', lw=1.5, ls='--', alpha=0.6, label='Twitch')
ax_t.axhline(0, color='white', lw=0.3, alpha=0.2)
ax_t.set_xlabel('Tiempo desde twitch (min)', color=C_TEXTO, fontsize=9)
ax_t.set_ylabel('Latente (media ± std)', color=C_TEXTO, fontsize=9)
ax_t.set_title('Dinamica temporal alrededor del twitch\n',
               color=C_TEXTO, fontsize=10, fontweight='bold')
ax_t.tick_params(colors=C_TEXTO, labelsize=7)
ax_t.spines[:].set_color(C_GRID)
ax_t.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXTO,
            framealpha=0.6, ncol=2)
ax_t.grid(color=C_GRID, alpha=0.3, lw=0.3)

fig37.text(0.01, 0.01,
    'Trayectoria media del embedding CEBRA-Time en ventana de ±5min alrededor de cada twitch. '
    'Si alta != baja: los twitches de sueno y vigilia ocurren en zonas distintas del espacio latente.',
    color='#555577', fontsize=7, style='italic')
savefig(fig37, '37_twitch_triggered_embedding', outdir=OUTPUT_DIR_TWITCHES)

# ══════════════════════════════════════════════════════════════════
# FIG 38 — Embedding multi-escala vs M0/M1 solo (Opcion C)
# ══════════════════════════════════════════════════════════════════
print("Fig 38 - Embedding multi-escala...")
fig38, axes38 = plt.subplots(2, 3, figsize=(22, 12), facecolor=C_FONDO,
                              gridspec_kw={'hspace':0.35,'wspace':0.22})
fig38.suptitle(
    'Embedding Multi-Escala (Opcion C): M0/M1 + Tasa Twitches\n'
    f'X original: {X.shape[1]} features | X multi: {X_multi.shape[1]} features | '
    f'InfoNCE M0/M1={modelos["time"]["loss"]:.4f} | Multi={loss_multi:.4f}',
    color=C_TEXTO, fontsize=11, fontweight='bold')

for col_i, (emb_plot, titulo, col_t) in enumerate([
    (emb_time,  f'M0/M1 solo\n({X.shape[1]} feat)', '#aaaaaa'),
    (emb_multi, f'M0/M1 + Twitches\n({X_multi.shape[1]} feat)', '#ef476f'),
    (emb_twitch_label, 'CEBRA-Behavior\n(label = twitch rate)', '#ef476f')]):

    # Fila 0: coloreado por hora
    ax = axes38[0, col_i]; ax.set_facecolor(C_PANEL)
    sc = ax.scatter(emb_plot[:,0], emb_plot[:,1],
                    c=hora_norm, cmap='hsv', s=0.6, alpha=0.4,
                    linewidths=0, rasterized=True)
    ax.set_title(titulo + '\nDim 0 vs 1 — hora',
                 color=col_t, fontsize=9, fontweight='bold')
    ax.set_xlabel('Dim 0', color=C_TEXTO, fontsize=8)
    ax.set_ylabel('Dim 1', color=C_TEXTO, fontsize=8)
    ax.tick_params(colors=C_TEXTO, labelsize=6)
    ax.spines[:].set_color(C_GRID)
    plt.colorbar(sc, ax=ax, shrink=0.7).ax.yaxis.set_tick_params(
        color=C_TEXTO, labelsize=5)

    # Fila 1: coloreado por tasa de twitches
    tw_norm = (twitch_rate_sm - twitch_rate_sm.min())
    tw_max  = tw_norm.max()
    if tw_max > 0: tw_norm /= tw_max
    ax2 = axes38[1, col_i]; ax2.set_facecolor(C_PANEL)
    sc2 = ax2.scatter(emb_plot[:,0], emb_plot[:,1],
                      c=tw_norm, cmap='hot', s=0.6, alpha=0.4,
                      linewidths=0, rasterized=True, vmin=0, vmax=1)
    ax2.set_title(titulo + '\nDim 0 vs 1 — tasa twitches',
                  color=col_t, fontsize=9, fontweight='bold')
    ax2.set_xlabel('Dim 0', color=C_TEXTO, fontsize=8)
    ax2.set_ylabel('Dim 1', color=C_TEXTO, fontsize=8)
    ax2.tick_params(colors=C_TEXTO, labelsize=6)
    ax2.spines[:].set_color(C_GRID)
    cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.7)
    cb2.set_label('Tasa twitches norm.', color=C_TEXTO, fontsize=6)
    cb2.ax.yaxis.set_tick_params(color=C_TEXTO, labelsize=5)

fig38.text(0.01, 0.01,
    f'Comparacion: M0/M1 solo vs M0/M1+twitches vs CEBRA-Behavior(twitch). '
    f'Si los twitches organizan el embedding: la tasa de twitches deberia '
    f'mostrar gradiente en fila 1.',
    color='#555577', fontsize=7, style='italic')
savefig(fig38, '38_twitch_multiscala_embedding', outdir=OUTPUT_DIR_TWITCHES)

# ══════════════════════════════════════════════════════════════════
# RESUMEN TWITCHES + CEBRA
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  RESUMEN CEBRA TWITCHES")
print("="*60)
print(f"  Tasa twitches: alta={tw_alta:.0f}/h  baja={tw_baja:.0f}/h  "
      f"ratio={tw_baja/max(tw_alta,0.01):.2f}x")
print(f"\n  InfoNCE:")
print(f"    Baseline (time):  {modelos['time']['loss']:.4f}")
print(f"    Opcion A (label): {loss_tw:.4f}  "
      f"-> {'GANA: twitches acoplados' if loss_tw < modelos['time']['loss'] else 'PIERDE: twitches independientes'}")
print(f"    Opcion C (multi): {loss_multi:.4f}  "
      f"-> {'GANA: twitches aportan info' if loss_multi < modelos['time']['loss'] else 'PIERDE: twitches no aportan'}")
print(f"\n  Trayectorias twitch-triggered (Opcion B):")
print(f"    Alta actividad: {len(traj_alta)} ventanas")
print(f"    Baja actividad: {len(traj_baja)} ventanas")
if len(traj_alta) > 0 and len(traj_baja) > 0:
    dist_traj = np.linalg.norm(
        traj_alta[:,:,:].mean(0)[WIN_TRIG_BINS] -
        traj_baja[:,:,:].mean(0)[WIN_TRIG_BINS])
    print(f"    Distancia en embedding en t=0: {dist_traj:.4f}")
    print(f"    {'DISTINTOS' if dist_traj > 0.1 else 'SIMILARES'}: "
          f"twitches de alta/baja en zonas "
          f"{'diferentes' if dist_traj > 0.1 else 'similares'} del embedding")
print(f"{'='*60}\n")
