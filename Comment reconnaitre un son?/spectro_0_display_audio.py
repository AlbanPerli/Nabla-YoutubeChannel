import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

multiplicateur = 4
# Paramètres audio
SAMPLE_RATE = 44100  # Fréquence d'échantillonnage
CHUNK_SIZE = 1024 * multiplicateur  # Taille des blocs
N_FFT = 1024 * multiplicateur       # Taille de la FFT (doit être un multiple de 2)
FREQ_MAX = 2000                     # Plage de fréquence maximale (2 kHz)

plt.rcParams['font.family'] = 'Chalkboard'

# Calcul de l'index correspondant à 2 kHz dans la FFT
freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
max_index = np.argmax(freqs >= FREQ_MAX)

# Configuration des graphiques avec GridSpec
plt.style.use('dark_background')  # Fond noir global
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4)

# Graphique des données brutes
ax2 = fig.add_subplot(gs[0, 0])
line_raw, = ax2.plot(np.zeros(CHUNK_SIZE), color="cyan", lw=1.5)
ax2.set_xlim(0, CHUNK_SIZE)
ax2.set_ylim(-0.5, 0.5)  # Échelle ajustée aux valeurs brutes audio
ax2.set_xlabel("Échantillons", color="white")
ax2.set_ylabel("Amplitude", color="white")
ax2.set_title("Données brutes (Signal temporel)", color="white")
ax2.tick_params(axis="x", colors="white")
ax2.tick_params(axis="y", colors="white")

# Ajouter du texte dynamique centré verticalement
ax_text = fig.add_subplot(gs[:, 1])  # Occupe toute la deuxième colonne
ax_text.axis("off")  # Supprime les axes
text_dynamic = ax_text.text(0.5, 0.5, "", color="white", fontsize=16,
                            ha="center", va="center", wrap=True, fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})

count = 0
# Fonction de mise à jour des graphiques
def update(frame):
    global count
    # Lire les données audio
    audio_data = stream.read(CHUNK_SIZE)[0]
    audio_data = np.frombuffer(audio_data, dtype=np.float32)

    # Mettre à jour le graphique des données brutes
    line_raw.set_ydata(audio_data)
    
    # calculer la somme des valeurs absolues
    sum = np.sum(np.abs(audio_data))
    
    # Mettre à jour le texte dynamique si la somme dépasse 100
    if sum > 150:
        text_dynamic.set_text("CLAP!!!")
    else:
        count += 1
        if count > 25:
            text_dynamic.set_text("")
            count = 0
    

    return line_raw

# Lancer le flux audio
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

# Animation
with stream:
    ani = FuncAnimation(fig, update, interval=50)  # Mise à jour toutes les 50 ms
    plt.show()
