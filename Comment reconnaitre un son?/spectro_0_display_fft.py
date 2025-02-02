import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

















multiplicateur = 12
# Paramètres audio
SAMPLE_RATE = 44100  # Fréquence d'échantillonnage
CHUNK_SIZE = 1024 * multiplicateur  # Taille des blocs
N_FFT = 1024* multiplicateur      # Taille de la FFT (doit être un multiple de 2)
FREQ_MAX = 2000      # Plage de fréquence maximale (2 kHz)









plt.rcParams['font.family'] = 'Chalkboard'

# Calcul de l'index correspondant à 2 kHz dans la FFT
freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
max_index = np.argmax(freqs >= FREQ_MAX)

# Paramètres du spectrogramme
TIME_WINDOW = 100  # Nombre de fenêtres temporelles affichées (pour heatmap)
spectrogram_data = np.zeros((max_index, TIME_WINDOW))  # Limité à 2 kHz

# Configuration des graphiques avec GridSpec pour partager la fenêtre
plt.style.use('dark_background')  # Fond noir global
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(1, 1, width_ratios=[1, ], height_ratios=[1,], hspace=0.4)

# Graphique FFT (instantané)
ax1 = fig.add_subplot(gs[0, 0])
line_fft, = ax1.plot(freqs[:max_index], np.zeros(max_index), color="white", lw=1.5)
ax1.set_xlim(0, FREQ_MAX)  # Limiter à 2 kHz
ax1.set_ylim(0, 350)        # Ajuster l'amplitude visible
ax1.set_xlabel("Fréquence", color="white", fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax1.set_ylabel("Intensité", color="white", fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax1.set_title("", color="white")
ax1.tick_params(axis="x", colors="white")
# set the font of the axis values
ax1.tick_params(axis="y", colors="white")

# Fonction de mise à jour des graphiques
def update(frame):
    global spectrogram_data

    # Lire les données audio
    audio_data = stream.read(CHUNK_SIZE)[0]
    audio_data = np.frombuffer(audio_data, dtype=np.float32)

    # Calculer la FFT
    fft_data = np.abs(np.fft.rfft(audio_data, n=N_FFT))

    # Limiter la FFT à la plage 0-2 kHz
    fft_data = fft_data[:max_index]

    # Mettre à jour le graphique FFT
    line_fft.set_ydata(fft_data)

    # Mettre à jour les données du spectrogramme
    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)  # Décale les colonnes
    spectrogram_data[:, -1] = fft_data  # Ajoute les nouvelles données

    # Trouver les 4 fréquences avec les amplitudes maximales
    top_indices = np.argpartition(fft_data, -4)[-4:]  # Indices des 4 plus grandes amplitudes
    top_indices = top_indices[np.argsort(freqs[top_indices])]  # Trier par fréquence croissante
    top_frequencies = freqs[top_indices]  # Fréquences correspondantes
    top_amplitudes = fft_data[top_indices]  # Amplitudes correspondantes

    # somme des amplitudes
    sum_amplitudes = np.sum(top_amplitudes)

    if sum_amplitudes > 20:
        # Formater le texte pour une largeur fixe
        text_lines = [
            "Intensités par fréquence :\n",
            f"{top_frequencies[0]:<8.0f} Hz   | Amplitude: {top_amplitudes[0]:<6.0f}",
            f"{top_frequencies[1]:<8.0f} Hz   | Amplitude: {top_amplitudes[1]:<6.0f}",
            f"{top_frequencies[2]:<8.0f} Hz   | Amplitude: {top_amplitudes[2]:<6.0f}",
            f"{top_frequencies[3]:<8.0f} Hz   | Amplitude: {top_amplitudes[3]:<6.0f}",
        ]
        #text_dynamic.set_text("\n".join(text_lines))
    else:
        #text_dynamic.set_text("")
        pass

    return line_fft, #text_dynamic

# Lancer le flux audio
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

# Animation
with stream:
    ani = FuncAnimation(fig, update, interval=50)  # Mise à jour toutes les 50 ms
    plt.show()
