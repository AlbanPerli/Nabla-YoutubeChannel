import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

plt.rcParams['font.family'] = 'Chalkboard'

# list of notes names in the chromatic scale over 3 octaves starting from DO_3, then DO#_3, Ré_3 etc..
notes = ['silence', 'DO_3', 'DO#_3', 'RE_3']
        #  , 'RE#_3', 'MI_3', 'FA_3', 'FA#_3', 
        #  'SOL_3', 'SOL#_3', 'LA_3', 'LA#_3', 
        #  'SI_3','DO_4', 'DO#_4', 'RE_4', 'RE#_4', 'MI_4', 
        #  'FA_4', 'FA#_4', 'SOL_4', 'SOL#_4', 'LA_4', 'LA#_4', 
        #  'SI_4','DO_5', 'DO#_5', 'RE_5', 'RE#_5', 'MI_5', 'FA_5', 
        #  'FA#_5', 'SOL_5', 'SOL#_5', 'LA_5', 'LA#_5', 'SI_5', 'DO_6']
notes_ref = ['silence', 'DO_3', 'DO#_3', 'RE_3']

recorded_freqs_X = []
recorded_freqs_Y = []


# Paramètres audio
SAMPLE_RATE = 44100  # Fréquence d'échantillonnage
CHUNK_SIZE = 1024*8  # Taille des blocs
N_FFT = 1024*8         # Taille de la FFT (doit être un multiple de 2)
FREQ_MAX = 1000      # Plage de fréquence maximale (2 kHz)

# Calcul de l'index correspondant à 2 kHz dans la FFT
freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
max_index = np.argmax(freqs >= FREQ_MAX)

# Paramètres du spectrogramme
TIME_WINDOW = 100  # Nombre de fenêtres temporelles affichées (pour heatmap)
spectrogram_data = np.zeros((max_index, TIME_WINDOW))  # Limité à 2 kHz

# Configuration des graphiques avec GridSpec pour partager la fenêtre
plt.style.use('dark_background')  # Fond noir global
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.4)

# Graphique FFT (instantané)
ax1 = fig.add_subplot(gs[0, 0])
line_fft, = ax1.plot(freqs[:max_index], np.zeros(max_index), color="white", lw=1.5)
ax1.set_xlim(0, FREQ_MAX)  # Limiter à 2 kHz
ax1.set_ylim(0, 100)        # Ajuster l'amplitude visible
ax1.set_xlabel("Fréquence", color="white", fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax1.set_ylabel("Intensité", color="white", fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax1.set_title("", color="white")
ax1.tick_params(axis="x", colors="white")
# set the font of the axis values
ax1.tick_params(axis="y", colors="white")

# Graphe Spectrogramme (heatmap)
ax2 = fig.add_subplot(gs[1, 0])
img = ax2.imshow(spectrogram_data, aspect="auto", origin="lower",
                 extent=[0, TIME_WINDOW, 0, FREQ_MAX], cmap="inferno")
ax2.set_xlabel("Temps -->", color="white",fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax2.set_ylabel("Fréquence", color="white",fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})
ax2.set_title("", color="white")
ax2.tick_params(axis="x", colors=(0,0,0,0))
ax2.tick_params(axis="y", colors="white")  # Masquer les graduations de l'axe y

# Ajouter du texte dynamique centré verticalement
ax_text = fig.add_subplot(gs[:, 1])  # Occupe toute la deuxième colonne
ax_text.axis("off")  # Supprime les axes
text_dynamic = ax_text.text(0.5, 0.5, "Texte initial", color="white", fontsize=16,
                            ha="center", va="center", wrap=True, fontdict={"fontsize": 12, "fontweight": "bold", "family": "chalkboard"})

new_recording = False
next_note = notes.pop(0)
current_note_index = 0
# Fonction de mise à jour des graphiques
def update(frame):
    global spectrogram_data, next_note, new_recording, current_note_index

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

    # Mettre à jour la heatmap
    img.set_data(spectrogram_data)
    img.set_clim(0, 40)  # Ajuste l'échelle de la heatmap

    # Trouver les 4 fréquences avec les amplitudes maximales
    top_indices = np.argpartition(fft_data, -4)[-4:]  # Indices des 4 plus grandes amplitudes
    top_indices = top_indices[np.argsort(freqs[top_indices])]  # Trier par fréquence croissante
    top_frequencies = freqs[top_indices]  # Fréquences correspondantes
    top_amplitudes = fft_data[top_indices]  # Amplitudes correspondantes

    # somme des amplitudes
    sum_amplitudes = np.sum(top_amplitudes)
    
    if sum_amplitudes > 20:
        new_recording = True
        # Formater le texte pour une largeur fixe
        text_lines = [
            "Intensités par fréquence :\n",
            f"{top_frequencies[0]:<8.0f} Hz   | Amplitude: {top_amplitudes[0]:<6.0f}",
            f"{top_frequencies[1]:<8.0f} Hz   | Amplitude: {top_amplitudes[1]:<6.0f}",
            f"{top_frequencies[2]:<8.0f} Hz   | Amplitude: {top_amplitudes[2]:<6.0f}",
            f"{top_frequencies[3]:<8.0f} Hz   | Amplitude: {top_amplitudes[3]:<6.0f}",
        ]
        recorded_freqs_X.append([top_frequencies[0],top_frequencies[1],top_frequencies[2],top_frequencies[3]])
        recorded_freqs_Y.append(current_note_index)
        text_dynamic.set_text("\n".join(text_lines))
      
    else:
        if new_recording:
            new_recording = False
            if len(notes) > 0:
                next_note = notes.pop(0)
                current_note_index += 1
                text_dynamic.set_text("Play note: " + next_note)
                np.savetxt('recorded_freqs_X.txt', np.round(recorded_freqs_X), fmt='%d')
                np.savetxt('recorded_freqs_Y.txt', np.round(recorded_freqs_Y), fmt='%d')
            else:
                text_dynamic.set_text("End of recording")
                np.savetxt('recorded_freqs_X.txt', np.round(recorded_freqs_X), fmt='%d')
                np.savetxt('recorded_freqs_Y.txt', np.round(recorded_freqs_Y), fmt='%d')

    return line_fft, img, text_dynamic

# Lancer le flux audio
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

# Animation
with stream:
    ani = FuncAnimation(fig, update, interval=50)  # Mise à jour toutes les 50 ms
    plt.show()
