import threading
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

folder = "chords"
song_name = "C-_arpegio"

plt.rcParams['font.family'] = 'Chalkboard'

# Paramètres audio
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024 * 4
N_FFT = 1024 * 128
FREQ_MAX = 2000

freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
max_index = np.argmax(freqs >= FREQ_MAX)

TIME_WINDOW = 40
spectrogram_data = np.zeros((max_index, TIME_WINDOW))

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
line_fft, = ax1.plot(freqs[:max_index], np.zeros(max_index), color="white", lw=1.5)
ax1.set_xlim(0, FREQ_MAX)
ax1.set_ylim(0, 100)
ax1.set_xlabel("Fréquence", color="white")
ax1.set_ylabel("Intensité", color="white")

ax2 = fig.add_subplot(gs[1, 0])
img = ax2.imshow(spectrogram_data, aspect="auto", origin="lower",
                 extent=[0, TIME_WINDOW, 0, FREQ_MAX], cmap="inferno")
ax2.set_xlabel("Temps -->", color="white")
ax2.set_ylabel("Fréquence", color="white")

ax_text = fig.add_subplot(gs[:, 1])
ax_text.axis("off")
text_dynamic = ax_text.text(0.5, 0.1, "Maintenez la souris ou une touche pour enregistrer", color="white",
                            fontsize=16, ha="center", va="center", wrap=True)

notes_display = ax_text.text(0.5, 1.1, "", color="white", fontsize=10, ha="center", va="top", wrap=True)

recording = False
notes_values = []
counter = 0

def on_key_press(event):
    """Callback pour démarrer l'enregistrement."""
    global recording, notes_values
    if not recording:  # Éviter de redémarrer si déjà en cours
        recording = True
        notes_values = []  # Réinitialiser les valeurs
        text_dynamic.set_text("Enregistrement en cours...")


def on_key_release(event):
    """Callback pour arrêter l'enregistrement et sauvegarder les données."""
    global recording, notes_values, counter, timer
    if recording:
        recording = False
        text_dynamic.set_text("Enregistrement arrêté. Sauvegarde en cours...")
        if notes_values:
            # zero-padding jusqu'à la taille de 30 éléments si nécessaire
            notes_values += [0] * (1024 - len(notes_values))
            # retirer les valeurs au delà de 30 éléments
            notes_values = notes_values[:1024]
            np.savetxt(f"./{folder}/{song_name}_{counter}.txt", np.array(notes_values), fmt='%d')
            notes_values = []
            counter += 1
            timer = fig.canvas.new_timer(interval=2000)
            timer = threading.Timer(2, create_timer)
            timer.start()
        text_dynamic.set_text("")

def create_timer():
    global timer
    #timer = fig.canvas.new_timer(interval=5000)
    on_key_press(None)
    #timer.add_callback(on_key_release, None)
    #timer.start()
    
timer = None
local_timer = None

create_timer()




def update(frame):
    global spectrogram_data, notes_values

    audio_data = stream.read(CHUNK_SIZE)[0]
    audio_data = np.frombuffer(audio_data, dtype=np.float32)

    fft_data = np.abs(np.fft.rfft(audio_data, n=N_FFT))
    fft_data = fft_data[:max_index]

    line_fft.set_ydata(fft_data)

    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    spectrogram_data[:, -1] = fft_data
    img.set_data(spectrogram_data)
    img.set_clim(0, 20)

    top_indices = np.argpartition(fft_data, -40)[-40:]
    top_indices = top_indices[np.argsort(freqs[top_indices])]
    top_frequencies = freqs[top_indices]
    top_amplitudes = fft_data[top_indices]

    if recording and np.sum(top_amplitudes) > 100:
        notes_values.extend(top_frequencies)

    # Mise à jour dynamique des notes
    #notes_str = "\n".join([f"{i + 1}: {note:.2f} Hz" for i, note in enumerate(notes_values)])
    notes_display.set_text(f"{len(notes_values)} fréquences enregistrées")
    

    return line_fft, img, text_dynamic, notes_display


fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)
fig.canvas.mpl_connect('button_press_event', on_key_press)
fig.canvas.mpl_connect('button_release_event', on_key_release)

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

with stream:
    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
