import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

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
word_values = []
counter = 0


words = ['silence', 'Misère', 'Ingratitude', 'Profit', 'Egoisme', "Oportunisme"]
words_ref = ['silence', 'Misère', 'Ingratitude', 'Profit', 'Egoisme', "Oportunisme"]
current_word = words.pop(0)

def on_key_press(event):
    """Callback pour démarrer l'enregistrement."""
    global recording, word_values, current_word, counter, words_ref, words
    if not recording:  # Éviter de redémarrer si déjà en cours
        recording = True
        word_values = []  # Réinitialiser les valeurs

        if len(words) == 0 and counter < 4:
            words = words_ref.copy()
            counter += 1
        current_word = words.pop(0)
        text_dynamic.set_text("Dites le mot: " + current_word)


def on_key_release(event):
    """Callback pour arrêter l'enregistrement et sauvegarder les données."""
    global recording, word_values, counter, current_word
    if recording:
        recording = False
        text_dynamic.set_text("Enregistrement arrêté. Sauvegarde en cours...")
        if word_values:
            word_values += [0] * (2000 - len(word_values))
            word_values = word_values[:2000]
            np.savetxt(f"./words/{current_word}_{counter}.txt", np.array(word_values), fmt='%d')
            word_values = []
        text_dynamic.set_text("")


def update(frame):
    global spectrogram_data, word_values

    audio_data = stream.read(CHUNK_SIZE)[0]
    audio_data = np.frombuffer(audio_data, dtype=np.float32)

    fft_data = np.abs(np.fft.rfft(audio_data, n=N_FFT))
    fft_data = fft_data[:max_index]

    line_fft.set_ydata(fft_data)

    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    spectrogram_data[:, -1] = fft_data
    img.set_data(spectrogram_data)
    img.set_clim(0, 20)

    top_indices = np.argpartition(fft_data, -90)[-90:]
    #top_indices = top_indices[np.argsort(freqs[top_indices])]
    top_frequencies = freqs[top_indices]
    top_amplitudes = fft_data[top_indices]

    if recording:
        word_values.extend(top_frequencies)

    # Mise à jour dynamique des notes
    #notes_str = "\n".join([f"{i + 1}: {note}" for i, note in enumerate(word_values)])
    #notes_display.set_text(f"Mots enregistrées:\n{notes_str}")

    return line_fft, img, text_dynamic, notes_display


fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)
fig.canvas.mpl_connect('button_press_event', on_key_press)
fig.canvas.mpl_connect('button_release_event', on_key_release)

stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

with stream:
    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
