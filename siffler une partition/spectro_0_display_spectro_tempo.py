import datetime
import os
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
import subprocess
from PIL import Image
from io import BytesIO
from matplotlib.widgets import Button
import threading  # Pour lancer la génération en arrière-plan
import json
import pandas as pd

gamme_temperee = {
    "c": [32.70, 65.41, 130.81, 261.63, 523.25, 1046.50, 2093.00, 4186.01, 8372.02, 16744.04],
    "des": [34.65, 69.30, 138.59, 277.18, 554.37, 1108.73, 2217.46, 4434.92, 8869.84, 17739.69],
    "d": [36.71, 73.42, 146.83, 293.66, 587.33, 1174.66, 2349.32, 4698.64, 9397.27, 18794.55],
    "ees": [38.89, 77.78, 155.56, 311.13, 622.25, 1244.51, 2489.02, 4978.03, 9956.06, 19912.13],
    "e": [41.20, 82.41, 164.81, 329.63, 659.25, 1318.51, 2637.02, 5274.04, 10548.08, 21096.16],
    "f": [43.65, 87.31, 174.61, 349.23, 698.46, 1396.91, 2793.83, 5587.65, 11175.30, 22350.61],
    "ges": [46.25, 92.50, 185.00, 369.99, 739.99, 1479.98, 2959.96, 5919.91, 11839.82, 23679.64],
    "g": [49.00, 98.00, 196.00, 392.00, 783.99, 1567.98, 3135.96, 6271.93, 12543.85, 25087.71],
    "aes": [51.91, 103.83, 207.65, 415.30, 830.61, 1661.22, 3322.44, 6644.88, 13289.75, 26579.50],
    "a": [55.00, 110.00, 220.00, 440.00, 880.00, 1760.00, 3520.00, 7040.00, 14080.00, 28160.00],
    "bes": [58.27, 116.54, 233.08, 466.16, 932.33, 1864.66, 3729.31, 7458.62, 14917.24, 29834.48],
    "b": [61.74, 123.47, 246.94, 493.88, 987.77, 1975.53, 3951.07, 7902.13, 15804.27, 31608.53]
}


freq_map = {}

for note, freqs in gamme_temperee.items():
    for i in range(2,len(freqs)):
        if i == 1:
            note_name = note + ","
        if i == 2:
            note_name = note
        if i == 3:
            note_name = note + "'"
        if i == 4:
            note_name = note + "''"
        if i == 5:
            note_name = note + "'''"

        delta = 1.3
        freq_map[note_name] = set(range(int(freqs[i] - delta), int(freqs[i] + delta)))     


def get_note_name_from_csv(frequency, frequency_map):
    """
    Trouve la note correspondant à une fréquence donnée en utilisant la carte de fréquences.
    """
    for note, freq_range in frequency_map.items():
        if frequency in freq_range:
            return note
    return None  # Si la fréquence ne correspond à aucune note

plt.rcParams['font.family'] = 'Chalkboard'

# Paramètres audio
SAMPLE_RATE = 44100  
CHUNK_SIZE = 1024 * 2
N_FFT = 1024 * 32        
FREQ_MAX = 2000       

note_sequence = [""]

freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
max_index = np.argmax(freqs >= FREQ_MAX)

TIME_WINDOW = 100  
spectrogram_data = np.zeros((max_index, TIME_WINDOW))  

# Configuration des graphiques
plt.style.use('dark_background')
fig = plt.figure(figsize=(25, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[0.5, 1], height_ratios=[1, 1], hspace=0.3)

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

ax_img = fig.add_subplot(gs[0, 1])
ax_img.axis("off")
img_lilypond = ax_img.imshow(np.zeros((1, 1, 3)), aspect="auto")  

ax_text = fig.add_subplot(gs[1, 1])
ax_text.axis("off")
text_annotation = ax_text.text(0.5, 0.6, "", ha="center", va="center", fontsize=12, color="white")

ax_text = fig.add_subplot(gs[1, 1])
ax_text.axis("off")
metronome_text = ax_text.text(0.2, 0.6, "", ha="center", va="center", fontsize=12, color="white")


button_axes = plt.axes([0.72, 0.07, 0.18, 0.05])
button = Button(button_axes, 'Reset score', color='gray', hovercolor='darkgray')

def generate_lilypond(notes, zoom=1, measures_per_line=4):
    """
    Génère un fichier LilyPond avec des notes de taille ajustable et un nombre de mesures défini.
    
    :param notes: Chaîne de notes LilyPond.
    :param zoom: Facteur d'agrandissement (1 = normal, >1 = plus grand, <1 = plus petit).
    :param measures_per_line: Nombre de mesures par ligne.
    """
    lilypond_code = f"""
    \\version "2.24.2"  % Vérifiez votre version de LilyPond
    \\absolute {{
        {notes}
    }}
    """
    
    with open("music.ly", "w") as file:
        file.write(lilypond_code)


def generate_image_in_background():
    """ Lance la génération d'image en arrière-plan sans bloquer """
    # hide the output
    with open(os.devnull, 'wb') as devnull:
        subprocess.Popen(["lilypond", "-dresolution=200", "--png", "-o", "output", "music.ly"], stdout=devnull, stderr=devnull)
    # Lancer un thread pour surveiller la génération de l'image
    threading.Thread(target=wait_for_image_and_update, daemon=True).start()

def wait_for_image_and_update():
    """ Attend que l'image soit créée et met à jour l'affichage """
    output_file = "output.png"

    for _ in range(20):  # Attendre 0.5 secondes max (5 x 0.1s)
        if os.path.exists(output_file):
            try:
                lily_img = Image.open(output_file)
                
                lily_img = crop_top_half(lily_img)
                # binariser l'image                
                lily_img = Image.fromarray(255 - np.array(lily_img))
                img_array = np.array(lily_img)
                img_lilypond.set_data(img_array)

                #text_annotation.set_text("Partition mise à jour !")
                fig.canvas.draw_idle()
                return
            except Exception as e:
                print(f"Erreur chargement image : {e}")
        time.sleep(0.1)  # Pause avant de réessayer

def crop_top_half(image):
    width, height = image.size
    return image.crop((80, 0, width, height // 3))


def generate_partition(event):
    global note_sequence
    # add a random note to the sequence

    current_notes = " ".join(note_sequence)  
    generate_lilypond(current_notes)

    # Lancer la génération en arrière-plan (non bloquante)
    generate_image_in_background()

    #text_annotation.set_text("Génération en cours...")

counter = 0
frame_counter = 0
signal = []
def update(frame):
    global spectrogram_data, freq_map, note_sequence, counter, tic, tac, frame_counter, signal
    frame_counter += 1

    audio_data = stream.read(CHUNK_SIZE)[0]
    audio_data = np.frombuffer(audio_data, dtype=np.float32)

    fft_data = np.abs(np.fft.rfft(audio_data, n=N_FFT))
    fft_data = fft_data[:max_index]
    line_fft.set_ydata(fft_data)

    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    spectrogram_data[:, -1] = fft_data
    # extract the top 10 most intense frequencies
    top_10_freqs = np.argsort(fft_data)[-1:]
    if fft_data[top_10_freqs[0]] > 20:
        top_10_freqs = freqs[top_10_freqs]
        signal.append(int(top_10_freqs[0]))
    else:
        signal.append("-1")
    
    if len(signal) > 30 or frame_counter > 25:
        print("Action!")
        print("signal: ", signal)
        current_seq = []
        for freq in signal:
            note = get_note_name_from_csv(int(top_10_freqs[0]), freq_map)
            if note:
                current_seq.append(note)
                counter += 1
        print("Found notes: ", current_seq)
        if len(current_seq) > 0:
            note_sequence.extend(current_seq)
            generate_partition(None)

        signal = []
        frame_counter = 0

    
    img.set_data(spectrogram_data)
    img.set_clim(0, 40)

    return line_fft, img, img_lilypond

def reset_score(event):
    global note_sequence
    note_sequence = [""]
    generate_partition(None)
    

button.on_clicked(reset_score)

# Paramètres du métronome
BPM = 60
INTERVAL = 70 / BPM  # Temps entre chaque battement
metronome_running = False
tic_tac = True
tic = None
tac = None
def metronome():
    global metronome_running,tac,tic,tic_tac
    while metronome_running:
        if tic_tac:
            # store the time of the tic
            tic = time.time()
            metronome_text.set_text("Tic")
        else:
            tic = time.time()
            metronome_text.set_text("Tac")
        tic_tac = not tic_tac
        fig.canvas.draw_idle()
        time.sleep(INTERVAL)

def toggle_metronome(event):
    global metronome_running
    if metronome_running:
        metronome_running = False
    else:
        metronome_running = True
        threading.Thread(target=metronome, daemon=True).start()

# Création du bouton pour le métronome
metronome_button_axes = plt.axes([0.5, 0.07, 0.18, 0.05])
metronome_button = Button(metronome_button_axes, 'Toggle Métronome', color='gray', hovercolor='darkgray')
metronome_button.on_clicked(toggle_metronome)
toggle_metronome(None)
output_file = "output.png"
if os.path.exists(output_file):
    os.remove(output_file)
music_file = "music.ly"
if os.path.exists(music_file):
    os.remove(music_file)
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SIZE)

with stream:
    ani = FuncAnimation(fig, update, interval=10)  
    plt.show()
