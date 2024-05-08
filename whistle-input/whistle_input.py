import pyaudio
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import linregress
from pynput.keyboard import Key, Controller

def find_prevalent_frequency(data:np.ndarray, sampling_rate:int) -> float:
    
    sos = signal.butter(5, [80, 1200], 'bandpass', fs=sampling_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    wd = np.hamming(len(data)) * data
    fourier = np.fft.fft(wd)
    
    frequencies = np.fft.fftfreq(len(wd), 1/sampling_rate)
    
    highest_frequency = np.max(frequencies[np.where(np.abs(fourier) == np.max(np.abs(fourier)))])
    return abs(highest_frequency)

def get_direction(data:np.ndarray):

    slope, intercept, rvalue, pvalue, std_error = linregress(np.arange(0,len(data),1), data)
    #print(f"SLOPE: {slope:.4}, STDERR:{std_error:.4}, RVAL:{rvalue:.4}")
    if abs(rvalue) > RVALUE_THRESH and abs(slope) > SLOPE_THRESH:
        if slope < 0:
            return -1
        else:
            return 1
    else:
        return 0


# from here on out, code is based on audio-sample.py

# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
WINDOW = 25

GRACE_PERIOD = 1.0

STDERR_THRESH = 5
RVALUE_THRESH = 0.8
SLOPE_THRESH = 15

SHOW_DEBUG_DIAGRAM = False
p = pyaudio.PyAudio()
keyboard = Controller()

# print info about audio devices
# let user select audio device
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

print('select audio device:')
input_device = int(input())

freqs = deque(maxlen=WINDOW)


# open audio input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=input_device)
if SHOW_DEBUG_DIAGRAM:
    # set up interactive plot
    fig = plt.figure()
    ax = plt.gca()
    line, = ax.plot(np.zeros(WINDOW))
    ax.set_ylim(-000, 3000)

    plt.ion()
    plt.show()

status = 0
chunks_since_last_input = 0
# continuously capture and plot audio singal
while True:
    chunks_since_last_input += 1
    # Read audio data from stream
    data = stream.read(CHUNK_SIZE)

    # Convert audio data to numpy array
    data = np.frombuffer(data, dtype=np.int16)

    freq = find_prevalent_frequency(data, RATE)
    freqs.append(freq)
    if len(freqs) >= WINDOW:
        if SHOW_DEBUG_DIAGRAM:
            line.set_ydata(freqs)
        direction = get_direction(np.array(freqs))
        if direction != status:
            if direction == 0:
                if status == -1:
                    keyboard.release(Key.down)
                if status == 1:
                    keyboard.release(Key.up)
                print("----")
                status = direction
            else:
                if chunks_since_last_input * CHUNK_SIZE / RATE >= GRACE_PERIOD:
                    if direction == -1:
                        if status == 1:
                            keyboard.release(Key.up)
                        keyboard.press(Key.down)
                        print("DOWN")
                        chunks_since_last_input = 0
                        
                    elif direction == 1:
                        if status == -1:
                            keyboard.release(Key.down)
                        keyboard.press(Key.up)
                        print("-UP-")
                        chunks_since_last_input = 0
                status = direction
                    
                
                

        if SHOW_DEBUG_DIAGRAM:
            # Redraw plot
            fig.canvas.draw()
            fig.canvas.flush_events()
