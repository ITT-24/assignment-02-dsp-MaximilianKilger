import pyaudio
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import pyglet
import math
import mido
from mido import MidiFile
import os, re
from threading import Thread
from collections import deque
# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 4096  # Number of audio frames per buffer
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Audio sampling rate (Hz)
p = pyaudio.PyAudio()

TRACK_NUMBER = 0 #track of the midi that the user has to sing to
TN = TRACK_NUMBER
STANDARD_TEMPO = 500000

#Range of Midi Notes displayed on screen
NOTE_RANGE = (36,84)

# Maximum allowed difference of sung note to actual note to qualify for a certain rating.
RATING_THRESHOLDS = {
    "!!Superstar!!" : 0.25, # in the same unit as midi notes
    "Amazing!" : 0.5,
    "Great" : 0.75,
    "Good" : 1.0,
    "Okay": 1.5,
    "Meh...": 3.0,
    "My ears!": 7000
}

def midi_num_to_frequency (m:int) -> float:
    if m <= 0:
        return None
    f = 2 ** ((m-69)/12) * 440 # newt.phys.unsw.edu.au/jw/notes.html
    return f # in Hz

def frequency_to_midi_num(f:float) -> float:
    if f <= 0:
        return None
    m = 12 * math.log(f/440,2) + 69 # newt.phys.unsw.edu.au/jw/notes.html
    return m

# given a ndarray (float) of samples, finds the most energetic frequency in the fourier transform of the signal
def find_prevalent_frequency(data:np.ndarray, sampling_rate:int, filter_low:float=None, filter_high:float=None) -> float:
    use_filter = filter_low != None and filter_high != None
    wd = np.zeros((len(data)))
    if use_filter:
        sos = signal.butter(5, [filter_low, filter_high], 'bandpass', fs=sampling_rate, output='sos') #apply bandpass filter
        filtered_data = signal.sosfiltfilt(sos, data)
        wd = np.hamming(len(filtered_data)) * filtered_data
    else:
        wd = np.hamming(len(data)) * data #apply hamming window
    fourier = np.fft.fft(wd)
    
    frequencies = np.fft.fftfreq(len(wd), 1/sampling_rate)
    
    highest_frequency = np.max(frequencies[np.where(np.abs(fourier) == np.max(np.abs(fourier)))])
    return abs(highest_frequency)

# from this part on, some parts for reading audio are stolen from audio-sample.py
# print info about audio devices
# let user select audio device
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

print('select audio device:')
input_device = int(input())

# open audio input stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
                input_device_index=input_device)

# set up interactive plot
#fig = plt.figure()
#ax = plt.gca()
#line, = ax.plot(np.zeros(CHUNK_SIZE))
#ax.set_ylim(0, 12)

#plt.ion()
#plt.show()
WIN_WIDTH = 1200
WIN_HEIGHT = 600
UI_HEIGHT = 100

RATING_POS_X = 15
RATING_POS_Y = 75 

SCORE_POS_X = WIN_WIDTH - 200
SCORE_POS_Y = 75

CURSOR_COLOR = (255,255,255)
SING_BAR_COLOR = (255,0,0)

window = pyglet.window.Window(WIN_WIDTH, WIN_HEIGHT)

filenames = ['freude.mid','berge.mid']

    

class GameManager():

    def __init__(self):
        
        self.filename = filenames[0]
        self.setup_song(self.filename)
        self.is_playing = False
        self.sung_note = -1
        self.cursor_pos = -1
        self.score = None
        self.rating = ""
        self.deltas = deque([],maxlen=int(RATE/CHUNK_SIZE))
        self.setup_notes_graphics()
        self.setup_ui()

    # reads song from file, converts midi messages to discrete numpy array with note values
    def setup_song(self,filename):
        filepath = os.path.join('songs', filename)
        file = MidiFile(filepath)

        length = file.length
        self.num_chunks_in_file = int(length * (RATE / CHUNK_SIZE))
        self.note_sheet = np.zeros((len(file.tracks),self.num_chunks_in_file), dtype=int) # a numpy array holding the note playing during each discrete chunk.
        tempo = STANDARD_TEMPO 

        #midi_parse_re = re.compile("{.*} channel={.*} note={.*} velocity={.*} time={.*}")
        for i, track in enumerate(file.tracks):
            cursor = 0
            for message in track:
                if message.is_meta:
                    #print(message)
                    continue
                #print(message)
                #mode, channel, note, velocity, time = midi_parse_re.match(message).groups()
                #print(f"mode: {mode}, note:{note}, time:{time}")
                midi_type = message.dict()["type"]
                time = mido.tick2second(message.dict()["time"], file.ticks_per_beat, tempo) #duration of midi note
                note = -1
                start = cursor
                stop = int(cursor + time * (RATE / CHUNK_SIZE)) #start and stop of the note in discrete numpy array
                if midi_type == "note_on":
                    note = message.dict()["note"]
                    stop = int(cursor + max(1, time * (RATE / CHUNK_SIZE))) #make sure that notes take up at least one chunk

                #print(note, "   ", time)

                
                #print(f"{start} - {stop}")
                for j in range(start, stop):
                    self.note_sheet[i][j] = note 
                cursor = stop
                
        self.chunk_pixel_width = WIN_WIDTH / self.num_chunks_in_file # for UI, the width of one chunk on screen
        self.chunk_pixel_height = (WIN_HEIGHT-UI_HEIGHT) / (NOTE_RANGE[1] - NOTE_RANGE[0]) # for UI, the height of one note on screen
            
        #print(self.note_sheet)        

    # creates the graphical representation of the notes.
    def setup_notes_graphics(self):
        self.notes_batch = pyglet.graphics.Batch()
        self.note_bars = []
        for i, note in enumerate(self.note_sheet[TN]):
            xpos = i * self.chunk_pixel_width
            ypos = UI_HEIGHT + self.chunk_pixel_height * (note - NOTE_RANGE[0])
            note_bar = pyglet.shapes.Rectangle(xpos,ypos,self.chunk_pixel_width, self.chunk_pixel_height, (255,255,0), batch=self.notes_batch)
            self.note_bars.append(note_bar)

    # creates UI for score and rating
    def setup_ui(self):
        self.score_label = pyglet.text.Label("", font_name="Century Gothic", font_size=30)
        self.score_label.x = SCORE_POS_X
        self.score_label.y = SCORE_POS_Y

        self.rating_label = pyglet.text.Label("", font_name="Century Gothic", font_size=30)
        self.rating_label.x = RATING_POS_X
        self.rating_label.y = RATING_POS_Y
    

    #draws notes to screen
    def draw_notes (self):
        self.notes_batch.draw()

    #draw cursor (that signifies where in the song we are) to the screen
    def draw_cursor(self):
        if self.cursor_pos != -1:
            xpos = self.cursor_pos * self.chunk_pixel_width
            cursor_bar = pyglet.shapes.Rectangle(xpos, UI_HEIGHT, self.chunk_pixel_width, WIN_HEIGHT-UI_HEIGHT, color=CURSOR_COLOR)
            cursor_bar.draw()

    #draw bar for currently sung note to the screen
    def draw_sung_note(self):
        note = self.sung_note
        if note != -1:
            bar_height = self.chunk_pixel_height / 4
            ypos = UI_HEIGHT + self.chunk_pixel_height * (note - NOTE_RANGE[0]) - self.chunk_pixel_height * 3 / 8
            bar = pyglet.shapes.Rectangle(0,ypos, WIN_WIDTH, bar_height, SING_BAR_COLOR)
            bar.draw()

    # draws score and rating labels
    def draw_ui(self):
        self.rating_label.text = self.rating
        self.rating_label.draw()

        self.score_label.text = f"Score :{int(self.score)}" if self.score != None else ""
        self.score_label.draw()
    
    # draws everything relevant
    def render(self):
        gm.draw_notes()
        gm.draw_sung_note()
        gm.draw_cursor()
        gm.draw_ui()
            
    # starts parallel thread that processes audio samples
    def play_round(self):

        if not self.is_playing:
            self.is_playing = True
        round_loop_thread = Thread(target=self.round_loop)
        round_loop_thread.start()

    # plays one round of the game. Blocking, don't call in main thread.
    def round_loop(self):
        #reset
        self.score = 0
        self.cursor_pos = 0
        last_note = -1
        #print(len(self.note_sheet[TN]))

        # continuously capture and plot audio singal # here, we start getting into some stolen code again
        while self.cursor_pos < len(self.note_sheet[TN]):
            # Read audio data from stream
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

            # Convert audio data to numpy array
            data = np.frombuffer(data, dtype=np.int16)
            # plot signal in frequency domain
            #line.set_ydata(np.abs(fourier))
            
            # find sung frequency
            frequency = find_prevalent_frequency(data, RATE)
            if frequency == None:
                continue
            
            midi_note = frequency_to_midi_num(frequency)
            if midi_note != None:
                #midi_note = midi_note % 12
                # instead of rating the note on its own, generate all notes that are one or more whole octaves above and below the note sung
                # and only rate the one closest to the note in the song
                base_note = midi_note % 8
                octaves = np.array([i*8+base_note for i in range(0,11)])
                
                if self.note_sheet[TN][self.cursor_pos] != -1 and self.note_sheet[TN][self.cursor_pos] != 0:
                    differences = np.abs(octaves - self.note_sheet[TN][self.cursor_pos])
                    last_note = self.note_sheet[TN][self.cursor_pos]
                else:
                    differences = np.abs(octaves - last_note)
                official_difference = np.min(differences)
                official_note = list(octaves)[list(differences).index(official_difference)]

                # share sung note with the main thread
                self.sung_note = official_note
                #print(official_difference)

                if self.note_sheet[TN][self.cursor_pos] != -1 and self.note_sheet[TN][self.cursor_pos] != 0:
                    #give a score
                    self.deltas.append(official_difference)
                    self.calc_score()

                    #difference = abs(self.note_sheet[TN][self.cursor_pos] - midi_note)
                #print(midi_note)
                #line.set_ydata(midi_note)
            #print(np.max(frequencies[np.where(np.abs(fourier) == np.max(np.abs(fourier)))]))
            #print(frequencies[mask])

            #advance cursor
            self.cursor_pos += 1
        
        #reset
        self.sung_note = -1
        self.cursor_pos = -1
        self.is_playing = False
    
    #calculate score and give rating
    def calc_score(self):
        avg_difference = np.mean(list(self.deltas))
        rating = ""
        for key in RATING_THRESHOLDS.keys():
            if avg_difference < RATING_THRESHOLDS[key]:
                rating = key
                break
        self.rating = rating

        self.score += max(10 - self.deltas[-1] * 5, 0)
        




gm = GameManager()

    
@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        if not gm.is_playing:
            gm.play_round()
    
    if symbol == pyglet.window.key.Q:
        window.close()
        os._exit(0)

    # switch to song 1
    if symbol == pyglet.window.key._1:
        gm.filename = filenames[0]
        gm.setup_song(gm.filename)
        gm.setup_notes_graphics()

    # switch to song 2
    if symbol == pyglet.window.key._2:
        gm.filename = filenames[1]
        gm.setup_song(gm.filename)
        gm.setup_notes_graphics()
        


@window.event
def on_draw():
    window.clear()
    gm.render()

pyglet.app.run()



    # Redraw plot1
    #fig.canvas.draw()
    #fig.canvas.flush_events()
