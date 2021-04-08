import pyaudio
import numpy as np
import matplotlib
import matplotlib.pyplot as pyplot
import time
import shutil

matplotlib.use('Agg')

RATE = 44100  # time resolution of the recording device (Hz), the sample frequency
UPDATES_PER_SECOND = 20 # the number times to output from the stream per second 
RUNTIME = 60 # the number of seconds to run the program
process_num = 0

CHUNK = int(RATE / UPDATES_PER_SECOND)  # RATE / number of updates per second
first = True

def process0(data):
    return data, { "peak": 0 }

def process1(data):
    data = data * np.hanning(len(data)) # Applying hanning smoothing

    fft = np.abs(np.fft.fft(data))
    fft = fft[: int(len(fft) / 2)] # only keep first half since it mirror the second half

    freq = np.fft.fftfreq(CHUNK, 1.0 / RATE) # creates an array of frequencies corresponding to the fft
    freq = freq[:int(len(freq) / 2)] # keep only first half again since these are corresponding to the fft
    freqPeak = freq[np.where(fft == np.max(fft))[0][0]] + 1

    # pass a threshold filter over the data
    THRESHOLD = 10000
    fft[fft < THRESHOLD] = 0

    return fft, { "peak": freqPeak }

processing_functions = [process0, process1]
process = processing_functions[process_num]

def soundplot(stream):
    global first
    start_time = time.time() # Start time

    # Read and process data
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    data, payload = process(data)

    # Backup old image to backup
    if first:
        first = False
    else:
        shutil.move("stream/realtime.png", "stream/backup.png")

    # Plot data and stream to website through file
    pyplot.plot(data)
    pyplot.title(i + 1)
    pyplot.grid()
    pyplot.axis([0, len(data), -(2 ** 16) / 2, 2 ** 16 / 2])
    pyplot.savefig("stream/realtime.png", dpi=50)
    pyplot.close("all")

    elapsed = int((time.time() - start_time) * 1000) # calculate elapsed time
    print(f"Max frequency: {payload['peak']}\t Time Elapsed: {elapsed}ms", " " * 10, end="\r") # print additional data


if __name__ == "__main__":
    p = pyaudio.PyAudio() # Initializes the PyAudio class
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=3,
    ) # Sets up the stream
    for i in range(RUNTIME * UPDATES_PER_SECOND):  # Loop for RUNTIME seconds
        soundplot(stream) # call to the plot function

    # Close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    shutil.move("stream/realtime.png", "stream/backup.png")