import pyaudio
import numpy as np
from reedsolo import RSCodec, ReedSolomonError
from scipy.signal import butter, sosfilt, sosfreqz, lfilter, iirpeak

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = int(round((0.1 / 2) * RATE))

HANDSHAKE_START_FREQ = 17500
HANDSHAKE_END_FREQ = HANDSHAKE_START_FREQ + 256
STEP_FREQ = 64
BITS = 4


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def dominant_freq(frame_rate, chunk):
    w = np.fft.fft(chunk)
    frequencies = np.fft.fftfreq(len(chunk))
    # iw = butter_bandpass_filter(w, 14000, 18000, 44100, order=10)
    # w = np.fft.ifft(iw)
    peak_coeff = np.argmax(np.abs(w))
    peak_freq = frequencies[peak_coeff]
    return abs(peak_freq * frame_rate)


def match(freq1, freq2):
    return abs(freq1 - freq2) < 8


def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill

        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset
        bits_left -= to_fill
        next_read_bit += to_fill

        if bits_left <= 0:
            out_bytes.append(byte)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes


def extract_packet(freqs):
    freqs = freqs[::2]
    bit_chunks = [int(round((f - HANDSHAKE_START_FREQ) / STEP_FREQ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks if 0 <= c < (2 ** BITS)]
    return bytearray(decode_bitchunks(BITS, bit_chunks))


def main():
    p = pyaudio.PyAudio()
    # print(CHUNK)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # print("* recording")

    frames = []
    handshake_done = False

    a = True
    while a:
        data = stream.read(CHUNK)
        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant_freq(RATE, chunk)
        print(dom)
        if handshake_done and match(dom, HANDSHAKE_END_FREQ):
            byte_stream = extract_packet(frames)
            try:
                byte_stream = RSCodec(4).decode(byte_stream)
                print(str(byte_stream))
            except ReedSolomonError as e:
                print("{}: {}".format(e, byte_stream))
        elif handshake_done:
            frames.append(dom)
        elif match(dom, HANDSHAKE_START_FREQ):
            handshake_done = True
        # a = False
    stream.stop_stream()
    stream.close()
    p.terminate()


main()
