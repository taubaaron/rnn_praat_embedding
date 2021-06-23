import parselmouth
import scipy.signal
from IPython.display import Audio
from parselmouth.praat import call
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import lfilter, filtfilt
from scipy.io.wavfile import write


def read_sound(file_name):
    return parselmouth.Sound(file_name)

def plot_sound(sound):
    plt.figure()
    plt.plot(sound.xs(), sound.values.T)
    plt.xlim([sound.xmin, sound.xmax])
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.show()

def save_plot_sound(sound, name):
    plt.figure()
    plt.plot(sound.xs(), sound.values.T)
    plt.xlim([sound.xmin, sound.xmax])
    plt.xlabel("Time [sec]")
    plt.ylabel("Amplitude")
    plt.savefig(f"{name}.png")

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    # plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range)
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


def pitch_manipulation(sound):
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")

    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, 2)
    call([pitch_tier, manipulation], "Replace pitch tier")
    sound_octave_up = call(manipulation, "Get resynthesis (overlap-add)")
    Audio(data=sound_octave_up.values, rate=sound_octave_up.sampling_frequency)
    sound_octave_up.save("whitney houston.wav", "WAV")
    Audio(filename="whitney houston.wav")


def lpc_matrix_extract(sound, lpc_order=64):
    lpc = parselmouth.praat.call(sound, "To LPC (burg)", lpc_order, 0.025, 0.005, 50.0)
    lpc_matrix = parselmouth.praat.call(lpc, "Down to Matrix (lpc)")
    return lpc_matrix.values
    # print(lpc_matrix.values)


def sound_to_f0_pulse(sound):
    sound_pitch = sound.to_pitch()
    pulses = sound_pitch.to_sound_pulses()
    pulses.save("pulses.wav", "WAV")
    Audio(filename="pulses.wav")
    return pulses


def mixing_voices(sound_a, sound_b):
    """
    !!Verify Capitalized Letters!!

    Whitney:                               Analyse periodicity -> to_pitch
    whitney.to_pitch:                      sound -> to_sound_pulses
    to_sound_pulses:                       lpf (?)
    Aaron:                                 Analyse Spectrum -> to_lpc(burg), 64 co, 0.025 w, ts 0.05, pre-emph 50
    Aaron_lpc + to_sound_pulses_lpf:       filter
    result:                                modify -> normalize/scale_peak
    """
    sound_a_pitch = sound_a.to_pitch()
    sound_a_pitch_pulse = sound_a_pitch.to_sound_pulses()
    sound_a_pitch_pulse.save("sound_a_pitch_pulse.wav", "WAV")
    Audio(filename="sound_a_pitch_pulse.wav")

    """   # LPF
    order = 5
    sampling_freq = 30
    cutoff_freq = 2

    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    numerator_coeffs, denominator_coeffs = scipy.signal.butter(order, normalized_cutoff_freq)
    sound_a_pitch_pulse_lpf = scipy.signal.lfilter(numerator_coeffs, denominator_coeffs, sound_a_pitch_pulse)
    sound_a_pitch_pulse_lpf.save("sound_a_pitch_pulse_lpf.wav", "WAV")
    Audio(filename="sound_a_pitch_pulse_lpf.wav")"""

    sound_b_lpc = parselmouth.praat.call(sound_b, "To LPC (burg)", 64, 0.025, 0.005, 50.0)
    sound_b_lpc_matrix = parselmouth.praat.call(sound_b_lpc, "Down to Matrix (lpc)")



    result = parselmouth.praat.call([sound_a_pitch_pulse, sound_b_lpc], "Filter...", False)

    result.save("result_before_scaled.wav", "WAV")
    Audio(filename="result_before_scaled.wav")
    # result_scaled = parselmouth.praat.call(result, "Scale peak...", 0.99)
    parselmouth.praat.call(result, "Scale peak...", 0.99)
    save_plot_sound(result, "Result")

    result_scaled = result
    result_scaled.save("result_scaled.wav", "WAV")
    Audio(filename="result_scaled.wav")
    save_plot_sound(result_scaled, "Result_Scaled")

###

def enframe(x, win, inc=None):
    nx = len(x)
    if isinstance(win, list) or isinstance(win, np.ndarray):
        nwin = len(win)
        nlen = nwin
    elif isinstance(win, int):
        nwin = 1
        nlen = win
    if inc is None:
        inc = nlen
    nf = (nx - nlen + inc) // inc
    frameout = np.zeros((nf, nlen))
    indf = np.multiply(inc, np.array([i for i in range(nf)]))
    for i in range(nf):
        frameout[i, :] = x[indf[i]:indf[i] + nlen][0]
    if isinstance(win, list) or isinstance(win, np.ndarray):
        frameout = np.multiply(frameout, np.array(win))
    return frameout


def Filpframe_OverlapS(x, win, inc):
    """
    :param x: should be size of: [window_size, num_of_frames
    :param win:
    :param inc:
    :return:
    """
    x = x.T
    win = win.T
    nf, slen = x.shape  # [wlen, number of frames] == [1102, 8409]
    nx = (nf - 1) * inc + slen
    frameout = np.zeros(nx)
    # x = x / win[None, :]
    for i in range(nf):
        frameout[slen + (i - 1) * inc:slen + i * inc] += x[i, slen - inc:]
    # frameout = frameout / max(frameout)
    return frameout


def lpc_synth_using_lflilter(lpc_matrix, f0_pulses, fs, wlen, inc):
    """
    :param lpc_matrix:
    :param f0_pulses:   [1, song_length]
    :param fs: sampling rate
    :param wlen: length of window
    :param inc: step size
    :return: synthed audio of new singer - lpc with f0 pulses
    """
    lpc_order = lpc_matrix.shape[0]
    msoverlap = wlen - inc

    f0_matrix = (enframe(f0_pulses[0,:], wlen, inc)).T  # TODO: check the size if its not transpose
    fn = min(f0_matrix.shape[1], lpc_matrix.shape[1])
    synFrame = np.zeros([wlen, fn])

    # Speech synthesis
    for i in range(fn):
        # synFrame[i, :] = lfilter([1], lpc_matrix[:, i], f0_matrix[:, i])  # Original
        if np.count_nonzero(lpc_matrix[:,i]) >= 1:  # check if not zero vector
            if i == 1082:
                asd=1
            # synFrame[:, i] = lfilter([1], np.trim_zeros(lpc_matrix[:, i], 'f'), f0_matrix[:, i])
            synFrame[:, i] = filtfilt([1], np.trim_zeros(lpc_matrix[:, i], 'f'), f0_matrix[:, i])
        else:  # if zeros vector
            synFrame[:, i] = np.zeros(synFrame[:, i].shape)

    outspeech = Filpframe_OverlapS(synFrame, np.hamming(wlen), inc)
    # plt.subplot(2, 1, 1)
    # plt.plot(lpc_matrix / np.max(np.abs(lpc_matrix)), 'k')
    # plt.title('Original Signal')
    # plt.subplot(2, 1, 2)
    # plt.title('Restore signal-LPC and error')
    # plt.plot(outspeech / np.max(np.abs(outspeech)), 'c')
    # plt.show()
    scipy.io.wavfile.write("outspeech.wav", int(fs), outspeech)
    return outspeech



if __name__ == '__main__':
    # Load Songs
    # x = read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f1_s5_song.wav")
    # x = read_sound("/Users/aarontaub/Desktop/specific_nhss copy/f1_s3_song.wav")
    sound = parselmouth.Sound("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav")

    # Prepare plotting
    # sns.set()  # default style for graphs
    # plt.rcParams['figure.dpi'] = 100  # show large images

    # plot_sound(sound)  # plotting the sound
    #
    # # Extract a part of the song
    # sound_part = sound.extract_part(from_time=0.9, preserve_times=True)
    # plot_sound(sound_part)

    # intensity = sound.to_intensity()
    # spectrogram = sound.to_spectrogram()
    # plt.figure()
    # draw_spectrogram(spectrogram)
    # plt.twinx()
    # draw_intensity(intensity)
    # plt.xlim([sound.xmin, sound.xmax])
    # plt.show()
    #
    # pitch = sound.to_pitch()
    # # If desired, pre-emphasize the sound fragment before calculating the spectrogram
    # pre_emphasized_snd = sound.copy()
    # pre_emphasized_snd.pre_emphasize()
    # spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency = 8000)
    #
    # plt.figure()
    # draw_spectrogram(spectrogram)
    # plt.twinx()
    # draw_pitch(pitch)
    # plt.xlim([sound.xmin, sound.xmax])
    # plt.show()


    # my functions

    # pitch_manipulation(sound)
    # Audio(data=sound.values, rate=sound.sampling_frequency)

    # LPC Extraction
    # lpc_matrix_extract(sound)

    # sound_a = parselmouth.Sound("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav")
    # sound_b = parselmouth.Sound("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/ExampleWithLPC/Audacity/fixed - Aaron talking whitney 2.wav")

    # What we did in praat
    # mixing_voices(sound_a, sound_b)

    # Synthesizing the LPC to sound
    sound = read_sound("/Users/aarontaub/Google Drive/AaronAndAmitBIU/FinalProject/Praat/whitney houston.wav")
    fs = sound.sampling_frequency
    sound_lpc_matrix = lpc_matrix_extract(sound, 64)
    f0 = sound_to_f0_pulse(sound)

    synth_sound = lpc_synth_using_lflilter(lpc_matrix=sound_lpc_matrix, f0_pulses=f0.values, fs=fs, wlen=round(0.025*fs), inc=round(0.005*fs))





