import pyaudio
import wave
import os
from faster_whisper import WhisperModel


# ANSI escape codes for colors
NEON_GREEN = "\033[92m"  # Bright Green or Neon Green
RESET_COLOR = "\033[0m"  # Reset color back to default


def record_chunk(p, stream, file_path, chunk_length = 10):

    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):

        data = stream.read(1024)            # reads 1024 bytes of audio data from the microphone
        frames.append(data)

    # write the audio data to the WAV file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def transcription(model, file_path):

    segments, _ = model.transcribe(file_path)
    transcriptions = ""
    for segment in segments:
        transcriptions += segment.text + " "
    return transcriptions.strip()


def main():

    model_size = "medium.en"
    model = WhisperModel(model_size, device = "cpu", compute_type = "float32")
    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16, 
                    channels = 1, 
                    rate = 16000, 
                    input = True, 
                    frames_per_buffer = 1024)
    
    final_transcription = ""

    try:
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)
            transcriptions = transcription(model, chunk_file)
            print(NEON_GREEN + transcriptions + RESET_COLOR)
            os.remove(chunk_file)
            final_transcription += transcriptions + " "

    except KeyboardInterrupt:
        print("Stopping...")
        with open("log.txt", "w") as log_file:
            log_file.write(final_transcription)
    
    finally:
        print("LOG : " + final_transcription)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
