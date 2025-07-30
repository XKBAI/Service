from faster_whisper import WhisperModel
import time

start=time.time()
#model_size = "large-v3"
#model_size = "small"

path = r"/home/xkb2/Desktop/QY/faster-whisper/large-v3"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16",local_files_only=True)
model = WhisperModel(model_size_or_path = path, device="cuda", compute_type="float16",local_files_only=True)


# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")


#segments, info = model.transcribe("audio3.mp3", beam_size=5)
segments, info = model.transcribe("audio3.mp3", beam_size=5,language="zh",vad_filter=True,vad_parameters=dict(min_silence_duration_ms=1000))

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
print(time.time()-start)
