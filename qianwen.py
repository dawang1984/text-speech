import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = (
    r"D:\code_center\models"
    r"\Qwen3-TTS-12Hz-0.6B-CustomVoice"
)

print("Loading model...")

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.float32,
    attn_implementation="eager",
)

print("Model loaded.")

wavs, sr = model.generate_custom_voice(
    text="Hell guys, I'll be away tomorrow. See you next Tuesday.",
    language="English",
    speaker="Uncle_Fu",
)

sf.write("test_custom_voice2.wav", wavs[0], sr)

print("合成完成：test_custom_voice.wav")