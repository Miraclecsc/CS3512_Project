import torch
import time
import os
from datasets import load_dataset, Audio
from jiwer import wer, cer, Compose, RemovePunctuation, RemoveWhiteSpace, RemoveMultipleSpaces, ToLowerCase, Strip
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSpeechSeq2Seq, Wav2Vec2FeatureExtractor
from faster_whisper import WhisperModel

# 经典whisper
asr_model_id = "openai/whisper-tiny.en"
transcriber = pipeline("automatic-speech-recognition", model=asr_model_id, device="cpu")

# fast whisper
# class FastWhisperPipeline:
#     def __init__(self, model_name, device="cpu"):
#         self.model = WhisperModel(model_name, device=device)
    
#     def __call__(self, audio_data):
#         # `audio_data` 可以是音频文件路径或音频数据数组，根据 `WhisperModel` 的实现调整
#         segments, info = self.model.transcribe(audio_data)
#         transcription_text = " ".join([segment.text for segment in segments])
#         return {"text": transcription_text}

# asr_model_id = "tiny.en"
# transcriber = FastWhisperPipeline(model_name=asr_model_id, device="cuda")


# distil whisper
# local_model_path = "./whisper-tiny-distil"
# transcriber = pipeline("automatic-speech-recognition", model=local_model_path)

# 下载模型及文件 
# local_model_path = "./fast-whisper-tiny"
# os.makedirs(local_model_path, exist_ok=True)
# model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en")
# model.save_pretrained(local_model_path)
# tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny.en")
# tokenizer.save_pretrained(local_model_path)
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("openai/whisper-tiny.en")
# feature_extractor.save_pretrained(local_model_path)

dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
dataset = dataset.take(100)
dataset = dataset.map(lambda x: {"audio": x["audio"]["array"]}, remove_columns=["file", "speaker_id", "chapter_id", "id"])

inference_times = []
wer_scores = []
memory_usages = []


transform = Compose([
    RemovePunctuation(),
    RemoveWhiteSpace(replace_by_space=True),
    RemoveMultipleSpaces(),
    ToLowerCase(),
    Strip()
])

for i, data in enumerate(dataset):
    audio_data = data["audio"]
    reference = data["text"]
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    start_time = time.time()
    result = transcriber(audio_data)
    end_time = time.time()
    
    max_memory_usage = torch.cuda.max_memory_allocated()
    memory_usages.append(max_memory_usage)
    
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    hypothesis = result['text']
    reference_processed = transform(reference)
    print(f"Reference : {reference_processed}")
    hypothesis_processed = transform(hypothesis)
    print(f"Hypothesis: {hypothesis_processed}")
    

    error_rate = wer(reference_processed, hypothesis_processed)
    print(f"Word Error Rate (WER): {error_rate}\n")
    wer_scores.append(error_rate)
    
    # print(f"Processed {i+1}/100")

print(f"Average Inference Time: {sum(inference_times) / len(inference_times)} seconds")
print(f"Average Word Error Rate (WER): {sum(wer_scores) / len(wer_scores)}")
print(f"Average Memory Usage: {sum(memory_usages) / len(memory_usages)} bytes")
