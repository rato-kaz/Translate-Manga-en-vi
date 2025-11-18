from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import time
from langdetect import detect as langdetect_detect
from fast_langdetect import detect as fast_detect

# Load model
model_name = "papluca/xlm-roberta-base-language-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

texts = [
    "こんにちは！",
    "Tôi đang đọc manga.",
    "This is a test.",
    "这是一个测试。"
]

# Map ID → label
labels = model.config.id2label

def detect_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        idx = torch.argmax(probs, dim=1).item()
    return labels[idx], probs[0][idx].item()

print("=" * 60)
print("Test with xlm-r")
print("=" * 60)

# Test
for t in texts:
    start_time = time.time()
    lang, prob = detect_language(t)
    end_time = time.time()
    print(f"Thời gian chạy: {end_time - start_time} giây")
    print(f"Text: {t}")
    print(f"Detected language: {lang}, probability={prob:.4f}")
    print("--------------")
    
print("=" * 60)
print("Test with langdetect")
print("=" * 60)

# Test with langdetect
for t in texts:
    start_time = time.time()
    lang = langdetect_detect(t)
    end_time = time.time()
    print(f"Thời gian chạy: {end_time - start_time} giây")
    print(f"Text: {t}")
    print(f"Detected language: {lang}")
    print("--------------")
    
print("=" * 60)
print("Test with fast-langdetect")
print("=" * 60)
# Test with fast-langdetect
for t in texts:
    start_time = time.time()
    results = fast_detect(t, model='full', k=1)
    end_time = time.time()
    result = results[0] if results else {}
    lang = result.get('lang', 'unknown')
    prob = result.get('score', 0.0)
    print(f"Thời gian chạy: {end_time - start_time} giây")
    print(f"Text: {t}")
    print(f"Detected language: {lang}, probability={prob:.4f}")
    print("--------------")