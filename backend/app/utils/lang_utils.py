from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    try:
        lang_code = detect(text)
        return lang_code
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"
