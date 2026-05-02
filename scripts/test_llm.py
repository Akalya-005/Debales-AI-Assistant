
from app.config import Settings
from app.llm import get_chat_model
import time

def test_llm():
    settings = Settings.from_env()
    print("Initializing LLM...")
    start = time.time()
    llm = get_chat_model(settings)
    print(f"LLM initialized in {time.time() - start:.2f}s")
    
    if llm is None:
        print("LLM is None. Check your .env file.")
        return

    print("Calling LLM (Groq)...")
    start = time.time()
    try:
        response = llm.invoke("Say 'Hello, I am ready' in 5 words.")
        print(f"LLM responded in {time.time() - start:.2f}s")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"LLM call failed: {e}")

if __name__ == "__main__":
    test_llm()
