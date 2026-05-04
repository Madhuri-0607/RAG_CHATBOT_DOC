import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.curdir))

try:
    from src.llm.ollama_llm import get_ollama_llm
    print("Import successful")
    
    llm = get_ollama_llm()
    print(f"Testing with model: {llm.model}")
    
    response = llm.invoke("Say hello")
    print(f"Response: {response}")
except Exception as e:
    print(f"Error: {e}")
