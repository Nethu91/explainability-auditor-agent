import os
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
or_key = os.getenv("OPENROUTER_API_KEY")

print("GROQ key loaded:", groq_key is not None, "- length:", len(groq_key) if groq_key else 0)
print("OpenRouter key loaded:", or_key is not None, "- length:", len(or_key) if or_key else 0)