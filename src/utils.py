import os
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken - centralized implementation"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def extract_workshop_id(filename: str) -> str:
    """Extract workshop ID from filename - centralized implementation"""
    return filename.split('-')[0] if '-' in filename else filename.split('.')[0]


def get_openai_client():
    """Initialize OpenAI client with API key - centralized implementation"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key) 