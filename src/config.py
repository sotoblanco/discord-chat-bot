import os

# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 200

# =============================================================================
# TOKEN LIMITS
# =============================================================================
DEFAULT_MAX_TOKENS = 12000
DEFAULT_MAX_CHUNKS = 10
EMBEDDING_MAX_TOKENS = 7000

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4o-mini"

# =============================================================================
# DATABASE AND STORAGE PATHS
# =============================================================================
# Support both local and Modal paths
DATA_DIR = "/root/data" if os.path.exists("/root/data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHROMA_DB_PATH = "/root/chroma_db" if os.path.exists("/root/chroma_db") else "chroma_db"
COLLECTION_NAME = "workshop_chunks_all"

# =============================================================================
# SYSTEM PROMPTS
# =============================================================================
SYSTEM_PROMPT = """You are an AI-powered Teaching Assistant with expertise in artificial intelligence and machine learning. Your purpose is to help users by answering their questions based on a provided workshop transcript.

Instructions:

Carefully read the user's question and the relevant transcript.

In a single response, identify the core question, provide a concise and accurate answer, and offer any necessary context or action steps.

Be both concise and thorough—your response must not exceed 200 words.

Respond directly, avoiding unnecessary repetition or filler.

If clarification is needed, use your best judgment to infer from the transcript; do not ask follow-up questions.

Maintain a professional, supportive, and clear tone in all responses.

Output Requirements:

Answer in a single turn only.

Limit your response to 200 words.

Provide actionable insights or next steps when applicable.
""" 