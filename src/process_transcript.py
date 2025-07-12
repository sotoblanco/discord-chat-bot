import re
import uuid
from typing import List, Dict, Any
import os

# Import shared utilities and configuration
from .utils import count_tokens, extract_workshop_id
from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

def load_vtt_content(file_path):
    """Load VTT file and extract clean text content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
    
    lines = content.split('\n')
    content_lines = []
    
    for line in lines:
        line = line.strip()
        if (not line or 
            line == 'WEBVTT' or 
            '-->' in line or 
            re.match(r'^\d+:\d+:\d+', line) or
            re.match(r'^[A-Z]+(\s*:.*)?$', line)):
            continue
        content_lines.append(line)
    
    return " ".join(content_lines)

def chunk_transcript(file_path: str, workshop_id: str) -> List[Dict[str, Any]]:
    """Simple, reliable chunking strategy with sentence-based splitting"""
    text = load_vtt_content(file_path)
    if not text:
        return []
    
    sentences = [s.strip() for s in re.split(r'[.!?]+\s+', text) if s.strip()]
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = []
    current_tokens = 0
    position = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if current_tokens + sentence_tokens > CHUNK_SIZE and current_chunk_sentences:
            chunk_text = ". ".join(current_chunk_sentences) + "."
            chunk = create_chunk(chunk_text, position, workshop_id)
            chunks.append(chunk)
            
            overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) >= 2 else current_chunk_sentences
            current_chunk_sentences = overlap_sentences + [sentence]
            current_tokens = count_tokens(". ".join(current_chunk_sentences))
            position += 1
        else:
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
    
    if current_chunk_sentences:
        chunk_text = ". ".join(current_chunk_sentences) + "."
        chunk = create_chunk(chunk_text, position, workshop_id)
        chunks.append(chunk)
    
    valid_chunks = [chunk for chunk in chunks if chunk['token_count'] >= MIN_CHUNK_SIZE]
    return valid_chunks

def create_chunk(text: str, position: int, workshop_id: str) -> Dict[str, Any]:
    """Create a chunk dictionary with metadata"""
    return {
        "chunk_id": str(uuid.uuid4()),
        "text": text.strip(),
        "position": position,
        "token_count": count_tokens(text),
        "source": "sentence_chunking",
        "timestamp": f"Chunk {position + 1}",
        "speaker": "Unknown",
        "workshop_id": workshop_id
    }

def chunk_workshop_transcript(transcript_path: str) -> List[Dict[str, Any]]:
    """Main chunking function - extracts workshop ID from path"""
    filename = os.path.basename(transcript_path)
    workshop_id = extract_workshop_id(filename)
    return chunk_transcript(transcript_path, workshop_id)