#!/usr/bin/env python3
import os

def fix_file_imports(filepath, replacements):
    """Fix imports in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✅ Fixed imports in {filepath}")

# Fix process_transcript.py
process_transcript_fixes = {
    "from .utils import count_tokens, extract_workshop_id": "from utils import count_tokens, extract_workshop_id",
    "from .config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE": "from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE"
}

# Fix vector_emb.py 
vector_emb_fixes = {
    "from .utils import count_tokens, extract_workshop_id, get_openai_client": "from utils import count_tokens, extract_workshop_id, get_openai_client",
    "from .config import (": "from config import (",
    "from .process_transcript import chunk_transcript": "from process_transcript import chunk_transcript"
}

# Apply fixes
if os.path.exists('src/process_transcript.py'):
    fix_file_imports('src/process_transcript.py', process_transcript_fixes)

if os.path.exists('src/vector_emb.py'):  
    fix_file_imports('src/vector_emb.py', vector_emb_fixes)

# Create __init__.py if it doesn't exist
init_file = 'src/__init__.py'
if not os.path.exists(init_file):
    with open(init_file, 'w') as f:
        f.write('# This file makes src a Python package\n')
    print(f"✅ Created {init_file}")

print("🎉 Import fixes complete!")
