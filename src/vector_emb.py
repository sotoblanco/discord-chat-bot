import os
import time
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import json
import glob

from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
import re
import uuid

from process_transcript import chunk_transcript


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken"""
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

# Support both local and Modal paths
DATA_DIR = "/root/data" if os.path.exists("/root/data") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
COLLECTION_NAME = "workshop_chunks_all"
CHROMA_DB_PATH = "/root/chroma_db" if os.path.exists("/root/chroma_db") else "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MAX_TOKENS = 12000
DEFAULT_MAX_CHUNKS = 10
COMPLETION_MODEL = "gpt-4o-mini"
EMBEDDING_MAX_TOKENS = 7000

SYSTEM_PROMPT = """You are an AI-powered Teaching Assistant with expertise in artificial intelligence and machine learning. Your purpose is to help users by answering their questions based on a provided workshop transcript.

Instructions:

Carefully read the user’s question and the relevant transcript.

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


def get_openai_client():
    """Initialize OpenAI client with API key"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

############### workshop discovery, chunking and embedding ##########

def discover_workshops(data_dir=DATA_DIR):
    """Discover all workshop VTT files in the data directory"""
    try:
        pattern = os.path.join(data_dir, "*.vtt")
        vtt_files = glob.glob(pattern)
        
        workshops = {}
        for vtt_file in vtt_files:
            filename = os.path.basename(vtt_file)
            workshop_id = filename.split('-')[0] if '-' in filename else filename.split('.')[0]
            
            workshops[workshop_id] = {
                'id': workshop_id,
                'filename': filename,
                'path': vtt_file
            }
        
        return workshops
        
    except Exception as e:
        print(f"Error discovering workshops: {e}")
        return {}

def split_large_chunk(text: str, max_tokens: int = EMBEDDING_MAX_TOKENS) -> List[str]:
    """Split a large chunk into smaller pieces that fit within token limits"""
    if count_tokens(text) <= max_tokens:
        return [text]
    
    sentences = re.split(r'[.!?]+\s+', text)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        if sentence_tokens > max_tokens:
            words = sentence.split()
            word_chunk = ""
            word_tokens = 0
            
            for word in words:
                word_token_count = count_tokens(word)
                if word_tokens + word_token_count > max_tokens and word_chunk:
                    chunks.append(word_chunk.strip())
                    word_chunk = word
                    word_tokens = word_token_count
                else:
                    word_chunk += " " + word if word_chunk else word
                    word_tokens += word_token_count
            
            if word_chunk:
                chunks.append(word_chunk.strip())
            continue
        
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            current_chunk += ". " + sentence if current_chunk else sentence
            current_tokens += sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text with safe token splitting"""
    client = get_openai_client()
    
    token_count = count_tokens(text)
    
    if token_count <= EMBEDDING_MAX_TOKENS:
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    else:
        split_texts = split_large_chunk(text, EMBEDDING_MAX_TOKENS)
        
        embeddings = []
        for split_text in split_texts:
            response = client.embeddings.create(
                input=split_text,
                model=EMBEDDING_MODEL
            )
            embeddings.append(response.data[0].embedding)
        
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        return avg_embedding

########## ChromaDB ##########

def get_chroma_client():
    """Initialize and return a ChromaDB client with persistence"""
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_or_create_collection(client, collection_name=COLLECTION_NAME):
    """Get or create a collection in ChromaDB"""
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Workshop transcript chunks"}
        )
    return collection

def add_chunks_to_collection(collection, chunks, workshop_id):
    """Add multiple chunks to the collection with workshop metadata"""
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk['text'])
            chunk_id = f"{workshop_id}_{chunk['chunk_id']}"
            
            ids.append(chunk_id)
            documents.append(chunk['text'])
            embeddings.append(embedding)
            
            metadata = {
                'workshop_id': workshop_id,
                'position': chunk['position'],
                'token_count': chunk['token_count'],
                'source': chunk['source'],
                'timestamp': chunk.get('timestamp', 'Unknown'),
                'speaker': chunk.get('speaker', 'Unknown'),
                'original_chunk_id': chunk['chunk_id']
            }
            metadatas.append(metadata)
            
        except Exception as e:
            continue
    
    if ids:
        try:
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            return len(ids)
        except Exception as e:
            return 0
    else:
        return 0

def process_workshop(workshop_data: Dict[str, str], collection_name: str = COLLECTION_NAME) -> int:
    """Process a single workshop"""
    try:
        workshop_id = workshop_data['id']
        workshop_path = workshop_data['path']
        
        client = get_chroma_client()
        collection = get_or_create_collection(client, collection_name)
        
        # Check if already processed
        try:
            existing_results = collection.query(
                query_embeddings=[[0.0] * 1536],
                n_results=1,
                where={"workshop_id": workshop_id}
            )
            if existing_results and existing_results['ids'] and len(existing_results['ids'][0]) > 0:
                return 0
        except:
            pass
        
        chunks = chunk_transcript(workshop_path, workshop_id)
        
        if not chunks:
            return 0
        
        num_added = add_chunks_to_collection(collection, chunks, workshop_id)
        return num_added
        
    except Exception as e:
        print(f"Error processing workshop {workshop_data.get('id', 'Unknown')}: {str(e)}")
        return 0

def process_all_workshops(collection_name=COLLECTION_NAME):
    """Process all discovered workshops"""
    workshops = discover_workshops()
    
    if not workshops:
        return []
    
    processed_workshops = []
    
    for workshop_id, workshop_info in workshops.items():
        try:
            num_chunks = process_workshop(workshop_info, collection_name)
            if num_chunks > 0:
                processed_workshops.append(workshop_id)
        except Exception as e:
            continue
    
    return processed_workshops

########## Retrieval ##########

def query_collection(collection, query_text, n_results=DEFAULT_MAX_CHUNKS, workshop_filter=None):
    """Query the collection for relevant documents"""
    query_embedding = generate_embedding(query_text)
    
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]  # <- ensure distances are returned
    }
    
    if workshop_filter:
        if isinstance(workshop_filter, str):
            query_params["where"] = {"workshop_id": workshop_filter}
        elif isinstance(workshop_filter, list):
            query_params["where"] = {"workshop_id": {"$in": workshop_filter}}
    
    results = collection.query(**query_params)
    return results

def retrieve_relevant_chunks(question, collection_name=COLLECTION_NAME, n_results=DEFAULT_MAX_CHUNKS, workshop_filter=None):
    """Retrieve chunks from vector database for a given question"""
    client = get_chroma_client()
    collection = get_or_create_collection(client, collection_name)
    
    results = query_collection(collection, question, n_results=n_results, workshop_filter=workshop_filter)
    
    chunks = []
    if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            relevance = max(0, 1 - (distance / 2))  # Normalize to 0–1 range

            chunk = {
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i],
                'relevance': relevance
            }
            chunks.append(chunk)
    
    return chunks

def combine_chunks(chunks, max_tokens=DEFAULT_MAX_TOKENS):
    """Combine multiple chunks into a single context"""
    if not chunks:
        return ""
    
    sorted_chunks = sorted(chunks, key=lambda x: int(x['metadata'].get('position', 0)))
    
    combined_text = ""
    total_tokens = 0
    
    for chunk in sorted_chunks:
        chunk_text = chunk['text']
        chunk_tokens = int(chunk['metadata'].get('token_count', 0))
        
        if chunk_tokens == 0:
            chunk_tokens = count_tokens(chunk_text)
        
        if total_tokens + chunk_tokens > max_tokens:
            break
        
        if combined_text:
            combined_text += "\n\n--- Next Section ---\n\n"
        
        combined_text += chunk_text
        total_tokens += chunk_tokens
    
    return combined_text

########## Answer Question ##########

def get_context_for_question(question, collection_name=COLLECTION_NAME, max_chunks=DEFAULT_MAX_CHUNKS, workshop_filter=None):
    """Get relevant context from the vector database for a question"""
    chunks = retrieve_relevant_chunks(question, collection_name, max_chunks, workshop_filter)
    
    sources = []
    for chunk in chunks:
        metadata = chunk['metadata']
        text = chunk['text']
        
        source = {
            'position': metadata.get('position', 'Unknown'),
            'timestamp': metadata.get('timestamp', "Unknown"),
            'speaker': metadata.get('speaker', "Unknown"),
            'workshop_id': metadata.get('workshop_id', "Unknown"),
            'text': text,
            'relevance': chunk.get('relevance')
        }
        sources.append(source)
    
    context = combine_chunks(chunks)
    return context, sources, chunks

def answer_question(question, workshop_filter=None):
    """Answer a question based on workshop transcripts"""
    client = get_chroma_client()
    collection = get_or_create_collection(client, COLLECTION_NAME)
    
    # Check if collection is empty and populate if needed
    count = collection.count()
    if count == 0:
        process_all_workshops(COLLECTION_NAME)
    
    context, sources, chunks = get_context_for_question(
        question=question,
        collection_name=COLLECTION_NAME,
        max_chunks=DEFAULT_MAX_CHUNKS,
        workshop_filter=workshop_filter
    )
    
    return context, sources, chunks

def llm_answer_question(client, context, sources, chunks, question):
    """Generate LLM answer with workshop awareness"""
    num_chunks = len(sources)
    workshops_used = list(set([source.get('workshop_id', 'Unknown') for source in sources]))
    
    client = get_openai_client()
    try:
        enhanced_system_prompt = SYSTEM_PROMPT + f"\n\nThe information provided comes from workshops: {', '.join(workshops_used)}."
        
        response = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": f"Workshop Transcript Sections:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0
        )
        
        message = response.choices[0].message.content
        
        completion_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
        prompt_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else count_tokens(context)
        
        context_info = {
             "num_chunks": num_chunks,
             "context_tokens": prompt_tokens,
             "completion_tokens": completion_tokens,
             "embedding_tokens": num_chunks * 1536,
             "workshops_used": workshops_used,
             "chunks": chunks
         }
        
        return message, context_info

    except Exception as e:
        error_message = f"Sorry, an error occurred: {str(e)}"
        return error_message, {"error": str(e)}

# ...existing code...

if __name__ == "__main__":
    # Example usage or test code here
    print("This runs only when vector_emb.py is executed directly.")