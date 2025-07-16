
# Discord Chat Bot

A Python-based Discord bot using RAG and Modal to answer questions about the transcription of the live sessions of the course [Building LLM Applications for Data Scientists and Software Engineers](https://maven.com/hugo-stefan/building-llm-apps-ds-and-swe-from-first-principles) using Retrieval-Augmented Generation (RAG). The bot integrates with Discord to respond to mentions, creates organized conversation threads, and uses ChromaDB for fast semantic search with OpenAI's GPT-4o-mini for generating responses.


## Model design

![image](images/DiscordLLMv2.jpg)

Note that an OpenAI API key is required.

## 🚀 Quick Start

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/sotoblanco/discord-chat-bot.git
cd discord-chat-bot
# Set environment variables
export OPENAI_API_KEY="your_openai_api_key"
```

```sh
# Option A: Install with pip
## Create a virtual environment
python -m venv venv
source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

# Option B: Install with uv
# Ensure uv is installed: https://docs.astral.sh/uv/getting-started/installation/
uv sync
```

### 2. Run Locally

```bash
# For local testing without Discord
python src/interactive_qa.py
```

## 📊 Evaluate Multiple Queries

Create a JSON file with questions:

```json
[
    {
      "id": "test_1",
      "question": "What are some best practices for prompt engineering?"
    }
]
```

Run evaluation:

```bash
python eval/test_retrieval.py
```


### 3. Discord Bot Setup

1. **Create a Discord Application**:
   - Go to the [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application
   - Under the "Bot" tab, add a bot to your application
   - Copy the bot token and set it as `DISCORD_BOT_TOKEN`

Set the bot token:

```bash
export DISCORD_BOT_TOKEN="your_discord_bot_token"
```

### 3. Deploy to Modal

```bash
# Deploy the Discord bot (use slashes, not dots)
modal deploy src/modal_discord.py

# Start the Discord bot
modal run src/modal_discord.py::discord_bot_runner

# (Optional) Initialize vector database (first run)
modal run src/modal_discord.py::clean_and_rechunk_workshops
```

### 4. Deploy the Feedback Database (Datasette)

This is a datasette database that stores the question-answer pairs and the feedback from the users.

```bash
modal deploy src/modal_datasette.py
```

---

## ✨ Key Features

- **Chunking**: Balanced chunk distribution with multiple fallback strategies
- **Discord Integration**: Responds to mentions and creates organized conversation threads
- **ChromaDB Vector Search**: Fast semantic search across workshop transcripts
- **OpenAI Integration**: Uses GPT-4o-mini for intelligent responses
- **Persistent Storage**: Data persists across deployments using Modal Volumes
- **Feedback Storage**: Thumbs up/down and thread messages are stored in the database
- **Workshop Source Display**: Bot answers include the workshops used as sources
- **Auto-restart**: Bot automatically restarts every 55 minutes to prevent timeouts
- **Diagnostic Tools**: Built-in functions to analyze and fix chunking issues

---

## 🏗️ Core Architecture

```txt
📁 src/
├── modal_discord.py      # Main bot deployment & diagnostic functions
├── modal_datasette.py    # Datasette database deployment
├── process_transcript.py # Robust chunking with fallback strategies
├── vector_emb.py         # Vector embeddings & retrieval
├── database.py           # Interaction logging

📁 eval/
├── evaluate_system.py    # LLM evaluation system
├── test_retrieval.py     # Test retrieval system
├── questions.json        # Questions for evaluation
├── eval_progress.json    # Progress tracking

📁 data/                  # Workshop VTT transcript files
📁 chroma_db/             # Persistent vector database
```

---

## 🤖 Discord Bot Usage

1. **Mention the bot**: `@DiscordBot What are debugging practice for RAG applications?`
2. **Use "bot" keyword**: `bot explain evaluation systems for AI applications`
3. **Get responses in threads**: Bot creates organized conversation threads
4. **Workshop-aware**: Responses include source workshop information (e.g., "Sources: This answer was based on information from WS1, WS2")
5. **Feedback**: React with 👍 or 👎, or reply in the thread—feedback is stored in the database

---

## 📝 Feedback Database

- All user feedback (thumbs up/down and thread replies) is stored in the persistent database (`discord-answer-logs.db`) on the Modal volume (`discord-bot-volume`).
- You can view the feedback using the datasette app deployed on Modal, or locally with sqlite3:

```bash
# Using sqlite3 locally
sqlite3 data/discord-answer-logs.db

# Using datasette (deployed on Modal)
modal deploy src/modal_datasette.py
```

---

## 📚 Workshop Source Display

- The bot's answers now include the names of the workshops from which the answer was derived.
- If no sources are found, the bot will indicate that the answer is based on the available transcripts.

---

---

## 🛠️ Resetting Modal Volumes and Redeploying

If you want to reset the data (database or vector DB) and redeploy with new data:

```bash
# List your volumes
modal volume list

# Delete a volume (WARNING: this deletes all data in the volume)
modal volume delete discord-bot-volume
modal volume delete chroma-db-volume

# (Optional) Create a new volume
modal volume create discord-bot-volume
modal volume create chroma-db-volume

# Update your code to use the new volume names if needed
# Then redeploy:
modal deploy src/modal_discord.py
modal deploy src/modal_datasette.py
```

---

## 🐞 Troubleshooting

### Bot Not Responding or Feedback Not Stored

- Ensure your Modal volumes are correctly mounted and not empty
- Check logs:

```bash
modal app logs discord-chat-bot
```

- If feedback is not being stored, check the logs for errors in the `store_user_feedback` function
- If workshop sources are not shown, ensure your vector database is populated:

```bash
modal run src/modal_discord.py::check_database_status
```

- If you need to reset the vector DB, use:

```bash
modal run src/modal_discord.py::clean_and_rechunk_workshops
```

---

## 🎛️ Configuration

### Key Constants (`vector_emb.py`)

- `DEFAULT_MAX_TOKENS = 12000` - Max context tokens for LLM
- `EMBEDDING_MAX_TOKENS = 7000` - Max tokens for embedding model
- `DEFAULT_MAX_CHUNKS = 10` - Chunks retrieved per query

### Chunking Parameters (`process_transcript.py`)

- `CHUNK_SIZE = 1000` - Target tokens per chunk
- `MIN_CHUNK_SIZE = 200` - Minimum tokens required per chunk
- `CHUNK_OVERLAP = 100` - Token overlap between chunks

---

## 🆘 Support

For issues with chunking imbalance, feedback storage, or bot functionality:

1. **Run diagnostics**: `modal run src/modal_discord.py::diagnose_chunking_issues`
2. **Check database**: `modal run src/modal_discord.py::analyze_chromadb_content`
3. **Clean & rechunk**: `modal run src/modal_discord.py::clean_and_rechunk_workshops`
4. **Verify fix**: `modal run src/modal_discord.py::debug_vector_database`
5. **Check feedback**: `modal run src/modal_discord.py::view_feedback_database`

---

## Version History

- **v2.0**: Feedback storage improved, workshop sources shown in Discord, Modal deployment commands updated, new diagnostics added.
- **v1.x**: Initial bot, basic RAG, Discord integration.

---

## License

MIT
