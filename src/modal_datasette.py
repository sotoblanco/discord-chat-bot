from pathlib import Path
import os
import sys

import modal

# Add the src directory to Python path for Modal environment
sys.path.append("/root/src")

DB_FILE = "discord-answer-logs.db"  # same as in database.py

app = modal.App("discord-datasette")

image = (
    modal.Image.debian_slim()
    .pip_install("setuptools")  # Add setuptools which includes pkg_resources
    .pip_install("datasette~=0.63.2")
    .add_local_dir("src", "/root/src")
)

db_storage = modal.Volume.from_name("discord-bot-volume", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data/db": db_storage},
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def ui():
    import asyncio
    import sqlite3

    from datasette.app import Datasette
    from database import init_db

    remote_db_path = Path("/data/db") / DB_FILE
    local_db_path = Path(".") / DB_FILE
    
    # Check if database exists in volume, if not create it
    if not remote_db_path.exists():
        print(f"Database file not found at {remote_db_path}, initializing...")
        # Initialize the database
        init_db()
        print("Database initialized successfully")
    
    # Copy database file to local path
    if remote_db_path.exists():
        local_db_path.write_bytes(remote_db_path.read_bytes())
        print(f"Database copied from {remote_db_path} to {local_db_path}")
    else:
        print(f"Warning: Database file still not found at {remote_db_path}")
        # Create empty database locally
        conn = sqlite3.connect(local_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                channel_id TEXT,
                question TEXT,
                response TEXT,
                feedback TEXT,
                thread_id TEXT
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                question TEXT,
                response TEXT,
                num_chunks INTEGER,
                context_tokens INTEGER,
                completion_tokens INTEGER,
                embedding_tokens INTEGER,
                total_tokens INTEGER,
                latency REAL,
                model TEXT,
                sources TEXT,
                success INTEGER,
                feedback_rating TEXT,
                feedback_reason TEXT,
                feedback_notes TEXT,
                feedback_user TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("Created empty database with tables")

    ds = Datasette(files=[local_db_path], settings={"sql_time_limit_ms": 10000})
    asyncio.run(ds.invoke_startup())
    return ds.app()
