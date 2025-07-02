import os
import sys
import time
import re
import asyncio
import datetime
from typing import Dict, Any
from contextlib import asynccontextmanager

import modal
from fastapi import FastAPI

# Add the src directory to Python path for Modal environment
sys.path.append("/root/src")

# Import Discord and other heavy dependencies only inside Modal functions
# from database import init_db, log_interaction, log_track_interaction
# from vector_emb import answer_question, llm_answer_question, get_openai_client

# Modal app setup
app = modal.App("discord-chat-bot")

# Define the Docker image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "discord.py>=2.3.0",
    "openai>=1.0.0",
    "chromadb",
    "tiktoken",
    "python-dotenv",
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0"
).add_local_dir("src", "/root/src").add_local_dir("data", "/root/data")

# Define secrets
secrets = [
    modal.Secret.from_name("openai-secret"),
    modal.Secret.from_name("discord-secret-2"),
    modal.Secret.from_dict({"MODAL_LOGLEVEL": "DEBUG"})  # Add debug logging
]

# Define Modal volumes for persistence
discord_bot_volume = modal.Volume.from_name("discord-bot-volume-2", create_if_missing=True)
chroma_volume = modal.Volume.from_name("chroma-db-volume-2", create_if_missing=True)

volume_mounts = {
    "/data/db": discord_bot_volume,
    "/root/chroma_db": chroma_volume
}

def bot_is_mentioned(content: str, client_user) -> bool:
    """Checks if the bot is mentioned or addressed in the message content."""
    # Use word boundaries (\b) to avoid matching parts of other words
    return (
        client_user.mention in content
        or re.search(r"\bbot\b", content, re.IGNORECASE) is not None
    )

@app.function(
    image=image, 
    secrets=secrets,
    volumes=volume_mounts,
    timeout=0
)
def fetch_api(question: str) -> Dict[str, Any]:
    """Get answer from OpenAI using context from vector database"""
    # Import dependencies inside the function
    from database import init_db, log_track_interaction
    from vector_emb import answer_question, llm_answer_question, get_openai_client
    
    start_time = time.time()
    
    try:
        # Initialize database
        init_db()
        
        # Get context and answer
        context, sources, chunks = answer_question(question)
        client = get_openai_client()
        response, context_info = llm_answer_question(client, context, sources, chunks, question)
        
        end_time = time.time()
        
        # Log the interaction
        log_id = log_track_interaction(
            question=question,
            response=response,
            context_info=context_info,
            model="gpt-4o-mini",
            start_time=start_time,
            end_time=end_time,
            success=True
        )
        
        return {
            "answer": response,
            "log_id": log_id,
            "context_info": context_info
        }
        
    except Exception as e:
        end_time = time.time()
        error_msg = f"Error generating response: {str(e)}"
        
        # Log failed interaction
        log_track_interaction(
            question=question,
            response=error_msg,
            context_info={"error": str(e)},
            model="gpt-4o-mini", 
            start_time=start_time,
            end_time=end_time,
            success=False
        )
        
        return {"answer": error_msg, "log_id": None, "context_info": {}}

@app.function(image=image, timeout=10)
def health_check():
    """Simple health check function"""
    return "alive"

# Global to track if bot is running (in-memory state)
bot_instance = None

@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    timeout=60*60,  # 1 hour timeout, will auto-restart
    schedule=modal.Period(minutes=55)  # Restart every 55 minutes to stay within timeout
)
@modal.concurrent(max_inputs=1)
async def discord_bot_runner():
    """Main function that runs the Discord bot with periodic restarts."""
    import discord
    from discord.ext import commands
    import datetime
    
    global bot_instance
    
    print(f"🚀 Starting Discord bot runner at {datetime.datetime.now()}")
    
    # Bot status tracking
    status = {"running": False, "last_restart": None, "last_error": None}
    
    def setup_discord_bot():
        """Set up and configure the Discord bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            print(f"Bot is ready! Logged in as {bot.user.name}")
            print(f"Guilds connected: {len(bot.guilds)}")
            status["running"] = True

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return

            # Check if the message is from the 'bwai-bot' channel
            if message.channel.name != 'bwai-bot':
                return

            if bot_is_mentioned(message.content, bot.user):
                print(f"Question received: {message.content}")
                thread = await message.create_thread(
                    name=f"Question from {message.author.display_name}",
                    auto_archive_duration=60,
                )
                await thread.send(f"Hey {message.author.mention}, let me think about that...")
                try:
                    result = fetch_api.remote(message.content)  # Remove await - Modal .remote() is synchronous
                    answer = result["answer"]
                    log_id = result["log_id"]
                    await thread.send(f"**Answer:** {answer}")
                    feedback_msg = await thread.send(
                        "\nWas this answer helpful? Please reply with:\n"
                        "👍 for helpful or 👎 for not helpful\n"
                        "Or provide additional feedback!"
                    )
                    await feedback_msg.add_reaction("👍")
                    await feedback_msg.add_reaction("👎")
                except Exception as e:
                    await thread.send(f"Sorry, I encountered an error: {str(e)}")
                    print(f"Error processing question: {e}")

            await bot.process_commands(message)

        return bot
    
    try:
        status["last_restart"] = datetime.datetime.now().isoformat()
        
        # Get Discord token with better error handling
        discord_token = os.environ.get("DISCORD_BOT_TOKEN")
        if not discord_token:
            raise ValueError(
                "DISCORD_BOT_TOKEN environment variable not set. "
                "Please ensure the discord-secret-2 secret is properly configured in Modal."
            )
        
        print(f"Discord token found (length: {len(discord_token)})")
        
        client = setup_discord_bot()
        bot_instance = client
        
        print("Starting Discord bot on Modal infrastructure...")
        print(f"Bot will run until {datetime.datetime.now() + datetime.timedelta(minutes=55)}")
        
        # Run the bot (this blocks until disconnected or timeout)
        await client.start(discord_token)

    except Exception as e:
        print(f"Bot encountered an error: {type(e).__name__}: {e}")
        status["running"] = False
        status["last_error"] = str(e)
        if bot_instance and not bot_instance.is_closed():
            await bot_instance.close()
    finally:
        print("Bot runner completed. Will restart via schedule.")


# Keep the old Bot class for backwards compatibility but mark it as deprecated
@app.cls(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    scaledown_window=300,
    timeout=0,
)
class Bot:
    """DEPRECATED: Use discord_bot_runner function instead"""
    @modal.enter()
    async def start(self):
        print("⚠️ WARNING: Bot class is deprecated. Use discord_bot_runner function instead.")
        pass

    @modal.exit()
    async def stop(self):
        pass

    @modal.fastapi_endpoint()
    def health(self):
        return {"status": "deprecated", "message": "Use discord_bot_runner function"}

    @modal.fastapi_endpoint() 
    def status(self):
        return {"status": "deprecated", "message": "Use discord_bot_runner function"}


# Add new monitoring endpoints
@app.function(image=image)
@modal.fastapi_endpoint()
def bot_health():
    """Health check endpoint for the Discord bot"""
    return {
        "status": "healthy",
        "service": "discord-chat-bot",
        "timestamp": datetime.datetime.now().isoformat()
    }


@app.function(image=image)
@modal.fastapi_endpoint()
def bot_info():
    """Information endpoint about the bot deployment"""
    return {
        "bot_name": "Discord Chat Bot",
        "deployment": "Modal",
        "version": "2.0",
        "features": [
            "RAG-based Q&A",
            "Thread creation for conversations",
            "Feedback collection",
            "Auto-restart on errors"
        ]
    }


@app.local_entrypoint()
def main():
    """Entry point for running the Discord bot."""
    print("Deploying Discord bot to Modal...")
    # When deployed, the run_bot function will be called automatically
    # For local testing, you can call it directly
    if os.environ.get("MODAL_ENVIRONMENT"):
        print("Running in Modal environment")
    else:
        print("To deploy: modal deploy src/modal_discord_bot.py")
        print("The bot will start automatically after deployment")


if __name__ == "__main__":
    print("This script should be run via Modal commands:")
    print("modal deploy src/modal_discord_bot.py")
    print("To run locally for testing: modal run src/modal_discord_bot.py")

# Add a function to manually trigger the bot runner
@app.function(
    image=image,
    secrets=secrets,
    volumes=volume_mounts,
    timeout=0,  # Run indefinitely
)
async def start_persistent_bot():
    """Start the Discord bot and keep it running indefinitely."""
    import discord
    from discord.ext import commands
    import datetime
    
    print(f"🚀 Starting persistent Discord bot at {datetime.datetime.now()}")
    
    def setup_discord_bot():
        """Set up and configure the Discord bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix="!", intents=intents)

        @bot.event
        async def on_ready():
            print(f"Bot is ready! Logged in as {bot.user.name}")
            print(f"Guilds connected: {len(bot.guilds)}")

        @bot.event
        async def on_message(message):
            if message.author == bot.user:
                return

            if bot_is_mentioned(message.content, bot.user):
                print(f"Question received: {message.content}")
                thread = await message.create_thread(
                    name=f"Question from {message.author.display_name}",
                    auto_archive_duration=60,
                )
                await thread.send(f"Hey {message.author.mention}, let me think about that...")
                try:
                    result = fetch_api.remote(message.content)  # Modal .remote() is synchronous, no await needed
                    answer = result["answer"]
                    log_id = result["log_id"]
                    await thread.send(f"**Answer:** {answer}")
                    feedback_msg = await thread.send(
                        "Was this answer helpful? Please reply with:\n"
                        "👍 for helpful\n"
                        "👎 for not helpful\n"
                        "Or provide additional feedback!"
                    )
                    await feedback_msg.add_reaction("👍")
                    await feedback_msg.add_reaction("👎")
                except Exception as e:
                    await thread.send(f"Sorry, I encountered an error: {str(e)}")
                    print(f"Error processing question: {e}")

            await bot.process_commands(message)

        return bot
    
    # Main bot loop with automatic restarts
    while True:
        try:
            # Get Discord token
            discord_token = os.environ.get("DISCORD_BOT_TOKEN")
            if not discord_token:
                raise ValueError(
                    "DISCORD_BOT_TOKEN environment variable not set. "
                    "Please ensure the discord-secret-2 secret is properly configured in Modal."
                )
            
            print(f"Discord token found (length: {len(discord_token)})")
            
            client = setup_discord_bot()
            
            print("Starting Discord bot connection...")
            print(f"Bot starting at {datetime.datetime.now()}")
            
            # This will run until the connection is lost or an error occurs
            await client.start(discord_token)

        except Exception as e:
            print(f"Bot encountered an error: {type(e).__name__}: {e}")
            print("Restarting in 30 seconds...")
            await asyncio.sleep(30) 