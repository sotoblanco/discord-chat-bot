#!/usr/bin/env python3
"""
Interactive Q&A Script for Workshop Transcripts

This script creates an interactive session where you can continuously ask questions
about workshop transcripts until you decide to exit.

Usage:
    python interactive_qa.py

Then just type your questions and press Enter. Type 'exit', 'quit', or 'q' to stop.
"""

import sys
import os
from src.vector_emb import (
    discover_workshops,
    process_all_workshops,
    answer_question,
    llm_answer_question,
    get_openai_client,
    get_chroma_client,
    get_or_create_collection,
    COLLECTION_NAME,
)
from dotenv import load_dotenv


def setup_environment():
    """Setup environment variables and check requirements"""
    load_dotenv()

    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in a .env file or export it:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        sys.exit(1)


def check_and_populate_database():
    """Check if database is populated, and populate if needed"""
    try:
        client = get_chroma_client()
        collection = get_or_create_collection(client, COLLECTION_NAME)
        count = collection.count()

        if count == 0:
            print(
                "📚 Database is empty. Let me populate it with workshop transcripts..."
            )

            # Discover workshops
            workshops = discover_workshops()
            if not workshops:
                print("❌ No workshop VTT files found in data directory")
                print("Please ensure your VTT files are in the ./data/ directory")
                sys.exit(1)

            print(f"Found {len(workshops)} workshops. Processing...")

            # Process all workshops
            processed = process_all_workshops()

            if processed:
                print(f"✅ Successfully processed {len(processed)} workshops")
                print(f"📊 Database now contains {collection.count()} chunks")
            else:
                print("❌ Failed to process workshops")
                sys.exit(1)
        else:
            print(f"✅ Database ready with {count} chunks")

    except Exception as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)


def answer_with_llm(question):
    """Get an AI-generated answer based on workshop transcripts"""
    try:
        # Get context from vector database
        context, sources, chunks = answer_question(question)

        if not chunks:
            print("❌ No relevant information found to answer your question.")
            print("Try rephrasing your question or asking about different topics.")
            return

        # Generate LLM response
        client = get_openai_client()
        answer, context_info = llm_answer_question(
            client, context, sources, chunks, question
        )

        print("\n" + "=" * 60)
        print("🎯 ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n" + "=" * 60)
        print(
            f"📊 Based on {context_info.get('num_chunks', 0)} chunks from workshops: {', '.join(context_info.get('workshops_used', []))}"
        )
        print("=" * 60)

        if os.environ.get("DEBUG"):
            print("\n🔍 DEBUG INFO:")
            print(f"Context: {context}")
            print(f"Sources: {sources}")
            print(f"Chunks: {chunks}")
            print("=" * 60)

    except Exception as e:
        print(f"❌ Error generating answer: {e}")


def main():
    print("🤖 Interactive Workshop Q&A")
    print("=" * 60)
    print("Ask questions about workshop transcripts!")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("=" * 60)

    # Setup environment
    setup_environment()
    print("✅ Environment setup complete")

    # Check and populate database if needed
    check_and_populate_database()

    print("\n🚀 Ready! Ask me anything about the workshops...")
    print("💡 Example questions:")
    print("  - What is Modal?")
    print("  - How does deployment work?")
    print("  - Explain the chunking strategy")
    print("  - What are the key features discussed?")

    # Interactive loop
    while True:
        try:
            print("\n" + "-" * 60)
            question = input("❓ Your question: ").strip()

            # Check for exit commands
            if question.lower() in ["exit", "quit", "q", ""]:
                print("\n👋 Goodbye!")
                break

            # Skip empty questions
            if not question:
                continue

            print(f"\n🔍 Searching for information about: '{question}'")
            answer_with_llm(question)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 End of input. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
