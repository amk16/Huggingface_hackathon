import argparse
import os
from textwrap import indent

from dotenv import load_dotenv

from src.database import VectorDB


def ensure_chroma_key():
    """Chroma's OpenAI embedding helper expects CHROMA_OPENAI_API_KEY."""
    if os.getenv("CHROMA_OPENAI_API_KEY"):
        return
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "Neither CHROMA_OPENAI_API_KEY nor OPENAI_API_KEY is set. "
            "Export one of them before running this command."
        )
    os.environ["CHROMA_OPENAI_API_KEY"] = openai_key


def pretty_print_documents(documents, metadatas):
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        heading = f"[{idx}] {meta.get('name', 'Unknown Firm')}"
        print(heading)
        print("-" * len(heading))

        summary_lines = [
            f"Tone: {meta.get('tone', 'n/a')}",
            f"Keywords: {meta.get('keywords', 'n/a')}",
            f"Recent wins: {meta.get('wins', 'n/a')}",
            f"Motto: {meta.get('motto', 'n/a')}",
            "",
            doc.strip(),
            "",
        ]
        print(indent("\n".join(summary_lines), "  "))


def list_all_entries(db: VectorDB):
    results = db.collection.get(include=["metadatas", "documents"])
    if not results.get("ids"):
        print("No entries found in the collection.")
        return
    pretty_print_documents(results["documents"], results["metadatas"])


def run_query(db: VectorDB, query_text: str, top_k: int):
    response = db.collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    if not documents:
        print("No matching entries found.")
        return

    for idx, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances), start=1):
        heading = (
            f"[{idx}] {meta.get('name', 'Unknown Firm')} "
            f"(similarity score: {1 - distance:.3f})"
        )
        print(heading)
        print("-" * len(heading))
        print(indent(doc.strip(), "  "))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or query the saved law firm embeddings."
    )
    parser.add_argument(
        "--query",
        help="Optional natural-language query. If omitted, the script prints every stored entry.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to return when --query is provided (default: 3).",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file containing keys (default: .env).",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    ensure_chroma_key()

    db = VectorDB()

    if args.query:
        run_query(db, args.query, args.top_k)
    else:
        list_all_entries(db)


if __name__ == "__main__":
    main()

