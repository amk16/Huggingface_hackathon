import argparse
import os
from textwrap import dedent

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.database import VectorDB


def ensure_required_keys():
    """Guarantee both ChatOpenAI and Chroma embedding helpers have API keys."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or add it to your .env file."
        )

    if not os.getenv("CHROMA_OPENAI_API_KEY"):
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key


def build_context_snippets(results):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    snippets = []
    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        name = meta.get("name", f"Result {idx}")
        snippet = dedent(
            f"""
            ### {name}
            Tone: {meta.get('tone', 'n/a')}
            Keywords: {meta.get('keywords', 'n/a')}
            Recent wins: {meta.get('wins', 'n/a')}
            Motto: {meta.get('motto', 'n/a')}

            {doc.strip()}
            """
        ).strip()
        snippets.append(snippet)

    return "\n\n".join(snippets)


def retrieve_context(db: VectorDB, question: str, top_k: int) -> str:
    response = db.collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    if not response.get("documents") or not response["documents"][0]:
        return ""

    return build_context_snippets(response)


def chat_loop(db: VectorDB, top_k: int, model: str):
    llm = ChatOpenAI(model=model, temperature=0)
    history = []

    print("Chat interface ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        context = retrieve_context(db, question, top_k)

        system_instruction = dedent(
            """
            You are a concise assistant that answers questions about London law firms
            using the provided context. If the context is empty, say you don't have
            enough information instead of guessing.
            """
        ).strip()

        conversation_summary = "\n".join(
            f"User: {q}\nAssistant: {a}" for q, a in history[-5:]
        )

        composed_prompt = dedent(
            f"""
            Conversation so far:
            {conversation_summary or 'None yet.'}

            Context documents:
            {context or '[no relevant context found]'}

            User question:
            {question}
            """
        ).strip()

        response = llm.invoke(
            [
                ("system", system_instruction),
                ("human", composed_prompt),
            ]
        )

        answer = response.content.strip()
        history.append((question, answer))

        print(f"Assistant: {answer}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat interface powered by the stored law-firm RAG data."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the environment file containing API keys (default: .env).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of context documents to retrieve per question (default: 3).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model to use (default: gpt-4o-mini).",
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)
    ensure_required_keys()

    db = VectorDB()
    chat_loop(db, args.top_k, args.model)


if __name__ == "__main__":
    main()

