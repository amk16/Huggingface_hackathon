import os
import subprocess
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.database import VectorDB


def ensure_required_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )

    if not os.getenv("CHROMA_OPENAI_API_KEY"):
        os.environ["CHROMA_OPENAI_API_KEY"] = openai_key


@st.cache_resource
def get_vector_db():
    ensure_required_keys()
    return VectorDB()


@st.cache_resource
def get_llm(model_name: str):
    return ChatOpenAI(model=model_name, temperature=0)


def retrieve_context(db: VectorDB, question: str, top_k: int):
    response = db.collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    if not documents:
        return ""

    snippets = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        snippet = dedent(
            f"""
            ### {meta.get('name', f'Result {idx}')}
            Tone: {meta.get('tone', 'n/a')}
            Keywords: {meta.get('keywords', 'n/a')}
            Recent wins: {meta.get('wins', 'n/a')}
            Motto: {meta.get('motto', 'n/a')}

            {doc.strip()}
            """
        ).strip()
        snippets.append(snippet)

    return "\n\n".join(snippets)


def answer_question(llm: ChatOpenAI, context: str, question: str, history: list):
    system_instruction = dedent(
        """
        You are a concise assistant that answers questions about London law firms
        using only the provided context snippets. If the context is empty, say that
        you don't have enough information instead of guessing.
        """
    ).strip()

    conversation_summary = "\n".join(
        f"User: {msg['user']}\nAssistant: {msg['assistant']}"
        for msg in history[-5:]
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
    return response.content.strip()


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""


def render_sidebar():
    st.sidebar.header("Settings")
    env_file = st.sidebar.text_input("Env file path", value=".env")
    load_env_clicked = st.sidebar.button("Reload Env", type="primary")

    if load_env_clicked:
        load_dotenv(env_file)
        st.sidebar.success(f"Environment reloaded from {env_file}")

    top_k = st.sidebar.slider("Results per query", min_value=1, max_value=5, value=3)
    model_name = st.sidebar.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        index=0,
    )
    show_context = st.sidebar.checkbox("Show retrieved context", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scraper control")
    scraper_max_targets = st.sidebar.number_input(
        "Firms per scraper run", min_value=1, max_value=50, value=10, step=1
    )
    run_scraper_clicked = st.sidebar.button("Run scraper now")

    return top_k, model_name, show_context, run_scraper_clicked, int(scraper_max_targets)


def run_scraper_from_ui(max_targets: int):
    """Invoke the scraper (main.py) as a subprocess and capture logs."""
    env = os.environ.copy()
    env["MAX_TARGETS"] = str(max_targets)
    # Ensure unbuffered output for real-time logs
    env["PYTHONUNBUFFERED"] = "1"

    # Railway / Docker image has `python` on PATH
    completed = subprocess.run(
        ["python", "-u", "main.py"],  # -u flag for unbuffered output
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.returncode, completed.stdout, completed.stderr


def render_chat_page():
    st.title("London Law Firm RAG Chat")
    st.caption("Ask natural-language questions grounded in your scraped RAG store.")

    init_session_state()
    top_k, model_name, show_context, run_scraper_clicked, scraper_max_targets = render_sidebar()

    # Load env on first run
    load_dotenv(".env")
    ensure_required_keys()

    # Optional: trigger scraper from the UI
    if run_scraper_clicked:
        with st.spinner(f"Running scraper for up to {scraper_max_targets} firm(s)..."):
            code, out, err = run_scraper_from_ui(scraper_max_targets)

        # Show status prominently
        if code == 0:
            st.success(f"✅ Scraper completed successfully! (Exit code: {code})")
        else:
            st.error(f"❌ Scraper finished with errors (Exit code: {code})")
        
        # Display logs in a prominent, scrollable area
        st.markdown("### 📋 Scraper Logs")
        
        # Combine stdout and stderr for better visibility
        all_output = []
        if out:
            all_output.append("=== STDOUT ===")
            all_output.append(out)
        if err:
            all_output.append("\n=== STDERR ===")
            all_output.append(err)
        
        if all_output:
            full_log = "\n".join(all_output)
            # Use st.code for better formatting and copy capability
            st.code(full_log, language="text")
            
            # Also show in expandable section for detailed view
            with st.expander("📊 Detailed Logs (scrollable)", expanded=True):
                st.text_area(
                    "Full scraper output",
                    value=full_log,
                    height=400,
                    disabled=True,
                    label_visibility="collapsed"
                )
        else:
            st.warning("No output captured from scraper.")

    db = get_vector_db()
    llm = get_llm(model_name)

    for message in st.session_state.history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["assistant"])

    prompt = st.chat_input("Ask about a law firm...")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Retrieving context..."):
            context = retrieve_context(db, prompt, top_k)

        with st.chat_message("assistant"):
            answer = answer_question(llm, context, prompt, st.session_state.history)
            st.write(answer)

        st.session_state.history.append({"user": prompt, "assistant": answer})
        st.session_state.last_context = context

    if show_context and st.session_state.last_context:
        with st.expander("Most recent retrieved context"):
            st.markdown(st.session_state.last_context)


def render_database_view():
    st.title("Database Entries")
    st.caption("View all law firm entries stored in the database.")

    # Load env on first run
    load_dotenv(".env")
    ensure_required_keys()

    db = get_vector_db()

    # Get all entries from the database
    with st.spinner("Loading database entries..."):
        results = db.collection.get(include=["metadatas", "documents", "ids"])
        
        if not results.get("ids"):
            st.info("No entries found in the database.")
            return

        entries_count = len(results["ids"])
        st.success(f"Found {entries_count} entry/entries in the database.")
        
        # Add a search/filter option
        search_term = st.text_input("🔍 Search by firm name", placeholder="Type to filter entries...")
        
        # Display entries
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        
        # Filter entries if search term is provided
        filtered_entries = []
        for idx, (doc, meta, entry_id) in enumerate(zip(documents, metadatas, ids)):
            firm_name = meta.get('name', 'Unknown Firm')
            if not search_term or search_term.lower() in firm_name.lower():
                filtered_entries.append((idx, doc, meta, entry_id))
        
        if not filtered_entries:
            st.warning(f"No entries found matching '{search_term}'.")
            return
        
        st.write(f"Showing {len(filtered_entries)} of {entries_count} entries.")
        
        # Display each entry in an expandable container
        for idx, doc, meta, entry_id in filtered_entries:
            firm_name = meta.get('name', 'Unknown Firm')
            with st.expander(f"🏛️ **{firm_name}**", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Tone:** {meta.get('tone', 'n/a')}")
                    st.markdown(f"**Keywords:** {meta.get('keywords', 'n/a')}")
                
                with col2:
                    st.markdown(f"**Recent Wins:** {meta.get('wins', 'n/a')}")
                    st.markdown(f"**Motto:** {meta.get('motto', 'n/a')}")
                
                st.markdown("---")
                st.markdown("**Full Content:**")
                st.markdown(doc.strip() if doc else "No content available.")


def main():
    st.set_page_config(page_title="Law Firm RAG Chat", layout="wide")
    
    # Create navigation with two pages
    page = st.navigation([
        st.Page(render_chat_page, title="Chat", icon="💬"),
        st.Page(render_database_view, title="Database View", icon="📊"),
    ])
    
    page.run()


if __name__ == "__main__":
    main()

