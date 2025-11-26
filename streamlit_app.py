import os
import subprocess
import sys
import logging
from textwrap import dedent

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.database import VectorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("streamlit_app")


def ensure_required_keys():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY is missing from environment")
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        logger.error("PINECONE_API_KEY is missing from environment")
        raise RuntimeError(
            "PINECONE_API_KEY is missing. Get your API key from https://app.pinecone.io/ "
            "and add it to your environment or .env file."
        )


@st.cache_resource
def get_vector_db():
    ensure_required_keys()
    logger.info("Initializing VectorDB")
    return VectorDB()


@st.cache_resource
def get_llm(model_name: str):
    logger.info(f"Initializing LLM with model: {model_name}")
    return ChatOpenAI(model=model_name, temperature=0)


def retrieve_context(db: VectorDB, question: str, top_k: int):
    logger.debug(f"Retrieving context for question: {question[:50]}... (top_k={top_k})")
    response = db.collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    if not documents:
        logger.warning(f"No documents found for query: {question[:50]}...")
        return ""

    logger.info(f"Retrieved {len(documents)} document(s) for query")
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
    logger.debug(f"Answering question: {question[:50]}... (history length: {len(history)})")
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

    try:
        response = llm.invoke(
            [
                ("system", system_instruction),
                ("human", composed_prompt),
            ]
        )
        logger.info(f"Successfully generated answer for question: {question[:50]}...")
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        raise


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
        logger.info(f"Reloading environment from {env_file}")
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
    import threading
    
    logger.info(f"Starting scraper subprocess with max_targets={max_targets}")
    env = os.environ.copy()
    env["MAX_TARGETS"] = str(max_targets)
    # Ensure unbuffered output for real-time logs
    env["PYTHONUNBUFFERED"] = "1"
    # Force Python to not buffer stdout/stderr
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        # Get the absolute path to main.py
        main_py_path = os.path.join(os.getcwd(), "main.py")
        if not os.path.exists(main_py_path):
            logger.error(f"main.py not found at: {main_py_path}")
            return 1, f"[ERROR] main.py not found at: {main_py_path}", ""
        
        logger.debug(f"Executing: {sys.executable} -u {main_py_path}")
        
        # Use Popen with PIPE to capture output and stream it to terminal
        process = subprocess.Popen(
            [sys.executable, "-u", main_py_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True,
            env=env,
            cwd=os.getcwd(),  # Ensure we're in the right directory
            bufsize=1,  # Line buffered
        )
        
        # Collect output lines for Streamlit display
        output_lines = []
        
        def read_output():
            """Read output from subprocess and stream to terminal while collecting."""
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    output_lines.append(line)
                    # Stream to terminal in real-time
                    print(line, flush=True)
            process.stdout.close()
        
        # Start reading output in a separate thread
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
        # Wait for process with timeout
        try:
            logger.debug("Waiting for scraper process to complete (timeout: 600s)")
            return_code = process.wait(timeout=600)  # 10 minute timeout
            logger.info(f"Scraper process completed with return code: {return_code}")
        except subprocess.TimeoutExpired:
            logger.warning("Scraper process timed out after 10 minutes")
            process.kill()
            process.wait()
            return_code = -1
        
        # Wait for output thread to finish reading
        output_thread.join(timeout=2)
        
        # Combine all output lines
        stdout_text = "\n".join(output_lines)
        
        # If we got no output, add diagnostic information
        if not stdout_text or stdout_text.strip() == "":
            logger.warning("No output captured from scraper process")
            stdout_text = f"[WARNING] No output captured from scraper process.\n"
            stdout_text += f"Process return code: {return_code}\n"
            stdout_text += f"Python executable: {sys.executable}\n"
            stdout_text += f"Working directory: {os.getcwd()}\n"
            stdout_text += "\nPossible issues:\n"
            stdout_text += "- The process may have failed silently\n"
            stdout_text += "- Output buffering may be preventing log capture\n"
            stdout_text += "- Check Railway deployment logs for more details\n"
        
        return return_code, stdout_text.strip(), ""
        
    except FileNotFoundError:
        error_msg = f"Python executable not found: {sys.executable}"
        logger.error(error_msg)
        return 1, f"[ERROR] {error_msg}", ""
    except Exception as e:
        error_msg = f"Failed to run scraper: {str(e)}"
        logger.error(error_msg, exc_info=True)
        import traceback
        traceback_str = traceback.format_exc()
        return 1, f"[ERROR] {error_msg}\n\nTraceback:\n{traceback_str}", ""


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
        logger.info(f"User triggered scraper run with max_targets={scraper_max_targets}")
        # Create a placeholder for status
        status_placeholder = st.empty()
        logs_placeholder = st.empty()
        
        status_placeholder.info(f"🔄 Starting scraper for up to {scraper_max_targets} firm(s)...")
        
        code, out, err = run_scraper_from_ui(scraper_max_targets)

        # Show status prominently
        if code == 0:
            logger.info("Scraper completed successfully")
            status_placeholder.success(f"✅ Scraper completed successfully! (Exit code: {code})")
        elif code == -1:
            logger.warning("Scraper timed out after 10 minutes")
            status_placeholder.error(f"⏱️ Scraper timed out after 10 minutes (Exit code: {code})")
        else:
            logger.error(f"Scraper finished with errors (exit code: {code})")
            status_placeholder.error(f"❌ Scraper finished with errors (Exit code: {code})")
        
        # Display logs in a prominent, scrollable area
        with logs_placeholder.container():
            st.markdown("### 📋 Scraper Logs")
        
        # Combine all output (stderr is redirected to stdout, so err should be empty)
        all_output = []
        if out:
            all_output.append(out)
        if err:
            all_output.append(f"\n=== Additional Errors ===\n{err}")
        
        if all_output:
            full_log = "\n".join(all_output)
            
            # Show summary first
            if full_log.strip():
                # Count lines for summary
                line_count = len(full_log.split('\n'))
                st.info(f"📊 Captured {line_count} lines of output")
            
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
            st.warning("⚠️ No output captured from scraper. This might indicate:")
            st.markdown("""
            - The scraper ran but produced no output
            - There was an issue capturing the subprocess output
            - Check Railway logs for more details
            """)
            
            # Show debug info
            with st.expander("🔍 Debug Information"):
                st.write(f"Exit code: {code}")
                st.write(f"Output length: {len(out) if out else 0} characters")
                st.write(f"Error length: {len(err) if err else 0} characters")

    db = get_vector_db()
    llm = get_llm(model_name)

    for message in st.session_state.history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["assistant"])

    prompt = st.chat_input("Ask about a law firm...")
    if prompt:
        logger.info(f"User question received: {prompt[:100]}...")
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Retrieving context..."):
            context = retrieve_context(db, prompt, top_k)

        with st.chat_message("assistant"):
            answer = answer_question(llm, context, prompt, st.session_state.history)
            st.write(answer)

        st.session_state.history.append({"user": prompt, "assistant": answer})
        st.session_state.last_context = context
        logger.debug(f"Added question to history (total messages: {len(st.session_state.history)})")

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
        logger.info("Loading database entries for view")
        results = db.collection.get(include=["metadatas", "documents"])
        
        if not results.get("ids"):
            logger.info("No entries found in the database")
            st.info("No entries found in the database.")
            return

        entries_count = len(results["ids"])
        logger.info(f"Found {entries_count} entry/entries in the database")
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

