## Railway deployment notes

### 1. Repo structure

- `main.py`: batch scraper that populates the Chroma DB using Playwright + OpenAI.
- `streamlit_app.py`: Streamlit UI that queries the Chroma DB (RAG chat).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image for Railway (includes Playwright browsers).
- `Procfile`: Declares a `web` process (Streamlit) and a `worker` process (`python main.py`).

### 2. Environment variables

Create these in Railway project settings:

- `OPENAI_API_KEY` (required)
- Optional:
  - `LANGCHAIN_API_KEY`
  - `LANGCHAIN_TRACING_V2=true`

`streamlit_app.py` will automatically mirror `OPENAI_API_KEY` into `CHROMA_OPENAI_API_KEY` if that variable is missing.

### 3. Deploying with Docker (recommended)

1. Push this repo to GitHub.
2. In Railway, create a new project -> "Deploy from GitHub" -> select this repo.
3. Railway will detect the `Dockerfile` and build using the Playwright base image:
   - Installs Python deps from `requirements.txt`.
   - Runs: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`.
4. Once built, Railway will expose the Streamlit UI on the generated URL.

### 4. Running the scraper on Railway

There are two common patterns:

**A. Run manually as a one-off job**

- In Railway, open the project -> "New Deployment" or "Run" (depending on UI):
  - Command: `python main.py`
  - This uses the same Docker image and environment as the web service.

**B. Use the `worker` process type**

- Railway can read the `Procfile`:
  - `web`: Streamlit UI.
  - `worker`: `python main.py` (continuous / batch scraper).
- You can choose whether to enable the `worker` service or run it only on demand.

Because scraping 100+ firms is slow and uses tokens, you’ll typically:

1. Enable `worker` temporarily (or run a one-off job) to populate/update the DB.
2. Stop/disable the worker and keep only `web` running for querying.

### 5. Data persistence

The Chroma DB files live under `chroma_db/` in the container filesystem.

For durable storage across deploys, configure:

- A Railway volume mounted at `/app/chroma_db`, or
- A remote vector store instead of local Chroma (future enhancement).

If you don’t attach a volume, data will be tied to the current container image and can be lost on rebuilds.


