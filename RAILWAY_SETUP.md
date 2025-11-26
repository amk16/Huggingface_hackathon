## Railway deployment notes

### 1. Repo structure

- `main.py`: batch scraper that populates the Pinecone vector database using Playwright + OpenAI.
- `streamlit_app.py`: Streamlit UI that queries the Pinecone database (RAG chat).
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image for Railway (includes Playwright browsers).
- `Procfile`: Declares a `web` process (Streamlit) and a `worker` process (`python main.py`).

### 2. Environment variables

Create these in Railway project settings:

- `OPENAI_API_KEY` (required) - Get from https://platform.openai.com/api-keys
- `PINECONE_API_KEY` (required) - Get from https://app.pinecone.io/
- Optional:
  - `PINECONE_INDEX_NAME` - Defaults to "london-law-firms" if not set
  - `LANGCHAIN_API_KEY`
  - `LANGCHAIN_TRACING_V2=true`

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

The vector database is now stored in **Pinecone** (cloud-hosted), so data persists automatically across deploys and container rebuilds. No volume configuration is needed.

**Setting up Pinecone:**

1. Sign up for a free account at https://app.pinecone.io/
2. Install the Pinecone CLI (if not already installed):
   ```bash
   # macOS
   brew tap pinecone-io/tap && brew install pinecone-io/tap/pinecone
   
   # Other platforms: Download from https://github.com/pinecone-io/cli/releases
   ```
3. Authenticate with Pinecone:
   ```bash
   pc login
   # Or set API key: export PINECONE_API_KEY="your-api-key"
   ```
4. Create the index using the CLI:
   ```bash
   pc index create --name london-law-firms --dimension 1536 --metric cosine \
     --cloud aws --region us-east-1
   ```
   - Name: `london-law-firms` (or set `PINECONE_INDEX_NAME` env var)
   - Dimension: `1536` (for text-embedding-3-small)
   - Metric: `cosine`
   - Cloud: `aws` (or `gcp`, `azure`)
   - Region: `us-east-1` (choose based on your location)
5. Copy your API key and add it as `PINECONE_API_KEY` in Railway

**Note:** Index creation must be done via CLI before running the scraper. The app will check if the index exists and raise an error if it doesn't.


