import os
import sys
import asyncio
import logging
import pandas as pd
from dotenv import load_dotenv
from src.scraper import LawFirmScraper
from src.processor import DataProcessor
from src.database import VectorDB


# Configure logging early so every module shares it
# Force logging to stdout for better capture in subprocess
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,  # Explicitly output to stdout
    force=True,  # Override any existing configuration
)
logger = logging.getLogger("law_firm_rag")

# Load environment variables from .env
load_dotenv()


def load_targets(csv_path: str = "targets.csv"):
    """Attempt to load target URLs from CSV, fall back to demo list."""
    try:
        df = pd.read_csv(csv_path)
        return df['url'].dropna().tolist()
    except Exception as read_error:
        logger.warning("Unable to load %s (%s). Falling back to demo targets.", csv_path, read_error)
        return [
            "https://www.mishcon.com",
            "https://www.kingsleynapley.co.uk",
        ]


async def process_firm(scraper, processor, db, url: str, idx: int, total: int):
    logger.info("(%d/%d) Processing %s", idx, total, url)
    data = await scraper.get_page_content(url)
    if not data.get("raw_text"):
        logger.warning("(%d/%d) Skipping %s - no content found", idx, total, url)
        return

    structured_data = processor.extract_intelligence(data["raw_text"])
    if structured_data:
        db.add_firm(structured_data)
        logger.info(
            "(%d/%d) Saved %s | keywords=%s",
            idx,
            total,
            structured_data.get("firm_name") or url,
            structured_data.get("hiring_keywords"),
        )
    else:
        logger.error("(%d/%d) Failed to extract structured data for %s", idx, total, url)


async def main(max_targets: int = 10):
    urls = load_targets()
    if max_targets and len(urls) > max_targets:
        logger.info("Capping run to first %d of %d targets.", max_targets, len(urls))
        urls = urls[:max_targets]

    total = len(urls)
    if not total:
        logger.error("No target URLs provided. Aborting.")
        return

    logger.info("Loaded %d target(s). Beginning scrape.", total)

    scraper = LawFirmScraper()
    processor = DataProcessor()
    db = VectorDB()

    for idx, url in enumerate(urls, start=1):
        await process_firm(scraper, processor, db, url, idx, total)

    logger.info("Completed scraping %d target(s).", total)


if __name__ == "__main__":
    # Print immediately to verify script is running
    print("=" * 60, flush=True)
    print("SCRAPER STARTED", flush=True)
    print("=" * 60, flush=True)
    
    max_targets_env = os.getenv("MAX_TARGETS")
    try:
        max_targets = int(max_targets_env) if max_targets_env else 10
    except (TypeError, ValueError):
        max_targets = 10
    
    print(f"Max targets: {max_targets}", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print("-" * 60, flush=True)

    asyncio.run(main(max_targets=max_targets))
    
    print("=" * 60, flush=True)
    print("SCRAPER COMPLETED", flush=True)
    print("=" * 60, flush=True)
