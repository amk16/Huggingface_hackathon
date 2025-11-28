import os
import sys
import asyncio
import logging
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from src.scraper import LawFirmScraper
from src.processor import DataProcessor
from src.database import VectorDB
from urllib.parse import urlparse


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

# Progress tracking file
PROGRESS_FILE = "scraper_progress.json"
TIMEOUT_SECONDS = 3600  # 1 hour timeout


def load_progress():
    """Load progress from JSON file if it exists."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                logger.info(f"Loaded progress: {len(progress.get('processed', []))} processed, {len(progress.get('all_urls', []))} total URLs")
                return progress
        except Exception as e:
            logger.warning(f"Failed to load progress file: {e}. Starting fresh.")
    return {
        "all_urls": [],
        "processed": [],
        "failed": [],
        "start_time": None
    }


def save_progress(progress):
    """Save progress to JSON file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


def load_targets(csv_path: str = "targets.csv"):
    """Attempt to load target URLs from CSV, fall back to demo list."""
    try:
        df = pd.read_csv(csv_path)
        urls = df['url'].dropna().tolist()
        logger.info(f"Loaded {len(urls)} URLs from {csv_path}")
        return urls
    except Exception as read_error:
        logger.warning("Unable to load %s (%s). Falling back to demo targets.", csv_path, read_error)
        return [
            "https://www.mishcon.com",
            "https://www.kingsleynapley.co.uk",
        ]


def extract_company_name_from_url(url: str) -> str:
    """Extract company name from URL for job searching."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    # Get the first part of the domain (before first dot)
    company_name = domain.split(".")[0]
    # Clean up common patterns
    company_name = company_name.replace("-", " ").replace("_", " ")
    # Capitalize words
    company_name = " ".join(word.capitalize() for word in company_name.split())
    return company_name


async def process_firm(scraper, processor, db, url: str, idx: int, total: int, progress):
    """Process a single firm and save progress after each success."""
    logger.info("(%d/%d) Processing %s", idx, total, url)
    try:
        data = await scraper.get_page_content(url)
        if not data.get("raw_text"):
            logger.warning("(%d/%d) Skipping %s - no content found", idx, total, url)
            progress["failed"].append({"url": url, "reason": "no_content"})
            save_progress(progress)
            return False

        structured_data = processor.extract_intelligence(data["raw_text"])
        if not structured_data:
            logger.error("(%d/%d) Failed to extract structured data for %s", idx, total, url)
            progress["failed"].append({"url": url, "reason": "extraction_failed"})
            save_progress(progress)
            return False

        firm_name = structured_data.get("firm_name") or extract_company_name_from_url(url)
        
        # Save firm data to Pinecone immediately
        db.add_firm(structured_data)
        logger.info(
            "(%d/%d) Saved %s to Pinecone | keywords=%s",
            idx,
            total,
            firm_name,
            structured_data.get("hiring_keywords"),
        )
        
        # Extract insights from the firm's own career pages instead of job boards
        try:
            career_insights = processor.extract_career_insights(
                data.get("career_sections", []),
                data.get("raw_text", "")
            )
            if career_insights:
                db.add_career_insights(firm_name, career_insights.model_dump())
                logger.info(
                    "(%d/%d) Saved career page insights for %s (openings=%d)",
                    idx,
                    total,
                    firm_name,
                    len(career_insights.current_openings),
                )
            else:
                logger.info("(%d/%d) No dedicated career insights detected for %s", idx, total, firm_name)
        except Exception as career_error:
            logger.warning(
                "(%d/%d) Error extracting career insights for %s: %s",
                idx,
                total,
                firm_name,
                str(career_error)
            )
        
        # Mark as processed and save progress
        if url not in progress["processed"]:
            progress["processed"].append(url)
            # Remove from failed if it was there
            progress["failed"] = [f for f in progress["failed"] if f.get("url") != url]
            save_progress(progress)
        return True
    except Exception as e:
        logger.error("(%d/%d) Error processing %s: %s", idx, total, url, str(e))
        progress["failed"].append({"url": url, "reason": str(e)})
        save_progress(progress)
        return False


def check_timeout(start_time):
    """Check if timeout has been reached."""
    if start_time is None:
        return False
    elapsed = time.time() - start_time
    return elapsed >= TIMEOUT_SECONDS


async def main(max_targets: int = None):
    """Main scraping function with timeout and progress tracking."""
    # Load or initialize progress
    progress = load_progress()
    
    # Load all URLs from CSV
    all_urls = load_targets()
    
    # If progress exists and has all_urls, use those (they may be from a previous run)
    # Otherwise, use the newly loaded URLs
    if progress.get("all_urls") and len(progress["all_urls"]) > 0:
        # Verify the URLs still match (CSV might have changed)
        if set(progress["all_urls"]) == set(all_urls):
            logger.info("Using URLs from previous progress file")
            all_urls = progress["all_urls"]
        else:
            logger.info("CSV has changed, merging with existing progress")
            # Merge: keep processed URLs, add new ones
            processed_set = set(progress["processed"])
            all_urls_set = set(all_urls)
            # Keep URLs that are either in CSV or were processed
            all_urls = list(all_urls_set | set(progress["all_urls"]))
            progress["all_urls"] = all_urls
            save_progress(progress)
    else:
        # First run, save all URLs to progress
        progress["all_urls"] = all_urls
        save_progress(progress)
    
    # Apply max_targets limit only on first run (no progress file or no processed URLs)
    # When resuming, always process all remaining URLs
    is_resuming = len(progress.get("processed", [])) > 0
    if max_targets and not is_resuming:
        # First run: limit to max_targets if specified
        logger.info("First run: Capping to first %d of %d targets.", max_targets, len(all_urls))
        all_urls = all_urls[:max_targets]
        progress["all_urls"] = all_urls
        save_progress(progress)
    elif is_resuming:
        logger.info("Resuming: Will process all remaining URLs (ignoring max_targets limit)")
    
    # Get URLs that haven't been processed yet
    processed_set = set(progress["processed"])
    remaining_urls = [url for url in all_urls if url not in processed_set]
    
    total_urls = len(all_urls)
    remaining_count = len(remaining_urls)
    processed_count = len(progress["processed"])
    
    if remaining_count == 0:
        logger.info("All %d URLs have already been processed. Nothing to do.", total_urls)
        return
    
    logger.info("Progress: %d/%d URLs processed. %d remaining to process.", 
                processed_count, total_urls, remaining_count)
    
    # Initialize start time if not set
    if progress.get("start_time") is None:
        progress["start_time"] = time.time()
        save_progress(progress)
    
    start_time = progress.get("start_time", time.time())
    
    # Initialize scraper components
    scraper = LawFirmScraper()
    processor = DataProcessor()
    db = VectorDB()
    
    # Process remaining URLs
    for idx, url in enumerate(remaining_urls, start=1):
        # Check timeout before processing next URL
        if check_timeout(start_time):
            elapsed = time.time() - start_time
            logger.warning(
                "Timeout reached (%d seconds). Saving progress. %d URLs remaining.",
                elapsed, remaining_count - idx + 1
            )
            # Reset start_time for next run
            progress["start_time"] = None
            save_progress(progress)
            logger.info("Progress saved. Please restart the scraper to continue.")
            return
        
        # Process the URL (saves progress internally)
        await process_firm(scraper, processor, db, url, processed_count + idx, total_urls, progress)
        
        # Update remaining count
        remaining_count = remaining_count - 1
    
    # All URLs processed
    logger.info("Completed processing all %d URLs.", total_urls)
    # Clean up progress file on completion
    if os.path.exists(PROGRESS_FILE):
        logger.info("Moving progress file to completed state")
        completed_file = PROGRESS_FILE.replace(".json", "_completed.json")
        try:
            os.rename(PROGRESS_FILE, completed_file)
            logger.info(f"Progress saved to {completed_file}")
        except Exception as e:
            logger.warning(f"Could not rename progress file: {e}")


if __name__ == "__main__":
    # Print immediately to verify script is running
    print("=" * 60, flush=True)
    print("SCRAPER STARTED", flush=True)
    print("=" * 60, flush=True)
    
    max_targets_env = os.getenv("MAX_TARGETS")
    try:
        max_targets = int(max_targets_env) if max_targets_env else None
    except (TypeError, ValueError):
        max_targets = None
    
    print(f"Max targets: {max_targets if max_targets else 'All URLs from CSV'}", flush=True)
    print(f"Timeout: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS / 60:.1f} minutes)", flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print("-" * 60, flush=True)

    try:
        asyncio.run(main(max_targets=max_targets))
        print("=" * 60, flush=True)
        print("SCRAPER COMPLETED", flush=True)
        print("=" * 60, flush=True)
    except KeyboardInterrupt:
        logger.info("Scraper interrupted by user. Progress has been saved.")
        print("=" * 60, flush=True)
        print("SCRAPER INTERRUPTED (Progress saved)", flush=True)
        print("=" * 60, flush=True)
    except Exception as e:
        logger.error(f"Scraper failed with error: {e}", exc_info=True)
        print("=" * 60, flush=True)
        print(f"SCRAPER FAILED: {e}", flush=True)
        print("=" * 60, flush=True)
        sys.exit(1)
