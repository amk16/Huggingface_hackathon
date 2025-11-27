import logging
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)


class JobScraper:
    """Scraper for multiple job platforms to find jobs by company name."""
    
    def __init__(self):
        self.location = "London"
        self.max_pages_per_platform = 3  # Limit pages to avoid excessive requests
        self.request_delay = 2  # Delay between requests to be respectful
        
    def _extract_company_name(self, url: str) -> str:
        """Extract company name from URL."""
        # Remove protocol and www
        name = url.replace("https://", "").replace("http://", "").replace("www.", "")
        # Remove domain extensions
        name = name.split(".")[0]
        # Clean up common patterns
        name = name.replace("-", " ").replace("_", " ")
        return name.strip()
    
    async def scrape_indeed(self, company_name: str) -> List[Dict]:
        """Scrape Indeed for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                for page_num in range(self.max_pages_per_platform):
                    start = page_num * 10
                    search_url = f'https://www.indeed.co.uk/jobs?q=company:"{company_name}"&l={self.location}'
                    if start > 0:
                        search_url += f"&start={start}"
                    
                    try:
                        logger.info(f"Scraping Indeed page {page_num + 1} for {company_name}")
                        await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                        await page.wait_for_timeout(3000)  # Wait for JS to render
                        
                        content = await page.content()
                        soup = BeautifulSoup(content, "html.parser")
                        
                        # Try multiple selectors
                        job_cards = soup.select(".jobsearch-SerpJobCard, [data-tn-component='organicJob']")
                        
                        if not job_cards:
                            logger.debug(f"No job cards found on Indeed page {page_num + 1}")
                            break
                        
                        for card in job_cards:
                            try:
                                title_elem = card.select_one("h2.jobTitle a, h2 a, .jobTitle")
                                title = title_elem.get_text(strip=True) if title_elem else "N/A"
                                
                                company_elem = card.select_one(".companyName, .company")
                                company = company_elem.get_text(strip=True) if company_elem else company_name
                                
                                location_elem = card.select_one(".companyLocation, .location")
                                location = location_elem.get_text(strip=True) if location_elem else self.location
                                
                                summary_elem = card.select_one(".summary, .job-snippet")
                                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                                
                                link_elem = card.select_one("h2.jobTitle a, h2 a")
                                link = link_elem.get("href", "") if link_elem else ""
                                if link and not link.startswith("http"):
                                    link = f"https://www.indeed.co.uk{link}"
                                
                                if title and title != "N/A":
                                    jobs.append({
                                        "title": title,
                                        "company": company,
                                        "location": location,
                                        "summary": summary,
                                        "link": link,
                                        "platform": "Indeed"
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing Indeed job card: {e}")
                                continue
                        
                        # If we got fewer than 10 results, likely no more pages
                        if len(job_cards) < 10:
                            break
                            
                        await asyncio.sleep(self.request_delay)
                    except Exception as e:
                        logger.warning(f"Error scraping Indeed page {page_num + 1}: {e}")
                        break
                
                await browser.close()
        except Exception as e:
            logger.error(f"Error in Indeed scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on Indeed for {company_name}")
        return jobs
    
    async def scrape_reed(self, company_name: str) -> List[Dict]:
        """Scrape Reed for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                for page_num in range(1, self.max_pages_per_platform + 1):
                    search_url = f"https://www.reed.co.uk/jobs?keywords={quote_plus(company_name)}&location={quote_plus(self.location)}&page={page_num}"
                    
                    try:
                        logger.info(f"Scraping Reed page {page_num} for {company_name}")
                        await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                        await page.wait_for_timeout(3000)
                        
                        content = await page.content()
                        soup = BeautifulSoup(content, "html.parser")
                        
                        job_cards = soup.select(".job-result, .job-result-info")
                        
                        if not job_cards:
                            logger.debug(f"No job cards found on Reed page {page_num}")
                            break
                        
                        for card in job_cards:
                            try:
                                title_elem = card.select_one("h2 a, h3 a, .job-result-title a")
                                title = title_elem.get_text(strip=True) if title_elem else "N/A"
                                
                                company_elem = card.select_one(".job-result-company, .company")
                                company = company_elem.get_text(strip=True) if company_elem else company_name
                                
                                location_elem = card.select_one(".job-result-location, .location")
                                location = location_elem.get_text(strip=True) if location_elem else self.location
                                
                                summary_elem = card.select_one(".job-result-description, .description")
                                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                                
                                link_elem = card.select_one("h2 a, h3 a, .job-result-title a")
                                link = link_elem.get("href", "") if link_elem else ""
                                if link and not link.startswith("http"):
                                    link = f"https://www.reed.co.uk{link}"
                                
                                if title and title != "N/A":
                                    jobs.append({
                                        "title": title,
                                        "company": company,
                                        "location": location,
                                        "summary": summary,
                                        "link": link,
                                        "platform": "Reed"
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing Reed job card: {e}")
                                continue
                        
                        if len(job_cards) < 10:
                            break
                            
                        await asyncio.sleep(self.request_delay)
                    except Exception as e:
                        logger.warning(f"Error scraping Reed page {page_num}: {e}")
                        break
                
                await browser.close()
        except Exception as e:
            logger.error(f"Error in Reed scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on Reed for {company_name}")
        return jobs
    
    async def scrape_cv_library(self, company_name: str) -> List[Dict]:
        """Scrape CV-Library for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                for page_num in range(1, self.max_pages_per_platform + 1):
                    search_url = f"https://www.cv-library.co.uk/search?phrases={quote_plus(company_name)}&location={quote_plus(self.location)}&page={page_num}"
                    
                    try:
                        logger.info(f"Scraping CV-Library page {page_num} for {company_name}")
                        await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                        await page.wait_for_timeout(3000)
                        
                        content = await page.content()
                        soup = BeautifulSoup(content, "html.parser")
                        
                        job_cards = soup.select(".search-result, .job-row")
                        
                        if not job_cards:
                            logger.debug(f"No job cards found on CV-Library page {page_num}")
                            break
                        
                        for card in job_cards:
                            try:
                                title_elem = card.select_one("h2 a, h3 a, .job-title a")
                                title = title_elem.get_text(strip=True) if title_elem else "N/A"
                                
                                company_elem = card.select_one(".company-name, .employer")
                                company = company_elem.get_text(strip=True) if company_elem else company_name
                                
                                location_elem = card.select_one(".location, .job-location")
                                location = location_elem.get_text(strip=True) if location_elem else self.location
                                
                                summary_elem = card.select_one(".job-description, .summary")
                                summary = summary_elem.get_text(strip=True) if summary_elem else ""
                                
                                link_elem = card.select_one("h2 a, h3 a, .job-title a")
                                link = link_elem.get("href", "") if link_elem else ""
                                if link and not link.startswith("http"):
                                    link = f"https://www.cv-library.co.uk{link}"
                                
                                if title and title != "N/A":
                                    jobs.append({
                                        "title": title,
                                        "company": company,
                                        "location": location,
                                        "summary": summary,
                                        "link": link,
                                        "platform": "CV-Library"
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing CV-Library job card: {e}")
                                continue
                        
                        if len(job_cards) < 10:
                            break
                            
                        await asyncio.sleep(self.request_delay)
                    except Exception as e:
                        logger.warning(f"Error scraping CV-Library page {page_num}: {e}")
                        break
                
                await browser.close()
        except Exception as e:
            logger.error(f"Error in CV-Library scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on CV-Library for {company_name}")
        return jobs
    
    async def scrape_totally_legal(self, company_name: str) -> List[Dict]:
        """Scrape TotallyLegal for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                search_url = f"https://www.totallylegal.com/jobs/search/?q={quote_plus(company_name)}&location={quote_plus(self.location)}"
                
                try:
                    logger.info(f"Scraping TotallyLegal for {company_name}")
                    await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(3000)
                    
                    content = await page.content()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    job_cards = soup.select(".result, .job-listing")
                    
                    for card in job_cards:
                        try:
                            title_elem = card.select_one("h2 a, h3 a, .job-title a")
                            title = title_elem.get_text(strip=True) if title_elem else "N/A"
                            
                            company_elem = card.select_one(".company, .employer")
                            company = company_elem.get_text(strip=True) if company_elem else company_name
                            
                            location_elem = card.select_one(".location")
                            location = location_elem.get_text(strip=True) if location_elem else self.location
                            
                            summary_elem = card.select_one(".description, .summary")
                            summary = summary_elem.get_text(strip=True) if summary_elem else ""
                            
                            link_elem = card.select_one("h2 a, h3 a, .job-title a")
                            link = link_elem.get("href", "") if link_elem else ""
                            if link and not link.startswith("http"):
                                link = f"https://www.totallylegal.com{link}"
                            
                            if title and title != "N/A":
                                jobs.append({
                                    "title": title,
                                    "company": company,
                                    "location": location,
                                    "summary": summary,
                                    "link": link,
                                    "platform": "TotallyLegal"
                                })
                        except Exception as e:
                            logger.debug(f"Error parsing TotallyLegal job card: {e}")
                            continue
                    
                    await browser.close()
                except Exception as e:
                    logger.warning(f"Error scraping TotallyLegal: {e}")
                    await browser.close()
        except Exception as e:
            logger.error(f"Error in TotallyLegal scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on TotallyLegal for {company_name}")
        return jobs
    
    async def scrape_law_careers(self, company_name: str) -> List[Dict]:
        """Scrape LawCareers.Net for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                search_url = f"https://www.lawcareers.net/Search/Vacancies?q={quote_plus(company_name)}"
                
                try:
                    logger.info(f"Scraping LawCareers.Net for {company_name}")
                    await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(3000)
                    
                    content = await page.content()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # Try common job listing selectors
                    job_cards = soup.select(".vacancy, .job-listing, .result, .job-card")
                    
                    for card in job_cards:
                        try:
                            title_elem = card.select_one("h2 a, h3 a, .title a")
                            title = title_elem.get_text(strip=True) if title_elem else "N/A"
                            
                            company_elem = card.select_one(".company, .employer, .firm")
                            company = company_elem.get_text(strip=True) if company_elem else company_name
                            
                            location_elem = card.select_one(".location")
                            location = location_elem.get_text(strip=True) if location_elem else self.location
                            
                            summary_elem = card.select_one(".description, .summary")
                            summary = summary_elem.get_text(strip=True) if summary_elem else ""
                            
                            link_elem = card.select_one("h2 a, h3 a, .title a")
                            link = link_elem.get("href", "") if link_elem else ""
                            if link and not link.startswith("http"):
                                link = f"https://www.lawcareers.net{link}"
                            
                            if title and title != "N/A":
                                jobs.append({
                                    "title": title,
                                    "company": company,
                                    "location": location,
                                    "summary": summary,
                                    "link": link,
                                    "platform": "LawCareers.Net"
                                })
                        except Exception as e:
                            logger.debug(f"Error parsing LawCareers.Net job card: {e}")
                            continue
                    
                    await browser.close()
                except Exception as e:
                    logger.warning(f"Error scraping LawCareers.Net: {e}")
                    await browser.close()
        except Exception as e:
            logger.error(f"Error in LawCareers.Net scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on LawCareers.Net for {company_name}")
        return jobs
    
    async def scrape_hays(self, company_name: str) -> List[Dict]:
        """Scrape Hays for jobs by company name."""
        jobs = []
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )
                page = await context.new_page()
                
                search_url = f"https://www.hays.co.uk/jobs?q={quote_plus(company_name)}&location={quote_plus(self.location)}"
                
                try:
                    logger.info(f"Scraping Hays for {company_name}")
                    await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(5000)  # Hays may need more time for JS
                    
                    content = await page.content()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # Try common selectors for Hays
                    job_cards = soup.select(".job-card, .job-result, .search-result, .job-listing")
                    
                    for card in job_cards:
                        try:
                            title_elem = card.select_one("h2 a, h3 a, .job-title a, .title a")
                            title = title_elem.get_text(strip=True) if title_elem else "N/A"
                            
                            company_elem = card.select_one(".company, .employer")
                            company = company_elem.get_text(strip=True) if company_elem else company_name
                            
                            location_elem = card.select_one(".location, .job-location")
                            location = location_elem.get_text(strip=True) if location_elem else self.location
                            
                            summary_elem = card.select_one(".description, .summary, .job-description")
                            summary = summary_elem.get_text(strip=True) if summary_elem else ""
                            
                            link_elem = card.select_one("h2 a, h3 a, .job-title a, .title a")
                            link = link_elem.get("href", "") if link_elem else ""
                            if link and not link.startswith("http"):
                                link = f"https://www.hays.co.uk{link}"
                            
                            if title and title != "N/A":
                                jobs.append({
                                    "title": title,
                                    "company": company,
                                    "location": location,
                                    "summary": summary,
                                    "link": link,
                                    "platform": "Hays"
                                })
                        except Exception as e:
                            logger.debug(f"Error parsing Hays job card: {e}")
                            continue
                    
                    await browser.close()
                except Exception as e:
                    logger.warning(f"Error scraping Hays: {e}")
                    await browser.close()
        except Exception as e:
            logger.error(f"Error in Hays scraper for {company_name}: {e}")
        
        logger.info(f"Found {len(jobs)} jobs on Hays for {company_name}")
        return jobs
    
    async def scrape_all_platforms(self, company_name: str) -> List[Dict]:
        """Scrape all platforms for a given company name."""
        logger.info(f"Starting job scraping for company: {company_name}")
        
        # Run all scrapers concurrently
        results = await asyncio.gather(
            self.scrape_indeed(company_name),
            self.scrape_reed(company_name),
            self.scrape_cv_library(company_name),
            self.scrape_totally_legal(company_name),
            self.scrape_law_careers(company_name),
            self.scrape_hays(company_name),
            return_exceptions=True
        )
        
        # Combine all results
        all_jobs = []
        for result in results:
            if isinstance(result, list):
                all_jobs.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Scraper returned exception: {result}")
        
        # Remove duplicates based on title and company
        seen = set()
        unique_jobs = []
        for job in all_jobs:
            key = (job["title"].lower(), job["company"].lower())
            if key not in seen:
                seen.add(key)
                unique_jobs.append(job)
        
        logger.info(f"Total unique jobs found for {company_name}: {len(unique_jobs)}")
        return unique_jobs

