import logging
from playwright.async_api import async_playwright
import html2text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


logger = logging.getLogger(__name__)


class LawFirmScraper:
    def __init__(self, section_keywords=None, static_paths=None):
        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.section_keywords = section_keywords or [
            "career",
            "about",
            "people",
            "team",
            "our people",
            "news",
            "insight",
            "join",
            "culture",
            "life",
        ]
        self.static_paths = static_paths or [
            "/careers",
            "/about-us",
            "/our-people",
            "/news",
            "/team",
            "/people",
            "/insights",
            "/join-us",
        ]

    async def get_page_content(self, base_url, additional_paths=None):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            page = await context.new_page()

            scraped_data = {"url": base_url, "raw_text": ""}

            try:
                # 1. Visit Homepage
                logger.info("Visiting %s", base_url)
                await page.goto(base_url, timeout=30000)
                await page.wait_for_load_state("domcontentloaded")

                # Extract Homepage Text (Good for Motto/Sector)
                content = await page.content()
                logger.debug("Homepage content length for %s: %d chars", base_url, len(content))
                scraped_data["raw_text"] += self.clean_html(content)

                discovered_links = self._discover_section_links(base_url, content)
                sitemap_links = await self._discover_from_sitemaps(page, base_url)
                static_links = self._build_static_links(base_url)
                extra_links = self._build_extra_links(base_url, additional_paths)

                candidate_links = (
                    discovered_links | sitemap_links | static_links | extra_links
                )
                
                # Limit to maximum 5 links per URL
                total_candidates = len(candidate_links)
                candidate_links = list(candidate_links)[:5]
                
                logger.info(
                    "Found %d candidate section links for %s, processing %d (max 5). Discovery breakdown: DOM=%d, sitemap=%d, static=%d, extra=%d",
                    total_candidates,
                    base_url,
                    len(candidate_links),
                    len(discovered_links),
                    len(sitemap_links),
                    len(static_links),
                    len(extra_links),
                )

                for link in candidate_links:
                    try:
                        logger.info("Visiting section link %s", link)
                        await page.goto(link, timeout=10000)
                        content = await page.content()
                        logger.debug("Section content length for %s: %d chars", link, len(content))
                        scraped_data["raw_text"] += f"\n--- SOURCE: {link} ---\n"
                        scraped_data["raw_text"] += self.clean_html(content)
                    except Exception as section_error:
                        logger.warning("Skipping %s due to %s", link, section_error)
                        continue

            except Exception as e:
                logger.error("Error scraping %s: %s", base_url, e)

            await browser.close()
            return scraped_data

    def clean_html(self, html_content):
        # Convert HTML to Markdown-like text to save tokens
        return self.converter.handle(html_content)

    def _discover_section_links(self, base_url, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        links = set()
        for anchor in soup.find_all("a", href=True):
            text = (anchor.get_text() or "").lower()
            href = anchor["href"]
            lower_href = href.lower()
            if any(keyword in text or keyword in lower_href for keyword in self.section_keywords):
                normalized = self._normalize_url(base_url, href)
                if normalized:
                    links.add(normalized)
        logger.debug("DOM discovery found %d candidate links on %s", len(links), base_url)
        return links

    async def _discover_from_sitemaps(self, page, base_url):
        candidate_links = set()
        sitemap_paths = ["/sitemap.xml", "/sitemap_index.xml"]

        for path in sitemap_paths:
            sitemap_url = urljoin(base_url, path)
            try:
                await page.goto(sitemap_url, timeout=8000)
                content = await page.content()
                candidate_links |= self._parse_sitemap_for_keywords(base_url, content)
            except Exception as sitemap_error:
                logger.debug("Skipping sitemap %s due to %s", sitemap_url, sitemap_error)
                continue

        return candidate_links

    def _parse_sitemap_for_keywords(self, base_url, xml_content):
        soup = BeautifulSoup(xml_content, "xml")
        links = set()
        for loc_tag in soup.find_all("loc"):
            href = loc_tag.get_text(strip=True)
            if not href:
                continue
            lower_href = href.lower()
            if any(keyword in lower_href for keyword in self.section_keywords):
                normalized = self._normalize_url(base_url, href)
                if normalized:
                    links.add(normalized)
        logger.debug("Sitemap discovery found %d candidate links for %s", len(links), base_url)
        return links

    def _build_static_links(self, base_url):
        links = set()
        for path in self.static_paths:
            normalized = self._normalize_url(base_url, path)
            if normalized:
                links.add(normalized)
        logger.debug("Static paths contributed %d links for %s", len(links), base_url)
        return links

    def _build_extra_links(self, base_url, additional_paths):
        links = set()
        if not additional_paths:
            return links
        for path in additional_paths:
            normalized = self._normalize_url(base_url, path)
            if normalized:
                links.add(normalized)
        logger.debug("Extra paths contributed %d links for %s", len(links), base_url)
        return links

    def _normalize_url(self, base_url, link):
        absolute = urljoin(base_url, link)
        base_netloc = urlparse(base_url).netloc
        absolute_netloc = urlparse(absolute).netloc
        if base_netloc != absolute_netloc:
            return None
        return absolute.rstrip("/")
