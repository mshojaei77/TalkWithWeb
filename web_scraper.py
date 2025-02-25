import aiohttp
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import random
import json
import re
import time
import logging
from urllib.parse import urljoin, urlparse, urlunparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('web_scraper')

async def get_contents(url: str) -> str:
    """
    Fetch and parse webpage contents quickly.
    Returns the text content of the webpage.
    """
    start_time = time.time()
    logger.info(f"Starting request to: {url}")
    
    async with aiohttp.ClientSession() as session:
        try:
            logger.debug(f"Sending GET request to {url}")
            async with session.get(url) as response:
                status_code = response.status
                logger.info(f"Received response from {url} with status code: {status_code}")
                
                if response.status == 403:
                    logger.warning(f"⌛ Detected Bot Protection for {url}. Falling back to Playwright.")
                    return await fallback_playwright(url)  # Fallback to Playwright
                elif response.status == 429:
                    delay = random.uniform(1, 3)
                    logger.warning(f"⌛ Too many requests for {url}. Retrying after {delay:.2f}s delay.")
                    await asyncio.sleep(delay)  # Wait before retrying
                    return await get_contents(url)  # Retry fetching the content
                elif response.status == 503:
                    delay = random.uniform(1, 3)
                    logger.warning(f"⌛ Service unavailable for {url}. Retrying after {delay:.2f}s delay.")
                    await asyncio.sleep(delay)  # Wait before retrying
                    return await get_contents(url)  # Retry fetching the content
                elif response.status == 504:
                    delay = random.uniform(1, 3)
                    logger.warning(f"⌛ Gateway timeout for {url}. Retrying after {delay:.2f}s delay.")
                    await asyncio.sleep(delay)  # Wait before retrying
                    return await get_contents(url)  # Retry fetching the content
                elif response.status == 502:
                    delay = random.uniform(1, 3)
                    logger.warning(f"⌛ Bad gateway for {url}. Retrying after {delay:.2f}s delay.")
                    await asyncio.sleep(delay)  # Wait before retrying
                    return await get_contents(url)  # Retry fetching the content
                elif response.status == 404:
                    logger.error(f"⌛ Page not found for {url}. Returning empty string.")
                    return ""  # Return an empty string for a 404 error

                logger.debug(f"Parsing HTML content from {url}")
                # Directly parse the response text without storing it in a variable
                soup = BeautifulSoup(await response.text(), 'html.parser')  # Use 'html.parser' for faster parsing
                
                # Remove unwanted elements using a single line with filter
                logger.debug(f"Removing unwanted elements from {url} content")
                list(map(lambda tag: tag.decompose(), soup(['script', 'style', 'nav', 'header', 'footer', 'ads'])))
                
                # Get main content with optimized extraction
                logger.debug(f"Extracting main content from {url}")
                content = ' '.join(soup.stripped_strings)  # More efficient content extraction
                
                elapsed_time = time.time() - start_time
                content_length = len(content)
                logger.info(f"Successfully scraped {url} - {content_length} chars in {elapsed_time:.2f}s")
                return content  # Return only the content
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            logger.warning(f"⌛ Falling back to Playwright for {url} due to error")
            return await fallback_playwright(url)  # Fallback to Playwright

async def fallback_playwright(url: str) -> str:
    """
    Fallback function to scrape webpage contents using Playwright.
    Returns the content of the webpage as a string.
    """
    start_time = time.time()
    logger.info(f"Starting Playwright fallback for: {url}")
    
    async with async_playwright() as p:
        logger.debug(f"Launching headless Chromium browser for {url}")
        browser = await p.chromium.launch(headless=True)  # Launch headless browser
        page = await browser.new_page()  # Create a new page in the browser
        try:
            logger.debug(f"Navigating to {url} with Playwright")
            await page.goto(url, wait_until='domcontentloaded')  # Navigate to the URL
            logger.debug(f"Getting page content for {url}")
            content = await page.content()  # Get the full page content
            
            # Check for "Just a moment..." in the content
            if "Just a moment..." in content:
                logger.warning(f"⌛ Detected Cloudflare protection for {url}. Using visible browser.")
                return await scrape_with_head(url)  # Call the handling function
            
            content_length = len(content)
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully scraped {url} with Playwright - {content_length} chars in {elapsed_time:.2f}s")
            return content  # Return only the content as a string
        except Exception as e:
            logger.error(f"Error using Playwright for {url}: {str(e)}")
            return ""
        finally:
            logger.debug(f"Closing Playwright browser for {url}")
            await browser.close()  # Ensure the browser is closed

async def scrape_with_head(url: str) -> str:
    """
    Handle the 'Just a moment...' security check during web scraping.
    This function implements strategies to bypass security screens.
    """
    start_time = time.time()
    logger.info(f"Starting visible browser scraping for: {url}")
    
    async with async_playwright() as p:
        logger.debug(f"Launching visible Chromium browser for {url}")
        browser = await p.chromium.launch(headless=False)  # Launch visible browser
        page = await browser.new_page()  # Create a new page in the browser
        try:
            logger.debug(f"Navigating to {url} with visible browser")
            await page.goto(url, wait_until='domcontentloaded')  # Navigate to the URL
            logger.debug(f"Getting page content for {url}")
            content = await page.content()  # Get the full page content
            
            content_length = len(content)
            elapsed_time = time.time() - start_time
            logger.info(f"Successfully scraped {url} with visible browser - {content_length} chars in {elapsed_time:.2f}s")
            return content  # Return only the content as a string
        except Exception as e:
            logger.error(f"Error using visible browser for {url}: {str(e)}")
            return ""
        finally:
            logger.debug(f"Closing visible browser for {url}")
            await browser.close()  # Ensure the browser is closed

def get_contents_sync(url: str) -> str:
    """
    Synchronous wrapper for the asynchronous get_contents function.
    """
    logger.info(f"Starting synchronous request to: {url}")
    result = asyncio.run(get_contents(url))
    logger.info(f"Completed synchronous request to: {url}")
    return result

def clean_content(html_content: str) -> str:
    """
    Highly optimized content cleaning function to extract the main article text.
    This function focuses on delivering the core content of the article efficiently.
    """
    start_time = time.time()
    content_length = len(html_content)
    logger.info(f"Starting content cleaning - Input size: {content_length} chars")
    
    from bs4 import BeautifulSoup
    import re

    # Parse HTML with a robust parser for better handling of malformed HTML
    logger.debug("Parsing HTML with lxml parser")
    soup = BeautifulSoup(html_content, 'lxml')

    # --- Step 1: Remove Unwanted Tags ---
    logger.debug("Removing unwanted HTML tags")
    # These tags are typically containers for non-article content or elements that clutter the main text.
    unwanted_tags = [
        'script', 'style', 'noscript', 'nav', 'header', 'footer', 'aside',
        'form', 'iframe', 'figure', 'figcaption', 'button', 'meta', 'link',
        'svg', 'canvas', 'audio', 'video', 'embed', 'amp-ad', 'amp-embed',
        'select', 'label', 'time',  # Added form elements and time tags
        'ul', 'ol', 'li',  # Remove lists which often contain related links
        'hr', 'br',  # Remove line breaks and horizontal rules
        'picture', 'source', 'img'  # Remove images and responsive image tags
    ]
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()

    # --- Step 2: Extract Main Content ---
    logger.debug("Extracting main content based on HTML structure")
    # These classes are identified as wrappers or containers for elements that are not part of the main article.
    # This list is expanded based on common patterns in news websites and the provided HTML context.
    unwanted_classes = [
        'ad-', 'promo', 'sponsored', 'hidden', 'cookie', 'consent',  # Advertising
        'sidebar', 'related-', 'popular-', 'trending-', 'recommended-',  # Side content
        'social-', 'share-', 'comment', 'author', 'bio', 'meta',  # Social and meta
        'header', 'footer', 'menu', 'nav', 'breadcrumb', 'pagination',  # Navigation
        'widget', 'tooltip', 'modal', 'overlay', 'loader',  # UI elements
        'disclaimer', 'legal', 'terms', 'privacy',  # Legal text
        'hidden', 'mobile-only', 'desktop-only',  # Responsive classes
        'newsletter', 'subscribe', 'cta', 'button',  # Calls to action
        'rating', 'score', 'badge', 'icon',  # Decorative elements
        'caption', 'credit', 'source',  # Media credits
        'table', 'chart', 'graph', 'infographic',  # Data visualization
        'lazy-load', 'placeholder',  # Dynamic loading elements
        'article-meta', 'post-tags', 'content-info'  # Article metadata
    ]

    paragraphs = (
        ' '.join(element.stripped_strings)
        for element in soup.find_all(['p', 'article', 'body', 'content-body', 'span', 'div', 'main', 'section']) # Added 'main' and 'section' for more semantic HTML5 content extraction
        if element.stripped_strings and not any(cls in element.get('class', []) for cls in unwanted_classes)
    )

    # Join paragraphs with double newlines for readability
    text = '\n\n'.join(paragraphs)

    # --- Step 3: Clean Up the Extracted Text ---
    logger.debug("Cleaning and normalizing extracted text")
    # Normalize whitespace and remove unwanted characters in a single regex pass.
    text = re.sub(
        r'[\x00-\x1F\x7F-\x9F]|[^\x00-\x7F]+|\.{2,}|["\'`]+|[\(\)\[\]\{\}]|([.,!?;:])\s*|\s+',
        ' ',
        text
    ).strip()

    cleaned_length = len(text)
    compression_ratio = (cleaned_length / content_length) * 100 if content_length > 0 else 0
    elapsed_time = time.time() - start_time
    logger.info(f"Content cleaning complete - Output: {cleaned_length} chars ({compression_ratio:.1f}% of original) in {elapsed_time:.2f}s")
    
    return text

async def scrape_urls(urls: list[str]) -> list[tuple[str, str]]:
    """
    Scrape content from a list of URLs asynchronously.
    Returns a list of tuples containing (url, content).
    """
    start_time = time.time()
    url_count = len(urls)
    logger.info(f"Starting batch scraping of {url_count} URLs")
    
    async def process_url(url: str) -> tuple[str, str]:
        url_start_time = time.time()
        logger.info(f"Processing URL: {url}")
        
        raw_content = await get_contents(url)
        raw_length = len(raw_content)
        logger.debug(f"Retrieved {raw_length} chars of raw content from {url}")
        
        cleaned_content = clean_content(raw_content)
        cleaned_length = len(cleaned_content)
        
        url_elapsed_time = time.time() - url_start_time
        logger.info(f"Completed processing {url} - Raw: {raw_length} chars, Cleaned: {cleaned_length} chars in {url_elapsed_time:.2f}s")
        
        return (url, cleaned_content)

    tasks = [process_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    total_elapsed_time = time.time() - start_time
    logger.info(f"Completed batch scraping of {url_count} URLs in {total_elapsed_time:.2f}s")
    
    return results

def scrape_urls_sync(urls: list[str]) -> list[tuple[str, str]]:
    """
    Synchronous wrapper for scraping multiple URLs.
    """
    logger.info(f"Starting synchronous batch scraping of {len(urls)} URLs")
    results = asyncio.run(scrape_urls(urls))
    logger.info(f"Completed synchronous batch scraping")
    return results

async def extract_links(html_content: str, base_url: str) -> list[str]:
    """
    Extract links from HTML content and normalize them.
    Returns a list of absolute URLs.
    """
    logger.info(f"Extracting links from content with base URL: {base_url}")
    
    # Check if we have content to parse
    if not html_content or len(html_content) < 100:
        logger.warning(f"HTML content for {base_url} is too short or empty: {len(html_content)} chars")
        return []
    
    # Use a more lenient parser
    soup = BeautifulSoup(html_content, 'lxml')
    links = []
    
    # Count total anchors for debugging
    all_anchors = soup.find_all('a')
    anchors_with_href = soup.find_all('a', href=True)
    logger.info(f"Found {len(all_anchors)} total anchors, {len(anchors_with_href)} with href attributes")
    
    # Find all anchor tags with href attributes
    for anchor in anchors_with_href:
        href = anchor['href'].strip()
        
        # Skip empty links, javascript, and anchor links
        if not href or href.startswith(('javascript:', '#', 'mailto:', 'tel:')):
            continue
            
        # Convert relative URLs to absolute
        if not href.startswith(('http://', 'https://')):
            href = urljoin(base_url, href)
            
        # Normalize URL (remove fragments and query parameters for deduplication)
        parsed = urlparse(href)
        normalized_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        
        # Skip empty URLs after normalization
        if not normalized_url or not parsed.netloc:
            continue
            
        links.append(normalized_url)
    
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    logger.info(f"Extracted {len(unique_links)} unique links from {base_url}")
    
    # Print first few links for debugging
    if unique_links:
        sample = unique_links[:5]
        logger.info(f"Sample links: {', '.join(sample)}")
    
    return unique_links

async def scrape_with_depth(start_urls: list[str], max_depth: int = 1, max_urls_per_level: int = 5) -> dict:
    """
    Scrape content from URLs and follow links up to a specified depth.
    
    Args:
        start_urls: List of URLs to start scraping from
        max_depth: Maximum depth of links to follow (0 = only start_urls, 1 = start_urls + their links, etc.)
        max_urls_per_level: Maximum number of URLs to scrape at each depth level
        
    Returns:
        Dictionary with structure {url: {'content': content, 'links': [urls], 'depth': depth}}
    """
    start_time = time.time()
    logger.info(f"Starting depth scraping with {len(start_urls)} seed URLs, max_depth={max_depth}")
    
    # Initialize result dictionary and tracking sets
    results = {}
    all_urls = set(start_urls)  # Track all URLs we've seen to avoid duplicates
    
    # Process each depth level
    current_urls = start_urls
    for depth in range(max_depth + 1):
        logger.info(f"Processing depth {depth} with {len(current_urls)} URLs")
        
        # Limit number of URLs at this level if needed
        if len(current_urls) > max_urls_per_level:
            logger.info(f"Limiting from {len(current_urls)} to {max_urls_per_level} URLs at depth {depth}")
            current_urls = current_urls[:max_urls_per_level]
        
        # Create tasks for both content scraping and link extraction
        async def process_url(url):
            # Get the raw HTML content directly - don't use get_contents which might return cleaned content
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            html_content = await response.text()
                            logger.info(f"Retrieved {len(html_content)} chars of raw HTML from {url}")
                        else:
                            logger.warning(f"Failed to get content from {url}: status {response.status}")
                            html_content = ""
                except Exception as e:
                    logger.error(f"Error fetching {url}: {str(e)}")
                    html_content = ""
            
            # Clean the content for storage
            cleaned_content = clean_content(html_content) if html_content else ""
            
            # Extract links from the raw HTML
            links = []
            if depth < max_depth and html_content:
                links = await extract_links(html_content, url)
                
            return url, cleaned_content, links
            
        # Process all URLs at this level concurrently
        tasks = [process_url(url) for url in current_urls]
        level_results = await asyncio.gather(*tasks)
        
        # Process results and collect links for next level
        next_level_urls = []
        for url, content, links in level_results:
            # Store results
            results[url] = {
                'content': content,
                'links': links,
                'depth': depth
            }
            
            # Add new links to the next level queue
            if depth < max_depth:
                for link in links:
                    if link not in all_urls:
                        next_level_urls.append(link)
                        all_urls.add(link)
        
        # Set up next iteration
        current_urls = next_level_urls
        if not current_urls:
            logger.info(f"No more links to follow after depth {depth}")
            break
    
    total_elapsed_time = time.time() - start_time
    logger.info(f"Completed depth scraping of {len(results)} total URLs in {total_elapsed_time:.2f}s")
    
    return results

def scrape_with_depth_sync(start_urls: list[str], max_depth: int = 1, max_urls_per_level: int = 5) -> dict:
    """
    Synchronous wrapper for depth-based scraping.
    """
    logger.info(f"Starting synchronous depth scraping with {len(start_urls)} seed URLs")
    results = asyncio.run(scrape_with_depth(start_urls, max_depth, max_urls_per_level))
    logger.info(f"Completed synchronous depth scraping with {len(results)} total URLs")
    return results

# Example usage
if __name__ == "__main__":
    '''logger.info("Starting web scraper")
    
    urls = [
        "https://news.ycombinator.com",
        "https://waitbutwhy.com",
        "https://arstechnica.com",
        "https://www.nature.com/news",
        "https://www.scientificamerican.com"
    ]
    
    logger.info(f"Scraping {len(urls)} URLs: {', '.join(urls)}")
    results = scrape_urls_sync(urls)
    
    for url, content in results:
        logger.info(f"URL: {url} - Content length: {len(content)} characters")
    
    logger.info("Web scraper execution complete")'''

    # Example URLs to scrape
    start_urls = [
        "https://news.ycombinator.com",
        "https://waitbutwhy.com"
    ]
    
    # Scrape with depth 2 and max 3 URLs per level
    results = scrape_with_depth_sync(
        start_urls=start_urls,
        max_depth=2,
        max_urls_per_level=3
    )
    
    # Print results
    logger.info(f"Scraped {len(results)} total URLs:")
    for url, data in results.items():
        logger.info(f"URL: {url}")
        logger.info(f"  Depth: {data['depth']}")
        logger.info(f"  Content length: {len(data['content'])} characters")
        logger.info(f"  Found {len(data['links'])} links")

