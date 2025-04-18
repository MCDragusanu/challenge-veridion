import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
from collections import Counter
import imghdr  # To determine image file type

def is_likely_logo_url(url):
    """Basic heuristic to check if a URL looks like it points to an image."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    return path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico'))

def fetch_image_size(url, timeout=5):
    """Fetches the image and tries to determine its size without downloading the whole thing."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        if 'content-length' in response.headers:
            return int(response.headers['content-length'])
        elif 'content-type' in response.headers and response.headers['content-type'].startswith('image/'):
            # Try a quick GET for the first few bytes to determine format and potentially size
            response_get = requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response_get.raise_for_status()
            for chunk in response_get.iter_content(chunk_size=128):
                img_format = imghdr.what(None, chunk)
                if img_format:
                    # We know it's an image, but don't have length reliably
                    return 1  # Return a small positive value to prioritize
                break
            return 0
        return 0
    except requests.exceptions.RequestException:
        return 0

def scrape_site_info(base_url):
    print(f"Scraping: {base_url}")
    logo = None
    keywords = []
    try:
        if not base_url.startswith("http"):
            base_url = "http://" + base_url

        response = requests.get(base_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # 1. Favicon (rel='icon', rel='shortcut icon')
        icon_links = soup.find_all("link", rel=lambda val: val and 'icon' in val.lower())
        for icon_link in icon_links:
            if icon_link.get("href"):
                potential_logo = urljoin(base_url, icon_link["href"])
                if is_likely_logo_url(potential_logo):
                    logo = potential_logo
                    print(f"  Found logo (favicon): {logo}")
                    return {"url": base_url, "logo": logo, "keywords": []} # Prioritize favicon

        # 2. Images with 'logo' in attributes
        logo_candidates_by_keyword = []
        for img in soup.find_all("img"):
            attrs = ' '.join([
                img.get('id') or '',
                ' '.join(img.get('class') or []),
                img.get('alt') or '',
                img.get('src') or ''
            ]).lower()
            src = img.get('src')
            if src and is_likely_logo_url(urljoin(base_url, src)) and 'logo' in attrs:
                logo_candidates_by_keyword.append(urljoin(base_url, src))

        if logo_candidates_by_keyword:
            logo = logo_candidates_by_keyword[0] # Take the first one found
            print(f"  Found logo (keyword): {logo}")
            return {"url": base_url, "logo": logo, "keywords": []} # Prioritize keyword matches

        # 3. Images in specific logo containers (ids, classes)
        logo_container_selectors = ['#logo', '.logo', '.header-logo', '#site-logo', '.brand-logo']
        for selector in logo_container_selectors:
            container = soup.select_one(selector)
            if container:
                img_in_container = container.find("img")
                if img_in_container and img_in_container.get("src") and is_likely_logo_url(urljoin(base_url, img_in_container["src"])):
                    logo = urljoin(base_url, img_in_container["src"])
                    print(f"  Found logo (container: {selector}): {logo}")
                    return {"url": base_url, "logo": logo, "keywords": []} # Prioritize container matches

        # 4. Largest image (with size check to avoid banners etc.)
        largest_logo_candidate = {"size": 0, "src": None}
        img_candidates_by_size = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and is_likely_logo_url(urljoin(base_url, src)):
                abs_src = urljoin(base_url, src)
                size = fetch_image_size(abs_src)
                if size > largest_logo_candidate["size"] and size > 500: # Basic size filter
                    largest_logo_candidate["size"] = size
                    largest_logo_candidate["src"] = abs_src
                img_candidates_by_size.append((size, abs_src))

        if largest_logo_candidate["src"]:
            logo = largest_logo_candidate["src"]
            print(f"  Found potential logo (largest, size > 500): {logo}")

        # --- Keyword Extraction (moved here to run even if logo is found early) ---
        keywords = set()

        # From meta tags
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords and meta_keywords.get("content"):
            keywords.update([kw.strip().lower() for kw in meta_keywords["content"].split(",") if kw.strip()])

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            keywords.update(re.findall(r'\b[a-zA-Z]{4,}\b', meta_desc["content"].lower()))

        # From title and headings
        content_sources = []
        if soup.title and soup.title.string:
            content_sources.append(soup.title.string)

        content_sources += [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3", "h4", "strong", "b"])]

        for text in content_sources:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            keywords.update(words)

        # Basic cleanup: remove overly generic terms
        blacklist = {"home", "about", "login", "click", "learn", "read", "more", "contact", "info", "page", "terms"}
        keywords = [kw for kw in keywords if kw not in blacklist]

        word_counter = Counter(keywords)
        top_keywords = [word for word, _ in word_counter.most_common(30)]

        return {
            "url": base_url,
            "logo": logo,
            "keywords": top_keywords
        }

    except Exception as e:
        print(f"[Error] Failed to scrape {base_url}: {e}")
        return {
            "url": base_url,
            "logo": None,
            "keywords": []
        }