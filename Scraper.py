import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from collections import Counter

def scrape_site_info(base_url):
    print(f"Scraping: {base_url}")
    try:
        if not base_url.startswith("http"):
            base_url = "http://" + base_url

        response = requests.get(base_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        logo = None

        # --- Logo Detection ---
        icon_link = soup.find("link", rel=lambda val: val and 'icon' in val.lower())
        if icon_link and icon_link.get("href"):
            logo = urljoin(base_url, icon_link["href"])

        if not logo:
            for img in soup.find_all("img"):
                attrs = ' '.join([
                    img.get('id') or '',
                    ' '.join(img.get('class') or []),
                    img.get('alt') or '',
                    img.get('src') or ''
                ]).lower()

                if 'logo' in attrs:
                    logo = urljoin(base_url, img.get('src'))
                    break

        if not logo:
            largest = {"area": 0, "src": None}
            for img in soup.find_all("img"):
                try:
                    width = int(img.get("width", 0))
                    height = int(img.get("height", 0))
                    area = width * height
                    if area > largest["area"] and img.get("src"):
                        largest = {"area": area, "src": img["src"]}
                except:
                    continue
            if largest["src"]:
                logo = urljoin(base_url, largest["src"])

        # --- Keyword Extraction ---
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
