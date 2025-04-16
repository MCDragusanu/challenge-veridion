import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_site_info(base_url):
    print(f"Scraping: {base_url}")
    try:
        # Ensure full URL
        if not base_url.startswith("http"):
            base_url = "http://" + base_url

        response = requests.get(base_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        logo = None

        # 1. Look for <link rel="icon" ...> and similar
        icon_link = soup.find("link", rel=lambda val: val and 'icon' in val.lower())
        if icon_link and icon_link.get("href"):
            logo = urljoin(base_url, icon_link["href"])

        # 2. If not found, look for <img> with 'logo' in any relevant attribute
        if not logo:
            for img in soup.find_all("img"):
                attrs = ' '.join([
                    img.get('id') or '',
                    img.get('class') and ' '.join(img.get('class')) or '',
                    img.get('alt') or '',
                    img.get('src') or ''
                ]).lower()

                if 'logo' in attrs:
                    logo = urljoin(base_url, img.get('src'))
                    break

        # 3. If still not found, fallback to largest image by size attributes
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

        # Extract keywords
        keywords = []
        meta_keywords = soup.find("meta", attrs={"name": "keywords"})
        if meta_keywords and meta_keywords.get("content"):
            keywords = [kw.strip() for kw in meta_keywords["content"].split(",") if kw.strip()]
        else:
            # fallback: collect visible strong headers
            headings = soup.find_all(["h1", "h2", "h3", "strong"])
            keywords = list({word.lower() for h in headings for word in h.get_text(strip=True).split() if len(word) > 3})

        return {
            "url": base_url,
            "logo": logo,
            "keywords": keywords
        }

    except Exception as e:
        print(f"[Error] Failed to scrape {base_url}: {e}")
        return {
            "url": base_url,
            "logo": None,
            "keywords": []
        }
