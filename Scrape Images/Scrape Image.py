"""
scrape_oil_images.py
Collect candidate images for an oil-leak detection dataset.

Sources (in order of preference):
  1. Openverse API       - Creative Commons images, license info included
  2. Wikimedia Commons   - freely licensed images, license info included
  3. Bing (via icrawler) - OPTIONAL, large volume but copyright unknown;
                           use only for private research/training, never redistribute

Every downloaded image is logged in metadata.csv (source, license, author, page URL)
so you can cite sources in your thesis/paper and drop anything with a bad license.

Install:
    pip install requests pillow icrawler tqdm

Usage:
    python scrape_oil_images.py                    # Openverse + Wikimedia only
    python scrape_oil_images.py --bing --max 150   # also use Bing, 150/query
"""

import argparse
import csv
import hashlib
import io
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Search terms grouped by the folder they should land in.
# "negative" images (clean engines, clean concrete) are just as important:
# they teach the model what is NOT a leak and reduce false positives.
# ---------------------------------------------------------------------------
QUERIES = {
    "oil_stain_ground": [
        "oil stain concrete driveway",
        "car oil leak puddle parking",
        "motor oil spill asphalt",
        "oil drip under car",
        "oil spot garage floor",
    ],
    "engine_oil_leak": [
        "engine oil leak",
        "valve cover gasket oil leak",
        "oil pan leak",
        "oily engine bay",
        "leaking engine oil residue",
        "oil filter leak",
    ],
    "fresh_vs_old_oil": [
        "fresh motor oil",
        "dirty used engine oil",
        "black engine oil",
    ],
    "negative_clean_engine": [
        "clean car engine bay",
        "car engine compartment",
        "detailed engine bay",
    ],
    "negative_clean_ground": [
        "clean concrete driveway",
        "parking lot asphalt",
        "wet asphalt road",       # hard negative: looks like oil, isn't
        "water puddle parking lot",  # hard negative
        "shadow under car",       # hard negative
    ],
}

HEADERS = {"User-Agent": "OilLeakDatasetStudentProject/1.0 (educational use)"}
MIN_SIDE = 320          # skip tiny images
REQUEST_DELAY = 1.0     # seconds between API calls - be polite


def save_image(content: bytes, out_dir: Path, seen: set) -> str | None:
    """Validate, deduplicate by hash, convert to JPEG. Returns filename or None."""
    digest = hashlib.md5(content).hexdigest()
    if digest in seen:
        return None
    try:
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")
    except Exception:
        return None
    if min(img.size) < MIN_SIDE:
        return None
    seen.add(digest)
    name = f"{digest}.jpg"
    img.save(out_dir / name, "JPEG", quality=92)
    return name


def download(url: str) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            return r.content
    except requests.RequestException:
        pass
    return None


# ---------------------------------------------------------------------------
# Source 1: Openverse (https://api.openverse.org) - no key needed for light use
# ---------------------------------------------------------------------------
def search_openverse(query: str, limit: int):
    results, page = [], 1
    while len(results) < limit:
        try:
            r = requests.get(
                "https://api.openverse.org/v1/images/",
                params={"q": query, "page": page, "page_size": 20,
                        "license_type": "all"},
                headers=HEADERS, timeout=20,
            )
            if r.status_code != 200:
                break
            data = r.json().get("results", [])
        except requests.RequestException:
            break
        if not data:
            break
        for item in data:
            results.append({
                "url": item.get("url"),
                "license": f'{item.get("license", "")} {item.get("license_version", "")}'.strip(),
                "author": item.get("creator", ""),
                "page": item.get("foreign_landing_url", ""),
                "source": "openverse",
            })
        page += 1
        time.sleep(REQUEST_DELAY)
    return results[:limit]


# ---------------------------------------------------------------------------
# Source 2: Wikimedia Commons API
# ---------------------------------------------------------------------------
def search_wikimedia(query: str, limit: int):
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query", "format": "json",
                "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": 6, "gsrlimit": min(limit, 50),
                "prop": "imageinfo", "iiprop": "url|extmetadata",
                "iiurlwidth": 1280,
            },
            headers=HEADERS, timeout=20,
        )
        pages = r.json().get("query", {}).get("pages", {})
    except (requests.RequestException, ValueError):
        return []
    results = []
    for p in pages.values():
        info = (p.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        results.append({
            "url": info.get("thumburl") or info.get("url"),
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "author": meta.get("Artist", {}).get("value", ""),
            "page": info.get("descriptionurl", ""),
            "source": "wikimedia",
        })
    time.sleep(REQUEST_DELAY)
    return results


# ---------------------------------------------------------------------------
# Source 3 (optional): Bing via icrawler - unknown licenses
# ---------------------------------------------------------------------------
def crawl_bing(query: str, out_dir: Path, limit: int):
    from icrawler.builtin import BingImageCrawler
    tmp = out_dir / "_bing_tmp" / query.replace(" ", "_")
    tmp.mkdir(parents=True, exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": str(tmp)}, log_level=40)
    crawler.crawl(keyword=query, max_num=limit, min_size=(MIN_SIDE, MIN_SIDE))
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw_images")
    ap.add_argument("--max", type=int, default=60, help="max images per query per source")
    ap.add_argument("--bing", action="store_true", help="also crawl Bing (license unknown)")
    args = ap.parse_args()

    root = Path(args.out)
    meta_path = root / "metadata.csv"
    root.mkdir(parents=True, exist_ok=True)
    new_file = not meta_path.exists()
    meta_f = open(meta_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(meta_f)
    if new_file:
        writer.writerow(["file", "category", "query", "source", "license", "author", "page_url"])

    seen: set = {p.stem for p in root.rglob("*.jpg")}

    for category, queries in QUERIES.items():
        cat_dir = root / category
        cat_dir.mkdir(exist_ok=True)
        for q in queries:
            candidates = search_openverse(q, args.max) + search_wikimedia(q, args.max)
            saved = 0
            for c in tqdm(candidates, desc=f"{category} | {q}", leave=False):
                if not c["url"]:
                    continue
                content = download(c["url"])
                if not content:
                    continue
                name = save_image(content, cat_dir, seen)
                if name:
                    writer.writerow([name, category, q, c["source"], c["license"],
                                c["author"], c["page"]])
                    saved += 1
            if args.bing:
                tmp = crawl_bing(q, root, args.max)
                for f in tmp.glob("*"):
                    name = save_image(f.read_bytes(), cat_dir, seen)
                    if name:
                        writer.writerow([name, category, q, "bing", "UNKNOWN", "", ""])
                        saved += 1
                    f.unlink()
            meta_f.flush()
            print(f"[{category}] '{q}': saved {saved}")

    meta_f.close()
    print(f"\nDone. Images in {root}/, sources logged in {meta_path}")
    print("Next: manually review every image and delete irrelevant ones before labeling.")


if __name__ == "__main__":
    main()