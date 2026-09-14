"""
Tabanan News RSS Scraper
- feedparser  : parse RSS feed
- Selenium    : resolve Google News redirect URLs
- newspaper3k : download + parse full article text
- Output      : structured CSV (raw fields, no 5W1H transformation)

CSV output is committed & pushed to a GitHub repo by the accompanying
GitHub Actions workflow, so it becomes reachable at a raw.githubusercontent.com URL.
"""

import csv
import re
import time
import os
import nltk
import feedparser

# Download required NLTK data silently
for _pkg in ("punkt", "punkt_tab"):
    nltk.download(_pkg, quiet=True)

from datetime import datetime
from newspaper import Article
from newspaper import network as newspaper_network
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# ── Config ────────────────────────────────────────────────────────────────────

RSS_URL = (
    "https://news.google.com/rss/search" "?q=Tabanan+when:1d&hl=id&gl=ID&ceid=ID:id"
)

MAX_RESULTS = 40

TODAY = datetime.now().strftime("%Y-%m-%d")
# NOTE: these paths are relative to the repo root when run from the GitHub
# Actions workflow, so the committed CSVs land inside the repo (and therefore
# become available via raw.githubusercontent.com once pushed).
#
# Flat, dated-filename layout (one file per day, no per-run timestamp folder)
# so downstream analysis can just glob("output/success/*.csv") without
# recursing through folders:
#   output/success/2026-09-14.csv
#   output/failed/2026-09-14.csv
SUCCESS_DIR = os.path.join("output", "success")
FAILED_DIR = os.path.join("output", "failed")
os.makedirs(SUCCESS_DIR, exist_ok=True)
os.makedirs(FAILED_DIR, exist_ok=True)

SUCCESS_CSV = os.path.join(SUCCESS_DIR, f"{TODAY}.csv")
FAILED_CSV = os.path.join(FAILED_DIR, f"{TODAY}.csv")

# Fixed values
FIXED = {
    "data_of_information": "2, 7",
    "category": "5",
    "source_agent_report": "Kilasbali",
    "hashtags": "kejaksaan, kejatibali, kejaritabanan, tabanan, news",
    "latitude": -8.536609490935,
    "longitude": 115.13552944186335,
    "location_name": "Dajan Peken, 82114, Tabanan, Tabanan, Bali, Indonesia",
    "where": "Kabupaten Tabanan, Bali, Indonesia",
}

# No more 5W1H fields (what/why/how) — just the raw article content + metadata.
SUCCESS_FIELDS = [
    "title",
    "published_date",
    "where",
    "source_agent_report",
    "url",
    "summary",
    "content",
    "location_name",
    "latitude",
    "longitude",
    "category",
    "data_of_information",
    "hashtags",
]

FAILED_FIELDS = ["title", "google_news_link", "reason"]

# ── Helpers ───────────────────────────────────────────────────────────────────


def clean_title(title: str) -> str:
    """Removes source suffix after '-' and strips special characters."""
    if "-" in title:
        title = title.rsplit("-", 1)[0]
    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)
    return " ".join(title.split())


def extract_source_from_title(title: str) -> str:
    """Original (uncleaned) title is needed here to find the source."""
    match = re.search(r"-\s*([^-]+)$", title)
    return match.group(1).strip() if match else "News"


def format_date_iso(pub_dt: datetime) -> str:
    return pub_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_existing_rows(csv_path: str, fields: list) -> list:
    """Load rows already written today (if the file exists from an earlier
    run today), so a second run the same day can append+dedupe instead of
    overwriting or creating a duplicate file."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row]


def dedupe_success_rows(rows: list) -> list:
    """De-duplicate by article URL, keeping the first occurrence."""
    seen = set()
    deduped = []
    for row in rows:
        key = row.get("url")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


def dedupe_failed_rows(rows: list) -> list:
    """De-duplicate by google_news_link, keeping the first occurrence."""
    seen = set()
    deduped = []
    for row in rows:
        key = row.get("google_news_link")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(row)
    return deduped


# ── Selenium URL resolver ─────────────────────────────────────────────────────


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    return webdriver.Chrome(options=opts)


def resolve_url(driver: webdriver.Chrome, google_url: str) -> str | None:
    try:
        driver.get(google_url)
        time.sleep(2)
        real = driver.current_url
        return real if "google.com" not in real else None
    except Exception:
        return None


# ── Article scraper ───────────────────────────────────────────────────────────


def scrape_article(url: str) -> tuple[str, str]:
    article = Article(url, language="id")
    article.download()
    article.parse()
    article.nlp()
    return article.text, article.summary


# ── Row builder ───────────────────────────────────────────────────────────────


def build_success_row(entry, pub_dt: datetime, full_text: str, summary: str, resolved_url: str) -> dict:
    original_title = entry.title
    cleaned_title = clean_title(original_title)

    return {
        "title": cleaned_title,
        "published_date": format_date_iso(pub_dt),
        "where": FIXED["where"],
        "source_agent_report": extract_source_from_title(original_title),
        "url": resolved_url,
        "summary": summary.strip(),
        "content": full_text.strip(),
        "location_name": FIXED["location_name"],
        "latitude": FIXED["latitude"],
        "longitude": FIXED["longitude"],
        "category": FIXED["category"],
        "data_of_information": FIXED["data_of_information"],
        "hashtags": FIXED["hashtags"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    newspaper_network.USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    print(f"[RSS] Fetching: {RSS_URL}")
    feed = feedparser.parse(RSS_URL)
    entries = feed.entries[:MAX_RESULTS]
    print(f"[RSS] Processing {len(entries)} articles.\n")

    driver = make_driver()
    success_rows = []
    failed_rows = []

    for i, entry in enumerate(entries, 1):
        title = entry.get("title", "(no title)")
        link = entry.get("link", "")
        print(f"[{i}/{len(entries)}] {title[:70]}...")

        try:
            pub_dt = datetime(*entry.published_parsed[:6])
        except Exception:
            failed_rows.append(
                {"title": title, "google_news_link": link, "reason": "Date error"}
            )
            continue

        resolved_url = resolve_url(driver, link)
        if not resolved_url:
            failed_rows.append(
                {
                    "title": title,
                    "google_news_link": link,
                    "reason": "URL resolve failed",
                }
            )
            continue

        try:
            full_text, summary = scrape_article(resolved_url)
        except Exception as e:
            failed_rows.append(
                {
                    "title": title,
                    "google_news_link": link,
                    "reason": f"Scrape failed: {e}",
                }
            )
            continue

        if not full_text or len(full_text.strip()) < 50:
            failed_rows.append(
                {
                    "title": title,
                    "google_news_link": link,
                    "reason": "Empty content (no article body text found)",
                }
            )
            print("    ✗ Skipped: No body content")
            continue

        row = build_success_row(entry, pub_dt, full_text, summary, resolved_url)
        success_rows.append(row)
        print("    ✓ Success")

        time.sleep(1)

    driver.quit()

    # Merge with anything already written today (handles re-runs on the same
    # day: manual dispatch + scheduled run, retries, etc.) and dedupe.
    all_success = dedupe_success_rows(
        load_existing_rows(SUCCESS_CSV, SUCCESS_FIELDS) + success_rows
    )
    all_failed = dedupe_failed_rows(
        load_existing_rows(FAILED_CSV, FAILED_FIELDS) + failed_rows
    )

    with open(SUCCESS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUCCESS_FIELDS)
        writer.writeheader()
        writer.writerows(all_success)

    with open(FAILED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_FIELDS)
        writer.writeheader()
        writer.writerows(all_failed)

    print(
        f"\n✅ Finished this run! New success: {len(success_rows)} | New failed: {len(failed_rows)}"
    )
    print(
        f"   Today's totals -> success: {len(all_success)} | failed: {len(all_failed)}"
    )

    # Emit paths so the GitHub Actions workflow can reference them if needed
    # (e.g. to build the raw.githubusercontent.com URL in a job summary).
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as f:
            f.write(f"success_csv={SUCCESS_CSV}\n")
            f.write(f"failed_csv={FAILED_CSV}\n")


if __name__ == "__main__":
    main()