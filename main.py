"""
Tabanan News RSS Scraper (multi-thread)

Fitur:
- Mengambil berita dari Google News RSS
- Selenium untuk resolve Google News redirect URL (1 Chrome per worker thread)
- newspaper4k/newspaper3k untuk mengambil isi berita
- Membentuk data 5W1H
- Menyimpan CSV success dan failed ke output/success/{date} dan output/failed/{date}
- Mendukung testing lokal
- Mendukung GitHub Actions (menulis GITHUB_OUTPUT: success_csv, failed_csv)
- Pemrosesan paralel dengan ThreadPoolExecutor (--workers)

Contoh penggunaan:

    python scraper.py
    python scraper.py --workers 6
    python scraper.py --max-results 5 --show-browser --workers 1
    python scraper.py --no-selenium --workers 8
    python scraper.py --rss "https://news.google.com/rss/search?q=Tabanan"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Tuple

import feedparser
import nltk

from newspaper import Article
from newspaper import network as newspaper_network

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_RSS_URL = (
    "https://news.google.com/rss/search"
    "?q=Tabanan+when:1d&hl=id&gl=ID&ceid=ID:id"
)

DEFAULT_MAX_RESULTS = 40
DEFAULT_WORKERS = 4

ID_DAYS = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}

ID_MONTHS = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
    8: "Agustus",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Desember",
}

FIXED = {
    "data_of_information": "2, 7",
    "category": "5",
    "source_agent_report": "Kilasbali",
    "hashtags": "kejaksaan, kejatibali, kejaritabanan, tabanan, news",
    "latitude": -8.536609490935,
    "longitude": 115.13552944186335,
    "location_name": (
        "Dajan Peken, 82114, Tabanan, Tabanan, Bali, Indonesia"
    ),
    "where": "Kabupaten Tabanan, Bali, Indonesia",
}

SUCCESS_FIELDS = [
    "title",
    "when_",
    "where",
    "who",
    "what",
    "why",
    "how",
    "source_agent_report",
    "location_name",
    "latitude",
    "longitude",
    "category",
    "data_of_information",
    "hashtags",
]

FAILED_FIELDS = [
    "title",
    "google_news_link",
    "reason",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


# ============================================================================
# NLTK
# ============================================================================

def prepare_nltk() -> None:
    """
    Mengunduh resource NLTK yang diperlukan oleh newspaper.
    Dipanggil sekali di thread utama sebelum worker dimulai.
    """

    for package in ("punkt", "punkt_tab"):
        try:
            nltk.download(package, quiet=True)
        except Exception as error:
            print(
                f"[WARNING] Gagal mengunduh NLTK resource "
                f"{package}: {error}"
            )


# ============================================================================
# TEXT HELPERS
# ============================================================================

def split_sentences(text: str) -> list[str]:
    """
    Memecah teks menjadi kalimat sederhana.
    """

    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text.strip())

    return [part.strip() for part in parts if part.strip()]


def get_sentences(text: str, start: int, end: int) -> str:
    """
    Mengambil kalimat dari index start sampai end.
    """

    return " ".join(split_sentences(text)[start:end])


def extract_source_from_title(title: str) -> str:
    """
    Mengambil nama media dari bagian akhir judul Google News.
    """

    match = re.search(r"-\s*([^-]+)$", title)

    if match:
        return match.group(1).strip()

    return "News"


def clean_title(title: str) -> str:
    """
    Membersihkan judul berita.

    - Menghapus suffix media setelah tanda "-"
    - Menghapus karakter khusus
    - Merapikan spasi
    """

    if not title:
        return ""

    if "-" in title:
        title = title.rsplit("-", 1)[0]

    title = re.sub(r"[^a-zA-Z0-9\s]", "", title)

    return " ".join(title.split())


def format_how_prefix(pub_dt: datetime) -> str:
    """
    Membuat prefix untuk kolom how.
    """

    day_name = ID_DAYS[pub_dt.weekday()]
    month_name = ID_MONTHS[pub_dt.month]

    return (
        f"Pada Hari {day_name} , "
        f"{pub_dt.day} {month_name} {pub_dt.year} "
        f"{pub_dt.strftime('%H:%M')}, "
        "Tabanan, Tabanan, Bali"
    )


def format_date_iso(pub_dt: datetime) -> str:
    """
    Format tanggal seperti:
        2026-09-16T10:20:30.123Z
    """

    return pub_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ============================================================================
# SELENIUM (satu driver per thread)
# ============================================================================

def make_driver(show_browser: bool = False) -> webdriver.Chrome:
    """
    Membuat Chrome WebDriver.
    """

    options = Options()

    if not show_browser:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=id-ID")
    options.add_argument(f"--user-agent={USER_AGENT}")

    # Percepat: tidak perlu menunggu semua resource selesai dimuat.
    options.page_load_strategy = "eager"

    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        return driver

    except Exception as error:
        raise RuntimeError(
            "Gagal menjalankan Chrome WebDriver. "
            "Pastikan Google Chrome sudah terinstall. "
            f"Detail: {error}"
        ) from error


class DriverPool:
    """
    Menyediakan satu WebDriver untuk setiap thread (WebDriver tidak
    thread-safe) dan menutup semuanya di akhir proses.
    """

    def __init__(self, show_browser: bool) -> None:
        self.show_browser = show_browser
        self._local = threading.local()
        self._all: list[webdriver.Chrome] = []
        self._lock = threading.Lock()

    def get(self) -> webdriver.Chrome:
        driver = getattr(self._local, "driver", None)

        if driver is None:
            driver = make_driver(show_browser=self.show_browser)
            self._local.driver = driver

            with self._lock:
                self._all.append(driver)

        return driver

    def close_all(self) -> None:
        with self._lock:
            drivers, self._all = self._all, []

        for driver in drivers:
            try:
                driver.quit()
            except Exception as error:
                print(f"[WARNING] Gagal menutup Chrome: {error}")

        if drivers:
            print(f"[SELENIUM] {len(drivers)} Chrome ditutup.")


def resolve_url(
    driver: webdriver.Chrome,
    google_url: str,
    timeout: float = 10.0,
) -> Optional[str]:
    """
    Membuka URL Google News dengan Selenium lalu menunggu sampai
    redirect ke URL artikel tujuan selesai (bukan sleep tetap),
    sehingga lebih cepat.
    """

    if not google_url:
        return None

    try:
        driver.get(google_url)

        try:
            WebDriverWait(driver, timeout, poll_frequency=0.25).until(
                lambda d: "google.com" not in d.current_url.lower()
            )
        except Exception:
            return None

        real_url = driver.current_url.strip()

        if not real_url or "google.com" in real_url.lower():
            return None

        return real_url

    except Exception:
        return None


# ============================================================================
# ARTICLE SCRAPER
# ============================================================================

def scrape_article(url: str) -> Tuple[str, str]:
    """
    Mengambil isi artikel dan ringkasannya menggunakan newspaper.
    """

    article = Article(url, language="id")

    article.download()
    article.parse()

    try:
        article.nlp()
    except Exception:
        pass

    return (
        (article.text or "").strip(),
        (article.summary or "").strip(),
    )


# ============================================================================
# CSV HELPERS
# ============================================================================

def create_output_paths() -> Tuple[str, str]:
    """
    Membuat path output untuk success dan failed:

        output/success/{YYYY-MM-DD}.csv
        output/failed/{YYYY-MM-DD}.csv

    File untuk tanggal yang sama akan ditimpa oleh run berikutnya.
    """

    today = datetime.now().strftime("%Y-%m-%d")

    success_dir = os.path.join("output", "success")
    failed_dir = os.path.join("output", "failed")

    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    return (
        os.path.join(success_dir, f"{today}.csv"),
        os.path.join(failed_dir, f"{today}.csv"),
    )


def write_csv(
    filename: str,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    """
    Menulis data ke CSV UTF-8.
    """

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_github_output(success_csv: str, failed_csv: str) -> None:
    """
    Menulis path CSV ke file GITHUB_OUTPUT (jika berjalan di GitHub Actions):

        ${{ steps.scrape.outputs.success_csv }}
        ${{ steps.scrape.outputs.failed_csv }}
    """

    github_output = os.environ.get("GITHUB_OUTPUT")

    if not github_output:
        return

    try:
        success_posix = success_csv.replace(os.sep, "/")
        failed_posix = failed_csv.replace(os.sep, "/")

        with open(github_output, "a", encoding="utf-8") as file:
            file.write(f"success_csv={success_posix}\n")
            file.write(f"failed_csv={failed_posix}\n")

    except Exception as error:
        print(f"[WARNING] Gagal menulis GITHUB_OUTPUT: {error}")


# ============================================================================
# ROW BUILDER
# ============================================================================

def build_success_row(
    entry,
    pub_dt: datetime,
    full_text: str,
    summary: str,
) -> dict:
    """
    Membentuk satu baris data success.
    """

    original_title = entry.get("title", "(no title)")
    cleaned_title = clean_title(original_title)

    what = get_sentences(full_text, 0, 2).strip() or summary.strip()
    why = get_sentences(full_text, 2, 4)

    prefix = format_how_prefix(pub_dt)
    how = f"{prefix}\n\n{full_text.strip()}"

    return {
        "title": cleaned_title,
        "when_": format_date_iso(pub_dt),
        "where": FIXED["where"],
        "who": original_title,
        "what": what,
        "why": why,
        "how": how,
        "source_agent_report": extract_source_from_title(original_title),
        "location_name": FIXED["location_name"],
        "latitude": FIXED["latitude"],
        "longitude": FIXED["longitude"],
        "category": FIXED["category"],
        "data_of_information": FIXED["data_of_information"],
        "hashtags": FIXED["hashtags"],
    }


# ============================================================================
# WORKER
# ============================================================================

print_lock = threading.Lock()


def log(lines: list[str]) -> None:
    """
    Mencetak beberapa baris sekaligus agar log antar thread tidak tercampur.
    """

    with print_lock:
        print("\n".join(lines), flush=True)


def process_entry(
    index: int,
    total: int,
    entry,
    args,
    pool: Optional[DriverPool],
) -> Tuple[str, dict]:
    """
    Memproses satu artikel. Mengembalikan ("success" | "failed", row).
    Dijalankan di worker thread.
    """

    title = entry.get("title", "(no title)")
    google_news_link = entry.get("link", "")

    lines = [f"[{index}/{total}] {title[:100]}"]

    def fail(reason: str, message: str) -> Tuple[str, dict]:
        lines.append(f"    ✗ {message}")
        log(lines)

        return (
            "failed",
            {
                "title": title,
                "google_news_link": google_news_link,
                "reason": reason,
            },
        )

    # ---- Tanggal publikasi ------------------------------------------------
    try:
        published_parsed = entry.get("published_parsed")

        if not published_parsed:
            raise ValueError("published_parsed tidak tersedia")

        pub_dt = datetime(*published_parsed[:6])

    except Exception as error:
        return fail(f"Date error: {error}", "Gagal: tanggal tidak valid")

    # ---- Resolve URL ------------------------------------------------------
    if args.no_selenium:
        resolved_url = google_news_link
    else:
        try:
            resolved_url = resolve_url(pool.get(), google_news_link)
        except Exception as error:
            return fail(
                f"Driver error: {error}",
                f"Gagal: driver error ({error})",
            )

    if not resolved_url:
        return fail("URL resolve failed", "Gagal: URL artikel tidak ditemukan")

    lines.append(f"    [URL] {resolved_url[:150]}")

    # ---- Scrape artikel ---------------------------------------------------
    try:
        full_text, summary = scrape_article(resolved_url)
    except Exception as error:
        return fail(f"Scrape failed: {error}", f"Scrape gagal: {error}")

    if not full_text or len(full_text.strip()) < 50:
        return fail(
            "Empty 'how' (No article body text found)",
            "Skipped: No body content",
        )

    # ---- Build row --------------------------------------------------------
    row = build_success_row(entry, pub_dt, full_text, summary)

    lines.append("    ✓ Success")
    log(lines)

    # Jeda per worker (sopan terhadap server target).
    if args.delay > 0:
        time.sleep(args.delay)

    return "success", row


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tabanan News RSS Scraper (multi-thread)",
    )

    parser.add_argument(
        "--rss",
        default=DEFAULT_RSS_URL,
        help="URL RSS Google News",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Jumlah maksimal artikel yang diproses",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Jumlah thread paralel "
            f"(default {DEFAULT_WORKERS}). Setiap thread membuka 1 Chrome "
            "jika Selenium aktif."
        ),
    )

    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Menampilkan Chrome saat Selenium berjalan",
    )

    parser.add_argument(
        "--no-selenium",
        action="store_true",
        help=(
            "Tidak menggunakan Selenium. "
            "Link RSS akan digunakan langsung sebagai URL artikel."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Jeda per worker setelah tiap artikel, dalam detik",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_arguments()

    if args.max_results < 1:
        print("[ERROR] --max-results harus lebih besar dari 0.")
        sys.exit(1)

    if args.workers < 1:
        print("[ERROR] --workers harus lebih besar dari 0.")
        sys.exit(1)

    if args.delay < 0:
        print("[ERROR] --delay tidak boleh negatif.")
        sys.exit(1)

    # Mode tampil browser lebih nyaman dengan 1 worker.
    if args.show_browser and args.workers > 1:
        print(
            "[INFO] --show-browser aktif: "
            "jumlah worker tetap sesuai --workers "
            f"({args.workers} jendela Chrome)."
        )

    prepare_nltk()

    newspaper_network.USER_AGENT = USER_AGENT

    success_csv, failed_csv = create_output_paths()

    print("=" * 80)
    print("TABANAN NEWS RSS SCRAPER (MULTI-THREAD)")
    print("=" * 80)
    print(f"[CONFIG] RSS URL       : {args.rss}")
    print(f"[CONFIG] Max results   : {args.max_results}")
    print(f"[CONFIG] Workers       : {args.workers}")
    print(f"[CONFIG] Selenium      : {not args.no_selenium}")
    print(f"[CONFIG] Show browser  : {args.show_browser}")
    print(f"[CONFIG] Delay         : {args.delay} detik")
    print()

    print(f"[RSS] Fetching: {args.rss}")

    try:
        feed = feedparser.parse(args.rss)
    except Exception as error:
        print(f"[ERROR] RSS gagal diproses: {error}")
        sys.exit(1)

    if getattr(feed, "bozo", False):
        print(
            "[WARNING] RSS memiliki kemungkinan format "
            "yang tidak sempurna."
        )

        if getattr(feed, "bozo_exception", None):
            print(f"[WARNING] Detail RSS: {feed.bozo_exception}")

    entries = feed.entries[: args.max_results]

    if not entries:
        print("[ERROR] Tidak ada artikel ditemukan dari RSS.")
        sys.exit(1)

    total = len(entries)
    workers = min(args.workers, total)

    print(f"[RSS] Processing {total} articles with {workers} workers.")
    print()

    pool = None if args.no_selenium else DriverPool(args.show_browser)

    # Hasil disimpan per index agar urutan CSV sama dengan urutan RSS.
    results: dict[int, Tuple[str, dict]] = {}

    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="scraper",
    )

    try:
        futures = {
            executor.submit(
                process_entry,
                index,
                total,
                entry,
                args,
                pool,
            ): index
            for index, entry in enumerate(entries, 1)
        }

        for future in as_completed(futures):
            index = futures[future]

            try:
                results[index] = future.result()
            except Exception as error:
                entry = entries[index - 1]

                results[index] = (
                    "failed",
                    {
                        "title": entry.get("title", "(no title)"),
                        "google_news_link": entry.get("link", ""),
                        "reason": f"Unexpected error: {error}",
                    },
                )

                log([f"[{index}/{total}] ✗ Unexpected error: {error}"])

    except KeyboardInterrupt:
        print("\n[STOP] Proses dihentikan oleh pengguna.")
        executor.shutdown(wait=False, cancel_futures=True)

    except Exception as error:
        print(f"\n[ERROR] Unexpected error: {error}")
        executor.shutdown(wait=False, cancel_futures=True)

    finally:
        executor.shutdown(wait=True, cancel_futures=True)

        if pool is not None:
            pool.close_all()

    success_rows = [
        results[i][1] for i in sorted(results) if results[i][0] == "success"
    ]
    failed_rows = [
        results[i][1] for i in sorted(results) if results[i][0] == "failed"
    ]

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    try:
        write_csv(success_csv, SUCCESS_FIELDS, success_rows)
        write_csv(failed_csv, FAILED_FIELDS, failed_rows)

    except Exception as error:
        print(f"[ERROR] Gagal menyimpan CSV: {error}")
        sys.exit(1)

    write_github_output(success_csv, failed_csv)

    print()
    print("=" * 80)
    print("SELESAI")
    print("=" * 80)
    print(f"Success : {len(success_rows)}")
    print(f"Failed  : {len(failed_rows)}")
    print(f"Success CSV: {success_csv}")
    print(f"Failed CSV : {failed_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
