"""
Tabanan News RSS Scraper

Fitur:
- Mengambil berita dari Google News RSS
- Selenium untuk resolve Google News redirect URL
- newspaper4k/newspaper3k untuk mengambil isi berita
- Membentuk data 5W1H
- Menyimpan CSV success dan failed ke output/success/{date} dan output/failed/{date}
- Mendukung testing lokal
- Mendukung GitHub Actions (menulis GITHUB_OUTPUT: success_csv, failed_csv)

Contoh penggunaan:

    python scraper.py
    python scraper.py --max-results 1
    python scraper.py --max-results 5 --show-browser
    python scraper.py --no-selenium
    python scraper.py --rss "https://news.google.com/rss/search?q=Tabanan"
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import feedparser
import nltk

from newspaper import Article
from newspaper import network as newspaper_network

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_RSS_URL = (
    "https://news.google.com/rss/search"
    "?q=Tabanan+when:1d&hl=id&gl=ID&ceid=ID:id"
)

DEFAULT_MAX_RESULTS = 40

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


# ============================================================================
# NLTK
# ============================================================================

def prepare_nltk() -> None:
    """
    Mengunduh resource NLTK yang diperlukan oleh newspaper.
    """

    packages = [
        "punkt",
        "punkt_tab",
    ]

    for package in packages:
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

    parts = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def get_sentences(
    text: str,
    start: int,
    end: int,
) -> str:
    """
    Mengambil kalimat dari index start sampai end.
    """

    sentences = split_sentences(text)

    return " ".join(
        sentences[start:end]
    )


def extract_source_from_title(title: str) -> str:
    """
    Mengambil nama media dari bagian akhir judul Google News.

    Contoh:
        "Berita Tabanan Hari Ini - Bali Post"

    Hasil:
        "Bali Post"
    """

    match = re.search(
        r"-\s*([^-]+)$",
        title,
    )

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

    title = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        title,
    )

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

    return (
        pub_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        + "Z"
    )


# ============================================================================
# SELENIUM
# ============================================================================

def make_driver(
    show_browser: bool = False,
) -> webdriver.Chrome:
    """
    Membuat Chrome WebDriver.

    show_browser=False:
        Chrome berjalan headless.

    show_browser=True:
        Chrome tampil secara normal untuk debugging lokal.
    """

    options = Options()

    if not show_browser:
        options.add_argument("--headless=new")

    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=id-ID")

    # User-Agent agar lebih menyerupai browser biasa.
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )

    try:
        driver = webdriver.Chrome(
            options=options,
        )

        driver.set_page_load_timeout(30)

        return driver

    except Exception as error:
        raise RuntimeError(
            "Gagal menjalankan Chrome WebDriver. "
            "Pastikan Google Chrome sudah terinstall. "
            f"Detail: {error}"
        ) from error


def resolve_url(
    driver: webdriver.Chrome,
    google_url: str,
    wait_seconds: float = 2.0,
) -> Optional[str]:
    """
    Membuka URL Google News menggunakan Selenium,
    lalu mengambil URL artikel tujuan.
    """

    if not google_url:
        return None

    try:
        driver.get(google_url)

        time.sleep(wait_seconds)

        real_url = driver.current_url.strip()

        if not real_url:
            return None

        # Jika masih berada di Google News, dianggap gagal.
        if "google.com" in real_url.lower():
            return None

        return real_url

    except Exception as error:
        print(
            f"    [WARNING] URL resolve error: {error}"
        )

        return None


# ============================================================================
# ARTICLE SCRAPER
# ============================================================================

def scrape_article(
    url: str,
) -> Tuple[str, str]:
    """
    Mengambil isi artikel dan ringkasannya menggunakan newspaper.
    """

    article = Article(
        url,
        language="id",
    )

    article.download()
    article.parse()

    # NLP digunakan untuk menghasilkan summary.
    try:
        article.nlp()
    except Exception:
        pass

    full_text = article.text or ""
    summary = article.summary or ""

    return (
        full_text.strip(),
        summary.strip(),
    )


# ============================================================================
# CSV HELPERS
# ============================================================================

def create_output_paths() -> Tuple[str, str]:
    """
    Membuat path output untuk success dan failed:

        output/success/{YYYY-MM-DD}.csv
        output/failed/{YYYY-MM-DD}.csv

    Jika file untuk tanggal yang sama sudah ada, file tersebut
    akan ditimpa (replace) oleh run berikutnya di hari yang sama.
    """

    today = datetime.now().strftime("%Y-%m-%d")

    success_dir = os.path.join(
        "output",
        "success",
    )

    failed_dir = os.path.join(
        "output",
        "failed",
    )

    os.makedirs(
        success_dir,
        exist_ok=True,
    )

    os.makedirs(
        failed_dir,
        exist_ok=True,
    )

    success_csv = os.path.join(
        success_dir,
        f"{today}.csv",
    )

    failed_csv = os.path.join(
        failed_dir,
        f"{today}.csv",
    )

    return (
        success_csv,
        failed_csv,
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


def write_github_output(
    success_csv: str,
    failed_csv: str,
) -> None:
    """
    Menulis path CSV ke file GITHUB_OUTPUT (jika berjalan di GitHub Actions)
    supaya bisa dipakai oleh step berikutnya, misalnya:

        ${{ steps.scrape.outputs.success_csv }}
        ${{ steps.scrape.outputs.failed_csv }}
    """

    github_output = os.environ.get("GITHUB_OUTPUT")

    if not github_output:
        return

    try:
        # Gunakan forward slash agar konsisten dengan URL raw.githubusercontent.com
        success_posix = success_csv.replace(os.sep, "/")
        failed_posix = failed_csv.replace(os.sep, "/")

        with open(
            github_output,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(f"success_csv={success_posix}\n")
            file.write(f"failed_csv={failed_posix}\n")

    except Exception as error:
        print(
            f"[WARNING] Gagal menulis GITHUB_OUTPUT: {error}"
        )


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

    original_title = entry.get(
        "title",
        "(no title)",
    )

    cleaned_title = clean_title(
        original_title,
    )

    what = (
        get_sentences(
            full_text,
            0,
            2,
        ).strip()
        or summary.strip()
    )

    why = get_sentences(
        full_text,
        2,
        4,
    )

    prefix = format_how_prefix(
        pub_dt,
    )

    how = (
        f"{prefix}\n\n"
        f"{full_text.strip()}"
    )

    return {
        "title": cleaned_title,
        "when_": format_date_iso(pub_dt),
        "where": FIXED["where"],
        "who": original_title,
        "what": what,
        "why": why,
        "how": how,
        "source_agent_report": extract_source_from_title(
            original_title,
        ),
        "location_name": FIXED["location_name"],
        "latitude": FIXED["latitude"],
        "longitude": FIXED["longitude"],
        "category": FIXED["category"],
        "data_of_information": FIXED[
            "data_of_information"
        ],
        "hashtags": FIXED["hashtags"],
    }


# ============================================================================
# ARGUMENTS
# ============================================================================

def parse_arguments():
    """
    Membaca argument command line.
    """

    parser = argparse.ArgumentParser(
        description="Tabanan News RSS Scraper",
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
        help="Jeda antarartikel dalam detik",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    args = parse_arguments()

    if args.max_results < 1:
        print(
            "[ERROR] --max-results harus lebih besar dari 0."
        )

        sys.exit(1)

    if args.delay < 0:
        print(
            "[ERROR] --delay tidak boleh negatif."
        )

        sys.exit(1)

    prepare_nltk()

    newspaper_network.USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    )

    success_csv, failed_csv = create_output_paths()

    print("=" * 80)
    print("TABANAN NEWS RSS SCRAPER")
    print("=" * 80)
    print(f"[CONFIG] RSS URL       : {args.rss}")
    print(f"[CONFIG] Max results   : {args.max_results}")
    print(f"[CONFIG] Selenium      : {not args.no_selenium}")
    print(f"[CONFIG] Show browser  : {args.show_browser}")
    print(f"[CONFIG] Delay         : {args.delay} detik")
    print()

    print(f"[RSS] Fetching: {args.rss}")

    try:
        feed = feedparser.parse(
            args.rss,
        )
    except Exception as error:
        print(
            f"[ERROR] RSS gagal diproses: {error}"
        )

        sys.exit(1)

    if getattr(feed, "bozo", False):
        print(
            "[WARNING] RSS memiliki kemungkinan format "
            "yang tidak sempurna."
        )

        if getattr(feed, "bozo_exception", None):
            print(
                f"[WARNING] Detail RSS: "
                f"{feed.bozo_exception}"
            )

    entries = feed.entries[:args.max_results]

    if not entries:
        print(
            "[ERROR] Tidak ada artikel ditemukan dari RSS."
        )

        sys.exit(1)

    print(
        f"[RSS] Processing {len(entries)} articles."
    )
    print()

    driver = None

    success_rows = []
    failed_rows = []

    try:
        if not args.no_selenium:
            print("[SELENIUM] Starting Chrome...")

            driver = make_driver(
                show_browser=args.show_browser,
            )

            print(
                "[SELENIUM] Chrome berhasil dijalankan."
            )
            print()

        for index, entry in enumerate(
            entries,
            1,
        ):
            title = entry.get(
                "title",
                "(no title)",
            )

            google_news_link = entry.get(
                "link",
                "",
            )

            print(
                f"[{index}/{len(entries)}] "
                f"{title[:100]}"
            )

            # ------------------------------------------------------------
            # Tanggal publikasi
            # ------------------------------------------------------------

            try:
                published_parsed = entry.get(
                    "published_parsed",
                )

                if not published_parsed:
                    raise ValueError(
                        "published_parsed tidak tersedia"
                    )

                pub_dt = datetime(
                    *published_parsed[:6],
                )

            except Exception as error:
                failed_rows.append(
                    {
                        "title": title,
                        "google_news_link": google_news_link,
                        "reason": f"Date error: {error}",
                    }
                )

                print(
                    "    ✗ Gagal: tanggal tidak valid"
                )

                continue

            # ------------------------------------------------------------
            # Resolve URL
            # ------------------------------------------------------------

            if args.no_selenium:
                resolved_url = google_news_link

                print(
                    "    [INFO] Selenium dinonaktifkan; "
                    "menggunakan link RSS langsung."
                )

            else:
                resolved_url = resolve_url(
                    driver,
                    google_news_link,
                )

            if not resolved_url:
                failed_rows.append(
                    {
                        "title": title,
                        "google_news_link": google_news_link,
                        "reason": "URL resolve failed",
                    }
                )

                print(
                    "    ✗ Gagal: URL artikel tidak ditemukan"
                )

                continue

            print(
                f"    [URL] {resolved_url[:150]}"
            )

            # ------------------------------------------------------------
            # Scrape artikel
            # ------------------------------------------------------------

            try:
                full_text, summary = scrape_article(
                    resolved_url,
                )

            except Exception as error:
                failed_rows.append(
                    {
                        "title": title,
                        "google_news_link": google_news_link,
                        "reason": f"Scrape failed: {error}",
                    }
                )

                print(
                    f"    ✗ Scrape gagal: {error}"
                )

                continue

            # ------------------------------------------------------------
            # Validasi isi artikel
            # ------------------------------------------------------------

            if (
                not full_text
                or len(full_text.strip()) < 50
            ):
                failed_rows.append(
                    {
                        "title": title,
                        "google_news_link": google_news_link,
                        "reason": (
                            "Empty 'how' "
                            "(No article body text found)"
                        ),
                    }
                )

                print(
                    "    ✗ Skipped: No body content"
                )

                continue

            # ------------------------------------------------------------
            # Build success row
            # ------------------------------------------------------------

            row = build_success_row(
                entry,
                pub_dt,
                full_text,
                summary,
            )

            success_rows.append(
                row,
            )

            print(
                "    ✓ Success"
            )

            if args.delay > 0:
                time.sleep(
                    args.delay,
                )

    except KeyboardInterrupt:
        print(
            "\n[STOP] Proses dihentikan oleh pengguna."
        )

    except Exception as error:
        print(
            f"\n[ERROR] Unexpected error: {error}"
        )

    finally:
        if driver is not None:
            try:
                driver.quit()

                print(
                    "[SELENIUM] Chrome ditutup."
                )

            except Exception as error:
                print(
                    f"[WARNING] Gagal menutup Chrome: {error}"
                )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    try:
        write_csv(
            success_csv,
            SUCCESS_FIELDS,
            success_rows,
        )

        write_csv(
            failed_csv,
            FAILED_FIELDS,
            failed_rows,
        )

    except Exception as error:
        print(
            f"[ERROR] Gagal menyimpan CSV: {error}"
        )

        sys.exit(1)

    # Tulis output untuk GitHub Actions (jika berjalan di sana)
    write_github_output(
        success_csv,
        failed_csv,
    )

    print()
    print("=" * 80)
    print("SELESAI")
    print("=" * 80)
    print(
        f"Success : {len(success_rows)}"
    )
    print(
        f"Failed  : {len(failed_rows)}"
    )
    print(
        f"Success CSV: {success_csv}"
    )
    print(
        f"Failed CSV : {failed_csv}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
