"""
Utility helpers for downloading MangaDex chapters.

Usage:
    python api_mangadex.py --manga 87ed0cea-cd84-40fe-ad94-71e9257536f8 --limit 5
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Iterable, List, Optional

import requests

BASE_URL = "https://api.mangadex.org"
DEFAULT_LANGS = ("en",)
DEFAULT_CONTENT_RATINGS = ("safe", "suggestive", "erotica")


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Translate-Manga-en-vi/1.0 (+https://github.com/)",
            "Accept": "application/json",
        }
    )
    return session


def fetch_chapters(
    manga_id: str,
    *,
    languages: Iterable[str] = DEFAULT_LANGS,
    content_ratings: Iterable[str] = DEFAULT_CONTENT_RATINGS,
    limit: Optional[int] = None,
    session: Optional[requests.Session] = None,
) -> List[dict]:
    """Return a list of chapter objects for the given manga."""
    owns_session = session is None
    session = session or _session()
    page_limit = 100
    collected: List[dict] = []
    for offset in itertools.count(0, page_limit):
        params = {
            "limit": page_limit,
            "offset": offset,
            "order[chapter]": "asc",
            "translatedLanguage[]": list(languages),
            "contentRating[]": list(content_ratings),
        }
        resp = session.get(f"{BASE_URL}/manga/{manga_id}/feed", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if not data:
            break

        collected.extend(data)
        if limit is not None and len(collected) >= limit:
            break

        total = payload.get("total")
        if total is not None and offset + page_limit >= total:
            break

    if limit is not None and len(collected) > limit:
        collected = collected[:limit]

    try:
        return collected
    finally:
        if owns_session:
            session.close()


def download_chapter(
    chapter_id: str,
    destination: Path,
    *,
    use_data_saver: bool = True,
    retries: int = 3,
    session: Optional[requests.Session] = None,
) -> Path:
    """Download a single chapter and return the directory path."""
    owns_session = session is None
    session = session or _session()
    resp = session.get(f"{BASE_URL}/at-home/server/{chapter_id}", timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    base_url = payload["baseUrl"]
    chapter_info = payload["chapter"]
    file_list = chapter_info["dataSaver"] if use_data_saver else chapter_info["data"]
    quality_segment = "data-saver" if use_data_saver else "data"
    hash_code = chapter_info["hash"]

    destination.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(file_list, start=1):
        url = f"{base_url}/{quality_segment}/{hash_code}/{filename}"

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                page_resp = session.get(url, timeout=30)
                page_resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt == retries:
                    raise
            else:
                ext = Path(filename).suffix or ".jpg"
                page_path = destination / f"{index:03}{ext}"
                page_path.write_bytes(page_resp.content)
                break
        if last_err and retries > 1:
            print(f"    retrying page {index}: {last_err}")

    try:
        return destination
    finally:
        if owns_session:
            session.close()


def sanitize_for_fs(value: Optional[str], fallback: str, *, max_length: int = 80) -> str:
    if not value:
        return fallback
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_", ".") else "_" for c in value)
    safe = safe.strip() or fallback
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip(" _-.")
    return safe or fallback


def fetch_manga_title(
    manga_id: str,
    *,
    prefer_langs: Iterable[str] = ("en", "ja-ro", "ja", "vi"),
    session: Optional[requests.Session] = None,
) -> str:
    owns_session = session is None
    session = session or _session()
    try:
        resp = session.get(f"{BASE_URL}/manga/{manga_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        attributes = data.get("attributes", {})

        title_map = attributes.get("title") or {}
        alt_titles = attributes.get("altTitles") or []

        for lang in prefer_langs:
            if lang in title_map:
                return title_map[lang]
            for alt in alt_titles:
                if lang in alt:
                    return alt[lang]

        if title_map:
            return next(iter(title_map.values()))
        for alt in alt_titles:
            if alt:
                return next(iter(alt.values()))
    finally:
        if owns_session:
            session.close()

    return manga_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Download chapters from MangaDex.")
    parser.add_argument("--manga", required=True, help="Manga UUID from mangadex.org")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of chapters to download (default: all available)",
    )
    parser.add_argument(
        "--lang",
        nargs="*",
        default=DEFAULT_LANGS,
        help="Translated language codes to include (default: en)",
    )
    parser.add_argument(
        "--dest",
        default="downloads",
        help="Destination root folder (default: ./downloads)",
    )
    parser.add_argument(
        "--full-quality",
        action="store_true",
        help="Download full quality images instead of data saver.",
    )

    args = parser.parse_args()

    session = _session()
    try:
        manga_title = fetch_manga_title(args.manga, session=session)
        sanitized_title = sanitize_for_fs(manga_title, args.manga, max_length=64)

        chapters = fetch_chapters(
            args.manga, languages=args.lang, limit=args.limit, session=session
        )
        if not chapters:
            print("No chapters found with the provided filters.")
            return

        dest_root = Path(args.dest).resolve() / sanitized_title
        dest_root.mkdir(parents=True, exist_ok=True)

        total_chapters = len(chapters)
        print(
            f"Found {total_chapters} chapters for '{manga_title}'. Downloading to {dest_root} ..."
        )

        for idx, chapter in enumerate(chapters, start=1):
            chapter_id = chapter["id"]
            attributes = chapter.get("attributes", {})
            chapter_num = sanitize_for_fs(
                attributes.get("chapter"), chapter_id, max_length=16
            )
            volume_dir = sanitize_for_fs(
                attributes.get("volume"), "vol", max_length=12
            )
            title = sanitize_for_fs(attributes.get("title"), "", max_length=40)

            chapter_dir_name = sanitize_for_fs(
                " ".join(filter(None, [f"ch_{chapter_num}", title])),
                f"ch_{chapter_num}",
                max_length=64,
            )
            chapter_folder = dest_root / volume_dir / chapter_dir_name

            print(f"[{idx}/{total_chapters}] Downloading chapter {chapter_num} -> {chapter_folder}")
            try:
                download_chapter(
                    chapter_id,
                    chapter_folder,
                    use_data_saver=not args.full_quality,
                    session=session,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  !! Failed to download chapter {chapter_num}: {exc}")
            else:
                print(f"  ✔ Saved {len(os.listdir(chapter_folder))} pages")
    finally:
        session.close()


if __name__ == "__main__":
    main()
    
# RUN
# python api_mangadex.py --manga {manga_id} 