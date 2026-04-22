"""
Multi-Source Data Scraper
Scrapes 4chan, Reddit, and Wikipedia and dumps everything into ./data/

Usage:
    py scrape.py                          # all three sources, defaults
    py scrape.py --sources 4chan reddit   # skip wikipedia
    py scrape.py --sources wikipedia      # just wikipedia
    py scrape.py --reddit-subs pol news askreddit worldnews
    py scrape.py --wiki-topics "World War 2" "United States" "China"
    py scrape.py --4chan-board pol --4chan-limit 3000
"""

import os
import re
import time
import json
import argparse
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

# ──────────────────────────────────────────────
# HTTP Session
# ──────────────────────────────────────────────
def make_session(extra_headers: dict = None) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    headers = {"User-Agent": "multi-scraper/1.0 (AI training data collector)"}
    if extra_headers:
        headers.update(extra_headers)
    session.headers.update(headers)
    return session


# ══════════════════════════════════════════════
# 4CHAN SCRAPER
# ══════════════════════════════════════════════
BASE_4CHAN = "https://a.4cdn.org"

def clean_4chan_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<wbr\s*/?>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    for k, v in {"&gt;": ">", "&lt;": "<", "&amp;": "&", "&quot;": '"', "&#039;": "'", "&apos;": "'"}.items():
        text = text.replace(k, v)
    return text.strip()


def format_4chan_thread(posts: list) -> str:
    if not posts:
        return ""
    lines = []
    op = posts[0]
    if op.get("sub"):
        lines.append(f"[THREAD: {op['sub']}]")
    lines.append(f"[OP #{op['no']}]")
    body = clean_4chan_text(op.get("com", ""))
    if body:
        lines.append(body)
    for post in posts[1:]:
        body = clean_4chan_text(post.get("com", ""))
        if body:
            lines.append(f"[#{post['no']}]")
            lines.append(body)
    lines.append("")
    return "\n".join(lines)


def scrape_4chan(board: str, limit: int, delay: float):
    session = make_session()
    log.info(f"[4chan] Starting /{board}/ scrape (limit={limit})")

    all_ids = []
    try:
        r = session.get(f"{BASE_4CHAN}/{board}/archive.json", timeout=15)
        r.raise_for_status()
        all_ids.extend(reversed(r.json()))
        log.info(f"[4chan] {len(all_ids):,} archived threads")
    except Exception:
        log.info("[4chan] No archive available, using catalog only")

    try:
        r = session.get(f"{BASE_4CHAN}/{board}/catalog.json", timeout=15)
        r.raise_for_status()
        for page in r.json():
            for t in page.get("threads", []):
                all_ids.append(t["no"])
        log.info(f"[4chan] +live threads, total: {len(all_ids):,}")
    except Exception as e:
        log.warning(f"[4chan] Catalog fetch failed: {e}")

    seen = set()
    unique_ids = [x for x in all_ids if not (x in seen or seen.add(x))]
    target = unique_ids[:limit]

    out_path = DATA_DIR / f"4chan_{board}.txt"
    scraped = skipped = total_posts = total_chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, tid in enumerate(target):
            try:
                r = session.get(f"{BASE_4CHAN}/{board}/thread/{tid}.json", timeout=15)
                if r.status_code == 404:
                    skipped += 1
                    continue
                r.raise_for_status()
                posts = r.json().get("posts", [])
                text = format_4chan_thread(posts)
                if text.strip():
                    f.write(text + "\n")
                    scraped += 1
                    total_posts += len(posts)
                    total_chars += len(text)
                if (i + 1) % 100 == 0:
                    log.info(f"[4chan] {i+1}/{len(target)} | saved={scraped} | {total_chars/1024:.0f} KB")
                time.sleep(delay)
            except KeyboardInterrupt:
                log.info("[4chan] Interrupted — saving progress")
                break
            except Exception as e:
                log.warning(f"[4chan] Thread {tid} failed: {e}")
                skipped += 1
                time.sleep(delay * 2)

    size = out_path.stat().st_size / 1024
    log.info(f"[4chan] Done → {out_path.name} | threads={scraped:,} | posts={total_posts:,} | {size:.0f} KB")


# ══════════════════════════════════════════════
# REDDIT SCRAPER
# ══════════════════════════════════════════════
REDDIT_BASE = "https://www.reddit.com"

def format_reddit_thread(post: dict, comments: list) -> str:
    lines = []
    title = post.get("title", "")
    selftext = post.get("selftext", "").strip()

    if title:
        lines.append(f"[THREAD: {title}]")
    if selftext and selftext not in ("[deleted]", "[removed]"):
        lines.append(f"[OP] {selftext}")

    def extract_comments(items, depth=0):
        for item in items:
            if not isinstance(item, dict):
                continue
            data = item.get("data", {})
            body = data.get("body", "").strip()
            if body and body not in ("[deleted]", "[removed]"):
                indent = "  " * depth
                lines.append(f"{indent}[comment] {body}")
            replies = data.get("replies", {})
            if isinstance(replies, dict):
                children = replies.get("data", {}).get("children", [])
                extract_comments([c for c in children if c.get("kind") == "t1"], depth + 1)

    extract_comments([c for c in comments if c.get("kind") == "t1"])
    lines.append("")
    return "\n".join(lines)


def scrape_reddit(subreddits: list, posts_per_sub: int, delay: float):
    session = make_session({"User-Agent": "multi-scraper/1.0 (data collection bot)"})
    log.info(f"[Reddit] Scraping {len(subreddits)} subreddits: {', '.join(subreddits)}")

    for sub in subreddits:
        log.info(f"[Reddit] r/{sub}")
        out_path = DATA_DIR / f"reddit_{sub}.txt"
        scraped = total_chars = 0

        with open(out_path, "w", encoding="utf-8") as f:
            after = None
            fetched = 0

            while fetched < posts_per_sub:
                batch = min(100, posts_per_sub - fetched)
                params = {"limit": batch, "raw_json": 1, "t": "all"}
                if after:
                    params["after"] = after

                try:
                    r = session.get(
                        f"{REDDIT_BASE}/r/{sub}/top.json",
                        params=params,
                        timeout=15,
                    )
                    if r.status_code == 403:
                        log.warning(f"[Reddit] r/{sub} is private or banned, skipping")
                        break
                    r.raise_for_status()
                    data = r.json().get("data", {})
                    posts = data.get("children", [])
                    after = data.get("after")

                    if not posts:
                        break

                    for post_wrap in posts:
                        post = post_wrap.get("data", {})
                        post_id = post.get("id")
                        permalink = post.get("permalink", "")
                        if not post_id:
                            continue

                        try:
                            time.sleep(delay)
                            cr = session.get(
                                f"{REDDIT_BASE}{permalink}.json",
                                params={"raw_json": 1, "limit": 200},
                                timeout=15,
                            )
                            cr.raise_for_status()
                            cdata = cr.json()
                            if len(cdata) >= 2:
                                comments = cdata[1].get("data", {}).get("children", [])
                                text = format_reddit_thread(post, comments)
                                if text.strip():
                                    f.write(text + "\n")
                                    scraped += 1
                                    total_chars += len(text)
                        except Exception as e:
                            log.warning(f"[Reddit] Comment fetch failed for {post_id}: {e}")

                    fetched += len(posts)
                    log.info(f"[Reddit] r/{sub} | fetched={fetched} | saved={scraped} | {total_chars/1024:.0f} KB")

                    if not after:
                        break
                    time.sleep(delay)

                except KeyboardInterrupt:
                    log.info("[Reddit] Interrupted — saving progress")
                    break
                except Exception as e:
                    log.warning(f"[Reddit] r/{sub} batch failed: {e}")
                    time.sleep(delay * 3)
                    break

        size = out_path.stat().st_size / 1024
        log.info(f"[Reddit] Done r/{sub} → {out_path.name} | posts={scraped:,} | {size:.0f} KB")


# ══════════════════════════════════════════════
# WIKIPEDIA SCRAPER
# ══════════════════════════════════════════════
WIKI_API = "https://en.wikipedia.org/w/api.php"

def get_wiki_article(session: requests.Session, title: str) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
        "redirects": True,
    }
    r = session.get(WIKI_API, params=params, timeout=15)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" not in page:
            return page.get("extract", "")
    return ""


def get_wiki_category_members(session: requests.Session, category: str, limit: int) -> list:
    titles = []
    cmcontinue = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(500, limit - len(titles)),
            "cmtype": "page",
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        r = session.get(WIKI_API, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return titles


def get_wiki_random_titles(session: requests.Session, count: int) -> list:
    titles = []
    while len(titles) < count:
        batch = min(500, count - len(titles))
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnlimit": batch,
            "format": "json",
        }
        r = session.get(WIKI_API, params=params, timeout=15)
        r.raise_for_status()
        items = r.json().get("query", {}).get("random", [])
        titles.extend(i["title"] for i in items)
        if len(items) < batch:
            break
    return titles


def scrape_wikipedia(topics: list, articles_per_topic: int, random_articles: int, delay: float):
    session = make_session({"User-Agent": "multi-scraper/1.0 (AI training; contact: local)"})
    log.info(f"[Wikipedia] Starting scrape")

    all_titles = []

    for topic in topics:
        log.info(f"[Wikipedia] Searching topic: {topic}")
        try:
            members = get_wiki_category_members(session, topic, articles_per_topic)
            if members:
                all_titles.extend(members[:articles_per_topic])
                log.info(f"[Wikipedia] Category '{topic}': {len(members)} articles")
            else:
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": topic,
                    "srlimit": articles_per_topic,
                    "format": "json",
                }
                r = session.get(WIKI_API, params=params, timeout=15)
                r.raise_for_status()
                results = r.json().get("query", {}).get("search", [])
                titles = [res["title"] for res in results]
                all_titles.extend(titles)
                log.info(f"[Wikipedia] Search '{topic}': {len(titles)} results")
        except Exception as e:
            log.warning(f"[Wikipedia] Topic '{topic}' failed: {e}")
        time.sleep(delay)

    if random_articles > 0:
        log.info(f"[Wikipedia] Fetching {random_articles} random articles")
        try:
            rand = get_wiki_random_titles(session, random_articles)
            all_titles.extend(rand)
            log.info(f"[Wikipedia] Got {len(rand)} random titles")
        except Exception as e:
            log.warning(f"[Wikipedia] Random fetch failed: {e}")

    seen = set()
    unique = [t for t in all_titles if not (t in seen or seen.add(t))]
    log.info(f"[Wikipedia] Total unique articles to fetch: {len(unique):,}")

    out_path = DATA_DIR / "wikipedia.txt"
    fetched = skipped = total_chars = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i, title in enumerate(unique):
            try:
                text = get_wiki_article(session, title)
                if text and len(text) > 200:
                    f.write(f"[ARTICLE: {title}]\n{text}\n\n")
                    fetched += 1
                    total_chars += len(text)
                else:
                    skipped += 1

                if (i + 1) % 50 == 0:
                    log.info(f"[Wikipedia] {i+1}/{len(unique)} | saved={fetched} | {total_chars/1024:.0f} KB")

                time.sleep(delay)
            except KeyboardInterrupt:
                log.info("[Wikipedia] Interrupted — saving progress")
                break
            except Exception as e:
                log.warning(f"[Wikipedia] '{title}' failed: {e}")
                skipped += 1
                time.sleep(delay * 2)

    size = out_path.stat().st_size / 1024
    log.info(f"[Wikipedia] Done → {out_path.name} | articles={fetched:,} | skipped={skipped} | {size:.0f} KB")


# ══════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Scrape 4chan, Reddit, Wikipedia into ./data/")

    parser.add_argument(
        "--sources",
        nargs="+",
        default=["4chan", "reddit", "wikipedia"],
        choices=["4chan", "reddit", "wikipedia"],
        help="Which sources to scrape (default: all three)",
    )

    # 4chan
    parser.add_argument("--4chan-board", default="pol", dest="chan_board")
    parser.add_argument("--4chan-limit", type=int, default=3000, dest="chan_limit")

    # Reddit
    parser.add_argument(
        "--reddit-subs",
        nargs="+",
        default=["worldnews", "news", "politics", "conspiracy", "history", "science", "technology"],
        dest="reddit_subs",
    )
    parser.add_argument("--reddit-posts", type=int, default=500, dest="reddit_posts")

    # Wikipedia
    parser.add_argument(
        "--wiki-topics",
        nargs="+",
        default=["History", "Science", "Politics", "Technology", "Philosophy", "Economics"],
        dest="wiki_topics",
    )
    parser.add_argument("--wiki-per-topic", type=int, default=200, dest="wiki_per_topic")
    parser.add_argument("--wiki-random", type=int, default=500, dest="wiki_random")

    # General
    parser.add_argument("--delay", type=float, default=1.0)

    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("═" * 50)
    log.info(f"Sources : {', '.join(args.sources)}")
    log.info(f"Output  : {DATA_DIR}")
    log.info(f"Delay   : {args.delay}s")
    log.info("═" * 50)

    if "4chan" in args.sources:
        log.info("\n── 4chan ──")
        scrape_4chan(args.chan_board, args.chan_limit, args.delay)

    if "reddit" in args.sources:
        log.info("\n── Reddit ──")
        scrape_reddit(args.reddit_subs, args.reddit_posts, args.delay)

    if "wikipedia" in args.sources:
        log.info("\n── Wikipedia ──")
        scrape_wikipedia(args.wiki_topics, args.wiki_per_topic, args.wiki_random, args.delay)

    log.info("\n" + "═" * 50)
    log.info("All done! Files saved to ./data/")
    total = sum(f.stat().st_size for f in DATA_DIR.glob("*.txt"))
    log.info(f"Total data: {total/1024/1024:.2f} MB")
    log.info("═" * 50)


if __name__ == "__main__":
    main()
