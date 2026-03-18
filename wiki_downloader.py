#!/usr/bin/env python3
"""
Simple MediaWiki content downloader.
Works with any MediaWiki-based wiki (Bulbapedia, Fandom wikis, etc.)
by changing BASE_URL and API_ENDPOINT.
"""

import json
import re
import requests
import argparse
from pathlib import Path

# Pokemon
BASE_URL = "https://onepiece.fandom.com"
API_ENDPOINT = f"{BASE_URL}/api.php"

def _strip_html(html: str) -> str:
    """Minimal HTML tag removal to get readable plain text from parsed output."""
    # Remove <style> and <script> blocks entirely
    text = re.sub(r"<(style|script)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
    # Remove [edit] / [modifier] links
    text = re.sub(r"\[(?:edit|modifier)\]", "", text)
    # Replace <br>, closing block tags with newlines
    text = re.sub(r"<br\s*/?>|</(?:p|div|li|tr|h[1-6])>", "\n", text)
    # Replace </td> and </th> with tab (for tables)
    text = re.sub(r"</(?:td|th)>", "\t", text)
    # Remove all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#039;", "'")
        .replace("&nbsp;", " ")
    )
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_page(title: str) -> dict:
    """Fetch a page via action=parse (works on all MediaWiki sites)."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|wikitext",
        "format": "json",
    }
    resp = requests.get(API_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        return {"title": title, "html": "", "wikitext": ""}
    return {
        "title": data["parse"]["title"],
        "html": data["parse"]["text"]["*"],
        "wikitext": data["parse"]["wikitext"]["*"],
    }


def get_page(title: str, fmt: str = "text") -> dict:
    """Fetch a single wiki page by title.

    Args:
        title: Page title (e.g. "Pikachu", "Goku").
        fmt: "text" for plain text, "html" for parsed HTML, "wikitext" for raw markup.
    """
    if fmt == "html":
        parsed = _parse_page(title)
        return {"title": parsed["title"], "content": parsed["html"], "format": "html"}

    if fmt == "wikitext":
        parsed = _parse_page(title)
        return {"title": parsed["title"], "content": parsed["wikitext"], "format": "wikitext"}

    # --- Plain text: try extracts first, fall back to parse + strip ---
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    resp = requests.get(API_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    content = page.get("extract", "")

    # Fallback: if extracts returned nothing, use action=parse and strip HTML
    if not content.strip():
        parsed = _parse_page(title)
        content = _strip_html(parsed["html"])
        return {"title": parsed["title"], "content": content, "format": "text"}

    return {
        "title": page.get("title", title),
        "content": content,
        "format": "text",
    }


def search_pages(query: str, limit: int = 10) -> list[dict]:
    """Search for pages matching a query string."""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }
    resp = requests.get(API_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()["query"]["search"]


def get_category_members(category: str, limit: int = 50) -> list[str]:
    """List page titles in a category (without 'Category:' prefix)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "format": "json",
    }
    resp = requests.get(API_ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    return [m["title"] for m in resp.json()["query"]["categorymembers"]]


def save(content: str, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(content, encoding="utf-8")
    print(f"Saved → {filepath}  ({len(content):,} chars)")


def main():
    parser = argparse.ArgumentParser(description="Download content from a MediaWiki")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- page ---
    p_page = sub.add_parser("page", help="Download a single page")
    p_page.add_argument("title", help="Page title, e.g. 'Pikachu'")
    p_page.add_argument("--format", choices=["text", "html", "wikitext"], default="text")
    p_page.add_argument("-o", "--output", help="Save to file instead of printing")

    # --- search ---
    p_search = sub.add_parser("search", help="Search for pages")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", "--limit", type=int, default=10)

    # --- category ---
    p_cat = sub.add_parser("category", help="List pages in a category")
    p_cat.add_argument("name", help="Category name (without 'Category:' prefix)")
    p_cat.add_argument("-n", "--limit", type=int, default=50)

    # --- bulk ---
    p_bulk = sub.add_parser("bulk", help="Download multiple pages (one file each)")
    p_bulk.add_argument("titles", nargs="+", help="Page titles")
    p_bulk.add_argument("--format", choices=["text", "html", "wikitext"], default="text")
    p_bulk.add_argument("-d", "--dir", default="wiki_pages", help="Output directory")

    args = parser.parse_args()

    if args.command == "page":
        result = get_page(args.title, args.format)
        if args.output:
            save(result["content"], args.output)
        else:
            print(result["content"])

    elif args.command == "search":
        results = search_pages(args.query, args.limit)
        for r in results:
            print(f"  {r['title']:30s}  (id={r['pageid']})")

    elif args.command == "category":
        titles = get_category_members(args.name, args.limit)
        for t in titles:
            print(f"  {t}")

    elif args.command == "bulk":
        ext = {"text": ".txt", "html": ".html", "wikitext": ".wiki"}[args.format]
        for title in args.titles:
            result = get_page(title, args.format)
            slug = title.replace(" ", "_").replace("/", "_")
            save(result["content"], f"{args.dir}/{slug}{ext}")


if __name__ == "__main__":
    main()
