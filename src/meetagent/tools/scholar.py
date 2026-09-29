"""Search real scholarly records using OpenAlex, Semantic Scholar, and arXiv."""

from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Protocol


class ScholarSearch(Protocol):
    async def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        """Return real records with title, year, and resolvable source URLs."""


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "MeetAgent/0.1 (local meeting research assistant)", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAlexSearch:
    """OpenAlex works search. An email can be supplied for its polite pool."""

    def search_sync(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        params = {
            "search": query,
            "per-page": str(max(1, min(limit, 25))),
            "select": "id,display_name,publication_year,doi,primary_location",
        }
        email = os.environ.get("OPENALEX_EMAIL")
        if email:
            params["mailto"] = email
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        payload = _get_json(url)
        records: list[dict[str, str]] = []
        for item in payload.get("results", []):
            title = str(item.get("display_name") or "").strip()
            identifier = str(item.get("id") or "")
            if not title or not identifier:
                continue
            doi = str(item.get("doi") or "")
            location = item.get("primary_location") or {}
            source_url = doi or (location.get("landing_page_url") if isinstance(location, dict) else None) or identifier
            records.append({
                "title": title,
                "year": str(item.get("publication_year") or ""),
                "url": str(source_url),
                "doi": doi.removeprefix("https://doi.org/") if doi else "",
                "source": "OpenAlex",
            })
        return records


class SemanticScholarSearch:
    """Semantic Scholar search; honors an optional API key if one is configured."""

    def search_sync(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({
            "query": query,
            "limit": str(max(1, min(limit, 25))),
            "fields": "title,year,externalIds,url",
        })
        headers = {}
        key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if key:
            headers["x-api-key"] = key
        payload = _get_json(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", headers)
        records: list[dict[str, str]] = []
        for item in payload.get("data", []):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            ids = item.get("externalIds") or {}
            doi = str(ids.get("DOI") or "")
            arxiv_id = str(ids.get("ArXiv") or "")
            url = (f"https://doi.org/{doi}" if doi else
                   f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else
                   str(item.get("url") or ""))
            records.append({
                "title": title,
                "year": str(item.get("year") or ""),
                "url": url,
                "doi": doi,
                "source": "Semantic Scholar",
            })
        return records


class ArxivSearch:
    """arXiv Atom API search."""

    def search_sync(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        params = urllib.parse.urlencode({"search_query": f"all:{query}", "start": "0", "max_results": str(max(1, min(limit, 25)))})
        request = urllib.request.Request(
            f"https://export.arxiv.org/api/query?{params}",
            headers={"User-Agent": "MeetAgent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            root = ET.fromstring(response.read())
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        records: list[dict[str, str]] = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns)).split())
            paper_url = entry.findtext("atom:id", default="", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns)
            match = re.search(r"\d{4}", published)
            if title and paper_url:
                records.append({
                    "title": title,
                    "year": match.group(0) if match else "",
                    "url": paper_url,
                    "doi": "",
                    "source": "arXiv",
                })
        return records


class CompositeScholarSearch:
    """Query scholarly indexes in parallel and deduplicate by DOI/title."""

    def __init__(self) -> None:
        self.backends = (OpenAlexSearch(), SemanticScholarSearch(), ArxivSearch())

    async def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        if not query.strip():
            return []
        calls = [asyncio.to_thread(backend.search_sync, query, limit) for backend in self.backends]
        responses = await asyncio.gather(*calls, return_exceptions=True)
        records: list[dict[str, str]] = []
        errors: list[str] = []
        for result in responses:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            records.extend(result)
        unique: dict[str, dict[str, str]] = {}
        for record in records:
            key = record.get("doi", "").casefold() or re.sub(r"\W+", "", record["title"].casefold())
            current = unique.get(key)
            if current is None or (not current.get("doi") and record.get("doi")):
                unique[key] = record
        if not unique and errors:
            raise RuntimeError("论文检索服务当前都不可用：" + "；".join(errors))
        return list(unique.values())[:max(1, min(limit, 25))]
