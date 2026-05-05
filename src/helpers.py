"""Low-level HTTP, NCBI/BLAST utilities (shared by biomedical tools)."""

from __future__ import annotations

import io
import json
import os
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from urllib3.exceptions import MaxRetryError, ProxyError

load_dotenv()

# Transient failures (rate limits, overloaded gateways). NCBI Datasets sometimes returns 400 briefly.
_RETRYABLE_HTTP_STATUS = frozenset({400, 408, 429, 500, 502, 503, 504})

bioontology_api_key = os.getenv("BIOONTOLOGY_API_KEY")
BLAST_URL = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
ENSEMBL_URL = "https://rest.ensembl.org"

# Public Datasets API (not v2alpha). Gene-by-symbol docs: datasets/v2/gene/symbol/{symbol}/taxon/{taxon}
NCBI_DATASETS_GENE_SYMBOL_BASE = "https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol"


def ncbi_gene_dataset_headers() -> dict[str, str]:
    """
    NCBI Datasets gene lookups. Omit api-key when unset — a missing env var sends api-key=null/empty,
    which NCBI often answers with HTTP 400, while the same URL in a browser works (browser sends no api-key).
    """
    h: dict[str, str] = {"Accept": "application/json"}
    key = (os.getenv("NCBI_API_KEY") or "").strip()
    if key:
        h["api-key"] = key
    return h


def is_id(s: str) -> bool:
    return s.startswith("HP:") and s[3:].isdigit()


def hpo_id(phenotype_name: str) -> Any:
    url_name = (
        f"https://ontology.jax.org/api/hp/search?q={quote(phenotype_name)}&page=0&limit=10"
    )
    response = requests.get(url_name)
    if response.status_code == 200:
        data = response.json()
        for term in data.get("terms", []):
            name = term.get("name", "")
            synonyms = term.get("synonyms", [])
            if phenotype_name.lower() in name.lower() or any(
                phenotype_name.lower() in synonym.lower() for synonym in synonyms
            ):
                return term.get("id")
    return None


def get_phenotype_id(phenotype_term: str) -> Any:
    if is_id(phenotype_term):
        return phenotype_term
    if phenotype_term.isdigit():
        return f"HP:{phenotype_term}"
    return hpo_id(phenotype_term)


def fetch_data(
    url: str,
    headers=None,
    params=None,
    max_retries: int = 5,
    delay: float = 2.0,
    timeout: float = 60.0,
):
    """GET JSON with retries. Retries common transient HTTP codes and connection errors."""
    merged: dict[str, str] = {}
    for k, v in (headers or {}).items():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        merged[str(k)] = s

    if "User-Agent" not in merged:
        merged["User-Agent"] = (
            os.getenv("HTTP_USER_AGENT")
            or "BioRAGent/1.0 (research; see https://www.ncbi.nlm.nih.gov/datasets/docs/)"
        )

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=merged, params=params, timeout=timeout)
            if response.status_code == 200:
                time.sleep(0.4)
                return response.json()
            if response.status_code in _RETRYABLE_HTTP_STATUS and attempt < max_retries - 1:
                wait = delay * (2**attempt)
                print(
                    f"HTTP {response.status_code}, backing off {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            time.sleep(0.4)
            return response.json()
        except (MaxRetryError, ProxyError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = delay * (2**attempt)
                print(f"Connection error: {e}. Retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"Request failed: {e}")
            break
        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait = delay * (2**attempt)
                print(f"Timeout: {e}. Retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"Request failed: {e}")
            break
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code in _RETRYABLE_HTTP_STATUS and attempt < max_retries - 1:
                wait = delay * (2**attempt)
                print(f"HTTP {code}, retry in {wait:.1f}s ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"Request failed: {e}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    return None


def extract_dna_sequence(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"[ACGTUNacgtun]{30,}", text)
    if not matches:
        return None
    seq = max(matches, key=len).upper().replace("U", "T")
    if len(seq) < 30:
        return None
    return seq


def submit_blast(sequence: str, database: str = "refseq_rna", program: str = "blastn") -> str:
    params = {
        "CMD": "Put",
        "PROGRAM": program,
        "DATABASE": database,
        "QUERY": sequence,
        "FORMAT_TYPE": "JSON2",
    }
    r = requests.get(BLAST_URL, params=params, timeout=60)
    r.raise_for_status()
    match = re.search(r"RID\s*=\s*([A-Z0-9-]+)", r.text)
    if not match:
        raise RuntimeError(f"Could not extract RID from BLAST response: {r.text[:500]}")
    return match.group(1)


def poll_blast_ready(rid: str, poll_seconds: int = 5, max_wait_seconds: int = 300) -> None:
    start = time.time()
    while True:
        params = {
            "CMD": "Get",
            "RID": rid,
            "FORMAT_OBJECT": "SearchInfo",
        }
        r = requests.get(BLAST_URL, params=params, timeout=60)
        r.raise_for_status()
        text = r.text

        if "Status=READY" in text:
            if (
                "ThereAreHits=yes" in text
                or "ThereAreHits = yes" in text
                or "ThereAreHits=Yes" in text
            ):
                return
            raise RuntimeError("BLAST finished but found no hits.")
        if "Status=FAILED" in text:
            raise RuntimeError("BLAST search failed.")
        if "Status=UNKNOWN" in text:
            raise RuntimeError("BLAST RID expired or unknown.")
        if time.time() - start > max_wait_seconds:
            raise TimeoutError(f"Timed out waiting for BLAST results: {text[:500]}")
        time.sleep(poll_seconds)


def fetch_blast_json(rid: str) -> Dict[str, Any]:
    params = {
        "CMD": "Get",
        "RID": rid,
        "FORMAT_TYPE": "JSON2",
    }
    r = requests.get(BLAST_URL, params=params, timeout=120)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        pass
    data = r.content
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            json_members = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_members:
                raise RuntimeError("BLAST ZIP contains no JSON file.")
            preferred = next((n for n in json_members if n.endswith("_1.json")), json_members[-1])
            with zf.open(preferred) as f:
                return json.load(f)
    raise RuntimeError("BLAST response was neither JSON nor ZIP.")


def find_hits_recursive(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "hits" and isinstance(v, list):
                    found.extend(item for item in v if isinstance(item, dict))
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return found


def extract_hit_description(hit: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    accession = None
    title = None
    if isinstance(hit.get("description"), list) and hit["description"]:
        desc0 = hit["description"][0]
        if isinstance(desc0, dict):
            accession = desc0.get("accession") or desc0.get("id")
            title = desc0.get("title")
    accession = accession or hit.get("accession")
    if not title:
        for key in ("title", "definition", "defline"):
            if key in hit and isinstance(hit[key], str):
                title = hit[key]
                break
    return accession, title


def looks_like_bad_clone_title(title: str) -> bool:
    if not title:
        return True
    t = title.lower()
    bad_terms = [
        "clone",
        "chromosome unknown",
        "unlocalized",
        "unplaced",
        "genomic contig",
        "bac",
        "complete sequence",
        "whole genome shotgun",
    ]
    return any(term in t for term in bad_terms)


def guess_gene_symbol_from_title(title: str) -> Optional[str]:
    if not title or looks_like_bad_clone_title(title):
        return None
    patterns = [
        r"\b(TR[ABDG][VJCD][A-Z0-9-]*)\b",
        r"\b(IG[HKL][VDJCMAGET][A-Z0-9-]*)\b",
        r"\(([A-Z0-9-]{2,20})\)",
        r"\b([A-Z][A-Z0-9-]{1,15})\s+(?:gene|mRNA|transcript)\b",
    ]
    for pat in patterns:
        m = re.search(pat, title)
        if m:
            symbol = m.group(1)
            if re.match(r"^(VMRC|RP\d+|CTD-|AC\d+|AL\d+|BX\d+|XM_|XR_)", symbol):
                continue
            return symbol
    return None


def choose_best_gene_like_hit(hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for hit in hits:
        _, title = extract_hit_description(hit)
        if title and not looks_like_bad_clone_title(title):
            return hit
    return hits[0] if hits else None


def fetch_gene_aliases(symbol: str) -> List[str]:
    aliases: List[str] = []
    symbol_resp = fetch_data(
        f"{NCBI_DATASETS_GENE_SYMBOL_BASE}/{symbol}/taxon/9606",
        headers=ncbi_gene_dataset_headers(),
    )
    if not symbol_resp or "reports" not in symbol_resp or not symbol_resp["reports"]:
        return aliases
    gene_id = symbol_resp["reports"][0]["gene"]["gene_id"]
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    summary_url = base_url + "esummary.fcgi"
    summary = fetch_data(summary_url, params={"db": "gene", "id": gene_id, "retmode": "json"})
    if not summary or "result" not in summary:
        return aliases
    uid_list = summary["result"].get("uids", [])
    if not uid_list:
        return aliases
    entry = summary["result"].get(uid_list[0], {})
    raw_aliases = entry.get("otheraliases", "")
    if isinstance(raw_aliases, str) and raw_aliases.strip():
        aliases.extend([a.strip() for a in raw_aliases.split(",") if a.strip()])
    for candidate in [entry.get("name"), entry.get("nomenclaturesymbol"), symbol]:
        if isinstance(candidate, str) and candidate.strip():
            aliases.append(candidate.strip())
    dedup = []
    seen = set()
    for a in aliases:
        k = a.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(a)
    return dedup
