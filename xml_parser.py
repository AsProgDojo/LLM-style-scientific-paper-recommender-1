from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------
# Configuration: admin filters + bucket mapping
# ---------------------------------------------------------------------

ADMIN_SECTION_TITLE_SUBSTRINGS = [
    "data availability",
    "author contributions",
    "conflict of interest",
    "competing interests",
    "publisher",
    "publisher's note",
    "ethics",
    "ethical",
    "funding",
    "acknowledg",  # matches acknowledgements/acknowledgments
    "abbreviations",
    "supplementary",
    "supporting information",
    "availability of data",
    "consent",
    "trial registration",
    "guarantor",
    "disclaimer",
    "patient and public involvement",
]

BUCKET_NAMES = ("introduction", "methods", "results", "discussion", "conclusion")


# ---------------------------------------------------------------------
# Text + XML helpers
# ---------------------------------------------------------------------

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")


def strip_xml_namespace(tag_name: str) -> str:
    """Convert '{namespace}tag' -> 'tag' """
    return tag_name.split("}", 1)[1] if "}" in tag_name else tag_name


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def extract_text_content(element: ET.Element) -> str:
    """Join all text nodes inside an element and normalize whitespace."""
    return normalize_whitespace(" ".join(t for t in element.itertext() if t))


def normalize_section_title(raw_title: str) -> str:
    """Lowercase + remove punctuation + normalize spaces."""
    title_lower = (raw_title or "").lower()
    title_alpha_num = NON_ALNUM_RE.sub(" ", title_lower)
    return normalize_whitespace(title_alpha_num)


def is_admin_section_title(normalized_title: str) -> bool:
    return any(substr in normalized_title for substr in ADMIN_SECTION_TITLE_SUBSTRINGS)

def map_section_title_to_bucket(section_title: str) -> Optional[str]:
    """
    Return one of BUCKET_NAMES or None.
    Returns None for admin titles and unknown titles.
    """

    normalized_title = normalize_section_title(section_title)
    if not normalized_title or is_admin_section_title(normalized_title):
        return None

    # Introduction / background
    if (
        normalized_title == "introduction"
        or normalized_title.startswith("background")
        or normalized_title in ("related work", "literature review")
    ):
        return "introduction"

    # Methods (expanded)
    if (
        "method" in normalized_title
        or "materials and methods" in normalized_title
        or "material and methods" in normalized_title
        or "methodology" in normalized_title
        or "experimental procedures" in normalized_title
        or "study design" in normalized_title
        or "patients and methods" in normalized_title
        or "data analysis" in normalized_title
        or "statistical analysis" in normalized_title
        or "endpoint" in normalized_title
        or "endpoints" in normalized_title
        or "measure" in normalized_title
        or "measures" in normalized_title
        or "variables" in normalized_title
        or "survey" in normalized_title
        or "questionnaire" in normalized_title
    ):
        return "methods"

    # Results
    if normalized_title.startswith("results") or normalized_title in ("findings", "outcome", "outcomes") or "outcomes" in normalized_title:
        return "results"

    # Discussion
    if normalized_title.startswith("discussion") or "interpretation" in normalized_title:
        return "discussion"

    # Conclusion
    if "conclusion" in normalized_title or normalized_title.startswith("concluding") or normalized_title in ("final remarks", "summary"):
        return "conclusion"

    return None


def find_first_descendant_by_tag(root: ET.Element, tag_without_namespace: str) -> Optional[ET.Element]:
    for element in root.iter():
        if strip_xml_namespace(element.tag) == tag_without_namespace:
            return element
    return None


def find_all_descendants_by_tag(root: ET.Element, tag_without_namespace: str) -> List[ET.Element]:
    return [element for element in root.iter() if strip_xml_namespace(element.tag) == tag_without_namespace]


def find_first_element_by_tag_path(root: ET.Element, tag_path: List[str]) -> Optional[ET.Element]:
    """
    Best-effort helper: find first element matching a sequence of tags.
    Used only for a few stable front-matter locations.
    """
    current_candidates = [root]
    for required_tag in tag_path:
        next_candidates: List[ET.Element] = []
        for candidate in current_candidates:
            for element in candidate.iter():
                if strip_xml_namespace(element.tag) == required_tag:
                    next_candidates.append(element)
        if not next_candidates:
            return None
        current_candidates = next_candidates
    return current_candidates[0]


# ---------------------------------------------------------------------
# Front-matter extraction
# ---------------------------------------------------------------------

def extract_article_identifiers(article_meta_element: ET.Element) -> Dict[str, Optional[str]]:
    """
    Extract IDs from <article-meta>:
      - PMCID: pub-id-type="pmc" or "pmcid"
      - PMID: pub-id-type="pmid"
      - DOI:  pub-id-type="doi" (choose best if multiple)
    """
    identifiers: Dict[str, Optional[str]] = {"pmcid": None, "pmid": None, "doi": None}
    doi_candidates: List[str] = []

    for article_id_element in find_all_descendants_by_tag(article_meta_element, "article-id"):
        id_type = (article_id_element.attrib.get("pub-id-type") or "").lower()
        id_value = normalize_whitespace(article_id_element.text or "")
        if not id_value:
            continue

        if id_type in ("pmc", "pmcid"):
            identifiers["pmcid"] = id_value if id_value.startswith("PMC") else f"PMC{id_value}"
        elif id_type == "pmid":
            identifiers["pmid"] = id_value
        elif id_type == "doi":
            doi_candidates.append(id_value)

    if doi_candidates:
        # Prefer canonical DOI; avoid revision-like suffixes when possible
        def doi_sort_key(doi: str) -> Tuple[int, int]:
            looks_like_revision = 1 if re.search(r"\.r\d{3,}$", doi) else 0
            return (looks_like_revision, len(doi))

        doi_candidates.sort(key=doi_sort_key)
        identifiers["doi"] = doi_candidates[0]

    return identifiers


def extract_article_title(article_meta_element: ET.Element) -> Optional[str]:
    title_element = find_first_element_by_tag_path(article_meta_element, ["title-group", "article-title"])
    return extract_text_content(title_element) if title_element is not None else None


def extract_journal_metadata(front_element: ET.Element) -> Dict[str, Optional[str]]:
    journal_info = {"name": None}
    journal_meta_element = find_first_descendant_by_tag(front_element, "journal-meta")
    if journal_meta_element is None:
        return journal_info

    journal_title_element = find_first_descendant_by_tag(journal_meta_element, "journal-title")
    if journal_title_element is not None:
        journal_info["name"] = extract_text_content(journal_title_element)

    return journal_info


def extract_publication_date(article_meta_element: ET.Element) -> Tuple[Optional[int], Optional[str]]:
    pub_date_elements = [el for el in article_meta_element.iter() if strip_xml_namespace(el.tag) == "pub-date"]
    if not pub_date_elements:
        return None, None

    def pub_date_priority(pub_date_el: ET.Element) -> int:
        pub_type = (pub_date_el.attrib.get("pub-type") or "").lower()
        preferred_order = ["epub", "ppub", "pub", "collection"]
        return preferred_order.index(pub_type) if pub_type in preferred_order else 999

    pub_date_elements.sort(key=pub_date_priority)
    chosen_pub_date = pub_date_elements[0]

    year_str = month_str = day_str = None
    for child in list(chosen_pub_date):
        child_tag = strip_xml_namespace(child.tag)
        if child_tag == "year":
            year_str = normalize_whitespace(child.text or "")
        elif child_tag == "month":
            month_str = normalize_whitespace(child.text or "")
        elif child_tag == "day":
            day_str = normalize_whitespace(child.text or "")

    publication_year = int(year_str) if year_str and year_str.isdigit() else None

    publication_date = None
    if year_str:
        if month_str and day_str:
            publication_date = f"{year_str}-{month_str.zfill(2)}-{day_str.zfill(2)}"
        elif month_str:
            publication_date = f"{year_str}-{month_str.zfill(2)}"
        else:
            publication_date = year_str

    return publication_year, publication_date


def extract_keywords_list(front_element: ET.Element) -> List[str]:
    keywords: List[str] = []
    for keyword_element in find_all_descendants_by_tag(front_element, "kwd"):
        keyword_text = extract_text_content(keyword_element)
        if keyword_text:
            keywords.append(keyword_text)

    # Case-insensitive de-dup preserving order
    seen_lower = set()
    unique_keywords: List[str] = []
    for keyword in keywords:
        key = keyword.lower()
        if key not in seen_lower:
            unique_keywords.append(keyword)
            seen_lower.add(key)

    return unique_keywords


def build_affiliation_id_to_text_map(front_element: ET.Element) -> Dict[str, str]:
    affiliation_map: Dict[str, str] = {}
    for affiliation_element in find_all_descendants_by_tag(front_element, "aff"):
        affiliation_id = affiliation_element.attrib.get("id")
        if not affiliation_id:
            continue
        affiliation_text = extract_text_content(affiliation_element)
        if affiliation_text:
            affiliation_map[affiliation_id] = affiliation_text
    return affiliation_map

def extract_author_list(article_meta_element: ET.Element, affiliation_map: Dict[str, str]) -> List[Dict[str, Optional[str]]]:
    authors: List[Dict[str, Optional[str]]] = []

    for contributor_element in find_all_descendants_by_tag(article_meta_element, "contrib"):
        contributor_type = (contributor_element.attrib.get("contrib-type") or "").lower()
        if contributor_type not in ("author", ""):
            continue

        name_element = None
        for child in list(contributor_element):
            if strip_xml_namespace(child.tag) == "name":
                name_element = child
                break
        if name_element is None:
            continue

        first_name = last_name = None
        for child in list(name_element):
            tag = strip_xml_namespace(child.tag)
            if tag == "given-names":
                first_name = extract_text_content(child)
            elif tag == "surname":
                last_name = extract_text_content(child)

        affiliations_for_author: List[str] = []
        for xref_element in find_all_descendants_by_tag(contributor_element, "xref"):
            if (xref_element.attrib.get("ref-type") or "").lower() != "aff":
                continue
            affiliation_ref_id = xref_element.attrib.get("rid")
            if affiliation_ref_id and affiliation_ref_id in affiliation_map:
                affiliations_for_author.append(affiliation_map[affiliation_ref_id])

        affiliations_for_author = list(dict.fromkeys(affiliations_for_author))
        affiliation_text = "; ".join(affiliations_for_author) if affiliations_for_author else None

        authors.append(
            {
                "first_name": first_name,
                "last_name": last_name,
                "affiliation": affiliation_text,
            }
        )

    return authors



def extract_abstract_text(article_meta_element: ET.Element) -> str:
    abstract_element = None
    for element in article_meta_element.iter():
        if strip_xml_namespace(element.tag) == "abstract":
            abstract_element = element
            break
    if abstract_element is None:
        return ""

    abstract_parts: List[str] = []

    # Structured abstract: may contain <sec> blocks with titles
    for child in list(abstract_element):
        child_tag = strip_xml_namespace(child.tag)
        if child_tag == "title":
            continue

        if child_tag == "sec":
            section_title = None
            for sec_child in list(child):
                if strip_xml_namespace(sec_child.tag) == "title":
                    section_title = extract_text_content(sec_child)
                    break
            if section_title:
                abstract_parts.append(f"{section_title}:")

            for paragraph in child.iter():
                if strip_xml_namespace(paragraph.tag) == "p":
                    paragraph_text = extract_text_content(paragraph)
                    if paragraph_text:
                        abstract_parts.append(paragraph_text)

        elif child_tag == "p":
            paragraph_text = extract_text_content(child)
            if paragraph_text:
                abstract_parts.append(paragraph_text)

    if not abstract_parts:
        abstract_parts.append(extract_text_content(abstract_element))

    return normalize_whitespace(" ".join(abstract_parts))


# ---------------------------------------------------------------------
# Body/sections extraction
# ---------------------------------------------------------------------

BLOCK_LEVEL_TAGS_TO_SKIP = {"fig", "table-wrap", "disp-formula", "inline-formula", "supplementary-material"}


def extract_direct_text_blocks_from_section(section_element: ET.Element) -> List[str]:
    """
    Extract text blocks belonging to THIS section (excluding nested <sec> content).
    """
    section_blocks: List[str] = []

    for direct_child in list(section_element):
        child_tag = strip_xml_namespace(direct_child.tag)

        if child_tag in ("title", "sec"):
            continue
        if child_tag in BLOCK_LEVEL_TAGS_TO_SKIP:
            continue

        if child_tag == "p":
            paragraph_text = extract_text_content(direct_child)
            if paragraph_text:
                section_blocks.append(paragraph_text)

        elif child_tag == "list":
            bullet_lines: List[str] = []
            for list_item in direct_child.iter():
                if strip_xml_namespace(list_item.tag) == "list-item":
                    paragraph_elements = [p for p in list_item.iter() if strip_xml_namespace(p.tag) == "p"]
                    if paragraph_elements:
                        for p in paragraph_elements:
                            t = extract_text_content(p)
                            if t:
                                bullet_lines.append(f"- {t}")
                    else:
                        t = extract_text_content(list_item)
                        if t:
                            bullet_lines.append(f"- {t}")
            if bullet_lines:
                section_blocks.append("\n".join(bullet_lines))

    return section_blocks


@dataclass
class SectionWalkContext:
    # Used to optionally inherit bucket assignment from previous sibling at the same depth.
    last_mapped_bucket_by_depth: Dict[int, Optional[str]]


def flatten_body_sections(body_element: ET.Element, *, enable_sibling_bucket_inheritance: bool = True,) -> List[Dict[str, Any]]:
    """
    Convert <body> into a flat list of section records:
      - path: list of titles (handles nested <sec>)
      - title: section title
      - bucket: one of BUCKET_NAMES or None
      - text: blocks directly under this <sec>
    """
    flattened_sections_in_traversal_order: List[Dict[str, Any]] = []
    walk_context = SectionWalkContext(last_mapped_bucket_by_depth={})

    traversal_section_index = 0

    def walk_section(
            section_element: ET.Element,
            current_path: List[str],
            parent_bucket: Optional[str],
            depth: int,
    ) -> None:
        nonlocal traversal_section_index

        title_element = None
        for child in list(section_element):
            if strip_xml_namespace(child.tag) == "title":
                title_element = child
                break

        section_title = extract_text_content(title_element) if title_element is not None else ""
        normalized_title = normalize_section_title(section_title)
        is_admin = bool(normalized_title) and is_admin_section_title(normalized_title)

        explicitly_mapped_bucket = map_section_title_to_bucket(section_title)
        effective_bucket = explicitly_mapped_bucket or parent_bucket

        if enable_sibling_bucket_inheritance and effective_bucket is None and not is_admin:
            effective_bucket = walk_context.last_mapped_bucket_by_depth.get(depth)

        section_path = current_path + ([section_title] if section_title else [])
        section_path_string = " > ".join([p for p in section_path if p])

        section_blocks = extract_direct_text_blocks_from_section(section_element)
        section_text = "\n\n".join(section_blocks).strip()

        section_record = {
            "section_index": traversal_section_index,
            "depth": depth,
            "path": section_path,
            "path_string": section_path_string,
            "title": section_title or None,
            "bucket": effective_bucket,
            "text": section_text,
        }
        flattened_sections_in_traversal_order.append(section_record)
        traversal_section_index += 1

        if explicitly_mapped_bucket in BUCKET_NAMES:
            walk_context.last_mapped_bucket_by_depth[depth] = explicitly_mapped_bucket

        for child in list(section_element):
            if strip_xml_namespace(child.tag) == "sec":
                walk_section(child, section_path, effective_bucket, depth + 1)

    for child in list(body_element):
        if strip_xml_namespace(child.tag) == "sec":
            walk_section(child, [], None, 0)

    # Keep only meaningful records
    meaningful_sections = [s for s in flattened_sections_in_traversal_order if (s.get("title") or s.get("text"))]

    # remove exact duplicates (same path_string + same text)
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in meaningful_sections:
        key = (s.get("path_string") or "", s.get("text") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped

def build_bucket_to_section_index_list(section_records: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Returns:
      {
        "introduction": [0, 3, ...],
        "methods": [2, 4, 5, ...],
        ...
      }

    We include only sections that:
    - have a valid bucket in BUCKET_NAMES
    - have non-empty text (so container sections like "Methods" with text="" are excluded)
    """

    bucket_to_section_indexes: Dict[str, List[int]] = {bucket: [] for bucket in BUCKET_NAMES}

    for section_record in section_records:
        bucket_name = section_record.get("bucket")
        section_text = (section_record.get("text") or "").strip()
        section_index = section_record.get("section_index")

        if bucket_name in bucket_to_section_indexes and section_text and isinstance(section_index, int):
            bucket_to_section_indexes[bucket_name].append(section_index)

    return bucket_to_section_indexes

def build_sections_by_index(paper_record: dict) -> dict[int, dict]:
    """
    Convenience index: section_index -> section record
    """
    sections = paper_record.get("sections", [])
    return {s["section_index"]: s for s in sections if isinstance(s.get("section_index"), int)}

def get_bucket_text(paper_record: dict, bucket_name: str, *, separator: str = "\n\n") -> str:
    """
    Returns the full text for a bucket by concatenating section texts in order.
    """
    buckets = paper_record.get("buckets", {})
    bucket_value = buckets.get(bucket_name)

    if isinstance(bucket_value, list):
        sections_by_index = build_sections_by_index(paper_record)
        parts: list[str] = []
        for section_index in bucket_value:
            section = sections_by_index.get(section_index)
            if not section:
                continue
            text = (section.get("text") or "").strip()
            if text:
                parts.append(text)
        return separator.join(parts).strip()

    return ""


# ---------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------

def parse_pmc_jats_xml_to_record(
    xml_file_path: str,
    *,
    enable_sibling_bucket_inheritance: bool = True,
) -> Dict[str, Any]:
    xml_tree = ET.parse(xml_file_path)
    article_root = xml_tree.getroot()

    if strip_xml_namespace(article_root.tag) != "article":
        raise ValueError(f"Not a JATS <article> XML: {xml_file_path}")

    front_element = find_first_descendant_by_tag(article_root, "front")
    if front_element is None:
        raise ValueError(f"Missing <front>: {xml_file_path}")

    article_meta_element = find_first_descendant_by_tag(front_element, "article-meta")
    if article_meta_element is None:
        raise ValueError(f"Missing <article-meta>: {xml_file_path}")

    identifiers = extract_article_identifiers(article_meta_element)
    stable_paper_id = (
        identifiers.get("pmcid")
        or identifiers.get("doi")
        or identifiers.get("pmid")
        or Path(xml_file_path).stem
    )

    article_title = extract_article_title(article_meta_element)
    journal_info = extract_journal_metadata(front_element)
    publication_year, publication_date = extract_publication_date(article_meta_element)
    keyword_list = extract_keywords_list(front_element)

    affiliation_map = build_affiliation_id_to_text_map(front_element)
    author_list = extract_author_list(article_meta_element, affiliation_map)

    abstract_text = extract_abstract_text(article_meta_element)

    body_element = find_first_descendant_by_tag(article_root, "body")
    section_records = (
        flatten_body_sections(body_element, enable_sibling_bucket_inheritance=enable_sibling_bucket_inheritance)
        if body_element is not None
        else []
    )
    bucket_section_index_lists = build_bucket_to_section_index_list(section_records)

    return {
        "schema_version": "1.0",
        "source": "pmc_jats",
        "paper_id": stable_paper_id,
        "ids": identifiers,
        "title": article_title,
        "journal": journal_info,
        "published": {"year": publication_year, "date": publication_date},
        "authors": author_list,
        "keywords": keyword_list,
        "abstract": abstract_text,
        "buckets": bucket_section_index_lists,
        "sections": section_records,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def collect_xml_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([file for file in input_path.rglob("*.xml") if file.is_file()])

def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("input", help="XML file or directory containing XMLs")
    argument_parser.add_argument("--out", required=True, help="Output path (.json for single file, .jsonl for many)")
    argument_parser.add_argument(
        "--no-sibling-inherit",
        action="store_true",
        help="Disable inheriting previous sibling bucket for unmapped, non-admin sections",
    )
    args = argument_parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    sibling_inherit_enabled = not args.no_sibling_inherit

    xml_files = collect_xml_files(input_path)
    if not xml_files:
        raise SystemExit(f"No .xml files found under: {input_path}")
    if len(xml_files) > 1 and output_path.suffix.lower() == ".json":
        raise SystemExit("For multiple XML files, use --out with a .jsonl filename (e.g., corpus.jsonl).")

    if len(xml_files) == 1 and output_path.suffix.lower() == ".json":
        paper_record = parse_pmc_jats_xml_to_record(str(xml_files[0]), enable_sibling_bucket_inheritance=sibling_inherit_enabled)
        output_path.write_text(json.dumps(paper_record, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"--out must be a file (e.g., papers.jsonl), not a directory: {output_path}")
    if output_path.suffix.lower() not in (".json", ".jsonl"):
        raise SystemExit(f"--out must end with .json or .jsonl. Got: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for xml_file in xml_files:
            try:
                paper_record = parse_pmc_jats_xml_to_record(str(xml_file), enable_sibling_bucket_inheritance=sibling_inherit_enabled)
                output_file.write(json.dumps(paper_record, ensure_ascii=False) + "\n")
            except Exception as exc:
                error_record = {
                    "schema_version": "1.0",
                    "source": "pmc_jats",
                    "paper_id": xml_file.stem,
                    "error": str(exc),
                    "file": str(xml_file),
                }
                output_file.write(json.dumps(error_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
