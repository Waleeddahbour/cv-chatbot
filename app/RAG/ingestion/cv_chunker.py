import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .di_cleaner import canonical_section

MAX_CHUNK_CHARS = 1500

DATE_RANGE_PATTERN = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+[-–]\s+"
    r"(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Section:
    name: str
    text: str


@dataclass(frozen=True)
class Block:
    text: str


def create_cv_chunks(
    cleaned_markdown: str,
    di_result: dict,
    source_file: str | Path,
) -> list[dict[str, Any]]:
    source_name = Path(source_file).name
    parent_id = build_parent_id(source_name)
    page_resolver = PageResolver(di_result)
    indexed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    chunks = []
    for section in parse_sections(cleaned_markdown):
        # Keep naturally related blocks together first, then split only the
        # blocks that exceed the chunk size limit.
        for text in split_blocks(parse_blocks(section.text)):
            text = text.strip()
            if not text:
                continue

            chunk_index = len(chunks)
            chunk_id = f"{parent_id}-{chunk_index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_id": parent_id,
                    "source_file": source_name,
                    "section": section.name,
                    "text": text,
                    "chunk_index": chunk_index,
                    "page_number": page_resolver.resolve(text),
                    "metadata_json": json.dumps(
                        {
                            "parser_version": "cv-jsonl-v1",
                            "char_count": len(text),
                        },
                        ensure_ascii=False,
                    ),
                    "indexed_at": indexed_at,
                }
            )

    return chunks


def build_parent_id(source_name: str) -> str:
    digest = hashlib.sha1(source_name.encode("utf-8")).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(source_name).stem).strip("-").lower()
    return f"{stem}-{digest}"


def parse_sections(markdown: str) -> list[Section]:
    sections: list[Section] = []
    current_section = "Contact"
    current_lines: list[str] = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section = extract_section_heading(line)
        if section:
            append_section(sections, current_section, current_lines)
            current_section = section
            current_lines = []
            continue

        current_lines.append(strip_heading_marks(line))

    append_section(sections, current_section, current_lines)
    return sections


def extract_section_heading(line: str) -> str | None:
    heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
    if heading_match:
        return canonical_section(heading_match.group(1))
    return canonical_section(line)


def strip_heading_marks(line: str) -> str:
    heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
    return heading_match.group(1).strip() if heading_match else line


def append_section(sections: list[Section], name: str, lines: list[str]) -> None:
    text = "\n".join(lines).strip()
    if text:
        sections.append(Section(name=name, text=text))


def parse_blocks(section_text: str) -> list[Block]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    blocks: list[Block] = []
    current: list[str] = []

    for index, line in enumerate(lines):
        previous_line = lines[index - 1] if index else ""
        next_line = lines[index + 1] if index + 1 < len(lines) else ""

        if current and starts_new_block(line, previous_line, next_line):
            blocks.append(Block(text="\n".join(current).strip()))
            current = []

        current.append(line)

    if current:
        blocks.append(Block(text="\n".join(current).strip()))

    return blocks


def starts_new_block(line: str, previous_line: str, next_line: str) -> bool:
    if is_bullet(line):
        return False
    if DATE_RANGE_PATTERN.search(line):
        # A title followed by a company/date line belongs in the same block.
        if looks_like_subheading(previous_line):
            return False
        return True
    if previous_line and line_continues_previous(previous_line):
        return False
    if line.endswith(":"):
        return True
    if looks_like_subheading(line) and DATE_RANGE_PATTERN.search(next_line):
        return True
    return looks_like_subheading(line) and is_bullet(next_line)


def line_continues_previous(line: str) -> bool:
    if is_bullet(line):
        return not line.endswith((".", ":", ";", ")"))
    return not line.endswith((".", ":", ";", ")"))


def is_bullet(line: str) -> bool:
    return bool(re.match(r"^[-*]\s+", line))


def looks_like_subheading(line: str) -> bool:
    if len(line) > 140:
        return False
    if line.endswith((".", ",", ";")):
        return False

    words = line.split()
    if not words or len(words) > 18:
        return False

    uppercase_or_title = sum(1 for word in words if word[:1].isupper())
    return uppercase_or_title / len(words) >= 0.45


def split_blocks(blocks: list[Block]) -> list[str]:
    chunks = []
    for block in blocks:
        chunks.extend(split_oversized_block(block.text))
    return [chunk for chunk in chunks if chunk.strip()]


def split_oversized_block(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    return pack_units(split_block_units(text))


def split_block_units(text: str) -> list[str]:
    units: list[str] = []
    current: list[str] = []

    for line in text.splitlines():
        if current and is_bullet(line):
            units.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        units.append("\n".join(current).strip())

    return units


def pack_units(units: list[str]) -> list[str]:
    chunks: list[str] = []
    current = ""

    for unit in units:
        if len(unit) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_hard_limit(unit))
            continue

        candidate = join_parts(current, unit)
        if current and len(candidate) > MAX_CHUNK_CHARS:
            chunks.append(current.strip())
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks


def split_hard_limit(text: str) -> list[str]:
    return [
        text[start : start + MAX_CHUNK_CHARS].strip()
        for start in range(0, len(text), MAX_CHUNK_CHARS)
        if text[start : start + MAX_CHUNK_CHARS].strip()
    ]


def join_parts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n{right}"


class PageResolver:
    def __init__(self, di_result: dict):
        # Markdown content is used for chunking; DI paragraph bounding regions
        # are used only to infer the most likely page number for each chunk.
        self.paragraphs = [
            {
                "page_number": self._paragraph_page(paragraph),
                "tokens": tokenize(paragraph.get("content", "")),
            }
            for paragraph in di_result.get("paragraphs", [])
            if paragraph.get("content")
        ]

    def resolve(self, text: str) -> int | None:
        chunk_tokens = tokenize(text)
        if not chunk_tokens:
            return None

        page_scores: dict[int, int] = {}
        for paragraph in self.paragraphs:
            page_number = paragraph["page_number"]
            if page_number is None:
                continue
            score = len(chunk_tokens & paragraph["tokens"])
            if score:
                page_scores[page_number] = page_scores.get(page_number, 0) + score

        if not page_scores:
            return None
        return max(page_scores, key=page_scores.get)

    @staticmethod
    def _paragraph_page(paragraph: dict) -> int | None:
        regions = paragraph.get("boundingRegions") or []
        if not regions:
            return None
        return regions[0].get("pageNumber")


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9+#.]+", text.casefold())
        if len(token) > 2
    }
