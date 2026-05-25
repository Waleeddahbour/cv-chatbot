import re

MAJOR_SECTIONS = {
    "Summary",
    "Skills",
    "Experience",
    "Education",
    "Certifications",
    "Languages",
    "Strengths",
    "Projects",
}

SECTION_SYNONYMS = {
    "Summary": (
        "summary",
        "profile",
        "professional summary",
        "career summary",
        "about me",
        "objective",
        "career objective",
    ),
    "Skills": (
        "skills",
        "technical skills",
        "technologies",
        "tech stack",
        "competencies",
        "tools",
    ),
    "Experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
    ),
    "Education": (
        "education",
        "academic background",
        "academic qualifications",
    ),
    "Certifications": (
        "certifications",
        "certificates",
        "licenses",
    ),
    "Languages": (
        "languages",
        "language skills",
    ),
    "Strengths": (
        "core strengths",
        "strengths",
    ),
    "Projects": (
        "projects",
        "selected projects",
        "portfolio",
        "personal projects",
        "academic projects",
    ),
}

SECTION_LOOKUP = {
    re.sub(r"\s+", " ", alias.strip().casefold()): section
    for section, aliases in SECTION_SYNONYMS.items()
    for alias in (section, *aliases)
}


def extract_markdown(di_result: dict) -> str:
    content = di_result.get("content")
    if not isinstance(content, str):
        raise ValueError("Document Intelligence result does not contain content.")
    return content


def clean_di_markdown(di_result: dict) -> str:
    return clean_markdown(extract_markdown(di_result))


def clean_markdown(markdown: str | None) -> str:
    text = markdown or ""
    # Remove common Document Intelligence markdown artifacts before section
    # detection and chunking.
    text = text.replace("☒", "")
    text = text.replace("☐", "")
    text = text.replace(":selected:", "")
    text = text.replace(":unselected:", "")
    text = text.replace("<!-- PageBreak -->", "")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = text.replace("•", "\n- ")
    text = text.replace("·", "\n- ")
    text = re.sub(r"(?m)^(\s*)\.\s+", r"\1- ", text)
    text = re.sub(r"\.\.\s+", ". ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = fix_major_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_major_headings(markdown: str) -> str:
    fixed_lines = []

    for line in markdown.splitlines():
        # Normalize only known CV section headings; leave job titles and other
        # non-section headings as regular text for the chunker.
        heading_match = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if heading_match:
            section = canonical_section(heading_match.group(1))
            fixed_lines.append(f"## {section}" if section else line)
            continue

        section = canonical_section(line)
        if section:
            fixed_lines.append(f"## {section}")
            continue

        fixed_lines.append(line)

    text = "\n".join(fixed_lines)
    for section in MAJOR_SECTIONS:
        text = re.sub(
            rf"(?<![#\w]){re.escape(section)}(?=[A-Z])",
            f"\n## {section}\n",
            text,
        )
    return text


def canonical_section(value: str) -> str | None:
    label = normalize_label(value)
    return SECTION_LOOKUP.get(label)


def normalize_label(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[*_`#|:-]", " ", value)
    return re.sub(r"\s+", " ", value).strip().casefold()
