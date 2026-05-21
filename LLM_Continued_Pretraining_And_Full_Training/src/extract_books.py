"""
extract_books.py
================
Extracts clean text from all books in the books_articles/ folder.
Handles PDF, EPUB, DOCX, and TXT formats.
Outputs one .txt file per book in data/raw/ with structure markers.

OCR strategy (PDFs):
  1.  Try pymupdf native text extraction on every page.
  2.  Compute an OCR-quality score for the page (ratio of printable ASCII-ish chars).
  3.  If quality < OCR_QUALITY_THRESHOLD, convert the page to a high-res PIL image
      and run Tesseract OCR on it (with medical-optimised pre-processing).
  4.  Post-process both paths: normalise whitespace, fix hyphenation, strip
      headers / footers / page numbers.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from pathlib import Path

# ── Configuration constants ───────────────────────────────────────────────────
BOOKS_DIR         = Path(__file__).parent.parent / "books_articles"
OUTPUT_DIR        = Path(__file__).parent.parent / "data" / "raw"
LOG_DIR           = Path(__file__).parent.parent / "logs"
LOG_FILE          = LOG_DIR / "extract_books.log"

# OCR quality: pages below this ratio of printable chars trigger Tesseract
OCR_QUALITY_THRESHOLD   = 0.60   # 60 % printable
# DPI for rasterising PDF pages when OCR is needed
OCR_DPI                 = 300
# Tesseract language(s) – eng covers most medical abbreviations
TESSERACT_LANG          = "eng"
# Minimum word count to consider a page worth keeping
MIN_PAGE_WORDS          = 10
# Fraction of digits above which a line is considered a page number / table row
DIGIT_LINE_THRESHOLD    = 0.70
# Repeated header/footer: if this exact string appears >N times in the doc, strip it
REPEATED_LINE_THRESHOLD = 5

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add(str(LOG_FILE), level="DEBUG", rotation="10 MB",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# ── Lazy imports with helpful error messages ──────────────────────────────────
def _require(module: str, pip_name: str = ""):
    import importlib
    pip_name = pip_name or module
    try:
        return importlib.import_module(module)
    except ImportError:
        logger.error(f"Missing dependency '{module}'. Install with: pip install {pip_name}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _printable_ratio(text: str) -> float:
    """Fraction of characters that are printable ASCII or common Unicode letters."""
    if not text:
        return 0.0
    printable = sum(
        1 for c in text
        if c.isprintable() or c in "\n\t"
    )
    return printable / len(text)


def _is_page_number_line(line: str) -> bool:
    """True if the line is just a page number (digits, roman numerals, dashes)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Pure digits
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    # Roman numerals
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,6}", stripped):
        return True
    # Dash-padded page numbers  "─ 42 ─" or "- 42 -"
    if re.fullmatch(r"[-─–—\s\d]{1,10}", stripped):
        return True
    return False


def _is_mostly_numeric(line: str) -> bool:
    """True if >70 % of the line's non-space chars are digits / punctuation."""
    chars = re.sub(r"\s", "", line)
    if not chars:
        return False
    digit_count = sum(1 for c in chars if c.isdigit() or c in ".,;:+-/|=<>")
    return digit_count / len(chars) > DIGIT_LINE_THRESHOLD


def _fix_hyphenation(text: str) -> str:
    """Re-join words hyphenated at line breaks (common in PDFs)."""
    # "treat-\nment" → "treatment"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def _normalise_whitespace(text: str) -> str:
    """Collapse multiple blank lines to one, strip trailing spaces."""
    lines = text.split("\n")
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        stripped = line.rstrip()
        if stripped:
            blank_run = 0
            cleaned.append(stripped)
        else:
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
    return "\n".join(cleaned)


def _remove_repeated_headers_footers(text: str, threshold: int = REPEATED_LINE_THRESHOLD) -> str:
    """Strip lines that appear verbatim more than `threshold` times (headers/footers)."""
    lines = text.split("\n")
    from collections import Counter
    line_counts = Counter(
        line.strip() for line in lines if len(line.strip()) > 5
    )
    repeated = {line for line, count in line_counts.items() if count >= threshold}
    if repeated:
        logger.debug(f"Removing {len(repeated)} repeated header/footer lines")
    return "\n".join(
        line for line in lines
        if line.strip() not in repeated
    )


def _clean_page_text(text: str) -> str:
    """Full cleaning pipeline applied to each page's raw text."""
    import ftfy
    text = ftfy.fix_text(text)
    text = _fix_hyphenation(text)
    # Remove lines that are purely page numbers
    lines = [l for l in text.split("\n") if not _is_page_number_line(l)]
    text = "\n".join(lines)
    text = _normalise_whitespace(text)
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# OCR IMAGE PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_for_ocr(image):
    """
    Apply image processing to maximise Tesseract OCR accuracy.
    Steps: greyscale → denoise → adaptive threshold → deskew.
    """
    cv2 = _require("cv2", "opencv-python")
    import numpy as np

    # Convert PIL to numpy BGR
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Greyscale
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(grey, h=10)

    # Adaptive threshold (works well for scanned medical books)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # Deskew using moments
    coords = np.column_stack(np.where(binary < 128))
    if coords.size > 0:
        angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.3:
            (h, w) = binary.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    from PIL import Image as PILImage
    return PILImage.fromarray(binary)


def _ocr_page(pil_image) -> str:
    """Run Tesseract on a PIL image and return extracted text."""
    pytesseract = _require("pytesseract")
    prepped = _preprocess_for_ocr(pil_image)
    custom_config = (
        "--oem 3 --psm 6 "           # OEM 3 = LSTM+legacy, PSM 6 = assume uniform block
        "-c preserve_interword_spaces=1 "
        "-c tessedit_do_invert=0"
    )
    return pytesseract.image_to_string(prepped, lang=TESSERACT_LANG, config=custom_config)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-COLUMN LAYOUT HANDLER (pymupdf)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_page_text_with_layout(page) -> str:
    """
    Extract text from a pymupdf page respecting multi-column layout.
    Uses text blocks sorted by (top, left) position so columns are read
    left-to-right, top-to-bottom.
    """
    blocks = page.get_text("blocks", sort=False)
    # Each block: (x0, y0, x1, y1, text, block_no, block_type)
    # block_type 0 = text, 1 = image
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if not text_blocks:
        return ""

    # Detect column boundaries via x-coordinate clustering
    page_width = page.rect.width
    mid_x = page_width / 2

    left_blocks  = [b for b in text_blocks if b[0] < mid_x * 0.7]
    right_blocks = [b for b in text_blocks if b[0] >= mid_x * 0.7]

    # If there's a meaningful right column, treat as two-column layout
    if len(right_blocks) > 2 and len(left_blocks) > 2:
        left_blocks  = sorted(left_blocks,  key=lambda b: (b[1], b[0]))
        right_blocks = sorted(right_blocks, key=lambda b: (b[1], b[0]))
        left_text  = "\n".join(b[4] for b in left_blocks)
        right_text = "\n".join(b[4] for b in right_blocks)
        return left_text + "\n" + right_text
    else:
        # Single-column: sort top-to-bottom
        text_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
        return "\n".join(b[4] for b in text_blocks)


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_CHAPTER_PATTERNS = [
    re.compile(r"^\s*CHAPTER\s+(\d+|[IVXLCDM]+)[:\s.–-]*(.{0,80})", re.IGNORECASE),
    re.compile(r"^\s*Chapter\s+(\d+|[IVXLCDMivxlcdm]+)[:\s.–-]*(.{0,80})"),
    re.compile(r"^\s*(\d+)\s*[.\s]\s*([A-Z][A-Z\s]{5,60})$"),    # "1. INTRODUCTION"
    re.compile(r"^(PART\s+[IVXLCDM\d]+)[:\s.–-]*(.{0,80})", re.IGNORECASE),
]

_SECTION_PATTERNS = [
    re.compile(r"^\s*(\d+\.\d+(?:\.\d+)?)\s+(.{3,80})"),    # "1.2 Section Name"
    re.compile(r"^\s*[A-Z][A-Z ]{4,60}$"),                   # ALL-CAPS section header
    re.compile(r"^\s*(INTRODUCTION|SUMMARY|CONCLUSION|REFERENCES|DISCUSSION|METHODS?|RESULTS?|BACKGROUND)\s*$", re.IGNORECASE),
]

_HIGH_VALUE_PATTERNS = [
    re.compile(r"(key\s+point|summary|take[\s-]?away|clinical\s+pearl|important|note\s+to\s+remember)", re.IGNORECASE),
]


def _classify_line(line: str) -> tuple[str, str]:
    """
    Returns (tag, content) where tag is 'chapter', 'section', 'text', or 'skip'.
    """
    stripped = line.strip()
    if not stripped:
        return "skip", ""

    for pat in _CHAPTER_PATTERNS:
        m = pat.match(stripped)
        if m:
            groups = m.groups()
            title = " ".join(g.strip() for g in groups if g and g.strip())
            return "chapter", title

    for pat in _SECTION_PATTERNS:
        m = pat.match(stripped)
        if m:
            groups = m.groups()
            title = " ".join(g.strip() for g in groups if g and g.strip()) or stripped
            return "section", title

    return "text", stripped


def _annotate_text(raw_text: str) -> str:
    """
    Walk through raw_text and insert [CHAPTER: …] and [SECTION: …] markers.
    Returns annotated text ready to write to data/raw/.
    """
    lines = raw_text.split("\n")
    output_lines: list[str] = []
    current_chapter = "Unknown Chapter"
    current_section = "General"

    for line in lines:
        tag, content = _classify_line(line)
        if tag == "chapter":
            current_chapter = content
            current_section = "General"
            output_lines.append(f"\n[CHAPTER: {current_chapter}]")
        elif tag == "section":
            current_section = content
            output_lines.append(f"\n[SECTION: {current_section}]")
        elif tag == "text":
            output_lines.append(line)
        # "skip" → drop blank lines handled by normalise_whitespace later

    return "\n".join(output_lines)


# ═══════════════════════════════════════════════════════════════════════════════
# FORMAT-SPECIFIC EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

def _is_index_page(text: str) -> bool:
    """
    Returns True if the page looks like a back-of-book index.
    Heuristic: >40% of lines contain an entry like "word ... 123" or "word, 45, 67".
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    index_pat = re.compile(r".+[,\.]\s*\d{1,4}")
    hits = sum(1 for l in lines if index_pat.search(l))
    return hits / len(lines) > 0.40


def extract_pdf(path: Path) -> tuple[str, dict]:
    """
    Extract text from a PDF file using pymupdf.
    Falls back to Tesseract OCR per-page when native extraction quality is low.
    Returns (full_text, stats).
    """
    fitz = _require("fitz", "pymupdf")

    stats = {
        "pages": 0, "ocr_pages": 0, "skipped_pages": 0,
        "word_count": 0, "chapters_found": 0,
    }

    doc = fitz.open(str(path))
    stats["pages"] = len(doc)
    all_page_texts: list[str] = []

    # Try to import pdf2image+PIL for OCR fallback
    try:
        from pdf2image import convert_from_path
        from PIL import Image as PILImage
        ocr_available = True
    except ImportError:
        logger.warning("pdf2image / Pillow not available – OCR fallback disabled")
        ocr_available = False

    for page_num in range(len(doc)):
        page = doc[page_num]

        # --- Native text extraction ---
        native_text = _extract_page_text_with_layout(page)
        quality     = _printable_ratio(native_text)

        if quality >= OCR_QUALITY_THRESHOLD and len(native_text.split()) >= MIN_PAGE_WORDS:
            page_text = native_text
        elif ocr_available:
            # --- OCR fallback ---
            try:
                logger.debug(f"Page {page_num + 1}: quality={quality:.2f}, falling back to OCR")
                mat    = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                pix    = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                from PIL import Image as PILImage
                import io
                img_bytes = pix.tobytes("png")
                pil_img   = PILImage.open(io.BytesIO(img_bytes))
                page_text = _ocr_page(pil_img)
                stats["ocr_pages"] += 1
            except Exception as ocr_err:
                logger.warning(f"OCR failed on page {page_num + 1}: {ocr_err}. Using raw native text.")
                page_text = native_text
        else:
            # Low quality, no OCR available
            if len(native_text.split()) >= MIN_PAGE_WORDS:
                page_text = native_text
            else:
                stats["skipped_pages"] += 1
                continue

        # Skip index / reference pages
        if _is_index_page(page_text):
            logger.debug(f"Page {page_num + 1} appears to be an index page – skipping")
            stats["skipped_pages"] += 1
            continue

        cleaned = _clean_page_text(page_text)
        if cleaned.strip():
            all_page_texts.append(cleaned)

    doc.close()

    full_text = "\n\n".join(all_page_texts)
    full_text = _remove_repeated_headers_footers(full_text)
    annotated = _annotate_text(full_text)

    stats["word_count"]     = len(annotated.split())
    stats["chapters_found"] = annotated.count("[CHAPTER:")

    return annotated, stats


def extract_epub(path: Path) -> tuple[str, dict]:
    """Extract text from an EPUB file preserving chapter structure."""
    ebooklib = _require("ebooklib", "ebooklib")
    from ebooklib import epub as epub_mod
    bs4_mod = _require("bs4", "beautifulsoup4")
    from bs4 import BeautifulSoup

    stats = {"pages": 0, "ocr_pages": 0, "skipped_pages": 0, "word_count": 0, "chapters_found": 0}

    book = epub_mod.read_epub(str(path), options={"ignore_ncx": True})
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    stats["pages"] = len(items)

    all_chapter_texts: list[str] = []

    for item in items:
        html_content = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "lxml")

        # Extract title from headings
        chapter_title = ""
        for h_tag in soup.find_all(["h1", "h2"]):
            chapter_title = h_tag.get_text(strip=True)
            break

        chapter_text_parts: list[str] = []

        if chapter_title:
            chapter_text_parts.append(f"[CHAPTER: {chapter_title}]")

        current_section = ""
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "div", "li"]):
            tag_name = elem.name
            text     = elem.get_text(separator=" ", strip=True)
            if not text:
                continue

            if tag_name in ("h1", "h2"):
                if text != chapter_title:
                    current_section = text
                    chapter_text_parts.append(f"\n[SECTION: {current_section}]")
            elif tag_name in ("h3", "h4"):
                chapter_text_parts.append(f"\n[SECTION: {text}]")
            else:
                if text and len(text.split()) >= 3:
                    chapter_text_parts.append(text)

        chapter_text = "\n".join(chapter_text_parts)
        if chapter_text.strip():
            all_chapter_texts.append(chapter_text)

    import ftfy
    full_text = "\n\n".join(all_chapter_texts)
    full_text = ftfy.fix_text(full_text)
    full_text = _remove_repeated_headers_footers(full_text)
    full_text = _normalise_whitespace(full_text)

    stats["word_count"]     = len(full_text.split())
    stats["chapters_found"] = full_text.count("[CHAPTER:")

    return full_text, stats


def extract_docx(path: Path) -> tuple[str, dict]:
    """Extract text from a DOCX file preserving heading-based structure."""
    docx_mod = _require("docx", "python-docx")
    from docx import Document

    stats = {"pages": 0, "ocr_pages": 0, "skipped_pages": 0, "word_count": 0, "chapters_found": 0}

    doc = Document(str(path))
    output_lines: list[str] = []

    for para in doc.paragraphs:
        text  = para.text.strip()
        style = para.style.name if para.style else ""

        if not text:
            continue

        if "Heading 1" in style:
            output_lines.append(f"\n[CHAPTER: {text}]")
        elif "Heading 2" in style:
            output_lines.append(f"\n[SECTION: {text}]")
        elif "Heading 3" in style or "Heading 4" in style:
            output_lines.append(f"\n[SECTION: {text}]")
        else:
            output_lines.append(text)

    import ftfy
    full_text = ftfy.fix_text("\n".join(output_lines))
    full_text = _normalise_whitespace(full_text)
    stats["word_count"]     = len(full_text.split())
    stats["chapters_found"] = full_text.count("[CHAPTER:")
    # Approximate pages from word count
    stats["pages"] = max(1, stats["word_count"] // 250)

    return full_text, stats


def extract_txt(path: Path) -> tuple[str, dict]:
    """Extract text from a plain-text file, detecting encoding."""
    chardet_mod = _require("chardet")
    import chardet as chardet_pkg

    stats = {"pages": 0, "ocr_pages": 0, "skipped_pages": 0, "word_count": 0, "chapters_found": 0}

    raw_bytes = path.read_bytes()
    detected  = chardet_pkg.detect(raw_bytes)
    encoding  = detected.get("encoding") or "utf-8"
    logger.debug(f"Detected encoding for {path.name}: {encoding} (confidence={detected.get('confidence', 0):.2f})")

    text = raw_bytes.decode(encoding, errors="replace")

    import ftfy
    text = ftfy.fix_text(text)
    text = _annotate_text(text)
    text = _remove_repeated_headers_footers(text)
    text = _normalise_whitespace(text)

    stats["word_count"]     = len(text.split())
    stats["chapters_found"] = text.count("[CHAPTER:")
    stats["pages"]          = max(1, stats["word_count"] // 250)

    return text, stats


# ═══════════════════════════════════════════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def extract_book(path: Path) -> tuple[str, dict] | None:
    """Auto-detect format and extract. Returns None on failure."""
    suffix = path.suffix.lower()
    extractors = {
        ".pdf":  extract_pdf,
        ".epub": extract_epub,
        ".docx": extract_docx,
        ".doc":  extract_docx,
        ".txt":  extract_txt,
    }

    extractor = extractors.get(suffix)
    if extractor is None:
        logger.warning(f"Unsupported format: {path.suffix} ({path.name}) – skipping")
        return None

    try:
        logger.info(f"Extracting: {path.name}")
        text, stats = extractor(path)
        return text, stats
    except Exception as exc:
        logger.error(f"FAILED to extract {path.name}: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    from tqdm import tqdm

    supported_exts = {".pdf", ".epub", ".docx", ".doc", ".txt"}
    book_files = sorted(
        p for p in BOOKS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    if not book_files:
        logger.error(f"No supported book files found in {BOOKS_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(book_files)} book file(s) in {BOOKS_DIR}")

    total_words    = 0
    total_chapters = 0
    success_count  = 0
    failure_count  = 0

    summary_rows: list[dict] = []

    for book_path in tqdm(book_files, desc="Extracting books", unit="book"):
        result = extract_book(book_path)
        if result is None:
            failure_count += 1
            continue

        text, stats = result

        # Sanitise filename for output
        safe_stem = re.sub(r'[^\w\s-]', '', book_path.stem)
        safe_stem = re.sub(r'\s+', '_', safe_stem.strip())[:80]
        out_path  = OUTPUT_DIR / f"{safe_stem}.txt"

        out_path.write_text(text, encoding="utf-8")
        logger.info(
            f"  Saved: {out_path.name} | "
            f"pages={stats['pages']} | "
            f"ocr_pages={stats['ocr_pages']} | "
            f"words={stats['word_count']:,} | "
            f"chapters={stats['chapters_found']}"
        )

        total_words    += stats["word_count"]
        total_chapters += stats["chapters_found"]
        success_count  += 1
        summary_rows.append({"book": book_path.name, **stats})

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"EXTRACTION COMPLETE")
    logger.info(f"  Processed : {success_count}/{len(book_files)} books")
    logger.info(f"  Failed    : {failure_count} books (see log for details)")
    logger.info(f"  Total words : {total_words:,}")
    logger.info(f"  Total chapters : {total_chapters}")
    logger.info(f"  Output dir : {OUTPUT_DIR}")
    logger.info("=" * 60)

    # Per-book table
    print("\n{'Book':<55} {'Pages':>6} {'OCR':>5} {'Words':>10} {'Chapters':>9}")
    print("-" * 90)
    for row in summary_rows:
        name = row["book"][:54]
        print(f"{name:<55} {row['pages']:>6} {row['ocr_pages']:>5} "
              f"{row['word_count']:>10,} {row['chapters_found']:>9}")


if __name__ == "__main__":
    main()
