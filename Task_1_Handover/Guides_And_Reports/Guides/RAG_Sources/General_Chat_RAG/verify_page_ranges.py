# -*- coding: utf-8 -*-
"""
verify_page_ranges.py
---------------------
For each source book in the final_structured_rag Qdrant collection, finds which
physical pages in the real source PDF the RAG chunks actually came from, and
reports the min–max page range with a match confidence count.

Requires: pymupdf  (pip install pymupdf)
Run with: python verify_page_ranges.py
"""
import json, re, collections
import fitz  # PyMuPDF — used to open PDFs and extract page text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Full JSON dump of the final_structured_rag Qdrant collection (2653 points).
# Produced by scrolling the collection via qdrant_client and writing to JSON.
# NOTE: this file lives in a temp folder — re-dump if it disappears.
DUMP = r"C:\Users\Krish\AppData\Local\Temp\claude\c--Users-Krish-Downloads-Cygnus-Med-Demo\274a17db-5786-459e-ae27-6aa361f16acd\scratchpad\final_structured_rag_dump.json"

# Folder containing the real source PDF files used to build the collection.
# These are the same books that were chunked by chunk_documents.py during ingestion.
PDFDIR = r"C:\Users\Krish\Downloads\Cygnus_Med_Demo\Miscellaneous\books_articles"

# ---------------------------------------------------------------------------
# Mapping: source_book field value (as stored in Qdrant) → actual PDF filename
# The source_book strings in the DB have underscores/truncation from the
# ingestion pipeline, so they don't match the original filenames directly.
# ---------------------------------------------------------------------------
PDF_MAP = {
    "0-Handbook_of_Venous_and_Lymphatic_Disorders_Guidelines_of_the_American_Venous_F":
        "0-Handbook of Venous and Lymphatic Disorders Guidelines of the American Venous Forum by Peter Gloviczki et al. (eds.) (z-lib.org).pdf",
    "Atlas_of_Endovascular_Venous_Surgery_by_Almeida_Jose_z-liborg":
        "Atlas of Endovascular Venous Surgery by Almeida, Jose (z-lib.org).pdf",
    "0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease":
        "0-Saphenous-Vein-Sparing-Strategies-in-Chronic-Venous-Disease.pdf",
    "0-duplex-ultrasound-of-superficial-leg-veins-2014":
        "0-duplex-ultrasound-of-superficial-leg-veins-2014.pdf",
    "2019_book_Principles_of_Venous_Hemodynamics_Franceschi_Zambo":
        "2019 book Principles of Venous Hemodynamics Franceschi Zambo.pdf",
    "2025-02-19-Theraclion-investor-presentation-extended-V332_compressed-2":
        "2025-02-19-Theraclion-investor-presentation-extended-V33.2_compressed-2.pdf",
    "adler-et-al-2022-varicose-veins-of-the-lower-extremity-doppler-us-evaluation-pro":
        "adler-et-al-2022-varicose-veins-of-the-lower-extremity-doppler-us-evaluation-protocols-patterns-and-pitfalls.pdf",
    "Carriazo2015": "Carriazo2015.pdf",
    "DelfrateR_CHIVA_article": "DelfrateR CHIVA article.pdf",
    "CHIVA_STRATEGY_Gianesini014": "CHIVA STRATEGY Gianesini014.pdf",
    "US_Lower_Extremity_Veins_Anatomy_and_Basic_Approach": "US Lower Extremity Veins_Anatomy and Basic Approach.pdf",
    "PrinciplesofVenousHemodynamicsFranceschiZamboni2009": "PrinciplesofVenousHemodynamicsFranceschiZamboni2009.pdf",
}

# The second Almeida atlas is handled separately because its source_book string
# in the DB contains a � replacement character (encoding corruption from
# ingestion) where the real "é" should be, making exact dict key lookup fail.
# We match it by prefix + substring instead (see resolve_pages).
ALMEIDA_1 = "1-Atlas of Endovascular Venous Surgery by José Almeida (z-lib.org).pdf"

# ---------------------------------------------------------------------------
# Load the collection dump and group chunks by source book
# ---------------------------------------------------------------------------

# all 2653 chunk records, each a dict with keys:
# id, text, token_count, source_book, chapter, section, position, high_value
data = json.load(open(DUMP, encoding="utf-8"))

# by_book[book_name] = list of all chunk records from that book
by_book = collections.defaultdict(list)
for r in data:
    by_book[r.get("source_book", "")].append(r)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def words(s):
    """
    Extracts all lowercase alphanumeric tokens from a string.
    Strips punctuation, special characters, and encoding-corrupted bytes.
    This is what makes matching robust — both the chunk text and the PDF page
    text are reduced to the same plain word sequence before comparison, so
    differences in en-dashes, ligatures (fi/fl), hyphenation, or �
    replacement chars don't cause false negatives.
    """
    return re.findall(r"[a-z0-9]+", s.lower())


def page_word_index(doc):
    """
    Pre-computes a word-normalized text string for every page in a PyMuPDF
    document. Returns a list where index i holds the joined word tokens of
    page i (0-based). Used so we only extract text from each PDF page once
    rather than re-extracting it for every chunk probe.
    """
    idx = []
    for page in doc:
        idx.append(" ".join(words(page.get_text())))
    return idx


def find_page_word(page_texts, probe_words, min_len=6):
    """
    Searches for a chunk probe (as a list of word tokens) inside the
    pre-indexed page texts. Uses the first `min_len` words as the search key.
    Returns the 1-based page number of the first page where the probe is found,
    or None if not found on any page.

    min_len=6 gives enough specificity to avoid false positives while being
    short enough to tolerate minor chunking differences at paragraph edges.
    """
    if len(probe_words) < min_len:
        return None  # probe too short to be reliable
    probe = " ".join(probe_words[:min_len])
    for i, pt in enumerate(page_texts):
        if probe in pt:
            return i + 1  # convert 0-based page index to 1-based page number
    return None


def probe_candidates(rec):
    """
    Generates up to 5 candidate word-token windows from a chunk record's body
    text to use as search probes. Each candidate is 20 words starting at a
    different offset (0, 20, 40, 60, 80 words into the body).

    Multiple offsets are tried because:
    - The first few words might fall on a page boundary in the PDF (split across
      two pages during extraction) causing a miss on offset 0.
    - The ingestion pipeline prepended a "### Medical Reference\\nSource: ..."
      header to each chunk's text field; we skip it by finding the first blank
      line (\\n\\n) and taking text after that.
    """
    t = rec.get("text", "")
    # skip the injected metadata header ("### Medical Reference\nSource: ...")
    idx = t.find("\n\n")
    body = t[idx + 2:] if idx != -1 else t
    w = words(body)
    # return one candidate window per offset; caller tries them in order and
    # stops at the first one that matches a page
    return [w[o:o + 20] for o in (0, 20, 40, 60, 80)]


def resolve_pages(book, recs):
    """
    Given a book name and all its chunk records, opens the corresponding source
    PDF and returns a string describing the page range those chunks span.

    Strategy: rather than checking all chunks (expensive for 1413-chunk books),
    sample ~9 chunks — first 3, last 3, and 3 evenly spaced through the middle.
    The min and max matched page numbers across those samples give the range.

    Returns a string like:
      "p.4-742 (of 869pp; 9/9 samples matched)"  — full coverage confirmed
      "NO_PDF_MAPPED"                             — no PDF filename known for this book
      "NO_MATCH (book has N pages)"               — PDF found but no chunk text located
      "OPEN_FAIL"                                 — PDF file couldn't be opened
    """
    # resolve PDF path — Almeida book 1 uses substring match due to encoding issue
    if book.startswith("1-Atlas") and "Almeida" in book:
        path = f"{PDFDIR}\\{ALMEIDA_1}"
    elif book in PDF_MAP:
        path = f"{PDFDIR}\\{PDF_MAP[book]}"
    else:
        return "NO_PDF_MAPPED"

    try:
        doc = fitz.open(path)
    except Exception:
        return "OPEN_FAIL"

    # build word index once for all pages, then close the PDF
    page_texts = page_word_index(doc)
    npages = len(doc)
    doc.close()

    # sort chunks by their ingestion position so first/last are meaningful
    recs_sorted = sorted(recs, key=lambda r: r.get("position", 0))

    found_pages = []  # page numbers (1-based) where chunk probes were located

    # pick ~9 sample indices: first 3, last 3, and fractional middle points
    sample_idx = sorted(set(
        [0, 1, 2, len(recs_sorted) - 1, len(recs_sorted) - 2, len(recs_sorted) - 3]
        + [int(len(recs_sorted) * f) for f in (0.25, 0.5, 0.75)]
    ))

    for i in sample_idx:
        if i < 0 or i >= len(recs_sorted):
            continue  # guard against duplicate indices on small books
        rec = recs_sorted[i]
        # try each candidate window until one matches a page
        for cand in probe_candidates(rec):
            p = find_page_word(page_texts, cand)
            if p:
                found_pages.append(p)
                break  # found a page for this chunk — move to next sample

    if not found_pages:
        return f"NO_MATCH (book has {npages} pages)"

    return (f"p.{min(found_pages)}-{max(found_pages)} "
            f"(of {npages}pp; {len(found_pages)}/{len(sample_idx)} samples matched)")


# ---------------------------------------------------------------------------
# Main: print page range for every book in the collection
# ---------------------------------------------------------------------------
for book, recs in by_book.items():
    print(book, "->", resolve_pages(book, recs))
