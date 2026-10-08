"""Read an arbitrary delimited text file or Excel sheet into (headers, rows) of strings.

Tolerates: UTF-8/UTF-16/Windows-1252 encodings, comma/tab/semicolon/pipe delimiters, preamble
lines above the header (report titles, "Exported by..."), trailing summary rows, duplicate or
blank headers, files with no header row at all, and single-column address lists.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sources.addresses import STATE_CODES, parse_full_address

csv.field_size_limit(10_000_000)
_DELIMITERS = [",", "\t", ";", "|"]


@dataclass
class Table:
    headers: list[str]
    rows: list[list[str]]
    delimiter: str | None = None
    encoding: str | None = None
    header_row_index: int = 0
    synthetic_headers: bool = False  # the file had no header row
    notes: list[str] = field(default_factory=list)


def decode_bytes(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def sniff_delimiter(text: str) -> str | None:
    lines = [ln for ln in text.splitlines()[:60] if ln.strip()]
    if not lines:
        return None
    best, best_score = None, 0.0
    for d in _DELIMITERS:
        counts = [len(next(csv.reader([ln], delimiter=d))) for ln in lines]
        multi = Counter(c for c in counts if c >= 2)  # preamble/footer lines have 1 field
        if not multi:
            continue
        modal, freq = multi.most_common(1)[0]
        score = (freq / len(counts)) * min(modal, 12)
        if score > best_score:
            best, best_score = d, score
    return best


def _looks_like_header(cells: list[str]) -> bool:
    """A header row is mostly short, non-numeric labels with no street-address patterns."""
    filled = [c.strip() for c in cells if c.strip()]
    if not filled:
        return False
    bad = 0
    for c in filled:
        if (
            re.fullmatch(r"[\d.,$%/\- ]+", c)
            or re.match(r"^\d+\s+\w+", c)
            or re.search(r"\b\d{5}(-\d{4})?\b", c)
            or c.upper() in STATE_CODES
            and len(filled) > 1
            or len(c) > 60
        ):
            bad += 1
    return bad / len(filled) < 0.3


def _dedupe_headers(headers: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    out = []
    for i, h in enumerate(headers):
        h = h.strip() or f"column_{i + 1}"
        seen[h] += 1
        out.append(h if seen[h] == 1 else f"{h}_{seen[h]}")
    return out


def table_from_rows(raw_rows: list[list[str]], **meta) -> Table:  # noqa: ANN003
    rows = [[str(c).strip() for c in r] for r in raw_rows if any(str(c).strip() for c in r)]
    if not rows:
        return Table([], [], **meta)
    width_counts = Counter(len(r) for r in rows)
    modal_width = max(width_counts, key=lambda w: (width_counts[w] if w >= 2 else 0, w))
    if width_counts[modal_width] < 2 and len(rows) > 1:
        modal_width = max(len(r) for r in rows)
    start = next((i for i, r in enumerate(rows) if len(r) == modal_width), 0)
    notes = []
    if start:
        notes.append(f"Skipped {start} line(s) above the table.")
    body = rows[start:]
    synthetic = False
    if _looks_like_header(body[0]):
        headers = _dedupe_headers(body[0])
        data = body[1:]
    else:
        headers = [f"column_{i + 1}" for i in range(modal_width)]
        data = body
        synthetic = True
        notes.append("No header row detected; columns were identified from their values.")

    norm: list[list[str]] = []
    dropped = 0
    for r in data:
        r = (r + [""] * modal_width)[:modal_width]
        # footer/summary rows such as "24 listings exported"
        if modal_width > 3 and sum(1 for c in r if c.strip()) < 2:
            dropped += 1
            continue
        norm.append(r)
    if dropped:
        notes.append(f"Ignored {dropped} summary/blank-ish row(s).")
    return Table(
        headers, norm, header_row_index=start, synthetic_headers=synthetic, notes=notes, **meta
    )


_ADDRESS_START = re.compile(r"^\s*\d+[A-Za-z]?\s+\S")
_NO_DELIMITER = "\x1f"  # never present in text; makes csv.reader keep each line as one cell


def looks_like_address_list(text: str) -> bool:
    """One full address per line (optionally under a header), e.g. a pasted list.

    Such lines contain commas ("123 Main St, Atlanta, GA 30301") that must not be treated as
    column separators. Lines with prices or coordinates are real tables, not address lists.
    """
    lines = [ln.strip().strip('"') for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if not _ADDRESS_START.match(lines[0]):
        if re.search(r"[,\t;|]", lines[0]):
            return False  # a multi-column header means this is a real table
        lines = lines[1:]  # allow one header line such as "Address"
    if not lines:
        return False
    ok = 0
    for ln in lines[:60]:
        if "$" in ln or re.search(r"[\t;|]|\d+\.\d", ln) or ln.count(",") > 5:
            continue
        tail = ln.rsplit(",", 1)[-1].strip() if "," in ln else ""
        if re.fullmatch(r"\d{1,4}", tail):
            continue  # "...,3": a bed count or similar, not part of an address
        if _ADDRESS_START.match(ln) and parse_full_address(ln).has_street:
            ok += 1
    return ok / len(lines[:60]) >= 0.8


def read_text_table(text: str, encoding: str | None = None) -> Table:
    if looks_like_address_list(text):
        reader = csv.reader(io.StringIO(text), delimiter=_NO_DELIMITER)
        t = table_from_rows(list(reader), delimiter=None, encoding=encoding)
        t.notes.append("Read as a list of addresses (one per line).")
        return t
    delimiter = sniff_delimiter(text)
    if delimiter is None:
        raw = [[ln.strip()] for ln in text.splitlines() if ln.strip()]
        return table_from_rows(raw, delimiter=None, encoding=encoding)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    return table_from_rows(list(reader), delimiter=delimiter, encoding=encoding)


def read_excel_table(source: Path | bytes) -> Table:
    import openpyxl  # lazy: only needed for .xlsx

    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    wb = openpyxl.load_workbook(stream, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    raw = [["" if c is None else str(c) for c in row] for row in ws.iter_rows(values_only=True)]
    t = table_from_rows(raw, encoding="xlsx")
    if len(wb.worksheets) > 1:
        t.notes.append(f"Workbook has {len(wb.worksheets)} sheets; read the first ('{ws.title}').")
    return t


def read_table(path_or_bytes: Path | str | bytes, filename: str | None = None) -> Table:
    """Entry point. `filename` is needed (for its extension) when passing raw bytes."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        name, data = path.name, path.read_bytes()
    else:
        name, data = filename or "upload.csv", path_or_bytes
    if name.lower().endswith((".xlsx", ".xlsm")):
        return read_excel_table(data)
    if name.lower().endswith(".xls"):
        raise ValueError(
            "Old .xls files are not supported. In Excel, use File > Save As > CSV or .xlsx."
        )
    text, enc = decode_bytes(data)
    return read_text_table(text, enc)
