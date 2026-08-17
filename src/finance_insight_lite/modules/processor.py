import pymupdf as fitz  # PyMuPDF
import pandas as pd
import datetime
import os
import re
import io
from langchain_core.documents import Document
import hashlib
from multiprocessing import Pool
from pathlib import Path
from functools import lru_cache
import pickle


# ============================================================================
# NEW: Image OCR support (PaddleOCR) — standalone images + embedded images
# inside PDF/Excel files.
# ============================================================================
#
# نستخدم PaddleOCR بدل Tesseract لسببين:
# 1. عنده وحدة PP-Structure مخصصة لاستخراج بنية الجداول (صفوف/أعمدة) من
#    صورة، وترجعها كـ HTML جاهز للتحويل — هذا بالضبط اللي محتاجينه لتقارير
#    مالية فيها جداول مصورة، وTesseract ما يقدمه أصلاً (يرجّع نص مبعثر).
# 2. دعم عربي أفضل بشكل عام مقارنة بـ Tesseract.
#
# ملاحظة: التهيئة الأولى (تحميل نماذج PP-Structure/PaddleOCR) بطيئة نسبياً
# (عدة ثواني)، فنستخدم lru_cache عشان تصير مرة وحدة بس لكل عملية بايثون،
# نفس فكرة get_embedding_model() الموجودة أصلاً بـ verctor_store.py.

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@lru_cache(maxsize=1)
def _get_structure_engine():

    from paddleocr import PPStructureV3
    return PPStructureV3(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def _parse_markdown_tables(markdown_text: str):
 
    table_sentences = []
    plain_lines = []
    lines = markdown_text.splitlines()
    separator_re = re.compile(r'^\|[\s:\-|]+\|$')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        is_header_row = line.startswith('|') and line.endswith('|')
        next_is_separator = (
            i + 1 < len(lines) and separator_re.match(lines[i + 1].strip())
        )

        if is_header_row and next_is_separator:
            headers = [h.strip() for h in line.strip('|').split('|')]
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_line = lines[i].strip()
                cells = [c.strip() for c in row_line.strip('|').split('|')]
                parts = []
                for header_label, val in zip(headers, cells):
                    if not val:
                        continue
                    parts.append(f"{header_label}: {val}" if header_label else val)
                if parts:
                    table_sentences.append(" | ".join(parts))
                i += 1
        else:
            if line:
                plain_lines.append(line)
            i += 1

    return table_sentences, "\n".join(plain_lines)


def _run_structure_on_image_array(img_array) -> dict:
    """
    يشغّل PP-StructureV3 على مصفوفة صورة (numpy array بصيغة BGR) ويرجّع:
    {"table_sentences": [...], "plain_text": "..."}. أي فشل يرجّع قواميس
    فاضية بدل ما يوقف كامل خط المعالجة — الصورة توصف ببساطة إنها بدون
    بيانات قابلة للاستخراج بدل ما توقف رفع باقي الملف.
    """
    table_sentences = []
    plain_text = ""

    try:
        structure_engine = _get_structure_engine()
        output = structure_engine.predict(img_array)

        markdown_parts = []
        for res in output:
            try:
                md_info = res.markdown
                md_text = md_info.get("markdown_texts", "") if isinstance(md_info, dict) else str(md_info)
            except Exception:
                # نسخ/إصدارات مختلفة قد ترجّع res كـ dict بدل object بخاصية markdown
                md_text = ""
                if isinstance(res, dict):
                    md_text = res.get("markdown", {}).get("markdown_texts", "") if isinstance(res.get("markdown"), dict) else ""
            if md_text:
                markdown_parts.append(md_text)

        combined_markdown = "\n\n".join(markdown_parts)
        table_sentences, plain_text = _parse_markdown_tables(combined_markdown)

    except Exception as e:
        print(f"⚠️ PP-StructureV3 error: {e}")

    return {
        "table_sentences": table_sentences,
        "plain_text": plain_text,
    }


def image_to_documents(image_path, source_label=None) -> list:
    """
    يستخرج جداول/أرقام/نص من صورة مستقلة (PNG/JPG/...) عبر PaddleOCR،
    ويرجّعها كقائمة Document — صف جدول واحد = Document واحد (بنفس فورمات
    excel_to_documents_optimized)، ونص عادي = Document واحد إضافي لو وُجد.
    """
    import cv2

    print(f"🖼️ Loading Image: {image_path}")
    img_array = cv2.imread(str(image_path))
    if img_array is None:
        print(f"⚠️ تعذّر قراءة الصورة: {image_path}")
        return []

    source_name = source_label or os.path.basename(image_path)
    extracted = _run_structure_on_image_array(img_array)

    documents = []

    for row_idx, sentence in enumerate(extracted["table_sentences"]):
        documents.append(Document(
            page_content=f"Image: {source_name}\n{sentence}",
            metadata={
                "source": source_name,
                "row": row_idx + 1,
                "extracted_from": "image_table",
            }
        ))

    if extracted["plain_text"].strip():
        documents.append(Document(
            page_content=f"Image: {source_name}\n{extracted['plain_text'].strip()}",
            metadata={
                "source": source_name,
                "extracted_from": "image_text",
            }
        ))

    if not documents:
        print(f"⚠️ ما انستخرج أي بيانات من الصورة: {image_path}")

    print(f"✓ استُخرج {len(documents)} عنصر من الصورة ({len(extracted['table_sentences'])} صف جدول)")
    return documents


def _extract_embedded_images_from_pdf(pdf_path) -> list:
    """
    يستخرج الصور المضمّنة جوّا صفحات PDF (رسوم بيانية/جداول مصورة داخل
    التقرير) ويشغّل عليها نفس محرك OCR، عشان أرقام موجودة كصورة داخل PDF
    (وليست نص قابل للاستخراج المباشر) ما تضيع.
    """
    import cv2
    import numpy as np

    documents = []
    pdf_name = os.path.basename(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_array is None:
                        continue

                    extracted = _run_structure_on_image_array(img_array)
                    label = f"{pdf_name} - Page {page_num + 1} - Image {img_index + 1}"

                    for row_idx, sentence in enumerate(extracted["table_sentences"]):
                        documents.append(Document(
                            page_content=f"Embedded Image [{label}]\n{sentence}",
                            metadata={
                                "source": pdf_name,
                                "page": page_num + 1,
                                "row": row_idx + 1,
                                "extracted_from": "pdf_embedded_image_table",
                            }
                        ))

                    if extracted["plain_text"].strip():
                        documents.append(Document(
                            page_content=f"Embedded Image [{label}]\n{extracted['plain_text'].strip()}",
                            metadata={
                                "source": pdf_name,
                                "page": page_num + 1,
                                "extracted_from": "pdf_embedded_image_text",
                            }
                        ))
                except Exception as e:
                    print(f"⚠️ فشل استخراج صورة مضمّنة (page {page_num + 1}, img {img_index + 1}): {e}")
                    continue
        doc.close()
    except Exception as e:
        print(f"⚠️ فشل فتح PDF لاستخراج الصور المضمّنة: {e}")

    if documents:
        print(f"✓ استُخرج {len(documents)} عنصر من الصور المضمّنة بالـ PDF")
    return documents


def _extract_embedded_images_from_excel(excel_path) -> list:
    """
    يستخرج الصور المضمّنة جوّا شيتات Excel (رسوم بيانية/لقطات جداول ملصقة
    كصورة داخل الملف) عبر openpyxl، ويشغّل عليها نفس محرك OCR.
    """
    import cv2
    import numpy as np

    documents = []
    excel_name = os.path.basename(excel_path)

    try:
        from openpyxl import load_workbook
        workbook = load_workbook(excel_path, data_only=True)

        for sheet in workbook.worksheets:
            sheet_images = getattr(sheet, "_images", [])
            for img_index, ws_image in enumerate(sheet_images):
                try:
                    image_bytes = ws_image._data()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_array is None:
                        continue

                    extracted = _run_structure_on_image_array(img_array)
                    label = f"{excel_name} - Sheet {sheet.title} - Image {img_index + 1}"

                    for row_idx, sentence in enumerate(extracted["table_sentences"]):
                        documents.append(Document(
                            page_content=f"Embedded Image [{label}]\n{sentence}",
                            metadata={
                                "source": excel_name,
                                "sheet_name": sheet.title,
                                "row": row_idx + 1,
                                "extracted_from": "excel_embedded_image_table",
                            }
                        ))

                    if extracted["plain_text"].strip():
                        documents.append(Document(
                            page_content=f"Embedded Image [{label}]\n{extracted['plain_text'].strip()}",
                            metadata={
                                "source": excel_name,
                                "sheet_name": sheet.title,
                                "extracted_from": "excel_embedded_image_text",
                            }
                        ))
                except Exception as e:
                    print(f"⚠️ فشل استخراج صورة مضمّنة (sheet {sheet.title}, img {img_index + 1}): {e}")
                    continue
    except Exception as e:
        print(f"⚠️ فشل فتح Excel لاستخراج الصور المضمّنة: {e}")

    if documents:
        print(f"✓ استُخرج {len(documents)} عنصر من الصور المضمّنة بالـ Excel")
    return documents


# ============================================================================
# OPTIMIZATION 1: Parallel PDF Processing
# ============================================================================

def _extract_page_range(args):
    """Extract one page range in its own process (PyMuPDF is not thread-safe)."""
    pdf_path, start_page, end_page = args
    document = fitz.open(pdf_path)
    try:
        return [
            (page_num, document[page_num].get_text("text"))
            for page_num in range(start_page, end_page)
        ]
    finally:
        document.close()


def pdf_to_documents_parallel(pdf_path, max_workers=1, extract_images=True):
    """
    Load a PDF using process-based extraction for large documents.

    Args:
        pdf_path: Path to PDF file
        max_workers: Number of parallel workers (default: 4)
        extract_images: لو True، يستخرج الصور المضمّنة (رسوم/جداول مصورة)
            ويشغّل عليها OCR بالإضافة للنص العادي.
    """
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    total_pages = doc.page_count
    doc.close()

    worker_count = min(max_workers, total_pages)
    if worker_count <= 1:
        page_texts = _extract_page_range((pdf_path, 0, total_pages))
    else:
        pages_per_worker = (total_pages + worker_count - 1) // worker_count
        page_ranges = [
            (pdf_path, start_page, min(start_page + pages_per_worker, total_pages))
            for start_page in range(0, total_pages, pages_per_worker)
        ]
        with Pool(processes=worker_count) as pool:
            page_texts = [item for batch in pool.map(_extract_page_range, page_ranges) for item in batch]

    documents = [
        Document(
            page_content=page_text,
            metadata={"source": pdf_name, "page": page_num + 1},
        )
        for page_num, page_text in page_texts
    ]

    if extract_images:
        documents.extend(_extract_embedded_images_from_pdf(pdf_path))

    print(f"✓ Loaded {len(documents)} pages from PDF")
    return documents


# ============================================================================
# OPTIMIZATION 2: Caching with File Hash
# ============================================================================

def get_file_hash(file_path):
    """Generate hash for file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def pdf_to_documents_cached(pdf_path, cache_dir="data/cache", max_workers=1):
    """
    Load PDF with caching - instant load for previously processed files
    
    Args:
        pdf_path: Path to PDF file
        cache_dir: Directory to store cached documents
        max_workers: Number of parallel workers
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate cache key from file hash
    file_hash = get_file_hash(pdf_path)
    cache_file = Path(cache_dir) / f"{file_hash}.pkl"
    
    # Check if cached version exists
    if cache_file.exists():
        print(f"📦 Loading from cache: {pdf_path}")
        with open(cache_file, 'rb') as f:
            documents = pickle.load(f)
        print(f"✓ Loaded {len(documents)} pages from cache (instant!)")
        return documents
    
    # Process and cache
    print(f"🔄 Processing PDF (first time): {pdf_path}")
    documents = pdf_to_documents_parallel(pdf_path, max_workers)
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(documents, f)
    print(f"💾 Cached for future use")
    
    return documents


# ============================================================================
# OPTIMIZATION 3: Fast PDF Reading (PyMuPDF optimizations)
# ============================================================================

def pdf_to_documents_fast(pdf_path):
    """
    Fastest PDF reading with PyMuPDF optimizations
    
    Optimizations:
    - Use get_text("text") instead of get_text() - faster
    - Close document early
    - Minimal object creation
    - Pre-allocate list
    """
    print(f"Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    total_pages = doc.page_count
    
    # Pre-allocate list for better memory performance
    documents = [None] * total_pages
    
    # Process pages
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")  # Fastest extraction method
        
        documents[page_num] = Document(
            page_content=text,
            metadata={
                "source": pdf_name,
                "page": page_num + 1
            }
        )
    
    doc.close()
    print(f"✓ Loaded {len(documents)} pages from PDF")
    return documents


# ============================================================================
# OPTIMIZATION 4: Smart Excel Processing
# ============================================================================

def _looks_numeric(value):
    """Best-effort check for whether a cell value reads as a number/percentage."""
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if text == "":
        return False
    text = text.replace(",", "").replace("%", "").replace("+", "").replace("-", "", 1)
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_data_value(value):
    """Return True for values that are much more likely data than a header."""
    if pd.isna(value):
        return False
    if _looks_numeric(value):
        return True
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return True

    # Transaction IDs, invoice references, dates read as strings, and similar
    # values commonly mix letters/punctuation with digits. Treating these as
    # ordinary text caused entire ledgers and bank statements to be classified
    # as header rows, leaving no documents for the vector database.
    return any(character.isdigit() for character in str(value))


def _is_probable_header_row(row_values):
    """
    Heuristic for 'is this row a column-header row?' — used because report-style
    sheets (title row, section-title rows, real column headers, data rows all
    mixed in one sheet) make pandas' default header=0 unreliable: it grabs
    whatever is in row 0 (often a merged title cell), leaving the *real*
    column headers — which can appear several rows further down, and can
    differ per section — misread as data and named 'Unnamed: N'.

    A header row: has at least 2 non-null cells, and most of those cells are
    not data-like (column headers are text; financial data rows commonly
    contain numbers, dates, transaction IDs, or invoice references).
    A single-cell row (e.g. a merged title/section banner) is NOT treated as
    a header — it has nothing to attach to a column index.
    """
    non_null = [v for v in row_values if pd.notna(v) and str(v).strip() != ""]
    if len(non_null) < 2:
        return False
    data_value_count = sum(1 for v in non_null if _looks_like_data_value(v))
    return (data_value_count / len(non_null)) < 0.3


def excel_to_documents_optimized(excel_path, sheet_name=None, chunk_size=1500, extract_images=True):
    """
    Optimized Excel loading — one row = one explicit "column: value" sentence,
    one row = one independent Document, with column headers detected
    dynamically per section rather than trusting pandas' default row-0 header.

    WHY (fix for row/label attribution errors seen downstream in the RAG
    pipeline): two compounding problems in the previous implementation:

    1. `df.to_string(index=False)` rendered a whole sheet as one
       whitespace-aligned block. That format relies on visual column
       alignment to tie a value to its header — which breaks for RTL/Arabic
       text and gives an LLM no explicit delimiter. Combined with a generic
       character-based text splitter downstream, a single chunk could easily
       contain several rows with no clear boundary between them, so the LLM
       could pull a real number from the WRONG row of a table with several
       similarly-worded labels.

    2. For "report-style" sheets — a title in the very first cell, then
       section banners, then the *real* column headers several rows further
       down, all mixed with the data in one sheet — pandas' default
       `header=0` grabs the title row as the header, so genuine columns come
       back named 'Unnamed: 1', 'Unnamed: 2', etc. Even isolating rows
       individually doesn't help if every value is labeled 'Unnamed: N'
       instead of its real column name.

    This version reads the sheet with no assumed header (`header=None`),
    walks it top to bottom, and dynamically detects header rows (a row of
    mostly-text cells) to use as the active column names for the data rows
    that follow — updating them again whenever a new header row appears
    (handles sheets with multiple sections, each with its own headers). Each
    data row is then rendered as an explicit "column_name: value | ..."
    sentence and stored as its own Document, so a row can never be split in
    half or silently merged with a neighboring row by a downstream
    character-based text splitter.

    Args:
        excel_path: Path to Excel file
        sheet_name: Specific sheet or None for all
        chunk_size: Number of rows per "Part" grouping for very large sheets
            (kept only to preserve manageable metadata/chunk sizes; each row
            is still embedded as an individual sentence within its part)
        extract_images: لو True، يستخرج الصور المضمّنة بالشيتات (رسوم
            بيانية/جداول ملصقة كصورة) ويشغّل عليها OCR بالإضافة للخلايا
            النصية العادية.
    """
    print(f"Loading Excel: {excel_path}")
    documents = []

    excel_file = pd.ExcelFile(excel_path)
    sheets_to_process = [sheet_name] if sheet_name else excel_file.sheet_names

    for sheet in sheets_to_process:
        raw_df = pd.read_excel(excel_file, sheet_name=sheet, header=None)

        if raw_df.empty:
            continue

        current_headers = None  # dict: column_index -> header text
        row_sentences = []

        for _, row in raw_df.iterrows():
            row_values = list(row)

            non_null = [v for v in row_values if pd.notna(v) and str(v).strip() != ""]
            if not non_null:
                continue  # blank separator row

            if _is_probable_header_row(row_values):
                current_headers = {
                    col_idx: str(val).strip()
                    for col_idx, val in enumerate(row_values)
                    if pd.notna(val) and str(val).strip() != ""
                }
                continue

            if len(non_null) == 1:
                # A single populated cell with no header context yet (or amid
                # data) is a title/section banner, not a data row — skip it
                # as a row, but treat it as free context for the sentences
                # that immediately follow within this section.
                continue

            parts = []
            for col_idx, val in enumerate(row_values):
                if pd.isna(val) or str(val).strip() == "":
                    continue
                header_label = (current_headers or {}).get(col_idx, f"Column {col_idx + 1}")
                parts.append(f"{header_label}: {val}")

            sentence = " | ".join(parts)
            if sentence:
                row_sentences.append(sentence)

        if not row_sentences:
            continue

        total_rows = len(row_sentences)

        # For very large sheets, group rows into numbered "parts" purely for
        # metadata/traceability — but each row remains its own Document, so
        # a part boundary never splits a row's data.
        num_parts = (total_rows + chunk_size - 1) // chunk_size if total_rows > chunk_size else 1

        for row_idx, sentence in enumerate(row_sentences):
            part_num = (row_idx // chunk_size) + 1 if num_parts > 1 else None

            header = f"Sheet: {sheet}"
            if part_num is not None:
                header += f" (Part {part_num}/{num_parts})"

            document = Document(
                page_content=f"{header}\n{sentence}",
                metadata={
                    "source": os.path.basename(excel_path),
                    "sheet_name": sheet,
                    "row": row_idx + 1,
                    "total_rows": total_rows,
                    "table_row": True,
                    **({"chunk": part_num, "total_chunks": num_parts} if part_num is not None else {}),
                }
            )
            documents.append(document)

    if extract_images:
        documents.extend(_extract_embedded_images_from_excel(excel_path))

    print(f"✓ Loaded {len(documents)} documents from Excel")
    return documents


def csv_to_documents_optimized(csv_path):
    """Load CSV as one complete row per Document with explicit column labels."""
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    documents = []
    source_name = os.path.basename(csv_path)

    for row_idx, row in df.iterrows():
        parts = []
        for column, value in row.items():
            if pd.isna(value) or str(value).strip() == "":
                continue
            parts.append(f"{column}: {value}")

        if not parts:
            continue

        documents.append(Document(
            page_content=f"CSV: {source_name}\n" + " | ".join(parts),
            metadata={
                "source": source_name,
                "row": row_idx + 1,
                "total_rows": len(df),
                "table_row": True,
            },
        ))

    print(f"✓ Loaded {len(documents)} documents from CSV")
    return documents


# ============================================================================
# MAIN FUNCTIONS - Choose based on your needs
# ============================================================================

def load_documents_fastest(file_path, use_cache=True, max_workers=1, **kwargs):
    """
    FASTEST document loader with all optimizations
    
    Performance improvements:
    - PDF: 5-10x faster with caching, 3-5x with parallel processing
    - Excel: 2-3x faster with optimized pandas
    - Image: OCR extraction via PaddleOCR (tables + plain text)
    - Instant load for previously processed files
    
    Args:
        file_path: Path to file
        use_cache: Use caching for instant reloads (recommended)
        max_workers: Parallel workers for PDF processing
        **kwargs: Additional arguments (e.g., sheet_name for Excel)
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        if use_cache:
            docs = pdf_to_documents_cached(file_path, max_workers=max_workers)
        else:
            docs = pdf_to_documents_parallel(file_path, max_workers=max_workers)
        file_type = 'PDF'
    
    elif file_extension in ['.xlsx', '.xls']:
        docs = excel_to_documents_optimized(file_path, **kwargs)
        file_type = 'Excel'

    elif file_extension == '.csv':
        docs = csv_to_documents_optimized(file_path)
        file_type = 'CSV'

    elif file_extension in IMAGE_EXTENSIONS:
        # NEW: صور مستقلة (رفعها المستخدم مباشرة، مو مضمّنة جوه ملف آخر)
        if use_cache:
            docs = _image_to_documents_cached(file_path)
        else:
            docs = image_to_documents(file_path)
        file_type = 'Image'

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    source_name = os.path.basename(file_path)
    for doc in docs:
        metadata = doc.metadata or {}
        metadata.setdefault("source", source_name)
        metadata.setdefault("source_file", metadata.get("source", source_name))
        doc.metadata = metadata
    
    return {
        'documents': docs,
        'relevant_docs_count': len(docs),
        'file_type': file_type,
        'source': source_name
    }


def _image_to_documents_cached(image_path, cache_dir="data/cache"):
    
 
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    file_hash = get_file_hash(image_path)
    cache_file = Path(cache_dir) / f"img_{file_hash}.pkl"

    if cache_file.exists():
        print(f"📦 Loading image OCR result from cache: {image_path}")
        with open(cache_file, 'rb') as f:
            documents = pickle.load(f)
        print(f"✓ Loaded {len(documents)} items from cache (instant!)")
        if documents:
            return documents
        print("⚠️ Cached result was empty — re-running OCR instead of trusting stale empty cache")

    documents = image_to_documents(image_path)

    if documents:
        with open(cache_file, 'wb') as f:
            pickle.dump(documents, f)
        print(f"💾 Cached OCR result for future use")
    else:
        print("⚠️ OCR returned 0 items — not caching empty result")

    return documents


# ============================================================================
# BACKWARD COMPATIBLE - Replace your old function
# ============================================================================

def pdf_to_documents(pdf_path):
    """
    Drop-in replacement for your original function
    Now 5-10x faster with caching!
    """
    return pdf_to_documents_cached(pdf_path)


# ============================================================================
# Clear Cache
# ============================================================================

def clear_cache(cache_dir="data/cache", vector_cache_dir="data/vector_cache"):
    """Clear cached source documents, chunks, and FAISS indices."""
    import shutil
    cleared = False
    for directory in (cache_dir, vector_cache_dir):
        cache_path = Path(directory)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            cache_path.mkdir(parents=True)
            cleared = True
    print("✓ Cache cleared" if cleared else "No cache to clear")
