import io
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from langchain_core.documents import Document


STRUCTURED_TABLE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def is_structured_table_path(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in STRUCTURED_TABLE_EXTENSIONS


def _read_table_frames(path: str | os.PathLike) -> list[tuple[str | None, pd.DataFrame]]:
    table_path = Path(path)
    suffix = table_path.suffix.lower()

    if suffix == ".csv":
        return [(None, pd.read_csv(table_path))]

    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(table_path, sheet_name=None)
        return list(sheets.items())

    raise ValueError(f"Unsupported structured table type: {suffix}")


def is_small_table(path: str, max_rows: int = 200, max_tokens: int = 6000) -> bool:
    frames = _read_table_frames(path)
    total_rows = sum(len(df) for _, df in frames)
    estimated_tokens = sum(len(df.to_string()) for _, df in frames) // 3
    return total_rows <= max_rows and estimated_tokens <= max_tokens


def render_full_table_document(path: str) -> Document:
    frames = _read_table_frames(path)
    source_name = os.path.basename(path)
    rendered_parts: list[str] = []
    total_rows = 0

    for sheet_name, df in frames:
        total_rows += len(df)
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)

        title = f"Source file: {source_name}"
        if sheet_name is not None:
            title += f"\nSheet: {sheet_name}"
        title += f"\nRows: {len(df)}"

        rendered_parts.append(f"{title}\nCSV:\n{buffer.getvalue().strip()}")

    return Document(
        page_content="\n\n".join(rendered_parts),
        metadata={
            "source": source_name,
            "source_file": source_name,
            "source_path": str(path),
            "table_full_context": True,
            "total_rows": total_rows,
        },
    )


def small_table_documents(source_paths: Iterable[str] | None) -> List[Document]:
    documents: list[Document] = []
    for source_path in source_paths or []:
        if not is_structured_table_path(source_path):
            continue
        try:
            if is_small_table(str(source_path)):
                documents.append(render_full_table_document(str(source_path)))
        except Exception as exc:
            print(f"⚠️ Could not inspect structured table {source_path}: {exc}")
    return documents


def is_tabular_row_document(document: Document) -> bool:
    metadata = document.metadata or {}
    source = str(metadata.get("source", "")).lower()
    extracted_from = str(metadata.get("extracted_from", ""))

    return (
        metadata.get("table_row") is True
        or source.endswith((".csv", ".xlsx", ".xls"))
        or extracted_from.endswith("_table")
    )
