import pdfplumber
from docx import Document
from pathlib import Path


class ParserError(Exception):
    pass


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext == ".docx":
        return _extract_docx(path)
    else:
        raise ParserError(f"Неподдерживаемый формат файла: «{ext}»")


def _extract_pdf(path: Path) -> str:
    pages = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

    if not pages:
        raise ParserError(
            "Файл содержит только изображения — текстовый слой не обнаружен"
        )

    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    doc = Document(path)
    parts = []

    for block in doc.element.body:
        tag = block.tag.split("}")[-1]

        if tag == "p":
            # Параграф
            text = "".join(run.text for run in block.iterchildren()
                           if run.tag.split("}")[-1] == "r"
                           for t in run.iterchildren()
                           if t.tag.split("}")[-1] == "t" and t.text)
            if text.strip():
                parts.append(text.strip())

        elif tag == "tbl":
            # Таблица: ячейки объединяются через табуляцию, строки через перевод строки
            rows = []
            for row in block.iterchildren():
                if row.tag.split("}")[-1] != "tr":
                    continue
                cells = []
                for cell in row.iterchildren():
                    if cell.tag.split("}")[-1] != "tc":
                        continue
                    cell_text = "".join(
                        t.text for p in cell.iterchildren()
                        for r in p.iterchildren()
                        for t in r.iterchildren()
                        if t.tag.split("}")[-1] == "t" and t.text
                    )
                    cells.append(cell_text.strip())
                if any(cells):
                    rows.append("\t".join(cells))
            if rows:
                parts.append("\n".join(rows))

    if not parts:
        raise ParserError("Файл не содержит текста")

    return "\n\n".join(parts)
