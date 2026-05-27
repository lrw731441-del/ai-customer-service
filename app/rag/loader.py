from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_text_file(filepath: str) -> str:
    """加载纯文本文件 (txt/md)"""
    return Path(filepath).read_text(encoding="utf-8")


def load_pdf(filepath: str) -> str:
    """加载 PDF 文件"""
    reader = PdfReader(filepath)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def load_docx(filepath: str) -> str:
    """加载 Word 文件"""
    doc = Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])


def load_document(filepath: str, file_type: str) -> str:
    """根据文件类型加载文档"""
    loaders = {
        "txt": load_text_file,
        "md": load_text_file,
        "pdf": load_pdf,
        "docx": load_docx,
    }
    loader = loaders.get(file_type)
    if not loader:
        raise ValueError(f"不支持的文件类型: {file_type}")
    return loader(filepath)
