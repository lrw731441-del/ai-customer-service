import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlmodel import Session, select

from app.models.database import get_session, KnowledgeDocument
from app.rag.loader import load_document
from app.rag.splitter import semantic_split
from app.rag.vectordb import add_documents, delete_collection, get_document_count
from app.dependencies import verify_jwt_token

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
UPLOAD_DIR = "knowledge_base"


@router.post("/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}，仅支持 {ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="文件大小不能超过 10MB")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(filepath, "wb") as f:
        f.write(content)

    doc = KnowledgeDocument(
        filename=file.filename,
        file_type=ext,
        file_size=len(content),
        status="processing",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        text = load_document(filepath, ext)
        chunks = semantic_split(text, source=file.filename)
        chunk_count = add_documents(chunks)
        doc.chunk_count = chunk_count
        doc.status = "ready"
    except Exception as e:
        doc.status = "error"
        db.add(doc)
        db.commit()
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")

    db.add(doc)
    db.commit()
    return {"ok": True, "document_id": doc.id, "filename": file.filename, "chunk_count": chunk_count}


@router.get("")
async def list_knowledge(
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    docs = db.exec(
        select(KnowledgeDocument).order_by(KnowledgeDocument.created_at.desc())
    ).all()
    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "file_size": d.file_size,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "total_chunks": get_document_count(),
    }


@router.delete("/{doc_id}")
async def delete_knowledge(
    doc_id: int,
    db: Session = Depends(get_session),
    token: dict = Depends(verify_jwt_token),
):
    doc = db.get(KnowledgeDocument, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")

    # 重建整个知识库
    delete_collection()

    # 重新处理剩余文档
    remaining = db.exec(
        select(KnowledgeDocument).where(
            KnowledgeDocument.id != doc_id,
            KnowledgeDocument.status == "ready"
        )
    ).all()
    for r in remaining:
        filename = r.filename
        # Find matching file in upload dir
        for f in os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []:
            if f.endswith("_" + filename):
                filepath = os.path.join(UPLOAD_DIR, f)
                try:
                    text = load_document(filepath, r.file_type)
                    chunks = semantic_split(text, source=filename)
                    add_documents(chunks)
                except Exception:
                    pass
                break

    db.delete(doc)
    db.commit()
    return {"ok": True, "deleted_id": doc_id}
