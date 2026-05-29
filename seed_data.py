"""初始化种子数据：创建管理员账户 + 导入知识库文件"""
import os
import logging
import bcrypt
from sqlmodel import Session, select, func

from app.models.database import engine, AdminUser, KnowledgeDocument
from app.rag.loader import load_document
from app.rag.splitter import semantic_split
from app.rag.vectordb import add_documents, get_document_count, delete_collection

logger = logging.getLogger(__name__)


def create_admin():
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "")
    if not password:
        print("错误: 请设置 ADMIN_PASSWORD 环境变量")
        return
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    with Session(engine) as session:
        existing = session.exec(
            select(AdminUser).where(AdminUser.username == username)
        ).first()
        if existing:
            print(f"管理员 {username} 已存在，跳过")
            return

        user = AdminUser(username=username, password_hash=password_hash, role="admin")
        session.add(user)
        session.commit()
        print(f"管理员 {username} 已创建")


def _check_and_heal_kb():
    """检测并修复知识库数据不一致（DB有记录但向量库为空）"""
    with Session(engine) as session:
        db_count = session.exec(
            select(func.count(KnowledgeDocument.id))
        ).one()
    vec_count = get_document_count()

    if db_count > 0 and vec_count == 0:
        print(f"⚠ 检测到数据不一致: 数据库有 {db_count} 条记录，但向量库为空。自动修复中...")
        with Session(engine) as session:
            for doc in session.exec(
                select(KnowledgeDocument).where(KnowledgeDocument.status == "ready")
            ).all():
                session.delete(doc)
            session.commit()
        delete_collection()
        print("  已清空旧记录和向量库，将重新导入")
        return True
    elif db_count > 0 and vec_count > 0:
        print(f"知识库状态正常: {db_count} 个文件, {vec_count} 个向量片段")
        return False
    return True


def import_knowledge_base():
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print("knowledge_base/ 目录不存在，跳过")
        return

    need_import = _check_and_heal_kb()

    with Session(engine) as session:
        for filename in sorted(os.listdir(kb_dir)):
            if not filename.endswith((".md", ".txt", ".pdf", ".docx")):
                continue

            filepath = os.path.join(kb_dir, filename)
            ext = filename.rsplit(".", 1)[-1]
            file_size = os.path.getsize(filepath)

            existing = session.exec(
                select(KnowledgeDocument).where(KnowledgeDocument.filename == filename)
            ).first()

            if existing and not need_import:
                continue

            if existing and need_import:
                session.delete(existing)
                session.commit()

            doc = KnowledgeDocument(
                filename=filename, file_type=ext, file_size=file_size, status="processing"
            )
            session.add(doc)
            session.commit()

            try:
                text = load_document(filepath, ext)
                chunks = semantic_split(text, source=filename)
                chunk_count = add_documents(chunks)
                doc.chunk_count = chunk_count
                doc.status = "ready"
                print(f"  ✓ {filename}: {chunk_count} 个片段")
            except Exception as e:
                doc.status = "error"
                print(f"  ✗ {filename}: {e}")

            session.add(doc)
            session.commit()

    print(f"导入完成，向量总数: {get_document_count()}")


if __name__ == "__main__":
    from app.models.database import init_db
    init_db()

    print("=== 创建管理员 ===")
    create_admin()
    print("\n=== 导入知识库 ===")
    import_knowledge_base()
    print("\n种子数据初始化完成")
