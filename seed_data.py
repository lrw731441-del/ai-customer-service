"""初始化种子数据：创建管理员账户 + 导入知识库文件"""
import os
from passlib.hash import bcrypt
from sqlmodel import Session, select

from app.models.database import engine, AdminUser, KnowledgeDocument
from app.rag.loader import load_document
from app.rag.splitter import semantic_split
from app.rag.vectordb import add_documents


def create_admin():
    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD", "admin123")
    password_hash = bcrypt.hash(password)

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
        print(f"管理员 {username} 已创建 (密码: {password})")


def import_knowledge_base():
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print("knowledge_base/ 目录不存在，跳过")
        return

    with Session(engine) as session:
        for filename in os.listdir(kb_dir):
            if not filename.endswith((".md", ".txt", ".pdf", ".docx")):
                continue

            filepath = os.path.join(kb_dir, filename)
            ext = filename.rsplit(".", 1)[-1]
            file_size = os.path.getsize(filepath)

            existing = session.exec(
                select(KnowledgeDocument).where(KnowledgeDocument.filename == filename)
            ).first()
            if existing:
                print(f"  文件 {filename} 已导入，跳过")
                continue

            doc = KnowledgeDocument(filename=filename, file_type=ext, file_size=file_size, status="processing")
            session.add(doc)
            session.commit()

            try:
                text = load_document(filepath, ext)
                chunks = semantic_split(text, source=filename)
                chunk_count = add_documents(chunks)
                doc.chunk_count = chunk_count
                doc.status = "ready"
            except Exception as e:
                doc.status = "error"
                session.add(doc)
                session.commit()
                print(f"  导入 {filename} 失败: {e}")
                continue

            session.add(doc)
            session.commit()
            print(f"  已导入 {filename}: {doc.chunk_count} 个片段")


if __name__ == "__main__":
    from app.models.database import init_db
    init_db()

    print("=== 创建管理员 ===")
    create_admin()
    print("\n=== 导入知识库 ===")
    import_knowledge_base()
    print("\n种子数据初始化完成")
