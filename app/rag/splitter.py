import re
from typing import List, Dict


def semantic_split(text: str, max_chunk_size: int = 300, source: str = "") -> List[Dict[str, str]]:
    """语义切分：按段落+句子边界切分，保持语义完整"""

    # 先按段落切分
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果加入当前段落不超限，合并
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n"
        else:
            # 当前 chunk 够大了，保存
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    # 保存最后一个 chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [{"content": chunk, "source": source} for chunk in chunks]
