
import json
import random
from collections import defaultdict
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from datasets import load_dataset

# ========== 配置 ==========
N_POPULAR = 20      # 热门书数量
N_NORMAL = 20       # 普通书数量
CHAPTERS_PER_BOOK = 3   # 每本书取前几章
MAX_CHARS_PER_BOOK = 15000  # 每本书最大字符数（约1.5万字）
OUTPUT_DIR = "data"
SEED = 42
# ==========================

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在加载数据集（首次运行会下载，可能需要几分钟）...")
ds = load_dataset("wdndev/webnovel-chinese", split="train")
print(f"数据集加载完成，共 {len(ds)} 条记录")

# ========== 1. 按书籍分组 ==========
print("正在按书籍分组...")
book_chapters = defaultdict(list)

for item in ds:
    title = item["title"].strip()
    chapter = item["chapter"].strip()
    text = item["text"].strip()
    if not title or not text:
        continue
    book_chapters[title].append({
        "chapter": chapter,
        "text": text
    })

# 只保留至少有3章的书
valid_books = {
    title: chapters
    for title, chapters in book_chapters.items()
    if len(chapters) >= CHAPTERS_PER_BOOK
}
print(f"有效书籍数量（至少{CHAPTERS_PER_BOOK}章）：{len(valid_books)}")

if len(valid_books) < N_POPULAR + N_NORMAL:
    print(f"警告：有效书籍只有 {len(valid_books)} 本，少于所需的 {N_POPULAR + N_NORMAL} 本")
    print("将自动调整热门/普通书数量")

# ========== 2. 模拟榜单：按章节数排序 ==========
# 章节数越多 → 说明书越完整 → 模拟热门书
sorted_books = sorted(
    valid_books.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

total_needed = min(N_POPULAR + N_NORMAL, len(sorted_books))
half = total_needed // 2

popular_books = sorted_books[:half]            # 热门：章节最多的一批
normal_books = sorted_books[-half:]             # 普通：章节最少的一批

print(f"热门书：{len(popular_books)} 本")
print(f"普通书：{len(normal_books)} 本")

# ========== 3. 保存为 txt ==========
def save_book(book_title, chapters, label, index):
    """将一本书的前N章保存为txt"""
    selected = chapters[:CHAPTERS_PER_BOOK]
    full_text = "\n\n".join([ch["text"] for ch in selected])
    full_text = full_text[:MAX_CHARS_PER_BOOK]

    # 清理文件名中的非法字符
    safe_title = "".join(
        c for c in book_title if c not in r'\/:*?"<>|'
    ).strip()
    if not safe_title:
        safe_title = f"书{index}"

    filename = f"{label}_{safe_title}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_text)

    return filename, len(full_text)

# 保存热门书
print("\n正在保存热门书...")
for i, (title, chapters) in enumerate(popular_books):
    fname, size = save_book(title, chapters, "热门", i + 1)
    if (i + 1) % 10 == 0:
        print(f"  已保存 {i+1}/{len(popular_books)} 本")

# 保存普通书
print("\n正在保存普通书...")
for i, (title, chapters) in enumerate(normal_books):
    fname, size = save_book(title, chapters, "普通", i + 1)
    if (i + 1) % 10 == 0:
        print(f"  已保存 {i+1}/{len(normal_books)} 本")

print(f"\n完成！文件已保存到 {os.path.abspath(OUTPUT_DIR)}")
print(f"总计：热门 {len(popular_books)} 本，普通 {len(normal_books)} 本")