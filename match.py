import os
import random
from collections import defaultdict
from datasets import load_from_disk

# ========== 配置（绝对路径）==========
TOP_FILE = r'C:\Users\whx\PycharmProjects\网文项目\qidian_top.txt'
CACHE_DIR = r'C:\Users\whx\PycharmProjects\网文项目\webnovel_cache'
OUTPUT_DIR = r'C:\Users\whx\PycharmProjects\网文项目\data'
CHAPTERS_PER_BOOK = 3
MAX_CHARS = 15000
SEED = 42
# =====================================

random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取榜单书名
print("检查榜单文件：", TOP_FILE)
print("文件是否存在：", os.path.exists(TOP_FILE))
print("文件大小（字节）：", os.path.getsize(TOP_FILE) if os.path.exists(TOP_FILE) else 0)

with open(TOP_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()

print("文件前200个字符：", repr(raw[:200]))
print("文件总长度：", len(raw))

top_books = set(line.strip() for line in raw.split('\n') if line.strip())
print(f"榜单书名：{len(top_books)} 本")

# 加载数据集
print("加载数据集缓存：", CACHE_DIR)
if not os.path.exists(CACHE_DIR):
    raise SystemExit(f"找不到缓存目录：{CACHE_DIR}\n请确认 webnovel_cache 文件夹存在。")

ds = load_from_disk(CACHE_DIR)
print(f"共 {len(ds)} 条记录")

# 按书名分组
book_chapters = defaultdict(list)
for item in ds:
    title = item["title"].strip()
    text = item["text"].strip()
    if not title or not text:
        continue
    book_chapters[title].append(text)

print(f"不同书名：{len(book_chapters)}")

# 模糊匹配：榜单书名出现在数据集书名里，或反过来
popular = []
matched_names = set()
for title, chapters in book_chapters.items():
    if len(chapters) < CHAPTERS_PER_BOOK:
        continue
    for top in top_books:
        if top in title or title in top:
            popular.append((title, chapters))
            matched_names.add(top)
            break

print(f"匹配到热门：{len(popular)} 本")
print(f"匹配到的榜单书名：{len(matched_names)} 个")
if matched_names:
    print("示例：", list(matched_names)[:10])

# 普通：书名不在榜单里，至少3章
normal_pool = []
for title, chapters in book_chapters.items():
    if title not in top_books and len(chapters) >= CHAPTERS_PER_BOOK:
        normal_pool.append((title, chapters))

random.shuffle(normal_pool)
N = len(popular)
normal = normal_pool[:N]
print(f"普通：{len(normal)} 本")

# 保存
def save_book(title, chapters, label, idx):
    text = "\n\n".join(chapters[:CHAPTERS_PER_BOOK])[:MAX_CHARS]
    safe = "".join(c for c in title if c not in r'\/:*?"<>|').strip() or f"书{idx}"
    filepath = os.path.join(OUTPUT_DIR, f"{label}_{safe}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

# 清空旧的 data 文件夹里的 txt
for f in os.listdir(OUTPUT_DIR):
    if f.endswith('.txt'):
        os.remove(os.path.join(OUTPUT_DIR, f))

for i, (t, ch) in enumerate(popular):
    save_book(t, ch, "热门", i+1)
for i, (t, ch) in enumerate(normal):
    save_book(t, ch, "普通", i+1)

print(f"\n完成！热门 {len(popular)} 本，普通 {len(normal)} 本")
print(f"文件已保存到 {OUTPUT_DIR}")