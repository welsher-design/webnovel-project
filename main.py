import os
import numpy as np
import torch
import torch.nn as nn
import jieba.posseg as pseg
from sklearn.metrics import roc_curve, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
# ---------- 1.读入 ----------
data_dir = 'data'
texts, labels, names = [], [], []
for fname in os.listdir(data_dir):
    if not fname.endswith('.txt'):
        continue
    if fname.startswith('热门_'):
        label = 1
    elif fname.startswith('普通_'):
        label = 0
    else:
        continue

    with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
        text = f.read()[:10000]

    texts.append(text)
    labels.append(label)
    names.append(fname)
print('总样本数：', len(texts))
print('热门数：', sum(labels))
print('普通数：', len(labels) - sum(labels))
if len(texts) < 4:
    raise SystemExit('样本太少，请先在 data 文件夹里放入至少 4 个 txt 文件')
# ---------- 2. 中文分词 ----------
# 停用词表：标点 + 常见虚词
STOPWORDS = set([
    '，', '。', '、', '；', '：', '？', '！', '“', '”', '‘', '’','.','·',
    '（', '）', '《', '》', '…', '—', '·', '\n', ' ',
    '的', '了', '是', '在', '和', '就', '都', '而', '及', '与',
    '着', '或', '一个', '没有', '我们', '你们', '他们', '它们',
    '这', '那', '有', '也', '不', '人', '上', '下', '中', '个',
    '他', '她', '它', '我', '你', '从', '到', '把', '被', '给',
    '对', '为', '以', '之', '其', '所', '但', '却', '还', '又',
    '很', '更', '最', '会', '能', '要', '可', '只', '才', '已',
    '呢', '吗', '吧', '啊', '呀', '哦', '嗯', '哈', '嘿',
])
NAME_BLACKLIST = set(['苏夏', '欧阳', '白晨', '小花','爷爷','院长','老子','哥哥','妹妹','李强','自己','家里'])
def clean_cut(text):
    words = pseg.cut(text)
    result = []
    for w, flag in words:
        if len(w) < 2:
            continue
        if w in STOPWORDS or w in NAME_BLACKLIST:
            continue
        if w.isdigit():
            continue
        if flag == 'nr':   # 过滤人名
            continue
        result.append(w)
    return ' '.join(result)
texts_cut = [clean_cut(t) for t in texts]
print('分词完成，示例：', texts_cut[0][:120])
# ---------- 3. TF-IDF 向量化 ----------
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1,2),
    min_df=2,
    tokenizer=str.split,
    token_pattern=None
)
X = vectorizer.fit_transform(texts_cut).toarray().astype('float32')
y = np.array(labels, dtype='float32')
print('特征矩阵大小：', X.shape)
# ---------- 4. 划分训练/测试集 ----------
stratify_arg = y if len(set(labels)) > 1 and min(np.bincount(y.astype(int))) >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=stratify_arg
)
print('训练集：', X_train.shape, '测试集：', X_test.shape)
# ---------- 5. 转成 PyTorch 张量 ----------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
train_ds = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
# ---------- 6. 定义 MLP 模型 ----------
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(X_train.shape[1]).to(device)
print('使用设备：', device)
# ---------- 7. 训练 ----------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f}')
# ---------- 8. 评估 ----------
model.eval()
with torch.no_grad():
    logits = model(X_test_t.to(device))
    probs = torch.sigmoid(logits).cpu().numpy().ravel()
# 用 ROC 曲线找最佳阈值
fpr, tpr, thresholds = roc_curve(y_test, probs)
best_idx = (tpr - fpr).argmax()
best_threshold = thresholds[best_idx]
# 用最佳阈值预测
preds_best = (probs > best_threshold).astype(int)
print('\n========== 评估结果 ==========')
print(f'最佳阈值：{best_threshold:.4f}')
print(classification_report(y_test, preds_best, zero_division=0))
try:
    print('AUC：', roc_auc_score(y_test, probs))
except ValueError:
    print('AUC 无法计算（测试集只有单一类别）')
# 训练集对比
with torch.no_grad():
    train_logits = model(X_train_t.to(device))
    train_probs = torch.sigmoid(train_logits).cpu().numpy().ravel()
    train_preds = (train_probs > best_threshold).astype(int)
print('训练集 accuracy：', accuracy_score(y_train, train_preds))
print('测试集 accuracy：', accuracy_score(y_test, preds_best))
# ---------- 9. 看哪些词最重要 ----------
feature_names = vectorizer.get_feature_names_out()
importances = model.net[0].weight.detach().cpu().numpy()
importance_scores = np.abs(importances).sum(axis=0)
top_idx = importance_scores.argsort()[::-1][:20]
print('\n========== 最重要的 20 个词/短语 ==========')
for i in top_idx:
    print(feature_names[i], round(float(importance_scores[i]), 4))