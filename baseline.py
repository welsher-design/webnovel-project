import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

data_dir = r'C:\Users\whx\PycharmProjects\网文项目\data'
texts, labels = [], []

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

print('总样本：', len(texts), '热门：', sum(labels), '普通：', len(labels)-sum(labels))

# 简单分词，不做复杂过滤
texts_cut = [' '.join(jieba.cut(t)) for t in texts]

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=1, tokenizer=str.split, token_pattern=None)
X = vectorizer.fit_transform(texts_cut)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

for name, clf in [('逻辑回归', LogisticRegression(max_iter=1000)), ('随机森林', RandomForestClassifier(n_estimators=100, random_state=42))]:
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)
    print(f'\n=== {name} ===')
    print(classification_report(y_test, preds, zero_division=0))
    try:
        print('AUC：', roc_auc_score(y_test, probs))
    except:
        print('AUC 无法计算')