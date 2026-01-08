# app.py
# =========================================
# Streamlit - LSTM 감정분류 (Review.xlsx 기반)
# - 입력: Review.xlsx (columns: Sentiment, Review)
# - 학습: Tokenizer + Embedding + BiLSTM
# - 기능: (1) 학습 (2) 성능 확인 (3) 리뷰 입력 -> 예측
# =========================================

import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models

import matplotlib.pyplot as plt

st.set_page_config(page_title="리뷰 감정분석 (LSTM)", layout="wide")

# ----------------------------
# 0) 기본 설정
# ----------------------------
DEFAULT_XLSX_PATH = "data/Review.xlsx"  # 프로젝트 폴더 안에 data/Review.xlsx 넣으면 자동 인식

# 감독관 요구 불용어(예시) + "너무" 추가
STOPWORDS = set([
    "이", "가", "을", "를", "은", "는", "에", "에서", "에게",
    "의", "와", "과", "도", "로", "으로",
    "하다", "되다", "있다", "없다",
    "그", "저", "것", "수",
    "좀", "잘", "매우", "정말",
    "때문", "같다",
    "너무"
])

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def simple_tokenize(s: str):
    """
    konlpy 없이도 돌아가게 '가벼운 토큰화' 버전.
    - 한글/영문/숫자만 남기고 나머지 제거
    - 공백 split
    - 불용어 제거
    """
    s = clean_text(s)
    s = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    tokens = [t for t in tokens if (t not in STOPWORDS and len(t) > 1)]
    return tokens

def preprocess_to_string(s: str) -> str:
    # Tokenizer는 문자열을 받으니까, 토큰들을 공백으로 다시 join
    return " ".join(simple_tokenize(s))

def load_review_xlsx(uploaded_file=None, fallback_path=DEFAULT_XLSX_PATH):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df
    # 업로드 안 했으면 로컬 파일(프로젝트 내 data/Review.xlsx) 시도
    try:
        df = pd.read_excel(fallback_path)
        return df
    except Exception:
        return None

def build_model(vocab_size: int, max_len: int, num_classes: int,
                emb_dim=128, lstm_units=64, dropout=0.3):
    model = models.Sequential([
        layers.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True),
        layers.Bidirectional(layers.LSTM(lstm_units)),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_confusion(cm, class_names):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    return fig

# ----------------------------
# UI
# ----------------------------
st.title("리뷰 감정 분석 (LSTM) — Positive / Negative (자동 감지)")
st.caption("엑셀(Review.xlsx)의 Sentiment, Review 컬럼으로 학습 → 리뷰 입력하면 바로 예측합니다.")

with st.sidebar:
    st.header("1) 데이터 로드")
    uploaded = st.file_uploader("Review.xlsx 업로드", type=["xlsx"])

    st.header("2) 학습 설정")
    vocab_size = st.number_input("VOCAB_SIZE (단어 사전 크기)", 5000, 50000, 20000, step=1000)
    max_len = st.number_input("MAX_LEN (패딩 길이)", 20, 300, 120, step=10)
    epochs = st.number_input("epochs", 1, 50, 10, step=1)
    batch_size = st.selectbox("batch_size", [8, 16, 32, 64], index=1)
    test_size = st.slider("test_size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("random_state", 0, 9999, 42, step=1)

    st.header("3) 불용어(Stopwords)")
    st.write(f"현재 불용어 개수: **{len(STOPWORDS)}**")
    extra_sw = st.text_input("추가 불용어(쉼표로 구분)", value="")
    if extra_sw.strip():
        for w in extra_sw.split(","):
            w = w.strip()
            if w:
                STOPWORDS.add(w)

    st.divider()
    train_btn = st.button("🚀 학습 시작", use_container_width=True)

# ----------------------------
# 데이터 로드
# ----------------------------
df = load_review_xlsx(uploaded_file=uploaded)

if df is None:
    st.warning("Review.xlsx를 업로드하거나, 프로젝트 폴더의 data/Review.xlsx를 확인해줘.")
    st.stop()

required_cols = {"Sentiment", "Review"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"엑셀에 필요한 컬럼이 없어: {missing}\n현재 컬럼: {list(df.columns)}")
    st.stop()

df = df[["Sentiment", "Review"]].copy()
df["Sentiment"] = df["Sentiment"].astype(str).fillna("").str.strip()
df["Review"] = df["Review"].astype(str).fillna("").str.strip()
df = df[(df["Sentiment"] != "") & (df["Review"] != "")].copy()

st.subheader("데이터 미리보기")
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(df.head(10), use_container_width=True)
with c2:
    st.write("라벨 분포")
    st.write(df["Sentiment"].value_counts())

# ----------------------------
# 학습 실행 (세션 캐시)
# ----------------------------
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.le = None
    st.session_state.max_len = None

def train_pipeline():
    texts_raw = df["Review"].tolist()
    labels_raw = df["Sentiment"].tolist()

    # 전처리
    texts = [preprocess_to_string(t) for t in texts_raw]

    # 라벨 인코딩
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    # Tokenizer + Padding
    tokenizer = Tokenizer(num_words=int(vocab_size), oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def to_pad(x_list):
        seq = tokenizer.texts_to_sequences(x_list)
        return pad_sequences(seq, maxlen=int(max_len), padding="post", truncating="post")

    X_train_pad = to_pad(X_train)
    X_test_pad = to_pad(X_test)

    num_classes = len(le.classes_)
    model = build_model(vocab_size=int(vocab_size), max_len=int(max_len), num_classes=num_classes)

    # EarlyStopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=int(epochs),
        batch_size=int(batch_size),
        callbacks=callbacks,
        verbose=0
    )

    # 평가
    probs = model.predict(X_test_pad, verbose=0)
    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, target_names=le.classes_, digits=4)
