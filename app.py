# app.py
# =========================================
# Streamlit - LSTM 감정분류 (Review.xlsx 기반)
# - 입력: Review.xlsx (columns: Sentiment, Review)
# - 학습: Tokenizer + Embedding + (Bi)LSTM
# - 기능: (1) 데이터 로드/미리보기 (2) 버튼으로 학습 (3) 입력 -> 예측
# =========================================

import re
import random
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models


# ----------------------------
# 0) Streamlit 설정
# ----------------------------
st.set_page_config(page_title="리뷰 감정분석 (LSTM)", layout="wide")

DEFAULT_XLSX_PATH = "data/Review.xlsx"  # 리포지토리에 data/Review.xlsx 넣으면 자동 로드

# 감독관 요구 불용어(예시) + "너무"
BASE_STOPWORDS = set([
    "이", "가", "을", "를", "은", "는", "에", "에서", "에게",
    "의", "와", "과", "도", "로", "으로",
    "하다", "되다", "있다", "없다",
    "그", "저", "것", "수",
    "좀", "잘", "매우", "정말",
    "때문", "같다",
    "너무"
])


# ----------------------------
# 1) 유틸
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokenize(s: str, stopwords: set) -> list[str]:
    """
    konlpy 없이도 돌아가게 만든 초간단 토큰화:
    - 한글/영문/숫자 외 문자 제거
    - 공백 split
    - 불용어 제거
    """
    s = clean_text(s)
    s = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    tokens = [t for t in tokens if (t not in stopwords and len(t) > 1)]
    return tokens


def preprocess_to_string(s: str, stopwords: set) -> str:
    return " ".join(simple_tokenize(s, stopwords))


@st.cache_data(show_spinner=False)
def load_review_xlsx_from_path(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(path, sheet_name=0)
        return df
    except Exception:
        return None


def load_review_xlsx(uploaded_file=None) -> pd.DataFrame | None:
    # 1) 업로드가 있으면 그걸 최우선
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file, sheet_name=0)
        except Exception:
            return None
    # 2) 없으면 기본 경로 시도
    return load_review_xlsx_from_path(DEFAULT_XLSX_PATH)


def build_model(vocab_size: int, max_len: int, num_classes: int,
                emb_dim=128, lstm_units=64, dropout=0.3) -> tf.keras.Model:
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


def train_lstm_pipeline(
    df: pd.DataFrame,
    stopwords: set,
    vocab_size: int,
    max_len: int,
    epochs: int,
    batch_size: int,
    test_size: float,
    seed: int
):
    set_seed(seed)

    # 필수 컬럼
    required_cols = {"Sentiment", "Review"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"엑셀에 필요한 컬럼이 없어: {missing}. 현재 컬럼: {list(df.columns)}")

    work = df[["Sentiment", "Review"]].copy()
    work["Sentiment"] = work["Sentiment"].astype(str).fillna("").str.strip()
    work["Review"] = work["Review"].astype(str).fillna("").str.strip()
    work = work[(work["Sentiment"] != "") & (work["Review"] != "")].copy()

    # 전처리
    texts_raw = work["Review"].tolist()
    labels_raw = work["Sentiment"].tolist()
    texts = [preprocess_to_string(t, stopwords) for t in texts_raw]

    # 라벨 인코딩
    le = LabelEncoder()
    y = le.fit_transform(labels_raw)

    # split (라벨 분포 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y
    )

    # Tokenizer + Padding
    tokenizer = Tokenizer(num_words=int(vocab_size), oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    def to_pad(x_list):
        seq = tokenizer.texts_to_sequences(x_list)
        return pad_sequences(seq, maxlen=int(max_len), padding="post", truncating="post")

    X_train_pad = to_pad(X_train)
    X_test_pad = to_pad(X_test)

    # 모델
    num_classes = len(le.classes_)
    model = build_model(vocab_size=int(vocab_size), max_len=int(max_len), num_classes=num_classes)

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

    acc = float(accuracy_score(y_test, pred))
    report = classification_report(y_test, pred, target_names=le.classes_, digits=4)
    cm = confusion_matrix(y_test, pred)

    # 샘플 몇개 뽑아서 예측 보기
    sample_idx = np.random.choice(len(X_test), size=min(6, len(X_test)), replace=False)
    samples = []
    for i in sample_idx:
        txt = X_test[i]
        true_label = le.inverse_transform([y_test[i]])[0]
        pred_label = le.inverse_transform([pred[i]])[0]
        samples.append((true_label, pred_label, txt[:140] + ("..." if len(txt) > 140 else "")))

    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": le,
        "max_len": int(max_len),
        "metrics": {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "classes": list(le.classes_),
            "samples": samples,
            "history": history.history,
            "label_dist": work["Sentiment"].value_counts().to_dict(),
        }
    }


def predict_one(text: str, stopwords: set, model, tokenizer, le, max_len: int):
    proc = preprocess_to_string(text, stopwords)
    seq = tokenizer.texts_to_sequences([proc])
    pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = le.inverse_transform([idx])[0]
    prob_dict = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    return label, float(np.max(probs)), prob_dict, proc


# ----------------------------
# 2) UI
# ----------------------------
st.title("리뷰 감정 분석 (LSTM)")
st.caption("Review.xlsx의 Sentiment/Review로 학습하고, 아래 입력창에서 바로 예측합니다.")

with st.sidebar:
    st.header("1) 데이터")
    uploaded = st.file_uploader("Review.xlsx 업로드(선택)", type=["xlsx"])
    st.caption("업로드 안 하면 리포지토리의 data/Review.xlsx를 자동으로 찾습니다.")

    st.header("2) 학습 설정")
    vocab_size = st.number_input("VOCAB_SIZE", 5000, 50000, 20000, step=1000)
    max_len = st.number_input("MAX_LEN", 20, 300, 120, step=10)
    epochs = st.number_input("epochs", 1, 50, 8, step=1)
    batch_size = st.selectbox("batch_size", [8, 16, 32, 64], index=1)
    test_size = st.slider("test_size", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("seed", 0, 9999, 42, step=1)

    st.header("3) 불용어")
    extra_sw = st.text_input("추가 불용어(쉼표로 구분)", value="")

    st.divider()
    train_btn = st.button("🚀 학습 시작", use_container_width=True)

# stopwords 확정
STOPWORDS = set(BASE_STOPWORDS)
if extra_sw.strip():
    for w in extra_sw.split(","):
        w = w.strip()
        if w:
            STOPWORDS.add(w)

# 데이터 로드
df = load_review_xlsx(uploaded)
if df is None:
    st.error("Review.xlsx를 못 읽었어. (업로드 파일/ data/Review.xlsx 경로/엑셀 손상 여부 확인)")
    st.stop()

# 데이터 미리보기
required_cols = {"Sentiment", "Review"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"엑셀에 필요한 컬럼이 없어: {missing}\n현재 컬럼: {list(df.columns)}")
    st.stop()

view_df = df[["Sentiment", "Review"]].copy()
view_df["Sentiment"] = view_df["Sentiment"].astype(str).fillna("").str.strip()
view_df["Review"] = view_df["Review"].astype(str).fillna("").str.strip()
view_df = view_df[(view_df["Sentiment"] != "") & (view_df["Review"] != "")].copy()

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("데이터 미리보기")
    st.dataframe(view_df.head(12), use_container_width=True)
with c2:
    st.subheader("라벨 분포")
    st.write(view_df["Sentiment"].value_counts())

# 세션 상태 초기화
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.bundle = None

# 학습 버튼 눌렀을 때만 학습 (핵심!)
if train_btn:
    with st.spinner("학습 중... (처음 한 번만 조금 기다리면 됨)"):
        try:
            bundle = train_lstm_pipeline(
                df=view_df,
                stopwords=STOPWORDS,
                vocab_size=int(vocab_size),
                max_len=int(max_len),
                epochs=int(epochs),
                batch_size=int(batch_size),
                test_size=float(test_size),
                seed=int(seed),
            )
            st.session_state.bundle = bundle
            st.session_state.trained = True
            st.success("학습 완료! 아래에서 성능 확인 + 예측해봐.")
        except Exception as e:
            st.session_state.trained = False
            st.session_state.bundle = None
            st.exception(e)

# 학습 결과 표시
st.divider()
st.subheader("학습/평가 결과")

if not st.session_state.trained:
    st.info("왼쪽에서 **학습 시작**을 눌러야 예측이 활성화돼.")
else:
    bundle = st.session_state.bundle
    m = bundle["metrics"]

    st.write(f"✅ 테스트 정확도: **{m['accuracy']:.4f}**")
    st.text("Classification Report:\n" + m["report"])

    cm = m["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=m["classes"], columns=m["classes"])
    st.write("Confusion Matrix")
    st.dataframe(cm_df, use_container_width=True)

    st.write("테스트 샘플 예측(일부)")
    sample_df = pd.DataFrame(m["samples"], columns=["true", "pred", "text_preview"])
    st.dataframe(sample_df, use_container_width=True)

    st.divider()
    st.subheader("리뷰 한 줄 입력 → 예측")

    user_text = st.text_area("리뷰를 입력해줘", value="", height=120, placeholder="예) 담배 냄새가 심하고 운전이 거칠었어요.")
    pred_btn = st.button("🔮 예측하기")

    if pred_btn:
        if not user_text.strip():
            st.warning("리뷰를 먼저 입력해줘!")
        else:
            label, conf, prob_dict, proc = predict_one(
                user_text,
                STOPWORDS,
                bundle["model"],
                bundle["tokenizer"],
                bundle["label_encoder"],
                bundle["max_len"],
            )
            st.success(f"예측: **{label}**  (confidence={conf:.3f})")
            st.write("클래스별 확률")
            st.json(prob_dict)
            with st.expander("전처리된 입력(모델에 들어간 텍스트) 보기"):
                st.code(proc)

st.caption("팁: 데이터가 적거나 라벨이 애매하면(중립/부정 경계) 정확도가 들쑥날쑥할 수 있어. 그건 네 실력이 아니라 데이터의 물리법칙이야 😵‍💫")
