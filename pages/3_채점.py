# pages/3_채점표.py — 채점표 페이지: 혼동행렬·네 가지 지표·틀린 사람 목록
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="채점표 · 뇌졸중 예측 실습실", page_icon="🩺", layout="wide")
st.title("📋 채점표")
st.write("'분류 모델' 페이지와 같은 방법으로 데이터를 나누고 두 모델을 학습한 뒤, 테스트 데이터로 채점합니다. "
         "뇌졸중(stroke=1)을 **양성**, 아님(stroke=0)을 **음성**으로 둡니다.")

데이터주소 = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
기준폴더 = Path(__file__).resolve().parent.parent           # pages 폴더의 위(= main.py가 있는 곳)
내려받은파일 = 기준폴더 / "data" / "stroke.csv"              # 인터넷이 안 될 때 대신 읽을 파일(선택)

# '분류 모델' 페이지에서 처음 골라 두는 네 속성(체질량지수만 뺀 것)과 똑같이 쓴다
입력속성 = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

이름_로지스틱 = "로지스틱 회귀(확률로 답하는 모델)"
이름_나무 = "의사결정트리(질문으로 답하는 모델)"
이름_기준 = "다수 클래스 기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"
못구함 = "계산할 수 없음"
목록_최대줄 = 20


@st.cache_data
def 데이터_읽기():
    """csv를 읽는다. 인터넷이 안 되면 data/stroke.csv를 대신 읽는다."""
    try:
        return pd.read_csv(데이터주소, encoding="utf-8")
    except Exception:
        if 내려받은파일.exists():
            return pd.read_csv(내려받은파일, encoding="utf-8")
        raise


try:
    df = 데이터_읽기()
except Exception as 오류:
    st.error(f"데이터를 읽지 못했습니다. 인터넷 연결을 확인해 주세요.\n\n({type(오류).__name__}: {오류})")
    st.stop()

# ── 데이터 나누기: '분류 모델' 페이지와 똑같이 ──────────────────────────
표 = df.sort_values("id", kind="stable").reset_index(drop=True)      # 번호 순으로 정렬
for 열 in 입력속성:
    표[열] = pd.to_numeric(표[열], errors="coerce")
테스트칸 = (np.arange(len(표)) % 10) < 3                              # 열 명씩 묶어 각 묶음의 앞 세 명
학습표, 테스트표 = 표[~테스트칸], 표[테스트칸]

X_train = 학습표[입력속성].copy()
X_test = 테스트표[입력속성].copy()
y_train = 학습표["stroke"].astype(int).to_numpy()
y_test = 테스트표["stroke"].astype(int).to_numpy()

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("입력 속성에 빈 값이 남아 있어 모델을 만들 수 없습니다.")
    st.stop()

크기맞추개 = StandardScaler().fit(X_train)                            # 학습용으로만 배운다
Xs_train = 크기맞추개.transform(X_train)
Xs_test = 크기맞추개.transform(X_test)

# ── 모델 만들기: '분류 모델' 페이지와 똑같이 ────────────────────────────
로지스틱 = LogisticRegression(max_iter=1000, random_state=42).fit(Xs_train, y_train)
나무 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
기준 = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

예측 = {
    이름_로지스틱: 로지스틱.predict(Xs_test),
    이름_나무: 나무.predict(X_test),
    이름_기준: 기준.predict(Xs_test),
}


def 네칸(실제, 답):
    """행은 실제(뇌졸중, 아님), 열은 예측(뇌졸중, 아님). 양성을 앞에 두므로 TP가 왼쪽 위에 온다."""
    (TP, FN), (FP, TN) = confusion_matrix(실제, 답, labels=[1, 0])
    return int(TP), int(FN), int(FP), int(TN)


네칸들 = {이름: 네칸(y_test, 답) for 이름, 답 in 예측.items()}
전체 = len(y_test)
실제양성 = int(y_test.sum())

st.caption(f"입력 속성: 나이, 평균 혈당, 고혈압, 심장병. 번호(id) 순으로 정렬해 열 명씩 묶고 각 묶음의 앞 세 명을 "
           f"테스트용으로 고정했습니다(테스트용 {전체:,}명, 학습용 {len(y_train):,}명).")

# ── 1. 혼동행렬 ─────────────────────────────────────────────────────
st.subheader("1. 혼동행렬 (행은 실제, 열은 예측)")


def 혼동표_html(이름, TP, FN, FP, TN):
    칸틀 = "border:1px solid rgba(128,128,128,.45); padding:10px 6px;"
    좋음 = "background:rgba(46,160,67,.16);"
    나쁨 = "background:rgba(214,39,40,.16);"
    합계색 = "background:rgba(128,128,128,.10);"

    def 칸(약자, 수, 뜻, 색):
        return (f'<td style="{칸틀}{색}"><div style="font-size:1.05rem;"><b>{약자}</b> · {수:,}명</div>'
                f'<div style="font-size:.75rem; opacity:.75;">{뜻}</div></td>')

    return "".join([
        f'<div style="font-weight:600; margin-bottom:6px;">{이름}</div>',
        '<table style="border-collapse:collapse; width:100%; text-align:center;">',
        f'<tr><th style="{칸틀}"></th><th style="{칸틀}">예측: 뇌졸중</th><th style="{칸틀}">예측: 아님</th>'
        f'<th style="{칸틀}{합계색}">합계</th></tr>',
        f'<tr><th style="{칸틀}">실제: 뇌졸중</th>{칸("TP", TP, "찾아낸 환자", 좋음)}{칸("FN", FN, "놓친 환자", 나쁨)}'
        f'<td style="{칸틀}{합계색}"><b>{TP + FN:,}명</b></td></tr>',
        f'<tr><th style="{칸틀}">실제: 아님</th>{칸("FP", FP, "헛짚은 사람", 나쁨)}{칸("TN", TN, "맞게 거른 사람", 좋음)}'
        f'<td style="{칸틀}{합계색}"><b>{FP + TN:,}명</b></td></tr>',
        f'<tr><th style="{칸틀}{합계색}">합계</th><td style="{칸틀}{합계색}"><b>{TP + FP:,}명</b></td>'
        f'<td style="{칸틀}{합계색}"><b>{FN + TN:,}명</b></td><td style="{칸틀}{합계색}"><b>{TP + FN + FP + TN:,}명</b></td></tr>',
        "</table>",
    ])


왼칸, 오른칸 = st.columns(2)
왼칸.markdown(혼동표_html(이름_로지스틱, *네칸들[이름_로지스틱]), unsafe_allow_html=True)
오른칸.markdown(혼동표_html(이름_나무, *네칸들[이름_나무]), unsafe_allow_html=True)
st.caption("TP 뇌졸중이라 예측했고 실제로 뇌졸중 · FN 아니라고 예측했지만 실제로는 뇌졸중 · "
           "FP 뇌졸중이라 예측했지만 실제로는 아님 · TN 아니라고 예측했고 실제로도 아님")

# ── 2. 네 가지 지표 ─────────────────────────────────────────────────


def 지표(TP, FN, FP, TN):
    """네 칸에서 정확도·재현율·정밀도·F1을 글로 만든다. 분모가 0이면 '계산할 수 없음'."""
    합 = TP + FN + FP + TN
    정확도 = f"{(TP + TN) / 합:.1%} ({TP + TN:,}/{합:,})"
    실제양성수 = TP + FN                 # 재현율의 분모: 실제 뇌졸중인 사람 수
    예측양성수 = TP + FP                 # 정밀도의 분모: 뇌졸중이라 예측한 사람 수
    재현율 = f"{TP / 실제양성수:.1%} ({TP:,}/{실제양성수:,})" if 실제양성수 else 못구함
    if 예측양성수 == 0:
        정밀도 = 못구함
        f1 = 못구함
    else:
        정밀도 = f"{TP / 예측양성수:.1%} ({TP:,}/{예측양성수:,})"
        p = TP / 예측양성수
        if 실제양성수 == 0:
            f1 = 못구함
        else:
            r = TP / 실제양성수
            f1 = f"{(2 * p * r / (p + r) if p + r > 0 else 0.0):.1%}"     # 둘 다 0이면 0으로 적는다
    return {"정확도": 정확도, "재현율": 재현율, "정밀도": 정밀도, "F1": f1}


st.subheader("2. 네 가지 지표 한눈에 비교")
st.write(f"테스트용 {전체:,}명 가운데 뇌졸중을 겪은 사람은 {실제양성:,}명입니다.")
지표표 = pd.DataFrame({이름: 지표(*네칸들[이름]) for 이름 in (이름_로지스틱, 이름_나무, 이름_기준)}).T
st.table(지표표[["정확도", "재현율", "정밀도", "F1"]])
st.caption("정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN), 분모는 실제 뇌졸중인 사람 수 · "
           "정밀도 = TP ÷ (TP+FP), 분모는 뇌졸중이라 예측한 사람 수 · "
           "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
           "괄호 안은 분자/분모 사람 수입니다. 정밀도와 재현율이 모두 0이면 F1은 0으로 적습니다.")

# ── 3. 틀린 사람 목록 ───────────────────────────────────────────────
st.subheader("3. 모델이 틀린 사람들")
고른이름 = st.radio("어느 모델의 목록을 볼까요?", [이름_로지스틱, 이름_나무], horizontal=True)
고른답 = 예측[고른이름]
TP, FN, FP, TN = 네칸들[고른이름]

놓친마스크 = (y_test == 1) & (고른답 == 0)       # 실제 뇌졸중인데 아니라고 예측
헛짚마스크 = (y_test == 0) & (고른답 == 1)       # 아닌데 뇌졸중이라고 예측


def 목록_화면표(마스크):
    사람들 = 테스트표.loc[마스크, ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]]
    사람들 = 사람들.sort_values(["age", "id"], ascending=[False, True], kind="stable")   # 나이가 많은 순서
    총인원 = len(사람들)
    보임 = 사람들.head(목록_최대줄)
    화면 = pd.DataFrame({
        "번호": 보임["id"].astype(int).to_numpy(),
        "나이": [f"{값:g}" for 값 in 보임["age"]],
        "평균 혈당": [f"{값:.1f}" for 값 in 보임["avg_glucose_level"]],
        "고혈압 여부": ["있음" if 값 == 1 else "없음" for 값 in 보임["hypertension"]],
        "심장병 여부": ["있음" if 값 == 1 else "없음" for 값 in 보임["heart_disease"]],
    }).set_index("번호")
    return 화면, 총인원


왼목록, 오른목록 = st.columns(2)
for 칸, 마스크, 이름, 설명 in [
    (왼목록, 놓친마스크, "놓친 환자", "실제 뇌졸중인데 아니라고 예측한 사람"),
    (오른목록, 헛짚마스크, "헛짚은 사람", "아닌데 뇌졸중이라고 예측한 사람"),
]:
    화면, 총인원 = 목록_화면표(마스크)
    제목 = f"#### {이름} — 전체 {총인원:,}명"
    if 총인원 > 목록_최대줄:
        제목 += f" (나이 많은 순으로 앞 {목록_최대줄}명만 보임)"
    칸.markdown(제목)
    칸.caption(설명)
    if 총인원 == 0:
        칸.write("해당하는 사람이 없습니다.")
    else:
        칸.table(화면)

놓친수, 헛짚수 = int(놓친마스크.sum()), int(헛짚마스크.sum())
if (놓친수, 헛짚수) == (FN, FP):
    st.write(f"위 두 목록의 사람 수(놓친 환자 {놓친수:,}명, 헛짚은 사람 {헛짚수:,}명)는 "
             f"{고른이름}의 혼동행렬 FN 칸({FN:,}명)과 FP 칸({FP:,}명)과 같습니다.")
else:
    st.error(f"목록의 사람 수(놓친 환자 {놓친수:,}명, 헛짚은 사람 {헛짚수:,}명)가 "
             f"혼동행렬의 FN 칸({FN:,}명), FP 칸({FP:,}명)과 다릅니다. 계산을 확인해 주세요.")
