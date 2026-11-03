# pages/4_성능_높이기.py — 성능 높이기 페이지: 양쪽의 가중치를 맞추기 전과 맞춘 뒤를 견준다
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="성능 높이기 · 뇌졸중 예측 실습실", page_icon="🩺", layout="wide")
st.title("🚀 성능 높이기")

데이터주소 = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
기준폴더 = Path(__file__).resolve().parent.parent           # pages 폴더의 위(= main.py가 있는 곳)
내려받은파일 = 기준폴더 / "data" / "stroke.csv"              # 인터넷이 안 될 때 대신 읽을 파일(선택)

우리말 = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
전체속성 = list(우리말)
처음고른것 = ["age", "avg_glucose_level", "hypertension", "heart_disease"]   # 체질량지수만 뺀 넷
영일속성 = {"hypertension", "heart_disease"}                                    # 0과 1만 있는 속성
빨강 = "#d62728"    # 뇌졸중(양성)
파랑 = "#1f77b4"    # 아님(음성)
주황 = "#e69f00"    # 가중치를 맞추기 전의 0.5 경계선
보라 = "#7b2cbf"    # 가중치를 맞춘 뒤의 0.5 경계선

이름_로지스틱 = "로지스틱 회귀(확률로 답하는 모델)"
이름_나무 = "의사결정트리(질문으로 답하는 모델)"
못구함 = "계산할 수 없음"

# ── 입력 속성 고르기 (화면 맨 위) ────────────────────────────────────
고른것 = st.multiselect(
    "입력으로 사용할 속성",
    options=전체속성,
    default=처음고른것,
    format_func=lambda 열: 우리말[열],
    key="성능_입력속성",
)
고른것 = [열 for 열 in 전체속성 if 열 in 고른것]          # 고른 차례와 상관없이 늘 같은 순서로
if len(고른것) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개보다 적으면 모델을 만들지 않고 여기서 멈춥니다.")
    st.stop()

st.write("뇌졸중을 겪은 사람은 아주 드뭅니다. 그래서 모델이 '아님'만 답해도 정확도가 높게 나오고, 정작 찾아야 할 환자는 놓치기 쉽습니다. "
         "여기서는 학습할 때 **뇌졸중과 아님 양쪽의 가중치를 맞추기 전과 맞춘 뒤**를 한 화면에서 견줍니다.")


@st.cache_data
def 데이터_읽기():
    """csv를 읽는다. 인터넷이 안 되면 data/stroke.csv를 대신 읽는다. bmi의 빈 값은 빈 값으로 남는다."""
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

# ── 데이터 나누기: '채점표'·'분류 모델' 페이지와 똑같이 ──────────────────
표 = df.sort_values("id", kind="stable").reset_index(drop=True)      # 번호 순으로 정렬
for 열 in 전체속성:
    표[열] = pd.to_numeric(표[열], errors="coerce")
테스트칸 = (np.arange(len(표)) % 10) < 3                              # 열 명씩 묶어 각 묶음의 앞 세 명
학습표, 테스트표 = 표[~테스트칸], 표[테스트칸]

X_train = 학습표[고른것].copy()
X_test = 테스트표[고른것].copy()
y_train = 학습표["stroke"].astype(int).to_numpy()
y_test = 테스트표["stroke"].astype(int).to_numpy()

채움글 = ""
if "bmi" in 고른것:
    빈훈련 = int(X_train["bmi"].isna().sum())
    빈테스트 = int(X_test["bmi"].isna().sum())
    훈련중앙값 = float(X_train["bmi"].median())
    X_train["bmi"] = X_train["bmi"].fillna(훈련중앙값)
    X_test["bmi"] = X_test["bmi"].fillna(훈련중앙값)
    채움글 = (f" 체질량지수가 비어 있던 사람(학습용 {빈훈련:,}명, 테스트용 {빈테스트:,}명)의 빈 값은 "
              f"학습용의 중앙값으로 채웠습니다(중앙값: {훈련중앙값:.1f}).")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("고른 속성에 빈 값이 남아 있어 모델을 만들 수 없습니다.")
    st.stop()

학습양성 = int(y_train.sum())
학습음성 = len(y_train) - 학습양성
if 학습양성 == 0 or 학습음성 == 0:
    st.error("학습용에 뇌졸중이 있는 사람과 없는 사람이 모두 있어야 모델을 만들 수 있습니다.")
    st.stop()

전체 = len(y_test)
실제양성 = int(y_test.sum())
st.caption(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"테스트용 {전체:,}명(그중 뇌졸중 {실제양성:,}명), 학습용 {len(y_train):,}명(그중 뇌졸중 {학습양성:,}명).{채움글}"
)
무게_음성 = len(y_train) / (2 * 학습음성)
무게_양성 = len(y_train) / (2 * 학습양성)
st.caption(
    f"가중치를 맞춘다는 것은 학습할 때 뇌졸중 한 명을 아님 한 명보다 약 {무게_양성 / 무게_음성:.1f}배 무겁게 세어, "
    f"양쪽의 합이 같아지게 하는 것입니다(학습용 뇌졸중 {학습양성:,}명, 아님 {학습음성:,}명 → "
    f"한 명의 무게 각각 {무게_양성:.2f}, {무게_음성:.2f}). 설정은 class_weight='balanced'입니다."
)

크기맞추개 = StandardScaler().fit(X_train)                            # 학습용으로만 배운다
Xs_train = 크기맞추개.transform(X_train)
Xs_test = 크기맞추개.transform(X_test)

# ── 모델 만들기: 가중치를 맞추기 전(None)과 맞춘 뒤('balanced') ───────────
로지스틱_전 = LogisticRegression(max_iter=1000, random_state=42).fit(Xs_train, y_train)
로지스틱_후 = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced").fit(Xs_train, y_train)
# 나무는 질문이 원래 값으로 나오도록 크기를 맞추기 전 값으로 학습한다
나무_전 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
나무_후 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42,
                             class_weight="balanced").fit(X_train, y_train)


def 네칸(실제, 답):
    """행은 실제(뇌졸중, 아님), 열은 예측(뇌졸중, 아님). 양성을 앞에 두므로 TP가 왼쪽 위에 온다."""
    (TP, FN), (FP, TN) = confusion_matrix(실제, 답, labels=[1, 0])
    return int(TP), int(FN), int(FP), int(TN)


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
        p = TP / 예측양성수
        정밀도 = f"{p:.1%} ({TP:,}/{예측양성수:,})"
        if 실제양성수 == 0:
            f1 = 못구함
        else:
            r = TP / 실제양성수
            f1 = f"{(2 * p * r / (p + r) if p + r > 0 else 0.0):.1%}"     # 둘 다 0이면 0으로 적는다
    return {"정확도": 정확도, "재현율": 재현율, "정밀도": 정밀도, "F1": f1}


def 전후표(전, 후):
    """한 모델의 네 칸과 네 지표를 가중치를 맞추기 전·후로 나란히 놓은 표."""
    전지표, 후지표 = 지표(*전), 지표(*후)
    행 = {
        "TP · 찾아낸 환자": (f"{전[0]:,}명", f"{후[0]:,}명"),
        "FN · 놓친 환자": (f"{전[1]:,}명", f"{후[1]:,}명"),
        "FP · 헛짚은 사람": (f"{전[2]:,}명", f"{후[2]:,}명"),
        "TN · 맞게 거른 사람": (f"{전[3]:,}명", f"{후[3]:,}명"),
    }
    for 이름 in ("정확도", "재현율", "정밀도", "F1"):
        행[이름] = (전지표[이름], 후지표[이름])
    return pd.DataFrame(행, index=["가중치 맞추기 전", "가중치 맞춘 뒤"]).T


def 바뀐줄(전, 후):
    return (f"찾아낸 환자 {전[0]:,}명 → {후[0]:,}명, 놓친 환자 {전[1]:,}명 → {후[1]:,}명, "
            f"헛짚은 사람 {전[2]:,}명 → {후[2]:,}명")


# ── 1. 전·후 비교 표 ────────────────────────────────────────────────
st.subheader("1. 가중치를 맞추기 전과 맞춘 뒤 (테스트 데이터)")
st.write(f"테스트용 {전체:,}명 가운데 뇌졸중을 겪은 사람은 {실제양성:,}명입니다. 뇌졸중을 양성으로 둡니다.")
비교 = [
    (이름_로지스틱, 네칸(y_test, 로지스틱_전.predict(Xs_test)), 네칸(y_test, 로지스틱_후.predict(Xs_test))),
    (이름_나무, 네칸(y_test, 나무_전.predict(X_test)), 네칸(y_test, 나무_후.predict(X_test))),
]
표칸들 = st.columns(2)
for 칸, (이름, 전, 후) in zip(표칸들, 비교):
    칸.markdown(f"#### {이름}")
    칸.table(전후표(전, 후))
    칸.write(바뀐줄(전, 후))
st.caption("재현율 = TP ÷ (TP+FN), 분모는 실제 뇌졸중인 사람 수 · 정밀도 = TP ÷ (TP+FP), 분모는 뇌졸중이라 예측한 사람 수 · "
           "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). 괄호 안은 분자/분모 사람 수입니다. "
           "정밀도와 재현율이 모두 0이면 F1은 0으로 적습니다.")

# ── 그림에 쓰는 공통 도구 ───────────────────────────────────────────
고정값 = {열: float(X_test[열].median()) for 열 in 고른것}      # 축이 아닌 속성은 테스트용 중앙값에 세운다


def 그림_범위(열):
    if 열 in 영일속성:
        return -0.5, 1.5
    낮 = float(X_test[열].min())
    높 = float(X_test[열].max())
    폭 = (높 - 낮) or 1.0
    return 낮 - 0.06 * 폭, 높 + 0.06 * 폭


def 점_위치(열, 난수):
    값들 = 테스트표[열].to_numpy(dtype=float)
    if 열 in 영일속성:   # 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍는다
        return 값들 + 난수.uniform(-0.12, 0.12, len(값들))
    return 값들


def 선형_기울기(모델, 축들, 낮들, 높들):
    """로지스틱 회귀의 확률 0.5 자리는 decision_function이 0인 곳이다. 축 위에서 이 값은 곧게 변하므로
    그림 한가운데의 값과 축마다 한 칸 움직였을 때의 값만 알면 기울기를 구할 수 있다."""
    중심 = [(l + h) / 2 for l, h in zip(낮들, 높들)]
    변화 = [(h - l) / 2 for l, h in zip(낮들, 높들)]
    행들 = [list(중심)]
    for k in range(len(축들)):
        행 = list(중심)
        행[k] += 변화[k]
        행들.append(행)
    틀 = pd.DataFrame({열: [고정값[열]] * len(행들) for 열 in 고른것})
    for k, 열 in enumerate(축들):
        틀[열] = [행[k] for 행 in 행들]
    값 = 모델.decision_function(크기맞추개.transform(틀))
    f0 = float(값[0])
    기울기 = [float(값[k + 1] - 값[0]) / 변화[k] for k in range(len(축들))]
    return f0, 기울기, 중심


def 선_자르기(f0, A, B, x0, y0, 왼, 오, 아래, 위):
    """f0 + A(x-x0) + B(y-y0) = 0 인 직선을 그림 네모 안으로 잘라 두 끝점을 돌려준다. 밖이면 None."""
    여유 = 1e-9
    후보 = []
    if abs(B) > 1e-12:
        for x in (왼, 오):
            y = y0 + (-f0 - A * (x - x0)) / B
            if 아래 - 여유 <= y <= 위 + 여유:
                후보.append((x, y))
    if abs(A) > 1e-12:
        for y in (아래, 위):
            x = x0 + (-f0 - B * (y - y0)) / A
            if 왼 - 여유 <= x <= 오 + 여유:
                후보.append((x, y))
    서로다른 = []
    for 점 in 후보:
        if all(abs(점[0] - q[0]) > 1e-7 * (오 - 왼) or abs(점[1] - q[1]) > 1e-7 * (위 - 아래) for q in 서로다른):
            서로다른.append(점)
    if len(서로다른) < 2:
        return None
    return max(((p, q) for i, p in enumerate(서로다른) for q in 서로다른[i + 1:]),
               key=lambda 쌍: (쌍[0][0] - 쌍[1][0]) ** 2 + (쌍[0][1] - 쌍[1][1]) ** 2)


def 평면_다각형(f0, 기울기, 중심, 낮들, 높들):
    """f0 + Σ 기울기×(값-중심) = 0 인 평면을 세 축의 상자 안으로 잘라 그 모서리 점들을 돌려준다. 밖이면 None."""
    낮 = np.array(낮들, dtype=float)
    폭 = np.array(높들, dtype=float) - 낮
    # 상자를 0~1짜리 단위 상자로 옮겨 계산한다
    아래모서리값 = f0 + sum(g * (l - c) for g, l, c in zip(기울기, 낮들, 중심))
    b = np.array([g * w for g, w in zip(기울기, 폭)])
    if np.max(np.abs(b)) < 1e-12:
        return None
    모서리 = [np.array(m) for m in itertools.product((0.0, 1.0), repeat=3)]
    값 = [아래모서리값 + float(np.dot(b, m)) for m in 모서리]
    점들 = []
    for i, j in itertools.combinations(range(8), 2):
        if int(np.sum(모서리[i] != 모서리[j])) != 1:        # 상자의 모서리(변)만 본다
            continue
        if 값[i] * 값[j] < 0:
            s = 값[i] / (값[i] - 값[j])
            점들.append(모서리[i] * (1 - s) + 모서리[j] * s)
    for m, v in zip(모서리, 값):
        if abs(v) < 1e-12:
            점들.append(m)
    유일 = []
    for p in 점들:
        if all(np.linalg.norm(p - q) > 1e-9 for q in 유일):
            유일.append(p)
    if len(유일) < 3:
        return None
    중점 = np.mean(유일, axis=0)
    n = b / np.linalg.norm(b)
    보조 = np.zeros(3)
    보조[int(np.argmin(np.abs(n)))] = 1.0
    u = np.cross(n, 보조)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    각 = [np.arctan2(float(np.dot(p - 중점, v)), float(np.dot(p - 중점, u))) for p in 유일]
    순서 = np.argsort(각)
    정렬 = np.array(유일)[순서]
    return 낮 + 폭 * 정렬


def 밖_문장(이름, 모양, f0):
    쪽 = "0.5 이상이라 모두 뇌졸중" if f0 > 0 else "0.5 미만이라 모두 아님"
    return (f"{이름}의 0.5 {모양}은 이 그림 범위 밖에 있어 그리지 못했습니다. "
            f"그림 안의 모든 자리에서 뇌졸중일 확률이 {쪽}이라고 답합니다.")


def 세워둔값_글(축들):
    남은 = [열 for 열 in 고른것 if 열 not in 축들]
    if not 남은:
        return f"고른 속성이 {len(축들)}개뿐이라 따로 세워 둔 값은 없습니다."
    조각 = ", ".join(f"{우리말[열]} {고정값[열]:g}" if 열 in 영일속성 else f"{우리말[열]} {고정값[열]:.1f}" for 열 in 남은)
    return f"축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {조각}"


# ── 2. 두 속성 산점도와 0.5 경계선 ──────────────────────────────────
st.subheader("2. 두 속성으로 본 0.5 경계선")
st.caption(f"0.5 경계선은 {이름_로지스틱}가 뇌졸중일 확률을 0.5로 보는 자리입니다. "
           "나무에는 이런 선이 없어서 로지스틱 회귀만 그립니다.")
선택칸1, 선택칸2 = st.columns(2)
x열 = 선택칸1.selectbox("가로축", 고른것, index=0, format_func=lambda 열: 우리말[열])
y열 = 선택칸2.selectbox("세로축", 고른것, index=1, format_func=lambda 열: 우리말[열])

if x열 == y열:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    xlo, xhi = 그림_범위(x열)
    ylo, yhi = 그림_범위(y열)
    결과 = {}
    for 이름, 모델 in [("전", 로지스틱_전), ("후", 로지스틱_후)]:
        f0, (A, B), (x0, y0) = 선형_기울기(모델, [x열, y열], [xlo, ylo], [xhi, yhi])
        결과[이름] = (f0, 선_자르기(f0, A, B, x0, y0, xlo, xhi, ylo, yhi))

    그림 = go.Figure()
    난수 = np.random.default_rng(42)
    점x = 점_위치(x열, 난수)
    점y = 점_위치(y열, 난수)
    진짜값 = np.column_stack([테스트표[x열].to_numpy(dtype=float), 테스트표[y열].to_numpy(dtype=float)])
    마우스글 = ("<b>%{fullData.name}</b><br>" + 우리말[x열] + ": %{customdata[0]:g}<br>"
             + 우리말[y열] + ": %{customdata[1]:g}<extra></extra>")
    for 정답, 이름, 색, 크기, 투명도 in [(0, "아님(0)", 파랑, 6, 0.5), (1, "뇌졸중(1, 양성)", 빨강, 8, 0.9)]:
        뽑기 = y_test == 정답
        그림.add_trace(go.Scatter(
            x=점x[뽑기], y=점y[뽑기], mode="markers", name=f"실제 {이름}",
            marker=dict(size=크기, color=색, opacity=투명도, line=dict(width=0.6, color="white")),
            customdata=진짜값[뽑기], hovertemplate=마우스글,
        ))
    for 열쇠, 이름, 색, 모양 in [("전", "가중치를 맞추기 전의 0.5 경계선", 주황, "dash"),
                           ("후", "가중치를 맞춘 뒤의 0.5 경계선", 보라, "solid")]:
        선 = 결과[열쇠][1]
        if 선 is not None:
            그림.add_trace(go.Scatter(
                x=[선[0][0], 선[1][0]], y=[선[0][1], 선[1][1]], mode="lines", name=이름,
                line=dict(color=색, width=3, dash=모양), hoverinfo="skip",
            ))
    그림.update_xaxes(title=우리말[x열], range=[xlo, xhi])
    그림.update_yaxes(title=우리말[y열], range=[ylo, yhi])
    for 열, 갱신 in [(x열, 그림.update_xaxes), (y열, 그림.update_yaxes)]:
        if 열 in 영일속성:
            갱신(tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
    그림.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    st.plotly_chart(그림)

    st.write(세워둔값_글([x열, y열]))
    문장들 = []
    for 열쇠, 이름, 색말 in [("전", "가중치를 맞추기 전", "주황색 점선"), ("후", "가중치를 맞춘 뒤", "보라색 실선")]:
        f0, 선 = 결과[열쇠]
        if 선 is not None:
            문장들.append(f"{색말}은 {이름}의 0.5 경계선입니다.")
        else:
            문장들.append(밖_문장(이름, "경계선", f0))
    if any(결과[k][1] is not None for k in 결과):
        문장들.append("선의 한쪽은 뇌졸중일 확률이 0.5 이상(뇌졸중이라 답함), 반대쪽은 0.5 미만(아님이라 답함)입니다.")
    st.write(" ".join(문장들))
    if x열 in 영일속성 or y열 in 영일속성:
        st.caption("0과 1만 있는 속성은 점이 겹치지 않도록 점만 살짝 흩어 찍었습니다. 마우스를 올리면 실제 값이 보입니다.")

# ── 3. 세 속성 3차원 그림과 0.5 평면 ────────────────────────────────
st.subheader("3. 세 속성으로 본 0.5 평면")
if len(고른것) < 3:
    st.info("입력 속성을 세 개 이상 고르면 3차원 그림이 나타납니다.")
else:
    st.caption("마우스로 끌면 돌려 볼 수 있고, 휠로 확대·축소할 수 있습니다.")
    축칸1, 축칸2, 축칸3 = st.columns(3)
    x3 = 축칸1.selectbox("가로축(3차원)", 고른것, index=0, format_func=lambda 열: 우리말[열])
    y3 = 축칸2.selectbox("세로축(3차원)", 고른것, index=1, format_func=lambda 열: 우리말[열])
    z3 = 축칸3.selectbox("높이축(3차원)", 고른것, index=2, format_func=lambda 열: 우리말[열])
    if len({x3, y3, z3}) < 3:
        st.info("세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        축들 = [x3, y3, z3]
        낮들, 높들 = zip(*[그림_범위(열) for 열 in 축들])
        낮들, 높들 = list(낮들), list(높들)
        결과3 = {}
        for 이름, 모델 in [("전", 로지스틱_전), ("후", 로지스틱_후)]:
            f0, 기울기, 중심 = 선형_기울기(모델, 축들, 낮들, 높들)
            결과3[이름] = (f0, 평면_다각형(f0, 기울기, 중심, 낮들, 높들))

        그림3 = go.Figure()
        난수 = np.random.default_rng(42)
        위치 = [점_위치(열, 난수) for 열 in 축들]
        진짜 = np.column_stack([테스트표[열].to_numpy(dtype=float) for 열 in 축들])
        마우스글3 = ("<b>%{fullData.name}</b><br>" + 우리말[x3] + ": %{customdata[0]:g}<br>"
                  + 우리말[y3] + ": %{customdata[1]:g}<br>" + 우리말[z3] + ": %{customdata[2]:g}<extra></extra>")
        for 정답, 이름, 색, 크기, 투명도 in [(0, "아님(0)", 파랑, 3, 0.45), (1, "뇌졸중(1, 양성)", 빨강, 5, 0.95)]:
            뽑기 = y_test == 정답
            그림3.add_trace(go.Scatter3d(
                x=위치[0][뽑기], y=위치[1][뽑기], z=위치[2][뽑기], mode="markers", name=f"실제 {이름}",
                marker=dict(size=크기, color=색, opacity=투명도),
                customdata=진짜[뽑기], hovertemplate=마우스글3,
            ))
        for 열쇠, 이름, 색 in [("전", "가중치를 맞추기 전의 0.5 평면", 주황), ("후", "가중치를 맞춘 뒤의 0.5 평면", 보라)]:
            점들 = 결과3[열쇠][1]
            if 점들 is not None:
                개수 = len(점들)
                그림3.add_trace(go.Mesh3d(
                    x=점들[:, 0], y=점들[:, 1], z=점들[:, 2],
                    i=[0] * (개수 - 2), j=list(range(1, 개수 - 1)), k=list(range(2, 개수)),
                    color=색, opacity=0.45, name=이름, showlegend=True, hoverinfo="skip", flatshading=True,
                ))

        장면 = {}
        for 이름, 열, 낮, 높 in zip(("xaxis", "yaxis", "zaxis"), 축들, 낮들, 높들):
            축설정 = dict(title=dict(text=우리말[열]), range=[낮, 높])
            if 열 in 영일속성:
                축설정.update(tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
            장면[이름] = 축설정
        그림3.update_layout(
            height=680, margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(aspectmode="cube", **장면),
            legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0),
        )
        st.plotly_chart(그림3)

        st.write(세워둔값_글(축들))
        문장들 = []
        for 열쇠, 이름, 색말 in [("전", "가중치를 맞추기 전", "주황색 평면"), ("후", "가중치를 맞춘 뒤", "보라색 평면")]:
            f0, 점들 = 결과3[열쇠]
            if 점들 is not None:
                문장들.append(f"{색말}은 {이름}의 0.5 평면입니다.")
            else:
                문장들.append(밖_문장(이름, "평면", f0).replace("이 그림 범위 밖", "이 그림 상자 밖"))
        if any(결과3[k][1] is not None for k in 결과3):
            문장들.append("평면의 한쪽은 뇌졸중일 확률이 0.5 이상, 반대쪽은 0.5 미만입니다.")
        st.write(" ".join(문장들))
        if any(열 in 영일속성 for 열 in 축들):
            st.caption("0과 1만 있는 속성은 점이 겹치지 않도록 점만 살짝 흩어 찍었습니다. 마우스를 올리면 실제 값이 보입니다.")
