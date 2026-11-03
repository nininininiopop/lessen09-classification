# pages/2_분류_모델.py — 분류 모델 페이지: 두 모델을 만들고 채점하고, 그림과 나무로 들여다본다
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="분류 모델 · 뇌졸중 예측 실습실", page_icon="🩺", layout="wide")
st.title("🧮 분류 모델")
st.write("뇌졸중을 겪었는지(1) 아닌지(0)를 맞히는 두 가지 모델을 만들고 채점합니다. "
         "뇌졸중을 **양성**, 아님을 **음성**으로 둡니다.")

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
빨강 = "#d62728"   # 뇌졸중(양성)
파랑 = "#1f77b4"   # 아님(음성)
글꼴 = "Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"

이름_로지스틱 = "로지스틱 회귀(확률로 답하는 모델)"
이름_나무 = "의사결정트리(질문으로 답하는 모델)"
이름_기준 = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"


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

# ── 1. 입력 속성 고르기 ─────────────────────────────────────────────
st.subheader("1. 입력으로 쓸 속성 고르기")
고른것 = st.multiselect(
    "입력으로 사용할 속성",
    options=전체속성,
    default=처음고른것,
    format_func=lambda 열: 우리말[열],
    key="분류_입력속성",
)
고른것 = [열 for 열 in 전체속성 if 열 in 고른것]          # 고른 차례와 상관없이 늘 같은 순서로
if len(고른것) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개보다 적으면 모델을 만들지 않고 여기서 멈춥니다.")
    st.stop()

# ── 2. 나누기와 준비 ────────────────────────────────────────────────
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
    채움글 = (f" 체질량지수가 비어 있던 사람(학습용 {빈훈련:,}명, 테스트용 {빈테스트:,}명)은 "
              f"학습용의 중앙값 {훈련중앙값:.1f}으로 채웠습니다.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("고른 속성에 빈 값이 남아 있어 모델을 만들 수 없습니다.")
    st.stop()

st.caption(
    f"번호(id) 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"테스트용 {len(X_test):,}명(그중 뇌졸중 {int(y_test.sum()):,}명), "
    f"학습용 {len(X_train):,}명(그중 뇌졸중 {int(y_train.sum()):,}명).{채움글}"
)

# 크기 맞추기는 학습용으로만 배우고, 테스트용에는 그 기준을 그대로 적용한다
크기맞추개 = StandardScaler().fit(X_train)
Xs_train = 크기맞추개.transform(X_train)
Xs_test = 크기맞추개.transform(X_test)

# ── 3. 모델 만들기 ──────────────────────────────────────────────────
로지스틱 = LogisticRegression(max_iter=1000, random_state=42).fit(Xs_train, y_train)
# 나무는 질문이 사람이 읽을 수 있는 원래 값으로 나오도록 크기를 맞추기 전 값으로 학습한다
# (크기 맞추기는 나무가 던지는 질문의 순서와 답을 바꾸지 않는다)
나무 = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
기준 = DummyClassifier(strategy="most_frequent").fit(Xs_train, y_train)

정확도 = {
    "로지스틱": (accuracy_score(y_train, 로지스틱.predict(Xs_train)),
              accuracy_score(y_test, 로지스틱.predict(Xs_test))),
    "나무": (accuracy_score(y_train, 나무.predict(X_train)),
            accuracy_score(y_test, 나무.predict(X_test))),
    "기준": (accuracy_score(y_train, 기준.predict(Xs_train)),
            accuracy_score(y_test, 기준.predict(Xs_test))),
}


def 카드(제목, 훈련, 테스트):
    return f"""<div style="border:1px solid rgba(128,128,128,.45); border-radius:12px; padding:16px 18px; background:rgba(128,128,128,.07);">
<div style="font-size:.95rem; font-weight:600; line-height:1.35; min-height:4.2em;">{제목}</div>
<div style="font-size:2.9rem; font-weight:800; line-height:1.2; margin:6px 0 0;">{테스트:.1%}</div>
<div style="font-size:.8rem; opacity:.7;">테스트 데이터의 정확도</div>
<div style="display:flex; gap:22px; margin-top:12px; font-size:.82rem; opacity:.85;">
<div>훈련 데이터의 정확도<br><b>{훈련:.1%}</b></div>
<div>테스트 데이터의 정확도<br><b>{테스트:.1%}</b></div>
</div>
</div>"""


st.subheader("2. 채점 결과 — 정확도")
카드칸 = st.columns(3)
for 칸, (제목, 열쇠) in zip(카드칸, [(이름_로지스틱, "로지스틱"), (이름_나무, "나무"), (이름_기준, "기준")]):
    칸.markdown(카드(제목, *정확도[열쇠]), unsafe_allow_html=True)
st.caption("정확도는 전체 가운데 맞힌 비율입니다. 뇌졸중을 겪은 사람이 드물어서 입력을 보지 않고 "
           "'아님'만 답해도 정확도가 높게 나옵니다. 두 모델이 기준 모델보다 얼마나 나은지 견주어 보세요.")

# ── 4. 두 속성으로 본 그림 ──────────────────────────────────────────
st.subheader("3. 두 속성으로 본 테스트 데이터")
선택칸1, 선택칸2 = st.columns(2)
x열 = 선택칸1.selectbox("가로축", 고른것, index=0, format_func=lambda 열: 우리말[열])
y열 = 선택칸2.selectbox("세로축", 고른것, index=1, format_func=lambda 열: 우리말[열])


def 그림_범위(열):
    if 열 in 영일속성:
        return -0.5, 1.5
    낮 = float(X_test[열].min())
    높 = float(X_test[열].max())
    폭 = (높 - 낮) or 1.0
    return 낮 - 0.06 * 폭, 높 + 0.06 * 폭


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
    가장먼 = max(((p, q) for i, p in enumerate(서로다른) for q in 서로다른[i + 1:]),
              key=lambda 쌍: (쌍[0][0] - 쌍[1][0]) ** 2 + (쌍[0][1] - 쌍[1][1]) ** 2)
    return 가장먼


if x열 == y열:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    고정값 = {열: float(X_test[열].median()) for 열 in 고른것}      # 축이 아닌 속성은 테스트용 중앙값에 세운다
    xlo, xhi = 그림_범위(x열)
    ylo, yhi = 그림_범위(y열)
    x0, y0 = (xlo + xhi) / 2, (ylo + yhi) / 2

    # (가) 의사결정트리가 그림을 나눈 칸: 격자 위의 모든 점에 대해 나무의 답을 구한다
    xs = np.linspace(xlo, xhi, 200)
    ys = np.linspace(ylo, yhi, 200)
    XX, YY = np.meshgrid(xs, ys)
    격자 = pd.DataFrame({열: np.full(XX.size, 고정값[열]) for 열 in 고른것})
    격자[x열] = XX.ravel()
    격자[y열] = YY.ravel()
    칸별답 = 나무.predict(격자).reshape(XX.shape)

    # (나) 로지스틱 회귀가 0.5로 가르는 선: 두 축에 대해 곧은 선이므로 세 점의 값만 알면 된다
    세점 = pd.DataFrame({열: [고정값[열]] * 3 for 열 in 고른것})
    세점[x열] = [x0, x0 + (xhi - xlo) / 2, x0]
    세점[y열] = [y0, y0, y0 + (yhi - ylo) / 2]
    값 = 로지스틱.decision_function(크기맞추개.transform(세점))      # 0이면 확률 0.5
    f0 = float(값[0])
    A = float(값[1] - 값[0]) / ((xhi - xlo) / 2)
    B = float(값[2] - 값[0]) / ((yhi - ylo) / 2)
    선 = 선_자르기(f0, A, B, x0, y0, xlo, xhi, ylo, yhi)

    # (다) 그림 그리기
    그림 = go.Figure()
    그림.add_trace(go.Heatmap(
        x=xs, y=ys, z=칸별답, zmin=0, zmax=1,
        colorscale=[[0, 파랑], [1, 빨강]], opacity=0.22,
        showscale=False, showlegend=False, hoverinfo="skip",
    ))
    for 색, 글 in [(파랑, "의사결정트리가 '아님'이라 답하는 칸"), (빨강, "의사결정트리가 '뇌졸중'이라 답하는 칸")]:
        그림.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=글,
            marker=dict(symbol="square", size=13, color=색, opacity=0.35),
        ))

    흩뿌리개 = np.random.default_rng(42)

    def 점_위치(열):
        값들 = 테스트표[열].to_numpy(dtype=float)
        if 열 in 영일속성:   # 0과 1만 있는 속성은 점이 겹치지 않게 그림에서만 살짝 흩어 찍는다
            return 값들 + 흩뿌리개.uniform(-0.12, 0.12, len(값들))
        return 값들

    점x = 점_위치(x열)
    점y = 점_위치(y열)
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
    if 선 is not None:
        그림.add_trace(go.Scatter(
            x=[선[0][0], 선[1][0]], y=[선[0][1], 선[1][1]], mode="lines",
            name=f"{이름_로지스틱}가 0.5로 가르는 선",
            line=dict(color="#8e44ad", width=3), hoverinfo="skip",
        ))

    그림.update_xaxes(title=우리말[x열], range=[xlo, xhi])
    그림.update_yaxes(title=우리말[y열], range=[ylo, yhi])
    for 열, 갱신 in [(x열, 그림.update_xaxes), (y열, 그림.update_yaxes)]:
        if 열 in 영일속성:
            갱신(tickvals=[0, 1], ticktext=["없음(0)", "있음(1)"])
    그림.update_layout(
        height=580, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0),
    )
    st.plotly_chart(그림)

    남은 = [열 for 열 in 고른것 if 열 not in (x열, y열)]
    if 남은:
        조각 = ", ".join(f"{우리말[열]} {고정값[열]:g}" if 열 in 영일속성 else f"{우리말[열]} {고정값[열]:.1f}"
                       for 열 in 남은)
        st.write(f"가로축·세로축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {조각}")
    else:
        st.write("고른 속성이 두 개뿐이라 따로 세워 둔 값은 없습니다.")

    if 선 is not None:
        st.write(f"보라색 선은 {이름_로지스틱}가 뇌졸중일 확률을 0.5로 보는 자리입니다. "
                 "선의 한쪽은 0.5 이상(뇌졸중이라 답함), 반대쪽은 0.5 미만(아님이라 답함)입니다.")
    else:
        쪽 = "0.5 이상이라 모두 뇌졸중" if f0 > 0 else "0.5 미만이라 모두 아님"
        st.write(f"확률 0.5로 가르는 선이 이 그림 범위 밖에 있어 그리지 못했습니다. "
                 f"그림 안의 모든 자리에서 뇌졸중일 확률이 {쪽}이라고 답합니다.")
    if x열 in 영일속성 or y열 in 영일속성:
        st.caption("0과 1만 있는 속성은 점이 겹치지 않도록 점만 살짝 흩어 찍었습니다. 마우스를 올리면 실제 값이 보입니다.")
    st.caption("옅은 색 칸은 의사결정트리가 같은 그림을 어떻게 나누는지 보여 줍니다.")


# ── 5. 의사결정트리가 던진 질문 ─────────────────────────────────────
st.subheader("4. 의사결정트리가 던진 질문")


def 나무_DOT(모델, X, y, 열이름들):
    """마디마다 훈련용 사람 수·뇌졸중 사람 수·비율을 적은 DOT 글을 만든다."""
    t = 모델.tree_
    경로 = 모델.decision_path(X)                                        # 사람 × 마디 (그 마디를 지나갔는가)
    총 = np.asarray(경로.sum(axis=0)).ravel().astype(int)
    양성 = np.asarray(경로.T.dot(np.asarray(y))).ravel().astype(int)
    줄 = [
        "digraph G {",
        f'graph [bgcolor="white", rankdir=TB, nodesep=0.35, ranksep=0.55];',
        f'node [shape=box, style="rounded,filled", width=2.6, margin="0.18,0.1", fontname="{글꼴}", '
        f'fontsize=13, color="#555555", fontcolor="#111111"];',
        f'edge [fontname="{글꼴}", fontsize=13, color="#555555", fontcolor="#111111"];',
    ]
    잎수 = 0
    아님잎수 = 0
    물은열 = []
    for i in range(t.node_count):
        비율 = 양성[i] / 총[i] if 총[i] else 0.0
        통계 = f"훈련용 {총[i]:,}명\\n뇌졸중 {양성[i]:,}명 ({비율:.1%})"
        왼, 오 = int(t.children_left[i]), int(t.children_right[i])
        if 왼 == -1:                                                    # 더 묻지 않고 답을 내는 마디
            답 = int(np.argmax(t.value[i][0]))
            잎수 += 1
            아님잎수 += int(답 == 0)
            색 = "#f4b6b6" if 답 == 1 else "#b9d7f0"
            줄.append(f'n{i} [label="답: {"뇌졸중" if 답 == 1 else "아님"}\\n{통계}", fillcolor="{색}", penwidth=2];')
        else:
            열 = 열이름들[int(t.feature[i])]
            if 열 not in 물은열:
                물은열.append(열)
            문턱 = round(float(t.threshold[i]), 2)
            질문 = f"{우리말[열]} = 0 (없음)?" if 열 in 영일속성 else f"{우리말[열]} ≤ {문턱:g}?"
            줄.append(f'n{i} [label="{질문}\\n{통계}", fillcolor="#eeeeee"];')
            줄.append(f'n{i} -> n{왼} [label="예"];')
            줄.append(f'n{i} -> n{오} [label="아니요"];')
    줄.append("}")
    return "\n".join(줄), 잎수, 아님잎수, 물은열


DOT글, 잎수, 아님잎수, 물은열 = 나무_DOT(나무, X_train, y_train, list(X_train.columns))
st.caption(f"{이름_나무}는 질문을 세 번까지만 던지고, 마지막 마디에 훈련용 사람이 다섯 명 미만이 되는 나눔은 하지 않습니다. "
           "마디 안의 숫자는 훈련용 사람 수와 그중 실제 뇌졸중인 사람 수·비율입니다. 가지의 '예'는 질문이 맞는 쪽입니다.")
st.graphviz_chart(DOT글)

st.write(f"답을 내는 마디는 모두 {잎수}칸이고, 그중 {아님잎수}칸이 '아님'이라고 답합니다.")
if 물은열:
    안물은 = [열 for 열 in 고른것 if 열 not in 물은열]
    덧붙임 = f" (묻지 않은 속성: {', '.join(우리말[열] for 열 in 안물은)})" if 안물은 else ""
    st.write(f"고른 속성 {len(고른것)}개 가운데 이 나무가 실제로 물은 것은 "
             f"{', '.join(우리말[열] for 열 in 물은열)}입니다.{덧붙임}")
else:
    st.write(f"고른 속성 {len(고른것)}개 가운데 이 나무가 실제로 물은 것은 없습니다. 질문 없이 바로 답을 냅니다.")
