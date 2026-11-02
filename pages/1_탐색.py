import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
st.set_page_config(page_title="뇌졸중 예측 실습실 · 탐색", page_icon="🧠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

STROKE_LABEL = {0: "겪지 않음", 1: "겪음"}
YESNO_LABEL = {0: "없음", 1: "있음"}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map(STROKE_LABEL)

st.title("🧠 뇌졸중 예측 실습실")
st.header("🔎 탐색")
st.caption("같은 데이터를 여러 방향에서 살펴봐요.")

# ---------------------------------------------------------------
# 1) 나이와 평균 혈당의 분포
# ---------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")

left, right = st.columns(2)

fig_age = px.histogram(
    df,
    x="age",
    nbins=30,
    title="나이 분포",
    labels={"age": "나이", "count": "사람 수"},
)
fig_age.update_layout(yaxis_title="사람 수", bargap=0.05)
left.plotly_chart(fig_age)

fig_glu = px.histogram(
    df,
    x="avg_glucose_level",
    nbins=40,
    title="평균 혈당 분포",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_glu.update_layout(yaxis_title="사람 수", bargap=0.05)
right.plotly_chart(fig_glu)

# ---------------------------------------------------------------
# 2) 뇌졸중 여부에 따른 비교
# ---------------------------------------------------------------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")

order = {"뇌졸중 여부": ["겪지 않음", "겪음"]}
left, right = st.columns(2)

fig_box_age = px.box(
    df,
    x="뇌졸중 여부",
    y="age",
    color="뇌졸중 여부",
    category_orders=order,
    title="나이 비교",
    labels={"age": "나이"},
)
fig_box_age.update_layout(showlegend=False)
left.plotly_chart(fig_box_age)

fig_box_glu = px.box(
    df,
    x="뇌졸중 여부",
    y="avg_glucose_level",
    color="뇌졸중 여부",
    category_orders=order,
    title="평균 혈당 비교",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_box_glu.update_layout(showlegend=False)
right.plotly_chart(fig_box_glu)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(["겪지 않음", "겪음"])
    .reset_index()
)
mean_table.columns = ["뇌졸중 여부", "사람 수", "평균 나이", "평균 혈당"]
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)

st.write("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True)

# ---------------------------------------------------------------
# 3) 고혈압·심장병 여부에 따른 뇌졸중 비율
# ---------------------------------------------------------------
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_by(col: str, title: str):
    """col 값(0/1)마다 뇌졸중 비율(%)을 계산해 막대그래프를 만든다."""
    g = (
        df.groupby(col)["stroke"]
        .agg(사람_수="size", 뇌졸중_사람_수="sum", 비율="mean")
        .reset_index()
    )
    g["비율"] = (g["비율"] * 100).round(2)
    g["구분"] = g[col].map(YESNO_LABEL)
    fig = px.bar(
        g,
        x="구분",
        y="비율",
        color="구분",
        category_orders={"구분": ["없음", "있음"]},
        title=title,
        labels={"구분": title.replace("별 뇌졸중 비율", ""), "비율": "뇌졸중 비율(%)"},
        hover_data={"사람_수": True, "뇌졸중_사람_수": True, "구분": False},
    )
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="뇌졸중 비율(%)")
    return fig


left, right = st.columns(2)
left.plotly_chart(stroke_rate_by("hypertension", "고혈압 여부별 뇌졸중 비율"))
right.plotly_chart(stroke_rate_by("heart_disease", "심장병 여부별 뇌졸중 비율"))

# ---------------------------------------------------------------
# 4) bmi가 비어 있는 사람들
# ---------------------------------------------------------------
st.subheader("4. 체질량지수(bmi)가 비어 있는 사람들")

bmi_missing = df[df["bmi"].isna()]
n_all = len(df)
n_all_stroke = int(df["stroke"].sum())
n_miss = len(bmi_missing)
n_miss_stroke = int(bmi_missing["stroke"].sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람들", "전체"],
        "사람 수": [n_miss, n_all],
        "뇌졸중 겪은 사람 수": [n_miss_stroke, n_all_stroke],
        "뇌졸중 비율(%)": [
            round(n_miss_stroke / n_miss * 100, 2) if n_miss else 0.0,
            round(n_all_stroke / n_all * 100, 2) if n_all else 0.0,
        ],
    }
)
st.dataframe(bmi_table, hide_index=True)

# ---------------------------------------------------------------
# 5) 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.subheader("5. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking_table["흡연 상태"] = smoking_table["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking_table, hide_index=True)
