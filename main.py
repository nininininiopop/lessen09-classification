import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ---------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

# 큰 숫자 카드 스타일
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 3rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# 저장 / 불러오기 (내_기록/기록.json)
# ---------------------------------------------------------------
def load_notes() -> dict:
    """저장된 기록을 불러온다. 파일이 없거나 깨졌으면 빈 기록을 돌려준다."""
    empty = {"meanings": {}, "source": ""}
    if not SAVE_FILE.exists():
        return empty
    try:
        data = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
        return {
            "meanings": dict(data.get("meanings", {})),
            "source": str(data.get("source", "")),
        }
    except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError):
        return empty


def save_notes(meanings: dict | None = None, source: str | None = None) -> None:
    """바뀐 부분만 갱신해서 json 파일에 저장한다."""
    notes = load_notes()
    if meanings is not None:
        notes["meanings"] = meanings
    if source is not None:
        notes["source"] = source
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_FILE.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt(v) -> str:
    """3.0 -> 3 처럼 보기 좋게 바꾼다."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return f"{v:g}" if isinstance(v, float) else str(v)


def describe_values(s: pd.Series) -> str:
    """열의 '값의 종류'를 글로 설명한다."""
    nonnull = s.dropna()
    n = nonnull.nunique()
    if n == 0:
        return "값 없음"
    if not pd.api.types.is_numeric_dtype(s):
        vals = sorted(str(v) for v in nonnull.unique())
        return f"{n}가지: " + ", ".join(vals)
    if n <= 5:
        vals = sorted(nonnull.unique())
        return f"{n}가지: " + ", ".join(fmt(v) for v in vals)
    return f"숫자 (서로 다른 값 {n:,}개, 최소 {fmt(nonnull.min())} ~ 최대 {fmt(nonnull.max())})"


try:
    df = load_data()
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결을 확인해 주세요.\n\n{e}")
    st.stop()

notes = load_notes()

# ---------------------------------------------------------------
# 화면 맨 위: 제목
# ---------------------------------------------------------------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면 · 이 데이터는 무엇일까요?")

# ---------------------------------------------------------------
# 1) 큰 숫자 카드 네 개
# ---------------------------------------------------------------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 사람 수", f"{total:,}명", border=True)
c2.metric("열 개수", f"{n_cols}개", border=True)
c3.metric("stroke가 1인 사람 수", f"{n_stroke:,}명", border=True)
c4.metric("stroke가 1인 비율", f"{ratio:.1f}%", border=True)

# ---------------------------------------------------------------
# 2) 열 설명 표 (우리말 뜻 칸은 직접 입력)
# ---------------------------------------------------------------
st.subheader("열 설명표")
st.write("교재를 보고 **우리말 뜻** 칸을 채운 뒤 **저장하기**를 눌러 주세요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [notes["meanings"].get(c, "") for c in df.columns],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

with st.form("meaning_form"):
    edited = st.data_editor(
        table,
        hide_index=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻", help="교재를 보고 직접 적어 주세요."
            ),
            "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
        },
        key="meaning_editor",
    )
    saved_meaning = st.form_submit_button("💾 저장하기")

if saved_meaning:
    new_meanings = {
        row["열 이름"]: ("" if pd.isna(row["우리말 뜻"]) else str(row["우리말 뜻"]))
        for _, row in edited.iterrows()
    }
    try:
        save_notes(meanings=new_meanings)
        st.success("우리말 뜻을 저장했어요.")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")

# ---------------------------------------------------------------
# 3) 데이터 처음 다섯 줄 (원본 그대로)
# ---------------------------------------------------------------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True)

# ---------------------------------------------------------------
# 4) 데이터 출처
# ---------------------------------------------------------------
st.subheader("데이터 출처")
with st.form("source_form"):
    source_text = st.text_area(
        "교재에 나온 출처를 적어 주세요.",
        value=notes["source"],
        height=120,
        placeholder="여기에 출처를 적어 주세요.",
    )
    saved_source = st.form_submit_button("💾 저장하기")

if saved_source:
    try:
        save_notes(source=source_text)
        st.success("출처를 저장했어요.")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")
