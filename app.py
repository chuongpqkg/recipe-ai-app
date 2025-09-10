import os
import random
import streamlit as st
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load biến môi trường
# Thử load từ st.secrets (deploy trên Cloud)
st.set_page_config(page_title="AI Gợi ý món ăn", page_icon="🍲", layout="wide")
DB_URL = st.secrets.get("DB_URL", None)

# Nếu không có (local) thì load từ .env
if not DB_URL:
    load_dotenv()
    DB_URL = os.getenv("DB_URL") or os.getenv("DATABASE_URL")

if not DB_URL:
    st.error("❌ Không tìm thấy DB_URL trong .env hoặc secrets.toml")
    st.stop()

# Kết nối DB
@st.cache_resource
def get_engine():
    # Giới hạn số connection trong pool để không vượt quá của Supabase (thường = 5)
    return create_engine(DB_URL, pool_size=5, max_overflow=0)

engine = get_engine()

# Load model embedding
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


# Hàm gợi ý món ăn theo embedding nguyên liệu
def get_recommendations(ingredient_query, top_k=5):
    query_embedding = model.encode(ingredient_query).tolist()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, ten_mon, anh, video, url, nguyen_lieu, cach_lam,
                       1 - (embedding <=> (:query_embedding)::vector) AS similarity
                FROM recipes
                ORDER BY embedding <=> (:query_embedding)::vector
                LIMIT :top_k
            """),
            {"query_embedding": query_embedding, "top_k": top_k}
        )
        return result.fetchall()

# Hàm gợi ý món ăn theo embedding tên món ăn
def get_recommendations_by_name(name_query, top_k=5):
    query_embedding = model.encode(name_query).tolist()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, ten_mon, anh, video, url, nguyen_lieu, cach_lam,
                       1 - (embedding_name <=> (:query_embedding)::vector) AS similarity
                FROM recipes
                ORDER BY embedding_name <=> (:query_embedding)::vector
                LIMIT :top_k
            """),
            {"query_embedding": query_embedding, "top_k": top_k}
        )
        return result.fetchall()


# Hàm lấy ngẫu nhiên món ăn
def get_random_recipes(top_k=5):
    with engine.connect() as conn:
        # Lấy MAX(id) để biết giới hạn
        max_id = conn.execute(text("SELECT MAX(id) FROM recipes")).scalar()

        # Sinh ra ngẫu nhiên top_k id
        random_ids = [random.randint(1, max_id) for _ in range(top_k)]

        # Query theo id
        result = conn.execute(
            text("""
                SELECT id, ten_mon, anh, video, url, nguyen_lieu, cach_lam, 1 AS similarity
                FROM recipes
                WHERE id = ANY(:ids)
            """),
            {"ids": random_ids}
        )
        return result.fetchall()


# 🚀 Cấu hình giao diện
#st.set_page_config(page_title="AI Gợi ý món ăn", page_icon="🍲", layout="wide")

# === Banner đầu trang ===
st.image("data/Food_Banner_1.jpg", width="stretch")

# Sidebar
st.sidebar.title("🍴 Sở thích của bạn")
search_mode = st.sidebar.radio("Bạn muốn tìm món ăn theo?", ["Nguyên liệu", "Tên món"])

# Main title
st.title("🍲 Gợi ý nấu ăn bằng AI")
st.write("Khám phá món ăn phù hợp bằng cách sử dụng AI và tìm tương đồng vector!")

# === Khu vực tìm kiếm ===
st.markdown(
    "<h2 style='color:#27AE60; font-size:28px; font-weight:bold; margin-bottom:15px;'>🔍 Tìm kiếm món ăn</h2>",
    unsafe_allow_html=True
)

# Dòng nhập tìm kiếm
col_input1, col_input2 = st.columns([1, 3])
with col_input1:
    if search_mode == "Nguyên liệu":
        st.markdown(
            "<p style='text-align:right; font-size:20px; font-weight:bold;'>Nhập nguyên liệu bạn đang có:</p>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<p style='text-align:right; font-size:20px; font-weight:bold;'>Nhập tên món ăn bạn cần tìm:</p>",
            unsafe_allow_html=True
        )

with col_input2:
    if search_mode == "Nguyên liệu":
        query = st.text_input("", placeholder="Ví dụ: gà, hành, ớt, tỏi", label_visibility="collapsed")
    else:
        query = st.text_input("", placeholder="Ví dụ: phở bò, gỏi cuốn, bún chả", label_visibility="collapsed")

# Dòng slider + nút gợi ý (nút nằm dưới slider cùng cột)
col_slider1, col_slider2 = st.columns([1, 3])
with col_slider1:
    st.markdown(
        "<p style='text-align:right; font-weight:bold;'>Số lượng món ăn bạn muốn gợi ý:</p>",
        unsafe_allow_html=True
    )
with col_slider2:
    top_k = st.slider("", 3, 10, 5, label_visibility="collapsed")

    # Nút nằm ngay dưới slider
    if st.button("🔍 Gợi ý món ăn"):
        if query.strip() == "":
            st.warning("⚠️ Vui lòng nhập ít nhất 1 nguyên liệu hoặc tên món.")
        else:
            if search_mode == "Nguyên liệu":
                st.session_state.results = get_recommendations(query, top_k=top_k)
            else:
                st.session_state.results = get_recommendations_by_name(query, top_k=top_k)

# Chia layout main
col1, col2 = st.columns([3, 1])

with col1:
    # Luôn load 5 món ăn hàng ngày (ngẫu nhiên)
    daily_recipes = get_random_recipes(top_k=5)

    # Hiển thị kết quả tìm kiếm nếu có
    if "results" in st.session_state and st.session_state.results:
        st.markdown(
            "<h2 style='color:#27AE60; font-size:28px; font-weight:bold;'>🍽️ Món ăn gợi ý cho bạn</h2>",
            unsafe_allow_html=True
        )
        for row in st.session_state.results:
            with st.container():
                st.subheader(f"{row.ten_mon}  (⭐ {row.similarity:.2f})")
                cols = st.columns([1, 2])

                with cols[0]:
                    if row.anh:
                        st.image(row.anh, width="stretch")
                    if row.video and "youtube" in row.video:
                        st.video(row.video)

                with cols[1]:
                    st.markdown(f"**Nguyên liệu:** {row.nguyen_lieu}")
                    st.markdown(f"**Cách làm:** {row.cach_lam}")
                    if row.url:
                        st.markdown(f"[🔗 Xem chi tiết]({row.url})")

                st.markdown("---")

    # Luôn hiển thị 5 món ăn hàng ngày
    if daily_recipes:
        st.markdown(
            "<h2 style='color:#27AE60; font-size:28px; font-weight:bold;'>🍴 Món ăn ngày mới dành cho bạn</h2>",
            unsafe_allow_html=True
        )
        for row in daily_recipes:
            with st.container():
                st.subheader(row.ten_mon)
                cols = st.columns([1, 2])

                with cols[0]:
                    if row.anh:
                        st.image(row.anh, width="stretch")
                    if row.video and "youtube" in row.video:
                        st.video(row.video)

                with cols[1]:
                    st.markdown(f"**Nguyên liệu:** {row.nguyen_lieu}")
                    st.markdown(f"**Cách làm:** {row.cach_lam}")
                    if row.url:
                        st.markdown(f"[🔗 Xem chi tiết]({row.url})")

                st.markdown("---")

with col2:
    st.markdown("### 💡 Mẹo")
    st.info(
        """
        - Cụ thể với nguyên liệu bạn có  
        - Kết hợp nhiều cách chế biến  
        - Thử phong cách mới  
        - Tìm món ngẫu nhiên để lấy cảm hứng  
        - Có thể tìm theo tên món trực tiếp
        """
    )

    st.markdown("### 📊 Thống kê cơ sở dữ liệu")
    st.metric("Món ăn có sẵn", "2400+")
    st.metric("Kết hợp nguyên liệu", "∞")
    st.metric("Độ chính xác AI", "95%+")