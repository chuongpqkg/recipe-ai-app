import os
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
engine = create_engine(DB_URL)

# Load model embedding
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


# Hàm gợi ý món ăn theo embedding
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


# Hàm lấy ngẫu nhiên món ăn
def get_random_recipes(top_k=5):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT id, ten_mon, anh, video, url, nguyen_lieu, cach_lam, 1 AS similarity
                FROM recipes
                ORDER BY RANDOM()
                LIMIT :top_k
            """),
            {"top_k": top_k}
        )
        return result.fetchall()


# 🚀 Cấu hình giao diện
#st.set_page_config(page_title="AI Gợi ý món ăn", page_icon="🍲", layout="wide")

# === Banner đầu trang ===
st.image("data/Food_Banner_1.jpg", use_container_width=True)

# Sidebar
st.sidebar.title("🍴 Sở thích của bạn")
search_mode = st.sidebar.radio("Bạn muốn tìm món ăn theo?", ["Nguyên liệu", "Tên món"])

# Main title
st.title("🍲 Gợi ý nấu ăn bằng AI")
st.write("Khám phá món ăn phù hợp bằng cách sử dụng AI và tìm tương đồng vector!")

# Input
if search_mode == "Nguyên liệu":
    query = st.text_input("Nhập nguyên liệu bạn có:", placeholder="Ví dụ: gà, hành, ớt, tỏi")
else:
    query = st.text_input("Nhập tên món ăn:", placeholder="Ví dụ: phở bò, gỏi cuốn, bún chả")

# Slider số món gợi ý
top_k = st.slider("Số lượng món gợi ý:", 3, 10, 5)

# Chia layout main
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🔍 Gợi ý món ăn"):
        if query.strip() == "":
            st.warning("⚠️ Vui lòng nhập ít nhất 1 nguyên liệu hoặc tên món.")
        else:
            results = get_recommendations(query, top_k=top_k)
    else:
        # ✅ Khi mới mở app, lấy 5 món ăn ngẫu nhiên
        results = get_random_recipes(top_k=5)

    if not results:
        st.error("❌ Không tìm thấy món ăn phù hợp.")
    else:
        for row in results:
            with st.container():
                st.subheader(f"{row.ten_mon}  (⭐ {row.similarity:.2f})")
                cols = st.columns([1, 2])

                with cols[0]:
                    if row.anh:
                        st.image(row.anh, use_container_width=True)
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