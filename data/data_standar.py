import pandas as pd
import re
import unicodedata

# Bước 1: Load dữ liệu
df = pd.read_csv("mon_an.csv")

# Bước 2: Xóa "Muỗng; Gram;" trong cột Nguyên liệu
df["Nguyên liệu"] = df["Nguyên liệu"].str.replace(r"^(Muỗng;)?\s*(Gram;)?", "", regex=True).str.strip()

# Bước 3: Hàm tách nội dung "Cách làm" thành 3 phần: Sơ chế, Thực hiện, Cách dùng
def tach_cach_lam(text):
    if pd.isna(text):
        return "", "", ""

    # Chuẩn hóa văn bản để dễ xử lý
    text = re.sub(r"\s+", " ", text.strip())  # bỏ các ký tự xuống dòng thừa
    text = unicodedata.normalize("NFC", text)  # chuẩn hóa unicode (tránh lỗi tiếng Việt)

    # Dùng regex để tách nội dung
    pattern = r"Sơ chế:\s*(.*?)(?=Thực hiện:|Cách dùng:|$)|Thực hiện:\s*(.*?)(?=Cách dùng:|$)|Cách dùng:\s*(.*)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)

    so_che = thuc_hien = cach_dung = ""

    for match in matches:
        if match[0]:
            so_che = match[0].strip()
        if match[1]:
            thuc_hien = match[1].strip()
        if match[2]:
            cach_dung = match[2].strip()

    return so_che, thuc_hien, cach_dung
# Áp dụng xử lý cho toàn bộ dữ liệu tách 3 cột
df[["So_che", "Thuc_hien", "Cach_dung"]] = df["Cách làm"].apply(lambda x: pd.Series(tach_cach_lam(x)))

# Bước 3: Xóa dòng dữ liệu bị trùngtrong 3 phần: Sơ chế, Thực hiện, Cách dùng
def chuan_hoa_cau(cau):
    # Chuẩn hóa unicode, lowercase, bỏ dấu câu để dễ phát hiện trùng
    cau = unicodedata.normalize("NFC", cau)
    cau = cau.lower().strip()
    cau = re.sub(r'[^\w\s]', '', cau)  # bỏ dấu câu
    cau = re.sub(r'\s+', ' ', cau)
    return cau

def xoa_cau_trung_lap(text):
    if pd.isna(text):
        return ""

        # Tách câu theo dấu kết thúc
    cau_list = re.split(r'(?<=[.!?])\s+', text.strip())

    cau_goc_khac_nhau = []
    cau_chuan_set = set()

    for cau in cau_list:
        cau_goc = cau.strip()
        cau_chuan = chuan_hoa_cau(cau_goc)

        if cau_chuan and cau_chuan not in cau_chuan_set:
            cau_chuan_set.add(cau_chuan)
            cau_goc_khac_nhau.append(cau_goc)

    return " ".join(cau_goc_khac_nhau)

# Xóa d liệu trùng lặp ở 3 cột
df["So_che"] = df["So_che"].apply(xoa_cau_trung_lap)
df["Thuc_hien"] = df["Thuc_hien"].apply(xoa_cau_trung_lap)
df["Cach_dung"] = df["Cach_dung"].apply(xoa_cau_trung_lap)


# Bước 4: Cột Thực hiện tác các bước cho dễ nhìn
def dinh_dang_thuc_hien(text):
    if pd.isna(text):
        return ""

    # Đảm bảo mỗi "Bước X" đều bắt đầu từ dòng mới
    text = re.sub(r"\s*(Bước\s*\d+[\.:])", r"\n\1", text, flags=re.IGNORECASE)

    # Loại bỏ khoảng trắng dư thừa và chuẩn hóa khoảng cách giữa các bước
    dong = text.strip().split("\n")
    ket_qua = []

    for d in dong:
        dong_sach = d.strip()
        if dong_sach:
            ket_qua.append(dong_sach)

    # Cách mỗi bước 1 dòng trắng
    return "\n\n".join(ket_qua)
# Tách các bước 1,2,3 thành đoạn
df["Thuc_hien"] = df["Thuc_hien"].apply(dinh_dang_thuc_hien)
# Xuất ra DataFrame kết quả
df_nguyen_lieu = pd.DataFrame(df)
# Lưu kết quả sạch thành file mới,Xuất lại file CSV với mã hóa đúng
df.to_csv("nguyen_lieu_sach.csv", index=False, encoding='utf-8-sig')
print("✅ Đã xuất file nguyen_lieu_sach.csv với mã hóa utf-8-sig (hỗ trợ tiếng Việt trong Excel)")


# Bước 5: Tách nguyên liệu thành dòng dạng: Tên nguyên liệu | Số lượng | Đơn vị
def tach_nguyen_lieu(text):
    if pd.isna(text):
        return []

    # Chuẩn hóa
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = text.replace("1/2", "0.5").replace("3/4", "0.75")
    text = re.sub(r'\s+', ' ', text)

    # Tách nguyên liệu theo dấu ;
    nglieu_raw = re.split(r';\s*', text)
    ket_qua = []
    seen = set()

    for item in nglieu_raw:
        item = item.strip()
        if not item:
            continue

        item_clean = re.sub(r'\([^)]*\)', '', item).strip()

        # Regex tìm số lượng và đơn vị
        match = re.match(r'^(.*?)(\d+[,\.]?\d*)\s*([a-zA-ZÀ-ỹà-ỹμµgGmMlLítítM]*)$', item_clean)
        if match:
            ten = match.group(1).strip(" :")
            sl = match.group(2)
            dv = match.group(3).strip()
            if not dv:
                dv = "lượng vừa đủ"
        else:
            # Nếu không bắt được số, gán số lượng = 1, đơn vị = lượng vừa đủ
            ten = item_clean.strip()
            sl = "1"
            dv = "lượng vừa đủ"

        key = ten.lower()
        if key not in seen:
            ket_qua.append({
                "Tên nguyên liệu": ten,
                "Số lượng": sl,
                "Đơn vị": dv
            })
            seen.add(key)

    return ket_qua

def tach_nguyen_lieu_dong_don(text):
    if pd.isna(text):
        return ""

    # Chuẩn hóa văn bản
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text)

    # Tách nguyên liệu thành dòng theo dấu . ; hoặc xuống dòng
    lines = re.split(r"[;\.\n\r]+", text)

    result_lines = []
    seen = set()

    for line in lines:
        line = line.strip(" .•–")
        if not line or re.search(r"GIA VỊ|BỮA|ĂN KÈM|TRÁNG MIỆNG", line, flags=re.IGNORECASE):
            continue

        # Nếu có dấu : -> tách tên và phần còn lại
        if ":" in line:
            parts = line.split(":", 1)
            ten = parts[0].strip()
            con_lai = parts[1].strip()
            line = f"{ten} {con_lai}"

        # Bắt tên + số lượng + đơn vị + đơn vị phụ (nếu có)
        match = re.match(r"^(.*?)(\d+(?:[.,]\d+)?)(?:\s*)([a-zA-ZÀ-ỹμMmlgGít]+)(?:\s*\(([^)]+)\))?$", line)
        if match:
            ten = match.group(1).strip(" :")
            so_luong = match.group(2).replace(",", ".")
            don_vi = match.group(3).strip()
            don_vi_phu = match.group(4)

            if don_vi_phu:
                line_out = f"{ten} {so_luong}{don_vi} ({don_vi_phu})."
            else:
                line_out = f"{ten} {so_luong}{don_vi}."
        else:
            # Nếu không match chuẩn, vẫn giữ lại dòng
            line_out = f"{line}."

        if line_out.lower() not in seen:
            result_lines.append(line_out)
            seen.add(line_out.lower())

    return "\n".join(result_lines)



# Giả sử bạn đã có DataFrame `df` và cột "Nguyên liệu"
ds_nguyen_lieu = df["Nguyên liệu"].apply(tach_nguyen_lieu)

# Gộp toàn bộ lại thành bảng dài (mỗi nguyên liệu 1 dòng)
df_nguyen_lieu = pd.DataFrame([row for sublist in ds_nguyen_lieu for row in sublist])

# Lưu kết quả sạch thành file mới,Xuất lại file CSV với mã hóa đúng
df_nguyen_lieu.to_csv("danh_sach_nguyen_lieu.csv", index=False, encoding="utf-8-sig")
print("✅ Đã xuất file danh_sach_nguyen_lieu.csv với mã hóa utf-8-sig (hỗ trợ tiếng Việt trong Excel)")
# Thêm cột nguyên liệu dạng dòng đơn
df["Nguyen_lieu"] = df["Nguyên liệu"].apply(tach_nguyen_lieu_dong_don)

# Lưu lại DataFrame chính kèm cột mới
df.to_csv("nguyen_lieu_sach.csv", index=False, encoding='utf-8-sig')
print("✅ Đã cập nhật và lưu thêm cột 'Nguyen_lieu_dong_don' vào nguyen_lieu_sach.csv")

# Đọc file gốc
df1 = pd.read_csv("nguyen_lieu_sach.csv")

def dinh_dang_cach_lam(so_che, thuc_hien, cach_dung):
    buoc_list = []

    if pd.notna(so_che) and so_che.strip().lower() != "không có nội dung.":
        buoc_list.append("🔹 Sơ chế:\n" + format_thanh_buoc(so_che))

    if pd.notna(thuc_hien) and thuc_hien.strip().lower() != "không có nội dung.":
        buoc_list.append("🔹 Thực hiện:\n" + format_thanh_buoc(thuc_hien))

    if pd.notna(cach_dung) and cach_dung.strip().lower() != "không có nội dung.":
        buoc_list.append("🔹 Cách dùng:\n" + format_thanh_buoc(cach_dung))

    return "\n\n".join(buoc_list)


def format_thanh_buoc(text):
    if pd.isna(text):
        return ""

    # Nếu text đã có dòng "Bước 1", "Bước 2"... thì giữ nguyên cách dòng
    if re.search(r"Bước\s*\d+", text, flags=re.IGNORECASE):
        text = re.sub(r"\s*(Bước\s*\d+[\.:]?)", r"\n\1", text, flags=re.IGNORECASE)

    # Tách câu theo dấu kết thúc để chia dòng nếu chưa có Bước
    lines = re.split(r'(?<=[.!?])\s+', text.strip())
    lines = [f"- {line.strip()}" for line in lines if line.strip()]
    return "\n".join(lines)

df1["Cach_lam"] = df1.apply(lambda row: dinh_dang_cach_lam(row["So_che"], row["Thuc_hien"], row["Cach_dung"]), axis=1)

# Xóa 2 cột "Nguyên liệu","Cách làm", "So_che","Thuc_hien","Cach_dung" nếu tồn tại
df1 = df1.drop(columns=["Nguyên liệu","Cách làm","So_che","Thuc_hien","Cach_dung"], errors="ignore")
print("✅ Đã xóa 5 cột 'Nguyên liệu', 'Cách làm' , 'So_che', 'Thuc_hien','Cach_dung' khỏi file nguyen_lieu_sach.csv")
# Ghi lại file (ghi đè hoặc đặt tên mới)
df1.to_csv("nguyen_lieu_sach2.csv", index=False, encoding="utf-8-sig")
print("✅ Đã xuất file danh_sach_nguyen_lieu2.csv với mã hóa utf-8-sig (hỗ trợ tiếng Việt trong Excel)")
