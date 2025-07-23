<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

# Tư Duy Trí Tuệ Nhân Tạo

# Thành viên nhóm

| STT |   MSSV   |                Họ và Tên |     Chức vụ | Github | Email |
| --- | :------: | -----------------------: | ----------: | ------------------------------------------------------: | ---------------------: |
| 1   | 25410098 |          Trần Hải Nguyên | Nhóm trưởng | [nguyentran070397](https://github.com/nguyentran070397) | 25410098@ms.uit.edu.vn |
| 2   | 25410139 |         Nguyễn Phước Thọ |  Thành viên | [nphuoctho](https://github.com/nphuoctho) | 25410139@ms.uit.edu.vn |
| 3   | 25410124 |             Đỗ Trọng Tài |  Thành viên | [taidotrong](https://github.com/taidotrong) | 25410124@ms.uit.edu.vn |
| 4   | 25410070 |             Đỗ Danh Khoa |  Thành viên | [dodanhkhoa](https://github.com/dodanhkhoa) | 25410070@ms.uit.edu.vn |
| 5   | 25410145 | Dương Phương Chương Toàn |  Thành viên | [ToanIT2004](https://github.com/ToanIT2004) | 25410145@ms.uit.edu.vn |

## GIỚI THIỆU NHÓM
- **Số thứ tự nhóm:** 12
- **Tên nhóm:** LangParse

## GIỚI THIỆU MÔN HỌC
- **Mã môn học:** AI002
- **Mã lớp:** AI002.E31.CN2.TTNT
- **Năm học:** Học kỳ 3 (2025-2026)
- **Giảng viên:** Duy Phan Thế - duypt@uit.edu.vn

## ĐỒ ÁN CUỐI KỲ
- **Tên đồ án:** Trợ lý AI theo dõi dùng thuốc và nhắc nhở liều dùng cá nhân hóa
- **Thư mục:** None

---

# Trợ Lý Nhắc Nhở Dùng Thuốc AI

Dự án này là một trợ lý AI giúp dự đoán và nhắc nhở bạn uống thuốc dựa trên ngữ cảnh hàng ngày của bạn.

## Tính năng
- Dự đoán khả năng bạn quên uống thuốc bằng AI
- Cung cấp nhắc nhở với các mức độ khẩn cấp khác nhau
- Giao diện dòng lệnh đơn giản

## Yêu cầu
- Python 3.8 trở lên
- Xem file `requirements.txt` để biết các thư viện Python cần thiết

## Cài đặt
1. **Tải hoặc clone kho lưu trữ này về máy.**
2. **Di chuyển vào thư mục dự án:**
   ```sh
   cd /Users/taidotrong/Desktop/Learning/AI-thinking
   ```
3. **Cài đặt các thư viện Python cần thiết:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Đảm bảo file `medication_model.pkl` có trong thư mục dự án.**

## Sử dụng
Chạy file chính:
```sh
python main.py
```

Bạn sẽ được yêu cầu nhập:
- Thứ trong tuần (0=Thứ 2, 1=Thứ 3, ..., 6=Chủ nhật)
- Buổi trong ngày (0=Sáng, 1=Tối)
- Chỉ số huyết áp tâm thu của bạn

AI sẽ phân tích dữ liệu bạn nhập và đưa ra nhắc nhở dựa trên nguy cơ bạn có thể quên uống thuốc.

## Lưu ý
- File mô hình (`medication_model.pkl`) phải nằm cùng thư mục với `main.py`.
- Nếu muốn huấn luyện lại mô hình, sử dụng file `train_model.py` (xem chi tiết trong script).

## Giấy phép
Dự án này chỉ dành cho mục đích học tập.

<!-- Footer -->
<p align='center'>Copyright © 2025 - LangParse</p> 