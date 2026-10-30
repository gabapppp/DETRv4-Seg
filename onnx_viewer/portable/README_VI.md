# DETRv4-Seg ONNX test

Gói này chứa app PyQt6, detector và mask ONNX, manifest cùng ảnh mẫu nếu có. Inference dùng ONNX Runtime CPU, không cần cài PyTorch, CUDA hay repo train.

## Chạy trên máy khác

1. Giải nén toàn bộ ZIP vào thư mục có quyền ghi, ví dụ `D:\DETRv4-Seg-ONNX-test` hoặc `~/DETRv4-Seg-ONNX-test`.
2. Cài Python 3.12 nếu máy chưa có; hỗ trợ Python 3.11–3.13.
3. Windows: nhấp đúp `run_windows.bat`. Linux: mở terminal tại thư mục đã giải nén và chạy `bash run_linux.sh`.
4. Lần đầu script tạo `.venv` ngay trong thư mục và cài các thư viện từ Internet. Các lần sau dùng lại môi trường đã cài. Nếu có lỗi, cửa sổ lệnh sẽ hiển thị nội dung để kiểm tra.
5. Model đã được chọn sẵn. Bấm **Open image folder**, chọn `sample_images` hoặc thư mục ảnh của bạn, rồi **Infer current** / **Infer folder**.

Không tách riêng file ONNX: giữ nguyên `models/manifest.json`, `models/detector.onnx` và `models/mask.onnx` cùng thư mục. Có thể đổi vị trí toàn bộ gói trước lần chạy đầu; sau khi chuyển sang máy khác, tạo lại môi trường `.venv` bằng script, không copy môi trường Python giữa các hệ điều hành.

App hiển thị **Original** và **Inference** cạnh nhau. Cuộn chuột hoặc bấm **+ / −** để zoom; giữ chuột trái kéo để di chuyển vùng xem. Hai khung đồng bộ zoom/pan. **Fit** hoặc nhấp đúp để vừa khung; **1:1** để xem ở tỷ lệ pixel thật. Inference trả kết quả giữ nguyên vị trí/zoom đang xem, chuyển ảnh mới tự Fit.

Trong khung **Inference**, rê chuột vào instance để xem ô nhỏ `C1 · Ø 20.30 px`. Bật/tắt ô này bằng **Diameter (hover)**; chữ giữ cỡ cố định khi zoom và không được vẽ vào ảnh preview. Giữ chuột trái trên instance để tạm ẩn overlay của nó; thả ra để hiện lại. Kéo chuột vẫn pan bình thường. Trỏ vào instance rồi nhấn **H** để giữ nó ẩn cho ảnh đó trong phiên làm việc, kể cả khi chuyển ảnh rồi quay lại. **Show all instances** hoặc checkbox **Show** trong bảng khôi phục instance; infer lại ảnh sẽ reset trạng thái ẩn. Các thao tác này không đổi mask hoặc phép đo, không chạy infer lại.

App vẽ contour và vòng tròn C từ `minEnclosingCircle`; Diameter là `2 × radius` theo pixel ảnh gốc. Bảng hiển thị tâm, đường kính và score. **Save results** lưu overlay PNG và `measurements.csv`. D hiện `N/A` nếu không có contour hợp lệ. Zoom không thay đổi phép đo Diameter.

## Cập nhật sau rsync

Sau khi rsync code mới vào `onnx_viewer/` và module `detrseg/roi.py`, chạy trong thư mục gói:

```bash
python3 onnx_viewer/portable/update_viewer.py --root .
bash run_linux.sh
```

Nếu đã chuyển thêm các model/manifest mới vào `models/`, thêm `--refresh-models` vào lệnh updater để nhận checksum model mới. Chỉ cập nhật code thì giữ nguyên checksum model. Launcher sẽ tự tạo pip nếu `.venv` thiếu pip.

## Kiểm tra không mở cửa sổ

Windows: `run_windows.bat --check`.

Linux: `bash run_linux.sh --check`.

Lệnh kiểm tra checksum, nạp cả hai model và chạy ảnh mẫu bằng CPU. Chỉ kiểm tra file, không cài thư viện: `python run_viewer.py --verify-only`.

Python từ https://www.python.org/downloads/. Linux cần có môi trường desktop để mở cửa sổ Qt; nếu Python thiếu module `venv`, cài gói `python3-venv` của hệ điều hành. Bản CPU không yêu cầu GPU.
