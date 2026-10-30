# Thử nghiệm S + P2, một nhãn, Tiny U-Net cho ROI nhỏ

Nhánh `experiment/s-p2-tiny-unet` dành cho thử nghiệm; chưa chọn hoặc thay model triển khai. Script dưới đây chỉ chạy khi người dùng gọi. Không cần mở dataset để cấu hình.

## Train thử

```bash
cd /home/gabap/prj/DETRv4-Seg
.venv-train/bin/python scripts/experiment_s_p2_tiny.py \
  --output runs/experiment_s_p2_tiny_v1 \
  --mlflow-uri http://192.168.122.112:5000
```

Script chạy tuần tự:

- Setup weights COCO **S**, train detector **S + P2**, input 640, 3000 steps, batch 4.
- Bật `--single-class --class-name ball`: các nhãn của input được ánh xạ về ID 1; giữ nguyên từng instance, bbox và image split. Head detector có đúng một class. `train.json`, `val.json`, `label_map.json` trong output dùng nhãn đã gộp; JSON nguồn không bị sửa.
- Train mask **`tiny_unet`** từ đầu, 1000 steps, batch 32, width 16, LR 0.001; ROI native pad vào canvas 64×64, không tải ResNet/ImageNet.

Thêm `--dry-run` để chỉ in lệnh, không đọc dữ liệu hoặc train. `--stage det`/`--stage seg` chạy riêng một phần. `--det-dataset` và `--seg-dataset` mặc định lần lượt là `dataset/sub_dataset/det` và `dataset/sub_dataset/seg`. Script dùng lại split đã xuất, và không tự hợp nhất hai dataset det/seg. Không chạy `fit` trên hai bộ riêng.

Output phải mới hoặc không có output stage tương ứng. Nếu chạy riêng một stage sau đó, có thể dùng cùng thư mục experiment khi stage đó chưa tồn tại. Checkpoint nằm tại `runs/experiment_s_p2_tiny_v1/detector/best.pt` và `runs/experiment_s_p2_tiny_v1/seg/mask/best.pt`.

MLflow cập nhật `val/*` và `best/*` ngay sau mỗi validation, chờ server ghi nhận rồi mới train tiếp. Mặc định experiment val det mỗi 250 steps, val seg mỗi 50 steps; loss theo step vẫn gửi bất đồng bộ. Cập nhật code không thay đổi tiến trình train đang chạy, chỉ áp dụng cho lần khởi động train tiếp theo.

## Giữ chi tiết đối tượng 20–30 px

`native` cắt ROI ở kích thước gốc rồi pad vào giữa canvas; RGB, bbox prior và mask giữ nguyên từng pixel, không được phóng/thu thành 20 px. Với padding 0.25, bbox 20–30 px thường cần ROI khoảng 30–46 px, phù hợp canvas 64. Nếu ROI lớn hơn canvas, script báo kích thước cần thiết để bạn tăng `--roi-size`; không tự downsample. Kích thước phải đủ cho **mọi** ROI trong bộ train/val.

Tiny U-Net mới có nhánh detail stride 1 và đưa cả input RGB/prior gốc trực tiếp vào decoder. Nhánh context chỉ pooling một lần. Giữ đường truyền chi tiết là đặc tính kiến trúc; chất lượng mask vẫn cần đo trên nhãn đã kiểm tra. Kiến trúc `tiny` cũ và `resnet18_unet` vẫn được hỗ trợ cho checkpoint cũ.

Augmentation native dùng dịch chuyển nguyên pixel và flip thay cho scale/warp làm mờ biên. Checkpoint lưu `roi_resize_mode=native` và width. PyTorch inference, export manifest và ONNX viewer dùng chung quy tắc pad/unpad; đường kính vẫn ở tọa độ ảnh gốc.

## Xuất thêm 500 ROI để gán/sửa mask

Lấy các instance đã có bbox từ **split train của det**. Chỉ người dùng chạy lệnh sau; mask xuất ra là bản nháp cần sửa/duyệt trong CVAT:

```bash
.venv-train/bin/python scripts/export_seg_coco.py \
  --annotations dataset/sub_dataset/det/train/annotations/instances_default.json \
  --images dataset/sub_dataset/det/train/images \
  --output dataset/sub_dataset/seg_review_500 \
  --mode roi --count 500 --require-count --seed 43 \
  --single-class --class-name ball \
  --exclude-reviewed reports/seg_review_100/annotations/instances_default.json \
  --zip
```

`--exclude-reviewed` nhận một hoặc nhiều COCO export của các bộ đã làm, khớp bằng đường dẫn ảnh nguồn + bbox nguồn thay vì ID có thể bị CVAT đánh lại. Dùng bản export ROI ban đầu có `source_file_name`/`source_bbox`; metadata `input_file_name` của split cũng được hỗ trợ. Nếu bộ cũ không còn mapping gốc thì không thể chứng minh không trùng chỉ bằng ID. `report.json` ghi số đã loại. Có thể bỏ tùy chọn này để chọn 500 mẫu khi không cần loại bộ cũ. `--require-count` báo lỗi trước khi đọc ảnh nếu không đủ 500 mẫu sau khi loại.

Có thể thêm `--mask-checkpoint runs/experiment_s_p2_tiny_v1/seg/mask/best.pt` sau khi đã có Tiny checkpoint để tạo nháp từ model, thay vì pseudo-mask. Mask có sẵn luôn được ưu tiên giữ. Mọi mẫu vẫn có `mask_review_status=pending`; nếu tạo mask thất bại thì giữ bbox placeholder và lý do trong `review.csv` để vẽ lại.

Kết quả gồm `images/default`, `annotations/instances_default.json`, `review.csv`, `report.json` và ZIP `dataset/sub_dataset/seg_review_500.zip`. Crop giữ pixel gốc. Đây là 500 instance, không nhất thiết từ 500 ảnh gốc khác nhau. Không ghép các mask pending vào train tự động; sau khi sửa, xuất COCO cùng ảnh rồi split seg riêng với mapping nguồn tương ứng.

## App thử inference

Agent UI bổ sung checkbox **Boxes / Contours / Circles / Diameter** để ẩn/hiện từng loại overlay. Từng dòng instance có **Show** (toàn bộ overlay của instance) và **Box** (riêng bbox), kèm **Show all / Hide all**. C1/C2 giữ nguyên số khi ẩn instance, không đánh số lại.

Đổi hiển thị dùng ảnh cache, không infer lại hoặc nạp lại ảnh, giữ zoom/pan và khung Original. Trạng thái instance được nhớ theo ảnh, xóa khi infer lại ảnh đó hoặc đổi settings. Save overlay dùng trạng thái đang chọn; CSV vẫn giữ toàn bộ phép đo và có thêm `instance_visible`, `box_visible`. Controls này phục vụ kiểm tra, không thay đổi kết quả inference.

Khi muốn kiểm tra ONNX của experiment sau khi train xong:

```bash
.venv-train/bin/python scripts/export_onnx_bundle.py \
  --detector-dir runs/experiment_s_p2_tiny_v1/detector \
  --mask-checkpoint runs/experiment_s_p2_tiny_v1/seg/mask/best.pt \
  --output runs/experiment_s_p2_tiny_v1/export
```

Manifest nằm tại `runs/experiment_s_p2_tiny_v1/export/onnx/manifest.json`. Không có bước deploy tự động. Khi cập nhật app ở máy test, rsync cả `onnx_viewer/` và `detrseg/roi.py` như hướng dẫn README để nhận đủ helpers native mới.
