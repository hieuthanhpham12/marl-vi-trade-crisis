# MARL–VI cho cân bằng mạng thương mại đa hàng hóa dưới khủng hoảng

Mã nguồn và dữ liệu thực nghiệm của đề tài cấp Học viện **"Giải bài toán bất đẳng thức biến phân cân bằng mạng thương mại đa hàng hóa bằng học tăng cường đa tác nhân dưới điều kiện khủng hoảng"**
(mã đề tài 09-2026-HV-CB1, Học viện Công nghệ Bưu chính Viễn thông – PTIT, 2026).
Chủ trì: TS. Phạm Thanh Hiếu (Khoa Cơ bản 1). Thành viên chính: TS. Nguyễn Kiều Linh (Khoa Trí tuệ nhân tạo).

> **English summary.** Code for *SR-PLR-MARL*, a shock-responsive, proximity-based step-size scheduler for projected stochastic variational-inequality (VI) dynamics on multi-commodity, multi-route trade-network equilibrium under crisis shocks (price spikes, route blockage, supply cuts). The repo contains (i) the original ICTA2026 code, (ii) supplementary experiments built on top of it (noise, missing data, repeated shocks, drifting prices, tight capacity with exact projection), and (iii) a limited MAPPO/MADDPG comparison on small networks (self-written numpy implementations).

## Sản phẩm khoa học liên quan

* **Sản phẩm chính (dự kiến):** Pham, T.H., Nguyen, K.L., *Shock-responsive proximity-based step-size scheduling for projected stochastic variational inequality equilibrium computation in trade networks*, ICTA 2026, Springer — đã được Ban tổ chức thông báo chấp nhận đăng; chưa xuất bản chính thức (camera-ready, DOI, lập chỉ mục Scopus: chờ xác nhận).
* **Kết quả đính kèm:** Pham, T.H., *Proximity-based adaptive step sizes for multi-agent reinforcement learning in variational inequality problems with rotational dynamics*, NSA2026, tr. 43–53. Mã của bài này **không** nằm trong repo (xem "Lưu ý").

## Cấu trúc

```
icta2026/            Mã gốc của bài ICTA2026 (không sửa): toán tử F, các bộ điều phối (PLR, SR-PLR-MARL, Fixed-LR,
                     Poly-decay), PG, EG, phép chiếu cài đặt; kèm hai script chạy đa thể hiện và cắt giảm cung.
supplementary/       Mở rộng trình chạy trên mã gốc (ext.py): nhiễu cập nhật/quan sát, dữ liệu thiếu, cú sốc lặp,
                     biến động giá, phép chiếu Euclide chính xác (Dykstra), nghiệm tham chiếu (SLSQP), tiêu chí G50.
  results/           Kết quả thô (JSON): res_A … res_G, thr_base.
marl_baselines/      MAPPO, MADDPG tự viết bằng numpy (npnn.py, algos_np.py), môi trường đa tác nhân (rlenv.py),
                     huấn luyện/triển khai (rlrun.py), dò siêu tham số (run_tune.py), chạy chính (run_main.py).
  results/           res_tune.json, res_main_none.json.
  (quick_check.py, repro_icta_tables.py: kiểm tra nhanh kết quả ICTA2026)
notebooks/           Ba notebook (đã chạy sẵn, Colab-ready).
tests/               Test nhanh (~3 giây).
```

## Cài đặt

```bash
git clone https://github.com/hieuthanhpham12/marl-vi-trade-crisis.git
cd marl-vi-trade-crisis
pip install -r requirements.txt          # numpy, scipy, pandas, matplotlib (+ jupyter cho notebook)
python tests/test_smoke.py               # kiểm tra nhanh: tái hiện RT=240 của bảng ICTA2026 trên mạng 2x2
```

Đã thử với Python 3.13, numpy 2.x, scipy 1.x, pandas 3.x. Mã gốc ICTA2026 có nhánh dự phòng cho `np.trapz` (numpy < 2).

## Notebook (mở trực tiếp trên Colab)

| Notebook | Nội dung | Thời gian |
|---|---|---|
| `01_reproduce_icta2026_price_shock.ipynb` | Tái hiện cú sốc giá ×2 trên mạng 2×2, 3×3 với 6 phương pháp, đồ thị $G_\lambda$ | vài phút (`SEEDS=5`) |
| `02_supplementary_noise_missing_capacity.ipynb` | Đọc kết quả đã lưu: nhiễu quan sát, dữ liệu thiếu, năng lực chặt | vài giây |
| `03_mappo_maddpg_small_networks.ipynb` | Bảng MAPPO/MADDPG đã lưu + huấn luyện lại một tác nhân MAPPO | ~30–60 giây |

## Chạy lại thực nghiệm

```bash
# Thực nghiệm bổ sung (ghi vào supplementary/results/)
cd supplementary
python run_A.py                   # cú sốc giá, 6 mạng x 6 phương pháp x 30 hạt giống (tái hiện bảng ICTA2026)
python run_rest.py B1             # nhiễu cập nhật;  B2: nhiễu quan sát;  C: dữ liệu thiếu;  D: cú sốc lặp/dồn dập
python run_rest.py E              # biến động giá;    F: cắt bỏ trên 5 mạng;  G: năng lực chặt (chậm: phép chiếu chính xác)
python thr_base.py                # ngưỡng tuyệt đối G<=1, G<=0.5 cho PG/EG/Fixed-LR/...
python summ.py results/res_B2.json

# MAPPO/MADDPG (ghi vào marl_baselines/results/)
cd ../marl_baselines
python run_tune.py                # lưới siêu tham số 2x2 trên mạng 3x3, hạt giống 100,101
python run_main.py none 100000    # 3 mạng nhỏ x 2 thuật toán x 10 hạt giống, 1e5 bước huấn luyện (~50 phút, 2 tiến trình)
```

Các script dùng `multiprocessing.Pool(2)`; chỉnh `procs` trong `drv.run_jobs` hoặc `Pool(...)` nếu máy có nhiều lõi.

Đối chiếu với báo cáo tổng kết: Mục 4.2–4.6 ↔ bảng bài ICTA2026 (`run_A.py`, `icta2026/`); Mục 4.7 ↔ `supplementary/`; Mục 4.8 ↔ `marl_baselines/`.

## Lưu ý trung thực về kết quả

* Toàn bộ mạng là **dữ liệu tổng hợp**, chưa hiệu chuẩn bằng dữ liệu thương mại thực (ví dụ BACI).
* Theo tiêu chí phục hồi không phụ thuộc $D_{\max}$ ($G_{50}$ hoặc $\rho_e$), **Fixed-LR và PG/EG phục hồi nhanh hơn SR-PLR-MARL**; ưu thế của SR-PLR-MARL là so với PLR gốc và Poly-decay, cùng bảo đảm hội tụ hầu chắc chắn sau cú sốc cuối cùng (giả thiết (A5) và phép chiếu chính xác).
* Nhiễu quan sát lớn gây báo động giả (vi phạm (A5) theo nghĩa thực nghiệm); không có lợi ích với biến động giá liên tục; khi ràng buộc năng lực chặt, $G^{\rm impl}$ **không** là chứng chỉ nghiệm (chỉ phép chiếu chính xác đưa PG/EG về nghiệm thật).
* MAPPO/MADDPG: cài đặt tự viết bằng numpy (chưa đối chiếu với Stable-Baselines/RLlib), chỉ 3 mạng nhỏ ($d\le27$), 10 hạt giống, huấn luyện không có cú sốc; không tái hiện nhận định "MAPPO/MADDPG không hội tụ" của De Natale và cs.
* Các lần chạy phép chiếu chính xác dùng 8 hạt giống; các thực nghiệm bổ sung không kèm khoảng tin cậy hay kiểm định thống kê.
* `supplementary/ext.py` là phần mở rộng trình chạy do chủ trì đề tài viết **sau** bài báo; không thuộc bài ICTA2026.
* Mã của bài NSA2026 không được đưa vào repo vì script hiện có chưa tái hiện được số liệu đã đăng (tập chuyển pha, độ gần cân bằng cuối); cần xác nhận mã đã sinh số liệu trước khi công bố.

## Trích dẫn

Xem `CITATION.cff`. Giấy phép: MIT (`LICENSE`).
