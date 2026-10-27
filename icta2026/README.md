# icta2026/ — mã gốc của bài ICTA2026

Ba tệp này là mã gốc của bài báo, **giữ nguyên, không chỉnh sửa**; các thực nghiệm bổ sung nằm ở `../supplementary/`.

| Tệp | Vai trò |
|---|---|
| `sr_plr_marl_v8_mc_mr.py` | Engine: mạng thương mại đơn/đa hàng hóa, toán tử F, PLR, SR-PLR-MARL, Fixed-LR, Poly-decay, PG, EG, phép chiếu cài đặt, các chỉ số phục hồi |
| `sr_plr_marl_v8_multiinstance.py` | Độ bền theo thể hiện toán tử (nhiều `param_seed`) |
| `sr_plr_supply_cut_runner.py` | Kịch bản cắt giảm cung |

Chạy thử: `python sr_plr_marl_v8_multiinstance.py --smoke`
