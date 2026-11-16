# Lotto649

一套大樂透候選組合研究工具。**統計先產生並排序候選組合，易卦微調同一統計分段內的順序，最後輸出前 N 組。**

## 快速開始

從 repo 根目錄執行，需 Python 3.10 以上。選號只需 NumPy 和 SciPy。

```powershell
python -m pip install "numpy>=1.26,<3" "scipy>=1.12,<2"
python -m lotto7x7 select --date 2026-11-17 --out results/selection_20261117.json
```

`--date` 指定目標開獎日。指令只使用該日期**以前**的開獎紀錄。省略 `--out` 會將 JSON 印到終端。輸出包含歷史截止日期、輸入指紋、當日卦象、每組候選的統計分數與分項、統計分段及易卦分數。

## 主架構

| 階段 | 職責 | 程式 |
| --- | --- | --- |
| 統計篩選 | 從固定種子產生不重複的候選池。依精確 6/49 總和分布、單號頻率與 7×7 形狀類別計算補償分數。 | [statistics.py](lotto7x7/selection/statistics.py) |
| 易卦微調 | 依目標日期的節氣、農曆日推導六爻，以候選組合六爻的相符數做段內排序。 | [hexagram.py](lotto7x7/selection/hexagram.py) |
| 結果輸出 | 依統計分數分段，保留跨段順序，輸出前 N 組。 | [select()](lotto7x7/selection/pipeline.py) |

設定集中在 [config/selection.json](config/selection.json)。預設候選池 2,000 組、輸出 4 組、三項統計權重相同。每段以尚未分段的最高統計分數為起點，與它的差距在 `epsilon`（預設 0.01）內的候選歸為同段。段內依易卦分數、統計分數、號碼順序排序。三項統計分數各在 −1 到 1 之間。

易卦資料來自 repo 保存的中央氣象署年度 PDF 衍生曆法，涵蓋 **2023–2026**。以台灣時間中午取當日有效節氣。24 節氣從冬至起每三節氣映射為一個上卦區段，農曆日以 30 天刻度映射為八個下卦區段。上下卦採 Gray code 編碼。候選號碼按小到大由下往上排，奇數為陽、偶數為陰。這是一套可重現的研究映射。超出曆法資料範圍時指令會明確報錯。

## 資料與研究紀錄

- [開獎 BIN 與來源清單](data/bin)提供開獎歷史。[年度 CSV](data/csv)是可查閱的匯出資料。
- [精確形狀參考表](data/reference/shape_class_counts.json)涵蓋全部 13,983,816 組。它與精確總和分布提供統計補償基準。
- [目錄、測試與重跑方法](docs/repository_layout.md)列出研究工具。舊指令可用 `python -m lotto7x7 research --help` 查詢，[早期影像研究](docs/archive/image_research.md)與[研究證據索引](docs/research_evidence_index.md)保留原始實驗脈絡。

補償分數描述候選組合對歷史分布缺口的方向與幅度。整套排序策略的樣本外預測能力仍待驗證。
