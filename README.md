# HW2_v2｜洪水風險分數多元線性迴歸

本專案以 Kaggle 的 Flood Prediction Dataset 練習多元線性迴歸，預測連續目標 `FloodProbability`，並比較特徵選擇與線性模型。交付內容包含 Python 分析程式、可重現結果、CRISP-DM 報告，以及含兩種分析案例和自訂 CSV 功能的互動網頁。

- 🔗 **儲存庫網址**：<https://github.com/nns2qgdq/HW2>
- 🌐 **Live Demo Page**：<https://nns2qgdq.github.io/HW2/>

![Regression Lab 線上展示預覽](assets/preview.png)

> **資料解讀界線**：資料來自 Kaggle 的 Flood Prediction Dataset／Playground Series，適合課程建模練習，但沒有可核實的地理座標、時間戳記及實際洪水事件標籤。模型是在預測資料集提供的連續分數，不是實地洪水預警，也不能視為已校準的洪水發生機率。

## 專案需求與資料來源

作業要求從 Kaggle 選擇 10–20 個特徵的資料，使用線性回歸，完成特徵選擇、模型評估、含區間的預測圖，並依 CRISP-DM 說明分析流程。本專案使用 20 個數值特徵，符合特徵數範圍。

- 資料集：[Kaggle — Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)
- 研究脈絡：[Kaggle Playground Series S4E5](https://www.kaggle.com/competitions/playground-series-s4e5/overview)
- 預測目標：連續數值 `FloodProbability`
- 模型特徵：20 個數值欄位；完整欄位名稱與概念說明見 [`data/DATASET_AUDIT.md`](data/DATASET_AUDIT.md)
- 識別欄：`id` 僅供資料列識別及排序檢視，不作為模型特徵

下載資料集並將以下四個原始 CSV 放入 `data/`：

| 檔案 | 用途與注意事項 |
|---|---|
| `train.csv` | 有標籤資料；主分析唯一使用的資料檔。檢查結果為 1,117,957 列、20 個特徵、目標及 `id`，必要欄位沒有缺值。 |
| `test.csv` | Kaggle 無標籤資料；因沒有 `FloodProbability`，不用於計算模型指標。 |
| `sample_submission.csv` | 提交格式範例；目標值為 0.5 的佔位值，不是模型預測結果。 |
| `flood.csv` | 額外的 50,000 筆有標籤資料；因其特徵範圍不同且無 `id`，不與 `train.csv` 混合。 |

原始 CSV 由 `.gitignore` 排除，不會隨 Git 提交；首次執行前需自行從資料集取得。四份檔案的欄位、筆數、範圍及使用判斷記錄在 [`data/DATASET_AUDIT.md`](data/DATASET_AUDIT.md)。

## 環境設定

Flood 主分析使用 Python 3.13 以上版本及 `uv` 管理套件。於專案根目錄執行：

```bash
uv sync
uv run python flood_regression_hw2.py
```

`uv sync` 依 [`pyproject.toml`](pyproject.toml) 和 [`uv.lock`](uv.lock) 安裝鎖定版本的相依套件。主分析需要 `data/train.csv`；若檔案不存在，程式會指出缺少的檔案位置。執行完成後，分析表格、摘要、診斷資料及圖表寫入 `results/flood/`。

如需使用傳統 Python/pip，依 [`requirements.txt`](requirements.txt) 安裝環境後執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python flood_regression_hw2.py
```

舊有 Startup Python 分析另可用 `uv run python 50_startups_regression.py` 執行；網頁使用已保存的分析結果，不需要先重跑該程式。

## 分析設計

1. **資料切分**：使用 seed 42 切出 80% 訓練端和 20% 最終保留測試集。模型與特徵選擇在訓練端進行；保留測試集只用於選擇完成後的最後評估。
2. **特徵選擇**：從訓練端抽取最多 40,000 列建立候選特徵組合，並使用另一組最多 12,000 列的訓練資料，以 5-fold CV 評分。比較 SelectKBest、RFE、Sequential Forward Selection、Lasso、Permutation Importance，以及完整 20 特徵基準。精簡方法設定 14 個特徵。
3. **模型比較**：在選出的特徵上比較 Linear Regression、Ridge（`alpha=1`）及 Lasso（`alpha=0.0005`），以訓練端 5-fold CV 的 RMSE 及 R² 評估。選擇最低 CV RMSE；若 Linear Regression 與最低分差距在 0.1% 內，優先採用較簡單的線性模型。
4. **最後評估**：選定 Linear Regression 後，以保留測試集計算 MAE、MSE、RMSE、R² 及 Adjusted R²。額外產生 OLS 預測區間與迴歸假設診斷。EFS 全子集合窮舉未執行，因其組合數會隨特徵數指數增加。

> 特徵選擇結果中的 Lasso 在本資料未刪除特徵，因此保留完整 20 特徵；最後的特徵數不是強制縮到 14。交叉驗證比較結果也不是最終測試集分數，請勿將兩者混為一談。

## 目前主要結果

以下為目前 `results/flood/` 中固定 seed 42 的結果。重新執行程式後，應以產生的 `best_model_summary.json` 和 CSV 為準。

| 最終模型 | 特徵數 | 測試集 MAE | 測試集 MSE | 測試集 RMSE | 測試集 R² | Adjusted R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 20 | 0.01579 | 0.0004032 | 0.02008 | 0.84488 | 0.84486 |

模型比較時，Ridge 的 CV RMSE 約為 0.02031017，Linear Regression 約為 0.02031018，差距小於預先設定的 0.1% 門檻，因此依簡約原則選擇 Linear Regression。以上測試集分數只描述此資料切分下的預測表現。

假設診斷中，RESET、Breusch–Pagan 與 Jarque–Bera 有顯著訊號，需留意可能的非線性、異質變異及殘差常態性偏離；大樣本下常態性檢定尤其敏感，應搭配診斷圖解讀。最大 VIF 約 1.013，沒有明顯共線性訊號。Durbin–Watson 約 1.979 是按資料列 ID 排序計算；ID 不是時間變數，不能據此宣稱時間序列誤差獨立。

## 分析輸出

程式會建立或更新 `results/flood/`：

| 輸出檔 | 說明 |
|---|---|
| `best_model_summary.json` | 最終模型、選定特徵、切分大小及保留測試指標。 |
| `feature_selection_comparison.csv` | 六種特徵集合的特徵數、欄位及 CV RMSE／R²。 |
| `model_comparison.csv` | Linear Regression、Ridge、Lasso 的 CV 比較。 |
| `holdout_predictions.csv` | 保留測試資料的實際值與預測值；部分固定樣本另含區間界限。 |
| `interactive_diagnostics_sample.csv` | 固定抽樣 5,000 列的 OLS 預測、實際值、信賴／預測區間及殘差，供網頁繪圖。 |
| `regression_assumptions.csv` | RESET、Breusch–Pagan、Jarque–Bera、Durbin–Watson、VIF 摘要。 |
| `vif_by_feature.csv` | 個別特徵的 VIF。 |
| `run_metadata.json` | 資料來源、切分、抽樣、交叉驗證設定及摘要指標。 |
| `01_target_and_correlations.png` | 目標分佈及特徵與目標的相關圖。 |
| `02_prediction_intervals.png` | 保留測試樣本的 OLS 預測、實際值及 95% 個別觀測預測區間。 |
| `03_regression_diagnostics.png` | 殘差對預測值、Q–Q、尺度位置及殘差分布圖。 |

## 互動網頁

網頁是靜態 HTML/CSS/JavaScript，不需要 Python 後端。

- 🌐 **線上展示（Live Demo）**：<https://nns2qgdq.github.io/HW2/>
- **本機啟動方式**：於專案根目錄啟動本機伺服器：

```bash
python3 -m http.server 8765
```

在瀏覽器開啟 <http://localhost:8765/>。請透過 HTTP 開啟，不要直接以 `file://` 開啟，否則瀏覽器可能阻擋結果檔載入。

### 內建案例

- **洪水機率預測**：查看預先計算的 Flood 指標和特徵選擇表；互動圖可切換 CV RMSE／R²、點選方法查看欄位、調整顯示樣本數、切換平均值信賴區間與個別觀測預測區間，並查看殘差診斷。
- **新創公司利潤**：比較既有 Startup 分析的五種方法；可切換 Nested CV／保留測試集 RMSE 與 R²，並點選方法查看入選欄位。底部連結可開啟依特徵數探索各方法的完整舊版互動實驗。

兩個案例的圖表均由既有 CSV 結果即時繪製，不是將分析圖直接貼成主頁圖片。

### 自訂 CSV（加分功能）

在「載入自己的 CSV」區選擇本機檔案，或貼上公開 CSV 網址。載入後選擇連續數值目標及特徵，再執行瀏覽器端的線性迴歸分析。網址來源必須允許 CORS；本機 CSV 不需上傳到伺服器。頁面會提供缺值與欄位錯誤訊息；大於 40,000 列的資料會以固定亂數種子抽樣，以維持互動效能。

自訂資料功能適合快速探索；其簡化的 train/test 評估流程不等同 Flood 主程式的訓練端特徵選擇、CV 與獨立保留測試設計。

## 報告與作業交付狀態

- CRISP-DM 報告 PDF 未納入 Git；`report/build_report.py` 與來源資料可用於產生本機報告。NotebookLM 摘要已納入報告來源；GPT pdfCrowd 對話匯出尚未收到，取得後需整合並重新檢查最終版。
- [`report/source/NotebookLM_summary.md`](report/source/NotebookLM_summary.md)：已收到的 NotebookLM 摘要來源檔。
- GPT pdfCrowd 匯出檔尚未提供；放入 `report/source/` 後需重新編入 PDF。附件命名及狀態見 [`report/source/README.md`](report/source/README.md)。
- 儲存庫網址（GitHub Repository）：[nns2qgdq/HW2](https://github.com/nns2qgdq/HW2)
- Live Demo Page：[https://nns2qgdq.github.io/HW2/](https://nns2qgdq.github.io/HW2/)

## 專案結構

```text
HW2_v2/
├── README.md                         # 本文件：資料、分析、執行及交付說明
├── pyproject.toml / uv.lock          # Flood 分析的環境與鎖定套件
├── flood_regression_hw2.py           # 本次 Flood 多元線性迴歸主程式
├── 50_startups_regression.py          # 既有 Startup 分析程式
├── 50_Startups.csv                   # 既有 Startup 資料
├── assets/                           # 專案預覽圖與展示靜態資源
├── data/
│   ├── DATASET_AUDIT.md              # Flood CSV 稽核與欄位摘要
│   └── *.csv                         # 下載後的原始資料；由 Git 忽略
├── results/
│   ├── flood/                        # Flood 表格、摘要、預測與圖表
│   └── *.csv, *.png                  # 既有 Startup 成果
├── report/
│   ├── build_report.py               # PDF 產生程式
│   └── source/                       # NotebookLM 與待補附件
├── index.html, styles.css, app.js    # 新版雙案例互動網頁
└── startup/                          # Startup 依特徵數探索的舊版互動頁
```

`requirements.txt` 保留傳統 pip 安裝方式；Flood 主程式的建議環境以 `pyproject.toml`／`uv.lock` 為準。資料 CSV 不會自動下載，請先放入 `data/` 再執行主程式。
