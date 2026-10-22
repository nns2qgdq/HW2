# Flood Prediction Dataset 資料稽核

來源：[Kaggle — Flood Prediction Dataset](https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset)

## 檔案檢查

| 檔案 | 資料列 | 欄位 | 用途 | 檢查結果 |
|---|---:|---:|---|---|
| `train.csv` | 1,117,957 | 22 | 有標籤訓練資料 | 20 個數值特徵、`FloodProbability`、`id`；無缺值、無重複列 |
| `test.csv` | 745,305 | 21 | Kaggle 競賽提交資料 | 有 20 個特徵及 `id`，沒有目標值；不能用來計算模型指標 |
| `sample_submission.csv` | 745,305 | 2 | 提交格式範例 | 只有 `id`、`FloodProbability`；目標值全為 0.5，屬佔位值 |
| `flood.csv` | 50,000 | 21 | 額外有標籤資料 | 有 20 個特徵及目標；無缺值、無重複列，無 `id` |

模型目標為連續數值 `FloodProbability`，因此本專案採用多元線性迴歸。`id` 只用來識別資料列及檢視排序，不納入特徵。`train.csv` 的目標範圍為 0.285–0.725，平均 0.50448；`flood.csv` 的目標範圍相同，平均 0.49966。

## 二十個候選特徵

| 特徵 | 欄位所表達的概念（依資料欄位名稱與發布者描述） |
|---|---|
| `MonsoonIntensity` | 季風強度 |
| `TopographyDrainage` | 地形排水能力 |
| `RiverManagement` | 河川管理 |
| `Deforestation` | 森林砍伐程度 |
| `Urbanization` | 都市化程度 |
| `ClimateChange` | 氣候變遷影響指標 |
| `DamsQuality` | 水壩品質 |
| `Siltation` | 河道／水庫淤積 |
| `AgriculturalPractices` | 農業方式指標 |
| `Encroachments` | 河川或洪氾區侵占程度 |
| `IneffectiveDisasterPreparedness` | 災害準備不足指標 |
| `DrainageSystems` | 排水系統指標 |
| `CoastalVulnerability` | 沿海脆弱度 |
| `Landslides` | 土石流／崩塌風險指標 |
| `Watersheds` | 集水區指標 |
| `DeterioratingInfrastructure` | 基礎設施劣化程度 |
| `PopulationScore` | 人口暴露程度指標 |
| `WetlandLoss` | 濕地流失程度 |
| `InadequatePlanning` | 規劃不足指標 |
| `PoliticalFactors` | 治理／政治因素指標 |

## 使用及解讀限制

- 主分析使用 `train.csv`，以固定亂數種子切出訓練資料與最後保留測試集；特徵選擇和模型選擇只用訓練資料。
- `test.csv` 沒有標籤，只能產生提交預測，不能用於效能評估。
- `flood.csv` 與 `train.csv` 的 50,000 筆資料中沒有完全相同的特徵加目標列，但其特徵觀察範圍略有不同（`flood.csv` 有特徵值到 22，`train.csv` 到 19），且它沒有 ID。為避免混合來源／分佈，主分析不將它併入訓練資料。
- Kaggle 資料適合課堂迴歸練習；現有外部研究指出其地理涵蓋與蒐集來源無法獨立確認。結果應解讀為對該資料中風險分數的預測，不應宣稱已驗證實際洪水事件。
