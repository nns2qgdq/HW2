# FloodProbability 多元線性迴歸研究摘要 (NotebookLM)

### 研究背景與資料集性質

本研究依據 CRISP-DM 流程，針對 Kaggle 發布者 Naiya Khalid 之 Flood Prediction Dataset（Playground Series S4E5）開展多元線性迴歸分析，旨在預測 20 個數值風險特徵與連續目標變數 `FloodProbability` 之間的關係[1]。該資料集屬於合成表格資料，缺乏可核實的地理座標與時間戳記[1][2]。必須明確區分的是：本模型僅用於預測 Kaggle 資料內部的 `FloodProbability` 基準分數，並不代表真實洪水事件的實際預警能力，亦不可將風險分數視為已校準的真實事件發生機率[1][2]。

### 特徵選擇與模型比較

研究於訓練集（894,365 列）進行 5-fold 交叉驗證（CV）。在特徵選擇方面，比較 SelectKBest、RFE、SFS 與 Permutation Importance 等方法後，精簡至 14 個特徵之 CV RMSE 約介於 0.03395 至 0.03412（CV R² 約 0.54934–0.55422）；相較於保留全 20 個特徵（CV RMSE 約 0.02031，CV R² 約 0.84046），精簡特徵會造成顯著預測資訊損失，故最終保留全部 20 個特徵[2]。模型比較方面，OLS 與 Ridge（`alpha=1`）兩者效能幾乎相同（CV RMSE 均約 0.020310），Lasso（`alpha=0.0005`）表現略遜（CV RMSE 約 0.020463）；依據簡約原則選用 OLS 多元線性迴歸[2]。

### 評估結果與迴歸診斷

最終於獨立保留測試集（223,592 列）評估，模型表現為 MAE 約 0.01579、RMSE 約 0.02008、R² 約 0.84488 及 Adjusted R² 約 0.84486[2]。 迴歸診斷顯示，Ramsey RESET 檢定（統計量 21.6, p &lt; 0.05）與 Breusch–Pagan 檢定（統計量 211, p &lt; 0.05）均達顯著，提示模型可能存在非線性關係與異質變異問題[2]；Jarque–Bera 檢定（統計量 751, p &lt; 0.05）在大樣本下極為敏感，須搭配 Q-Q 圖解讀殘差常態性[2]。Durbin–Watson 統計量接近 2（約 1.98），僅表示未見明顯的一階自相關訊號；由於資料列順序（`id`）並非真實時間順序，不可據此推論時間上的誤差獨立性[1][2]。

### 外部比較與結論

外部討論區雖有集成方法達到 R² 約 0.8695，但因採用不同競賽切分與特徵工程，不可當成同條件下的直接比較[2][3]。本研究選用多元線性迴歸，在保留模型係數可解讀性的前提下完成擬合與評估；惟受限於非線性與異質變異，預測區間之實際涵蓋率可能偏離標稱 95%[2]。

---

### 參考來源

1. **DATASET\_AUDIT.md** — *Flood Prediction Dataset 資料稽核與特徵說明*
2. **HW2\_report.pdf** — *Flood Probability 多元線性迴歸分析報告 (HW2\_v2)*
3. [Kaggle Discussion — Ensemble solution](https://www.kaggle.com/competitions/playground-series-s4e5/discussion/506793) — 外部集成方法的競賽結果（不同切分與特徵工程，僅作背景比較）
