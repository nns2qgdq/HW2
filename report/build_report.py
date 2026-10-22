from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

root=Path(__file__).resolve().parents[1]
out=root/'report/HW2_report.pdf'
summary=json.loads((root/'results/flood/best_model_summary.json').read_text())
model=pd.read_csv(root/'results/flood/model_comparison.csv')
sel=pd.read_csv(root/'results/flood/feature_selection_comparison.csv')
ass=pd.read_csv(root/'results/flood/regression_assumptions.csv')
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
pdfmetrics.registerFontFamily('STSong-Light', normal='STSong-Light', bold='STSong-Light', italic='STSong-Light', boldItalic='STSong-Light')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleTC',fontName='STSong-Light',fontSize=23,leading=31,alignment=TA_CENTER,textColor=colors.HexColor('#12304a'),spaceAfter=8))
styles.add(ParagraphStyle(name='SubTC',fontName='STSong-Light',fontSize=10,leading=16,alignment=TA_CENTER,textColor=colors.HexColor('#516579'),spaceAfter=15))
styles.add(ParagraphStyle(name='H1TC',fontName='STSong-Light',fontSize=16,leading=22,textColor=colors.HexColor('#12304a'),spaceBefore=5,spaceAfter=9))
styles.add(ParagraphStyle(name='H2TC',fontName='STSong-Light',fontSize=11,leading=16,textColor=colors.HexColor('#15616d'),spaceBefore=6,spaceAfter=4))
styles.add(ParagraphStyle(name='BodyTC',fontName='STSong-Light',fontSize=9.3,leading=15,spaceAfter=6,wordWrap='CJK'))
styles.add(ParagraphStyle(name='SmallTC',fontName='STSong-Light',fontSize=8,leading=12,spaceAfter=4,wordWrap='CJK',textColor=colors.HexColor('#425466')))
styles.add(ParagraphStyle(name='WarningTC',fontName='STSong-Light',fontSize=9.5,leading=15,textColor=colors.HexColor('#8a3b12'),backColor=colors.HexColor('#fff1db'),borderPadding=8,spaceBefore=5,spaceAfter=9,wordWrap='CJK'))
P=lambda s,style='BodyTC':Paragraph(s,styles[style])
def page(canvas,doc):
    canvas.saveState(); w,h=A4
    canvas.setStrokeColor(colors.HexColor('#d9e2ea'));canvas.line(18*mm,15*mm,w-18*mm,15*mm)
    canvas.setFont('STSong-Light',7.5);canvas.setFillColor(colors.HexColor('#607080'))
    canvas.drawString(18*mm,10*mm,'HW2_v2｜洪水風險分數多元線性迴歸')
    canvas.drawRightString(w-18*mm,10*mm,f'{doc.page}')
    canvas.restoreState()
def table(rows,widths=None):
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('FONTNAME',(0,0),(-1,0),'STSong-Light'),('FONTSIZE',(0,0),(-1,-1),8),('LEADING',(0,0),(-1,-1),11),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f5')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#12304a')),('GRID',(0,0),(-1,-1),.35,colors.HexColor('#d1dbe4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t
story=[]
story += [Spacer(1,24*mm),P('Flood Probability<br/>多元線性迴歸分析','TitleTC'),P('HW2_v2｜CRISP-DM 流程與模型評估','SubTC'),P('資料來源：Kaggle Flood Prediction Dataset｜主目標：FloodProbability｜20 個候選特徵','SubTC'),Spacer(1,7*mm),P('報告狀態：草稿。NotebookLM 研究摘要已納入附錄；尚待加入 GPT pdfCrowd 對話匯出後重新輸出最終版。','WarningTC'),P('本報告使用 Kaggle 教學／競賽型風險分數資料，評估的是對該資料標籤的預測能力，不代表真實洪水事件的預警效能。','BodyTC'),Spacer(1,15*mm),P('執行摘要','H1TC'),P(f"固定 seed 42 的 80/20 切分中，最終模型為 {summary['Model']}（{summary['Number of Features']} 個特徵）。保留測試集 RMSE={summary['Test RMSE']:.5f}、R²={summary['Test R2']:.5f}。模型具有可觀的資料內預測力，但殘差診斷提示線性形式與等變異假設並不完全成立，因此應保留解讀限制。",'BodyTC'),PageBreak()]
# business/data
story += [P('1. Business Understanding｜商業理解','H1TC'),P('本作業的目標是用多個數值風險指標，預測資料所提供的連續 FloodProbability 分數，藉此完整練習多元線性迴歸、特徵選擇、模型評估與部署呈現。假想使用情境是依指標排序並檢視分數差異；本分析不做因果推論，也不輸出具地理位置和時間資訊的實際洪水警報。','BodyTC'),P('成功標準是：使用訓練資料完成特徵／模型選擇，以從未參與選擇的保留測試集報告 MAE、MSE、RMSE、R²、Adjusted R²，並呈現預測區間和診斷。','BodyTC'),P('2. Data Understanding｜資料理解','H1TC'),P('資料取自 Kaggle 發布者 Naiya Khalid 的 Flood Prediction Dataset。Kaggle Playground Series S4E5 官方說明其競賽資料由真實世界資料生成，屬合成 tabular data；因此這裡的 target 是 benchmark 分數，並非可直接驗證的實際事件紀錄。主檔 train.csv 有 1,117,957 列、20 個數值候選特徵、連續目標 FloodProbability 及識別欄 id；本地檢查未發現缺值或重複列。目標平均約 0.5045，標準差約 0.0510，範圍 0.285–0.725。完整四檔稽核及欄位說明見 data/DATASET_AUDIT.md。','BodyTC'),P('資料有四個 CSV：train.csv 有標籤；test.csv 缺目標，故不拿來算分；sample_submission.csv 的 0.5 是佔位預測；flood.csv 有 50,000 筆標籤列，與主訓練檔無完全相同列，但特徵值範圍較寬，因此不混入主資料。資料欄位是風險概念指標，沒有可核實的地理、時間與事件標籤脈絡；指標不應外推至現實防災決策。','BodyTC'),P('目標分數及特徵分佈圖','H2TC'),Image(str(root/'results/flood/01_target_and_correlations.png'),width=174*mm,height=61*mm),P('圖 1. FloodProbability 分布與各特徵和目標的 Pearson 相關。相關僅作描述，不代表因果關係。','SmallTC'),PageBreak()]
# preparation modeling
story += [P('3. Data Preparation｜資料準備','H1TC'),P('從主訓練檔排除 id，因它是列識別碼而非有意義的預測因子；其餘 20 欄及目標皆為數值型，且無缺值，故不需補值或類別編碼。以 random_state=42 分割 80% 訓練端（894,365 列）與 20% 最終保留測試端（223,592 列）。','BodyTC'),P('特徵選擇樣本 40,000 列及模型比較樣本 12,000 列皆由訓練端抽樣。模型比較採訓練端 5-fold CV；測試端直到選擇完成後才用於一次最終評估。區間與殘差診斷使用 statsmodels OLS。此流程避免讓保留測試結果回饋模型選擇。','BodyTC'),P('4. Modeling｜建模','H1TC'),P('特徵選擇比較 SelectKBest（f_regression）、RFE、Sequential Forward Selection、標準化後 Lasso 與 permutation importance。設定 14 特徵供精簡方案比較；Lasso 在本資料的係數未能刪除特徵，因此也比較完整 20 特徵基準。EFS 未執行，因 20 個特徵的子集合窮舉成本呈指數成長。','BodyTC'),P('以相同特徵集比較 Linear Regression、Ridge（alpha=1）及 Lasso（alpha=0.0005），交叉驗證為 5 folds。選擇規則是最低 CV RMSE；若 Linear Regression 落在最低值 0.1% 以內，優先選較簡單的 OLS。Ridge 與 OLS 的 CV RMSE 幾乎相同，因此選 OLS。','BodyTC'),P('交叉驗證比較','H2TC')]
rows=[[P(x,'SmallTC') for x in ['模型','CV RMSE','CV R²']]]
for _,r in model.iterrows(): rows.append([P(str(r['Model']),'SmallTC'),P(f"{r['CV RMSE']:.6f}",'SmallTC'),P(f"{r['CV R2']:.6f}",'SmallTC')])
story += [table(rows,[77*mm,45*mm,45*mm]),Spacer(1,3*mm),P('特徵選擇比較','H2TC')]
rows=[[P(x,'SmallTC') for x in ['方法','特徵數','CV RMSE','CV R²']]]
for _,r in sel.iterrows(): rows.append([P(str(r['Method']),'SmallTC'),P(str(int(r['Number of Features'])),'SmallTC'),P(f"{r['CV RMSE']:.5f}",'SmallTC'),P(f"{r['CV R2']:.5f}",'SmallTC')])
story += [table(rows,[76*mm,27*mm,34*mm,30*mm]),Spacer(1,5*mm),P('14 特徵方法 CV RMSE 約 0.03395；完整 20 特徵方案約 0.02031。這表示在目前資料與比較程序中，縮減至 14 特徵損失明顯預測資訊。','SmallTC'),PageBreak()]
# evaluation
story += [P('5. Evaluation｜評估','H1TC'),P('保留測試集最終指標','H2TC')]
rows=[[P(x,'SmallTC') for x in ['MAE','MSE','RMSE','R²','Adjusted R²']],[P(f"{summary['Test MAE']:.5f}",'SmallTC'),P(f"{summary['Test MSE']:.7f}",'SmallTC'),P(f"{summary['Test RMSE']:.5f}",'SmallTC'),P(f"{summary['Test R2']:.5f}",'SmallTC'),P(f"{summary['Test Adjusted R2']:.5f}",'SmallTC')]]
story += [table(rows,[31*mm]*5),Spacer(1,4*mm),Image(str(root/'results/flood/02_prediction_intervals.png'),width=174*mm,height=74*mm),P('圖 2. 保留測試資料中依預測值排序的樣本；藍帶為 OLS 95% prediction interval，紅點為實際值。Prediction interval 描述個別新觀測的不確定性，通常比平均值 confidence interval 寬。','SmallTC'),P('假設診斷','H2TC')]
rows=[[P(x,'SmallTC') for x in ['檢查','統計量','p 值／摘要']]]
for _,r in ass.iterrows():
    pv='—' if pd.isna(r['p_value']) else f"{r['p_value']:.2g}"
    rows.append([P(str(r['Assumption']),'SmallTC'),P(f"{r['Statistic']:.3g}",'SmallTC'),P(pv+'；'+str(r['Interpretation']),'SmallTC')])
story += [table(rows,[56*mm,24*mm,91*mm]),Spacer(1,4*mm),Image(str(root/'results/flood/03_regression_diagnostics.png'),width=174*mm,height=74*mm),P('RESET 與 Breusch–Pagan 檢定顯著，提示可能非線性與異質變異；Jarque–Bera 在大樣本下對偏離非常敏感，須搭配 Q-Q 圖解讀。VIF 最大約 1.013，未見顯著共線性。Durbin–Watson 約 1.979；id 順序只是資料列順序代理，不能據此主張時間獨立性。線性迴歸的預測力良好不代表所有推論假設均成立。','SmallTC'),PageBreak()]
# deployment and limitations, refs, appendices pending
story += [P('6. Deployment｜部署','H1TC'),P('互動網頁以靜態 HTML、CSS、JavaScript 提供兩個內建 problem 切換：Flood Probability 與專案既有的 Startup Profit。頁面顯示已產生的 Flood 指標、特徵選擇比較及圖表；另可匯入本機 CSV 或公開 URL，選擇連續目標及預測特徵，執行瀏覽器端 80/20 評估。URL 資料須允許 CORS；自訂資料結果為快速探索用途，不等同本報告採用的訓練端 CV 與獨立保留集程序。','BodyTC'),P('網路解法比較與研究脈絡','H2TC'),P('Kaggle Playground Series S4E5「Regression with a Flood Prediction Dataset」的公開討論可見集成方案回報 R² 約 0.8695。這項分數來自不同競賽切分、提交與特徵工程，不能直接當成本作業同一測試集上的公平對照；它僅說明競賽中有人採用更複雜的集成解法。本作業以線性模型為主，是為符合課程目標並保留係數可解讀性。','BodyTC'),P('限制及後續改進','H2TC'),P('資料來源沒有可驗證的事件、地理及時間欄位；資料標籤為連續風險分數，不能等同事件機率校準。顯著 RESET 與異質變異提示可在延伸研究比較非線性項、穩健標準誤或樹模型，但若做此比較，必須只在訓練端調參並保留測試端。模型區間也依賴指定模型形式，假設違反時涵蓋率可能偏離標稱 95%。','BodyTC'),P('參考來源','H2TC'),P('1. Kaggle，Flood Prediction Dataset，https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset<br/>2. Kaggle，Regression with a Flood Prediction Dataset（Playground Series S4E5），https://www.kaggle.com/competitions/playground-series-s4e5/overview<br/>3. Kaggle Discussion，ensemble solution 報告 R² 約 0.8695，https://www.kaggle.com/competitions/playground-series-s4e5/discussion/506793<br/>4. Kaggle，Playground Series S4E5 說明（合成資料背景），https://www.kaggle.com/competitions/playground-series-s4e5<br/>5. statsmodels，PredictionResults：conf_int 支援預測與信賴區間，https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.html','SmallTC'),P('必交附件狀態','H2TC'),P('NotebookLM 研究摘要已納入下一頁附錄，來源檔為 report/source/NotebookLM_summary.md。GPT 對話過程（pdfCrowd 匯出）仍待提供原始匯出檔；加入後須重新輸出最終 PDF。','WarningTC'),P('本地來源檔：flood_regression_hw2.py、results/flood/、data/DATASET_AUDIT.md；重現指令與互動網頁說明見 README.md。','SmallTC'),PageBreak(),P('附錄 A｜NotebookLM 研究摘要','H1TC'),P('資料背景與適用界線','H2TC'),P('NotebookLM 摘要指出，FloodProbability 是 Kaggle Playground S4E5 的合成表格基準資料，包含 20 個數值風險特徵，沒有可核實的地理座標與時間戳記。因此模型評估的是對資料內連續分數的擬合能力，不能解讀成真實洪水事件的預警效能，也不能視為已校準的事件發生機率。','BodyTC'),P('特徵選擇與模型比較','H2TC'),P('使用訓練資料的 5-fold CV 比較特徵子集：14 特徵方案 RMSE 約 0.03395–0.03412、R² 約 0.549–0.554；完整 20 特徵方案 RMSE 約 0.02031、R² 約 0.84046，故保留 20 個特徵。OLS 與 Ridge 的 CV RMSE 幾乎一致（約 0.020310），Lasso 略高（約 0.020463），依簡約原則選 OLS。','BodyTC'),P('測試集表現與診斷','H2TC'),P('獨立保留測試集有 223,592 列，MAE 約 0.01579、RMSE 約 0.02008、R² 約 0.84488、Adjusted R² 約 0.84486。RESET 與 Breusch–Pagan 檢定提示可能有非線性與異質變異；Jarque–Bera 在大樣本下敏感，應搭配 Q-Q 圖解讀。Durbin–Watson 約 1.98 只表示未見明顯一階自相關訊號；資料 ID 不是時間順序，不能用來推論時間誤差獨立。','BodyTC'),P('外部比較','H2TC'),P('Kaggle 討論中的集成方法回報 R² 約 0.8695，但使用不同切分和特徵工程，不能作為同條件直接比較。線性模型符合本作業的練習目的，也保留係數可解讀性；模型假設偏離時，預測區間涵蓋率仍可能不同於標稱 95%。','BodyTC'),P('來源：report/source/NotebookLM_summary.md；對應資料稽核、模型結果及 Kaggle 討論來源見本文參考資料。','SmallTC')]
doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=20*mm,title='Flood Probability Multiple Linear Regression - HW2',author='HW2_v2')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(out)
