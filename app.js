const $ = id => document.getElementById(id);
const FLOOD_FEATURES = [
  'MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation','Urbanization',
  'ClimateChange','DamsQuality','Siltation','AgriculturalPractices','Encroachments',
  'IneffectiveDisasterPreparedness','DrainageSystems','CoastalVulnerability','Landslides',
  'Watersheds','DeterioratingInfrastructure','PopulationScore','WetlandLoss',
  'InadequatePlanning','PoliticalFactors'
];
let loadedDataset = null;
let featureSelectionRows = [];
let intervalRows = [];
let startupRows = [];
let selectedFeatureMethod = 0;
let selectedStartupMethod = 0;
let problemRequestToken = 0;

function parseCsv(text) {
  const rows=[]; let row=[], cell='', quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(quoted){
      if(c==='"'&&text[i+1]==='"'){cell+='"';i++;}
      else if(c==='"') quoted=false;
      else cell+=c;
    } else if(c==='"') quoted=true;
    else if(c===','){row.push(cell.trim());cell='';}
    else if(c==='\n'||c==='\r'){
      if(c==='\r'&&text[i+1]==='\n')i++;
      row.push(cell.trim());if(row.some(v=>v!==''))rows.push(row);row=[];cell='';
    } else cell+=c;
  }
  row.push(cell.trim());if(row.some(v=>v!==''))rows.push(row);
  if(rows.length<2)throw Error('CSV 沒有資料列。');
  const headers=rows.shift().map((v,i)=>i===0?v.replace(/^\uFEFF/,''):v);
  if(headers.some(v=>!v))throw Error('CSV 欄位名稱不能空白。');
  if(new Set(headers).size!==headers.length)throw Error('CSV 欄位名稱有重複，請先重新命名。');
  const records=rows.map((cells,i)=>{
    if(cells.length!==headers.length)throw Error(`第 ${i+2} 列欄位數與標題不一致。`);
    return Object.fromEntries(headers.map((h,j)=>[h,cells[j]]));
  });
  return {headers,records};
}
function numericColumns(data){
  return data.headers.filter(h=>{
    const values=data.records.map(r=>r[h]).filter(v=>v!=='');
    return values.length>1&&values.every(v=>Number.isFinite(Number(v)));
  });
}
function showProblemInfo(items){$('problem-info').innerHTML=items.map(x=>`<span class="info-pill">${escapeHtml(x)}</span>`).join('');}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function formatNumber(v,d=4){return Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});}
function setMetrics(container,items){container.innerHTML=items.map(([label,value])=>`<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join('');}
function parseSimpleCsv(text){return parseCsv(text);}
function svgText(x,y,text,cls='axis-label',anchor='middle'){return `<text class="${cls}" x="${x}" y="${y}" text-anchor="${anchor}">${escapeHtml(text)}</text>`;}
function renderStartupChart(metric=$('startup-chart').dataset.metric||'Nested CV RMSE'){
  const svg=$('startup-chart');if(!svg||!startupRows.length)return;
  svg.dataset.metric=metric;
  const lowerBetter=metric.includes('RMSE'),values=startupRows.map(r=>Number(r[metric])),dataLow=Math.min(...values),dataHigh=Math.max(...values);
  const low=lowerBetter?0:Math.min(0,dataLow),high=dataHigh*1.08,span=(high-low)||Math.max(Math.abs(high)*.08,.0001);
  const W=900,H=330,left=260,right=95,top=48,bottom=38,sx=v=>left+((lowerBetter?high-v:v-low)/span)*(W-left-right);
  const bestIndex=lowerBetter?values.indexOf(dataLow):values.indexOf(dataHigh);
  let out=`<title>Startup ${metric} 方法比較</title>`;
  for(let t=0;t<=4;t++){const value=lowerBetter?high-span*t/4:low+span*t/4,x=sx(value);out+=`<line class="grid-line" x1="${x}" y1="${top-12}" x2="${x}" y2="${H-bottom}"/>`+svgText(x,H-11,formatNumber(value,metric.includes('RMSE')?0:3),'axis-label');}
  startupRows.forEach((r,i)=>{const y=top+i*48,value=Number(r[metric]),x=sx(value);out+=`<g class="selection-row ${i===selectedStartupMethod?'selected':''} ${i===bestIndex?'best':''}" data-index="${i}" role="button" tabindex="0" aria-label="${escapeHtml(r.Method)}，${r['Number of Features']} 個特徵，${metric} ${formatNumber(value,metric.includes('RMSE')?2:4)}"><title>點選查看 ${escapeHtml(r.Method)} 的特徵與表現</title><text class="row-label" x="${left-12}" y="${y+4}" text-anchor="end">${escapeHtml(r.Method)}</text><line class="bar-track" x1="${left}" y1="${y}" x2="${W-right}" y2="${y}"/><line class="selection-bar" x1="${left}" y1="${y}" x2="${x}" y2="${y}"/><circle class="selection-point" cx="${x}" cy="${y}" r="7"/><text class="bar-value" x="${Math.min(x+13,W-right)}" y="${y+4}">${formatNumber(value,metric.includes('RMSE')?1:4)}</text></g>`;});
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;
  svg.querySelectorAll('[data-index]').forEach(g=>{const select=()=>selectStartupMethod(Number(g.dataset.index),metric);g.addEventListener('click',select);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});});
}
function selectStartupMethod(index,metric=$('startup-chart').dataset.metric||'Nested CV RMSE'){
  selectedStartupMethod=index;const r=startupRows[index];if(!r)return;
  const features=String(r['Selected Features']).match(/'((?:\\'|[^'])*)'/g)?.map(v=>v.slice(1,-1).replaceAll("\\'","'"))||[];
  $('startup-detail').innerHTML=`<strong>${escapeHtml(r.Method)}</strong><span>${r['Number of Features']} 個特徵</span><span>Nested CV RMSE ${formatNumber(r['Nested CV RMSE'],2)}</span><span>Nested CV R² ${formatNumber(r['Nested CV R2'],4)}</span><span>Test RMSE ${formatNumber(r['Test RMSE'],2)}</span><span>Test R² ${formatNumber(r['Test R2'],4)}</span><span class="selection-features">欄位：${escapeHtml(features.join('、'))}</span>`;
  renderStartupChart(metric);
}
function selectFeatureMethod(index,metric=$('selection-chart').dataset.metric||'CV RMSE'){
  selectedFeatureMethod=index;
  const r=featureSelectionRows[index];
  if(!r)return;
  $('selection-detail').innerHTML=`<strong>${escapeHtml(r.Method)}</strong><span>${r['Number of Features']} 個特徵</span><span>CV RMSE ${formatNumber(r['CV RMSE'],6)}</span><span>CV R² ${formatNumber(r['CV R2'],6)}</span><span class="selection-features">欄位：${escapeHtml(r['Selected Features'])}</span>`;
  renderSelectionChart(metric);
}
function renderSelectionChart(metric=$('selection-chart').dataset.metric||'CV RMSE'){
  const svg=$('selection-chart');if(!svg||!featureSelectionRows.length)return;
  svg.dataset.metric=metric;
  const W=900,H=Math.max(300,featureSelectionRows.length*43+88),left=245,right=60,top=50,bottom=38;
  const vals=featureSelectionRows.map(r=>Number(r[metric]));const dataLow=Math.min(...vals),dataHigh=Math.max(...vals),low=metric==='CV RMSE'?0:Math.min(0,dataLow),high=dataHigh*1.08,span=(high-low)||Math.max(Math.abs(high)*.08,.0001);
  const sx=v=>left+((metric==='CV RMSE'?(high-v):(v-low))/span)*(W-left-right);
  const bestIndex=metric==='CV RMSE'?vals.indexOf(dataLow):vals.indexOf(dataHigh);
  let out=`<title>${metric==='CV RMSE'?'交叉驗證均方根誤差':'交叉驗證決定係數'}比較</title>`;
  for(let t=0;t<=4;t++){const v=metric==='CV RMSE'?high-span*t/4:low+span*t/4,x=sx(v);out+=`<line class="grid-line" x1="${x}" y1="${top-14}" x2="${x}" y2="${H-bottom}"/>`+svgText(x,H-11,formatNumber(v,metric==='CV RMSE'?4:3),'axis-label');}
  featureSelectionRows.forEach((r,i)=>{
    const y=top+i*43, v=Number(r[metric]), x=sx(v), active=i===selectedFeatureMethod, width=Math.max(2,x-left);
    out+=`<g class="selection-row ${active?'selected':''} ${i===bestIndex?'best':''}" data-index="${i}" role="button" tabindex="0" aria-label="${escapeHtml(r.Method)}，${r['Number of Features']} 個特徵，${metric} ${formatNumber(v,6)}"><title>點選查看 ${escapeHtml(r.Method)} 的特徵與分數</title><text class="row-label" x="${left-12}" y="${y+4}" text-anchor="end">${escapeHtml(r.Method)}</text><line class="bar-track" x1="${left}" y1="${y}" x2="${W-right}" y2="${y}"/><line class="selection-bar" x1="${left}" y1="${y}" x2="${x}" y2="${y}"/><circle class="selection-point" cx="${x}" cy="${y}" r="7"/><text class="bar-value" x="${Math.min(x+13,W-right)}" y="${y+4}">${formatNumber(v,5)}</text></g>`;
  });
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;
  svg.querySelectorAll('[data-index]').forEach(g=>{g.addEventListener('click',()=>selectFeatureMethod(Number(g.dataset.index),metric));g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectFeatureMethod(Number(g.dataset.index),metric);}});});
}
function renderIntervalChart(){
  const svg=$('interval-chart');if(!svg||!intervalRows.length)return;
  const type=$('interval-type').value,count=Number($('interval-count').value),lowKey=type==='obs'?'obs_ci_lower':'mean_ci_lower',highKey=type==='obs'?'obs_ci_upper':'mean_ci_upper';
  $('interval-count-value').value=String(count);$('interval-legend-label').textContent=type==='obs'?'95% 預測區間':'95% 平均值信賴區間';
  const sorted=[...intervalRows].sort((a,b)=>Number(a.ols_mean)-Number(b.ols_mean));const stride=Math.max(1,Math.ceil(sorted.length/count));let shown=sorted.filter((_,i)=>i%stride===0).slice(0,count);if(shown.length>count)shown=shown.slice(0,count);
  const W=900,H=355,left=64,right=23,top=24,bottom=52;const allY=sorted.flatMap(r=>[Number(r.actual),Number(r.ols_mean),Number(r[lowKey]),Number(r[highKey])]);let min=Math.min(...allY),max=Math.max(...allY),pad=(max-min)*.08||.01;min-=pad;max+=pad;
  const x=i=>left+i*(W-left-right)/Math.max(1,shown.length-1),y=v=>top+(max-v)*(H-top-bottom)/(max-min);
  let out='<title>保留測試集 OLS 預測、實際值及信賴區間</title>';
  for(let t=0;t<=4;t++){const v=min+(max-min)*t/4,yy=y(v);out+=`<line class="grid-line" x1="${left}" y1="${yy}" x2="${W-right}" y2="${yy}"/>`+svgText(left-10,yy+4,formatNumber(v,3),'axis-label','end');}
  const upper=shown.map((r,i)=>`${x(i)},${y(Number(r[highKey]))}`).join(' '),lower=shown.map((r,i)=>`${x(i)},${y(Number(r[lowKey]))}`).reverse().join(' ');
  out+=`<polygon class="interval-band" points="${upper} ${lower}"/>`;
  out+=`<polyline class="prediction-line" points="${shown.map((r,i)=>`${x(i)},${y(Number(r.ols_mean))}`).join(' ')}"/>`;
  shown.forEach((r,i)=>{const cx=x(i),cy=y(Number(r.actual)),tip=`ID ${r.id}｜實際 ${formatNumber(r.actual,5)}｜OLS ${formatNumber(r.ols_mean,5)}｜區間 ${formatNumber(r[lowKey],5)}–${formatNumber(r[highKey],5)}`;out+=`<circle class="actual-point" cx="${cx}" cy="${cy}" r="3.3" tabindex="0"><title>${escapeHtml(tip)}</title></circle>`;});
  out+=`<text class="axis-title vertical-label" x="16" y="${(top+H-bottom)/2}" transform="rotate(-90 16 ${(top+H-bottom)/2})">FloodProbability</text>`;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;
  svg.querySelectorAll('.actual-point').forEach((pt,i)=>{const r=shown[i];const detail=()=>{$('interval-detail').textContent=`ID ${r.id}｜實際值 ${formatNumber(r.actual,5)}｜OLS 預測 ${formatNumber(r.ols_mean,5)}｜${type==='obs'?'預測區間':'平均值信賴區間'} ${formatNumber(r[lowKey],5)} 至 ${formatNumber(r[highKey],5)}`;};pt.addEventListener('pointerenter',detail);pt.addEventListener('focus',detail);});
}
function renderDiagnosticChart(){
  const svg=$('diagnostic-chart');if(!svg||!intervalRows.length)return;const type=$('diagnostic-type').value,W=900,H=330,left=65,right=25,top=24,bottom=50;
  const fitted=intervalRows.map(r=>Number(r.ols_mean)),res=intervalRows.map(r=>Number(r.residual));let xs,ys,xLabel,yLabel;
  if(type==='scale'){xs=fitted;ys=res.map(v=>Math.sqrt(Math.abs(v)));xLabel='OLS 預測值';yLabel='√|殘差|';}
  else if(type==='histogram'){const n=32,min=Math.min(...res),max=Math.max(...res),step=(max-min)/n||1,bins=Array(n).fill(0);res.forEach(v=>bins[Math.min(n-1,Math.floor((v-min)/step))]++);const peak=Math.max(...bins);let out='<title>殘差直方圖</title>';for(let i=0;i<n;i++){const bw=(W-left-right)/n-1,bh=(H-top-bottom)*bins[i]/peak,xx=left+i*(W-left-right)/n,yy=H-bottom-bh;out+=`<rect class="hist-bar" x="${xx}" y="${yy}" width="${bw}" height="${bh}"><title>殘差區間 ${formatNumber(min+i*step,4)}–${formatNumber(min+(i+1)*step,4)}：${bins[i]} 筆</title></rect>`;}for(let t=0;t<=4;t++){const xx=left+t*(W-left-right)/4, count=Math.round(peak*t/4);out+=svgText(xx,H-12,formatNumber(min+(max-min)*t/4,4),'axis-label')+`<line class="grid-line" x1="${left}" y1="${H-bottom-(H-top-bottom)*t/4}" x2="${W-right}" y2="${H-bottom-(H-top-bottom)*t/4}"/>`+svgText(left-9,H-bottom-(H-top-bottom)*t/4+4,String(count),'axis-label','end');}out+=`<text class="axis-title vertical-label" x="16" y="${(top+H-bottom)/2}" transform="rotate(-90 16 ${(top+H-bottom)/2})">筆數</text>`;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;$('diagnostic-detail').textContent=`殘差平均 ${formatNumber(res.reduce((a,b)=>a+b,0)/res.length,6)}；樣本數 ${res.length.toLocaleString()}。`;return;}
  else {xs=fitted;ys=res;xLabel='OLS 預測值';yLabel='殘差';}
  let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(type==='residuals'){const span=Math.max(Math.abs(ymin),Math.abs(ymax));ymin=-span*1.1;ymax=span*1.1;}const xp=(x)=>left+(x-xmin)*(W-left-right)/(xmax-xmin||1),yp=(y)=>top+(ymax-y)*(H-top-bottom)/(ymax-ymin||1);let out=`<title>${escapeHtml(yLabel)}對OLS預測值</title>`;
  for(let t=0;t<=4;t++){const xv=xmin+(xmax-xmin)*t/4,yv=ymin+(ymax-ymin)*t/4;out+=`<line class="grid-line" x1="${xp(xv)}" y1="${top}" x2="${xp(xv)}" y2="${H-bottom}"/><line class="grid-line" x1="${left}" y1="${yp(yv)}" x2="${W-right}" y2="${yp(yv)}"/>`+svgText(xp(xv),H-12,formatNumber(xv,3),'axis-label')+svgText(left-9,yp(yv)+4,formatNumber(yv,4),'axis-label','end');}
  if(type==='residuals')out+=`<line class="zero-line" x1="${left}" y1="${yp(0)}" x2="${W-right}" y2="${yp(0)}"/>`;
  xs.forEach((v,i)=>{const yy=ys[i];out+=`<circle class="diagnostic-point" cx="${xp(v)}" cy="${yp(yy)}" r="2.5" data-index="${i}"><title>預測 ${formatNumber(v,5)}｜${yLabel} ${formatNumber(yy,5)}</title></circle>`;});out+=`<text class="axis-title vertical-label" x="16" y="${(top+H-bottom)/2}" transform="rotate(-90 16 ${(top+H-bottom)/2})">${escapeHtml(yLabel)}</text>`;svg.setAttribute('viewBox',`0 0 ${W} ${H}`);svg.innerHTML=out;svg.querySelectorAll('.diagnostic-point').forEach((point,i)=>point.addEventListener('pointerenter',()=>{$('diagnostic-detail').textContent=`OLS 預測 ${formatNumber(fitted[i],5)}；${yLabel} ${formatNumber(ys[i],5)}。`}));$('diagnostic-detail').textContent=`固定抽樣 ${intervalRows.length.toLocaleString()} 筆保留測試資料；移至點上可查看個別觀測值。`;
}
function renderFloodCharts(){renderSelectionChart();if(!$('selection-detail').innerHTML)selectFeatureMethod(selectedFeatureMethod);renderIntervalChart();renderDiagnosticChart();if(!$('startup-charts').hidden)renderStartupChart();}

async function showFlood(requestToken){
  $('problem-error').hidden=true;$('precomputed-results').hidden=true;$('flood-charts').hidden=true;$('startup-charts').hidden=true;
  showProblemInfo(['Kaggle Flood Prediction Dataset','20 個候選特徵','連續目標：FloodProbability','訓練資料：1,117,957 筆']);
  try{
    const [summary,selection,models,diagnostics]=await Promise.all([
      fetch('results/flood/best_model_summary.json').then(checkResponse).then(r=>r.json()),
      fetch('results/flood/feature_selection_comparison.csv').then(checkResponse).then(r=>r.text()).then(parseSimpleCsv),
      fetch('results/flood/model_comparison.csv').then(checkResponse).then(r=>r.text()).then(parseSimpleCsv),
      fetch('results/flood/interactive_diagnostics_sample.csv').then(checkResponse).then(r=>r.text()).then(parseSimpleCsv)
    ]);
    if(requestToken!==problemRequestToken)return;
    setMetrics($('metric-cards'),[
      ['Selected model',summary.Model],['Selected features',`${summary['Number of Features']} / 20`],
      ['Holdout RMSE',formatNumber(summary['Test RMSE'],5)],['Holdout R²',formatNumber(summary['Test R2'],4)]
    ]);
    $('comparison-note').textContent='獨立驗證樣本上的 5-fold CV；測試集僅作最後評估';
    $('comparison-head').innerHTML='<tr><th>特徵選擇方法</th><th>特徵數</th><th>CV RMSE</th><th>CV R²</th></tr>';
    $('comparison-body').innerHTML=selection.records.map(r=>`<tr><td>${escapeHtml(r.Method)}</td><td>${r['Number of Features']}</td><td>${formatNumber(r['CV RMSE'],5)}</td><td>${formatNumber(r['CV R2'],4)}</td></tr>`).join('');
    featureSelectionRows=selection.records;intervalRows=diagnostics.records;
    $('selection-chart').dataset.metric='CV RMSE';selectedFeatureMethod=featureSelectionRows.findIndex(r=>r.Method==='Lasso');if(selectedFeatureMethod<0)selectedFeatureMethod=0;
    $('selection-detail').innerHTML='';
    $('interval-detail').textContent='將游標移至樣本點檢視明細。';
    $('method-note').textContent=`模型比較：${models.records.map(r=>`${r.Model} CV RMSE ${formatNumber(r['CV RMSE'],5)}`).join('；')}。預測區間與殘差診斷使用固定抽樣的保留測試資料。`;
    $('precomputed-results').hidden=false;$('startup-charts').hidden=true;$('flood-charts').hidden=false;
    renderFloodCharts();
  }catch(e){if(requestToken!==problemRequestToken)return;$('problem-error').textContent=`載入或繪製預先計算結果時發生錯誤：${e.message}。請確認結果檔存在且格式正確。`;$('problem-error').hidden=false;}
}
async function showStartup(requestToken){
  $('problem-error').hidden=true;$('precomputed-results').hidden=true;$('flood-charts').hidden=true;$('startup-charts').hidden=true;
  showProblemInfo(['Startup legacy example','5 個模型特徵','連續目標：Profit','資料列：50 筆']);
  try{
    const file=await fetch('results/final_comparison.csv').then(checkResponse).then(r=>r.text());
    if(requestToken!==problemRequestToken)return;const table=parseSimpleCsv(file);startupRows=table.records;const best=[...table.records].sort((a,b)=>Number(a['Nested CV RMSE'])-Number(b['Nested CV RMSE']))[0];
    setMetrics($('metric-cards'),[
      ['Best nested-CV method',best.Method],['Selected features',best['Number of Features']],
      ['Nested CV RMSE',formatNumber(best['Nested CV RMSE'],2)],['Test R²',formatNumber(best['Test R2'],4)]
    ]);
    $('comparison-note').textContent='原 Startup 專案結果；訓練集 nested CV 與保留測試集';
    $('comparison-head').innerHTML='<tr><th>方法</th><th>特徵數</th><th>Nested CV RMSE</th><th>Nested CV R²</th><th>Test RMSE</th></tr>';
    $('comparison-body').innerHTML=table.records.map(r=>`<tr><td>${escapeHtml(r.Method)}</td><td>${r['Number of Features']}</td><td>${formatNumber(r['Nested CV RMSE'],2)}</td><td>${formatNumber(r['Nested CV R2'],4)}</td><td>${formatNumber(r['Test RMSE'],2)}</td></tr>`).join('');
    $('startup-chart').dataset.metric='Nested CV RMSE';selectedStartupMethod=table.records.findIndex(r=>r.Method===best.Method);$('startup-detail').innerHTML='';
    document.querySelectorAll('[data-startup-metric]').forEach(button=>{const active=button.dataset.startupMetric==='Nested CV RMSE';button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
    $('method-note').textContent='互動圖比較既有 Startup 分析的五種方法；Nested CV 用於方法選擇，保留測試集指標僅供最終評估。';
    $('flood-charts').hidden=true;$('startup-charts').hidden=false;$('precomputed-results').hidden=false;
    renderStartupChart();selectStartupMethod(selectedStartupMethod);
  }catch(e){if(requestToken!==problemRequestToken)return;$('problem-error').textContent=`無法讀取 Startup 成果：${e.message}`;$('problem-error').hidden=false;}
}
function checkResponse(response){if(!response.ok)throw Error(`HTTP ${response.status}`);return response;}
function showProblem(){const requestToken=++problemRequestToken;return $('problem-select').value==='flood'?showFlood(requestToken):showStartup(requestToken);}

function populateSetup(data,label){
  loadedDataset=data;
  const nums=numericColumns(data);
  if(nums.length<2)throw Error('至少需兩個數值欄位，才能選一個目標及一個數值特徵。');
  $('dataset-meta').textContent=`${label}：${data.records.length.toLocaleString()} 筆資料，${data.headers.length} 個欄位。選擇連續數值目標與特徵欄位。`;
  $('target-select').innerHTML=nums.map(c=>`<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
  const targetGuess=nums.find(c=>/target|price|profit|saleprice|floodprobability|value|y$/i.test(c));
  if(targetGuess)$('target-select').value=targetGuess;
  renderFeatureList();$('custom-setup').hidden=false;$('custom-results').hidden=true;$('custom-error').hidden=true;
}
function renderFeatureList(){
  if(!loadedDataset)return;
  const target=$('target-select').value;
  const usable=loadedDataset.headers.filter(h=>h!==target&&h!==''&&loadedDataset.records.some(r=>r[h]!==''));
  $('feature-list').innerHTML=usable.map(h=>{
    const values=new Set(loadedDataset.records.slice(0,10000).map(r=>r[h]).filter(v=>v!==''));
    const numeric=loadedDataset.records.slice(0,10000).every(r=>r[h]===''||Number.isFinite(Number(r[h])));
    const allowed=numeric||values.size<=30;
    const checked=allowed&&h.toLowerCase()!=='id';
    const suffix=numeric?'數值':`類別 ${values.size} 種`;
    return `<label class="feature-option"><input type="checkbox" value="${escapeHtml(h)}" ${checked?'checked':''} ${allowed?'':'disabled'}><span>${escapeHtml(h)} <small>${suffix}${allowed?'':'（類別太多）'}</small></span></label>`;
  }).join('');
  updateFeatureCount();
}
function updateFeatureCount(){const count=$('feature-list').querySelectorAll('input:checked').length;$('feature-count').textContent=`(${count} 個)`;}
function seededRandom(seed){let t=seed>>>0;return ()=>{t+=0x6D2B79F5;let x=t;x=Math.imul(x^(x>>>15),x|1);x^=x+Math.imul(x^(x>>>7),x|61);return ((x^(x>>>14))>>>0)/4294967296;};}
function solveLinear(a,b){
  const n=b.length,m=a.map((row,i)=>[...row,b[i]]);
  for(let col=0;col<n;col++){
    let pivot=col;for(let r=col+1;r<n;r++)if(Math.abs(m[r][col])>Math.abs(m[pivot][col]))pivot=r;
    if(Math.abs(m[pivot][col])<1e-10)throw Error('選取特徵存在完全共線或常數欄，請移除後重試。');
    [m[col],m[pivot]]=[m[pivot],m[col]];const d=m[col][col];for(let j=col;j<=n;j++)m[col][j]/=d;
    for(let r=0;r<n;r++){if(r===col)continue;const f=m[r][col];for(let j=col;j<=n;j++)m[r][j]-=f*m[col][j];}
  }
  return m.map(row=>row[n]);
}
function fitModel(rows,features,target){
  const valid=rows.filter(r=>Number.isFinite(Number(r[target]))&&features.every(f=>r[f]!==''));
  if(valid.length<30)throw Error('移除缺值後需至少 30 筆完整資料。');
  const order=Array.from({length:valid.length},(_,i)=>i),rng=seededRandom(42);
  for(let i=order.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[order[i],order[j]]=[order[j],order[i]];}
  const testN=Math.max(1,Math.ceil(valid.length*.2)),testIds=order.slice(0,testN),trainIds=order.slice(testN);
  const trainRows=trainIds.map(i=>valid[i]);
  const numeric=features.filter(f=>trainRows.every(r=>r[f]===''||Number.isFinite(Number(r[f]))));
  const categoricals=features.filter(f=>!numeric.includes(f)),categoryLevels={};
  for(const f of categoricals)categoryLevels[f]=[...new Set(trainRows.map(r=>r[f]).filter(v=>v!==''))].sort().slice(1);
  const expanded=[...numeric,...categoricals.flatMap(f=>categoryLevels[f].map(v=>`${f}=${v}`))];
  if(!expanded.length||expanded.length>50)throw Error('轉換後特徵數需介於 1 到 50。');
  const encode=r=>[...numeric.map(f=>Number(r[f])),...categoricals.flatMap(f=>categoryLevels[f].map(v=>r[f]===v?1:0))];
  const matrix=valid.map(encode),y=valid.map(r=>Number(r[target]));
  const trainMatrix=trainIds.map(i=>matrix[i]);
  const means=expanded.map((_,j)=>trainMatrix.reduce((s,row)=>s+row[j],0)/trainMatrix.length);
  const scales=expanded.map((_,j)=>Math.sqrt(trainMatrix.reduce((s,row)=>s+(row[j]-means[j])**2,0)/trainMatrix.length)||1);
  const z=matrix.map(row=>[1,...row.map((v,j)=>(v-means[j])/scales[j])]);
  const dim=expanded.length+1,xtx=Array.from({length:dim},()=>Array(dim).fill(0)),xty=Array(dim).fill(0);
  for(const i of trainIds){for(let a=0;a<dim;a++){xty[a]+=z[i][a]*y[i];for(let b=0;b<dim;b++)xtx[a][b]+=z[i][a]*z[i][b];}}
  for(let j=1;j<dim;j++)xtx[j][j]+=1e-8;
  const beta=solveLinear(xtx,xty),predict=i=>z[i].reduce((s,v,j)=>s+v*beta[j],0);
  const actual=testIds.map(i=>y[i]),predicted=testIds.map(predict),mean=actual.reduce((s,v)=>s+v,0)/actual.length;
  const mse=actual.reduce((s,v,i)=>s+(v-predicted[i])**2,0)/actual.length;
  const mae=actual.reduce((s,v,i)=>s+Math.abs(v-predicted[i]),0)/actual.length;
  const sst=actual.reduce((s,v)=>s+(v-mean)**2,0),r2=sst?1-mse*actual.length/sst:NaN;
  return {actual,predicted,expanded,coefficients:beta.slice(1).map((v,j)=>({name:expanded[j],value:v/scales[j]})),metrics:{n:valid.length,train:trainIds.length,test:testIds.length,rmse:Math.sqrt(mse),mae,r2},dropped:rows.length-valid.length};
}
function drawCustomChart(actual,predicted){
  const canvas=$('custom-chart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,rect.width*dpr);canvas.height=Math.max(1,rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=rect.width,h=rect.height,p={l:62,r:20,t:20,b:48};
  const values=[...actual,...predicted],min=Math.min(...values),max=Math.max(...values),span=(max-min)||1;
  const x=v=>p.l+(v-min)/span*(w-p.l-p.r),y=v=>h-p.b-(v-min)/span*(h-p.t-p.b);
  ctx.strokeStyle='rgba(160,178,202,.18)';ctx.fillStyle='#9baac0';ctx.font='11px sans-serif';ctx.textAlign='right';
  for(let i=0;i<=4;i++){const v=min+span*i/4,yy=y(v);ctx.beginPath();ctx.moveTo(p.l,yy);ctx.lineTo(w-p.r,yy);ctx.stroke();ctx.fillText(v.toPrecision(3),p.l-8,yy+4);}
  ctx.beginPath();ctx.moveTo(x(min),y(min));ctx.lineTo(x(max),y(max));ctx.strokeStyle='#f6b94f';ctx.setLineDash([6,5]);ctx.lineWidth=1.5;ctx.stroke();ctx.setLineDash([]);
  const count=Math.min(2000,actual.length),stride=Math.max(1,Math.floor(actual.length/count));ctx.fillStyle='#44c7ec';ctx.globalAlpha=.55;
  for(let i=0;i<actual.length;i+=stride){ctx.beginPath();ctx.arc(x(actual[i]),y(predicted[i]),3,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
  ctx.textAlign='center';ctx.fillStyle='#9baac0';ctx.fillText('Actual',p.l+(w-p.l-p.r)/2,h-11);ctx.save();ctx.translate(15,p.t+(h-p.t-p.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('Predicted',0,0);ctx.restore();
  ctx.fillStyle='#44c7ec';ctx.fillRect(w-190,17,10,10);ctx.fillStyle='#c8d2e2';ctx.textAlign='left';ctx.fillText('Holdout prediction',w-174,26);ctx.fillStyle='#f6b94f';ctx.fillRect(w-190,36,10,3);ctx.fillText('Ideal prediction',w-174,43);
}
function runCustomAnalysis(){
  $('custom-error').hidden=true;$('custom-results').hidden=true;
  try{
    const target=$('target-select').value;
    const features=[...$('feature-list').querySelectorAll('input:checked')].map(x=>x.value);
    if(!features.length)throw Error('至少選擇一個輸入特徵。');
    let rows=loadedDataset.records;
    const originalN=rows.length;
    if(rows.length>40000){const rng=seededRandom(2026),picked=new Set();while(picked.size<40000)picked.add(Math.floor(rng()*rows.length));rows=[...picked].sort((a,b)=>a-b).map(i=>rows[i]);}
    $('custom-status').textContent='計算中…';
    setTimeout(()=>{
      try{
        const result=fitModel(rows,features,target);
        setMetrics($('custom-metrics'),[
          ['完整資料列',result.metrics.n.toLocaleString()],['Train / test',`${result.metrics.train.toLocaleString()} / ${result.metrics.test.toLocaleString()}`],
          ['RMSE',formatNumber(result.metrics.rmse,5)],['MAE',formatNumber(result.metrics.mae,5)],['R²',formatNumber(result.metrics.r2,4)]
        ]);
        $('coefficient-rows').innerHTML=result.coefficients.sort((a,b)=>Math.abs(b.value)-Math.abs(a.value)).map(c=>`<tr><td>${escapeHtml(c.name)}</td><td>${formatNumber(c.value,6)}</td></tr>`).join('');
        drawCustomChart(result.actual,result.predicted);$('custom-results').hidden=false;
        const sampled=originalN>40000?`；已從 ${originalN.toLocaleString()} 筆固定抽樣為 40,000 筆`:'';
        $('custom-status').textContent=`分析完成；排除 ${result.dropped} 筆缺值列${sampled}。`;
      }catch(e){$('custom-status').textContent='';$('custom-error').textContent=e.message;$('custom-error').hidden=false;}
    },20);
  }catch(e){$('custom-status').textContent='';$('custom-error').textContent=e.message;$('custom-error').hidden=false;}
}
async function loadCustom(url){
  $('custom-error').hidden=true;$('load-button').disabled=true;$('load-button').textContent='載入中…';
  try{const response=await fetch(url);checkResponse(response);populateSetup(parseCsv(await response.text()),url);}
  catch(e){$('custom-error').textContent=`載入失敗：${e.message}。請確認網址公開、CSV 格式正確且允許 CORS。`;$('custom-error').hidden=false;}
  finally{$('load-button').disabled=false;$('load-button').textContent='載入資料 →';}
}
$('problem-select').addEventListener('change',showProblem);
document.querySelectorAll('[data-selection-metric]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-selection-metric]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});renderSelectionChart(button.dataset.selectionMetric);}));
document.querySelectorAll('[data-startup-metric]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-startup-metric]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});renderStartupChart(button.dataset.startupMetric);}));
$('interval-type').addEventListener('change',renderIntervalChart);$('interval-count').addEventListener('input',renderIntervalChart);$('diagnostic-type').addEventListener('change',renderDiagnosticChart);window.addEventListener('resize',renderFloodCharts);
$('target-select').addEventListener('change',()=>{renderFeatureList();$('custom-results').hidden=true;});
$('feature-list').addEventListener('change',updateFeatureCount);
$('analyze-button').addEventListener('click',runCustomAnalysis);
$('custom-form').addEventListener('submit',e=>{e.preventDefault();const url=$('csv-url').value.trim();if(!url){$('custom-error').textContent='請輸入 CSV 網址或選擇本機 CSV。';$('custom-error').hidden=false;return;}loadCustom(url);});
$('csv-file').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;$('custom-error').hidden=true;try{populateSetup(parseCsv(await file.text()),file.name);}catch(err){$('custom-error').textContent=err.message;$('custom-error').hidden=false;}});
showProblem();
