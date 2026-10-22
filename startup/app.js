const DEFAULT_URL='https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/refs/heads/master/datasets/50_Startups.csv';
const REQUIRED=['R&D Spend','Administration','Marketing Spend','State','Profit'];
const METHODS=[
  {id:'EFS',label:'窮舉特徵選擇 (EFS)',title:'方法 1：窮舉特徵選擇 (Exhaustive Feature Selection)',note:'逐一比較每種特徵組合，依交叉驗證 RMSE 找出每個特徵數下表現最佳的組合。'},
  {id:'RFE',label:'遞迴特徵消除 (RFE)',title:'方法 2：遞迴特徵消除 (Recursive Feature Elimination, RFE)',note:'反覆擬合線性迴歸並移除影響較小的特徵，呈現由完整特徵集逐步消除的路徑。'},
  {id:'Lasso',label:'Lasso 正規化 (Lasso)',title:'方法 3：Lasso 正規化 (Lasso)',note:'以標準化特徵擬合 Lasso 正規化模型，調整 alpha 觀察非零係數特徵與迴歸表現。'},
  {id:'Backward',label:'後向消除法 (Backward)',title:'方法 4：後向消除法 (Backward Elimination)',note:'從完整特徵集開始，每一步移除統計解釋力最低的欄位，再評估各階段特徵組合。'},
  {id:'SFS',label:'循序前向選取 (SFS)',title:'方法 5：循序前向選取 (Sequential Forward Selection)',note:'從單一特徵開始，每一步加入最能降低交叉驗證誤差的特徵。'},
];
let records=[],categories=[],features=[],folds=[],analysisToken=0,currentResults=[];
const $=id=>document.getElementById(id);
const fmt=(n,d=2)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function parseCSV(text){
  const rows=[];let row=[],cell='',quoted=false;
  for(let i=0;i<text.length;i++){const c=text[i];if(quoted){if(c==='"'&&text[i+1]==='"'){cell+='"';i++;}else if(c==='"')quoted=false;else cell+=c;}else if(c==='"')quoted=true;else if(c===','){row.push(cell.trim());cell='';}else if(c==='\n'||c==='\r'){if(c==='\r'&&text[i+1]==='\n')i++;row.push(cell.trim());if(row.some(v=>v!==''))rows.push(row);row=[];cell='';}else cell+=c;}
  row.push(cell.trim());if(row.some(v=>v!==''))rows.push(row);if(rows.length<2)throw Error('CSV 內沒有資料列。');
  const headers=rows.shift().map(v=>v.replace(/^\uFEFF/,''));const missing=REQUIRED.filter(v=>!headers.includes(v));if(missing.length)throw Error(`CSV 缺少必要欄位：${missing.join(', ')}`);
  const data=rows.map((cells,i)=>{const r={};headers.forEach((h,j)=>r[h]=cells[j]??'');for(const key of REQUIRED.filter(k=>k!=='State')){r[key]=Number(String(r[key]).replace(/[$,]/g,''));if(!Number.isFinite(r[key]))throw Error(`第 ${i+2} 列的 ${key} 不是有效數字。`);}r.State=String(r.State).trim();if(!r.State)throw Error(`第 ${i+2} 列的 State 為空。`);return r;});
  if(data.length<6)throw Error('至少需要 6 筆資料才能切分訓練集並進行 4-fold 交叉驗證。');return data;
}

function prepareData(){
  categories=[...new Set(records.map(r=>r.State))].sort();
  features=['R&D Spend','Administration','Marketing Spend',...categories.slice(1).map(s=>`State_${s}`)];
  const encoded=records.map(r=>({...r,x:[r['R&D Spend'],r.Administration,r['Marketing Spend'],...categories.slice(1).map(s=>r.State===s?1:0)]}));
  // Match sklearn train_test_split(test_size=.2, random_state=42), then evaluate
  // feature-count curves with KFold(n_splits=4, shuffle=True, random_state=43).
  const ids=encoded.map((_,i)=>i),splitRng=makeMT19937(42);
  for(let i=ids.length-1;i>0;i--){const j=splitRng.interval(i);[ids[i],ids[j]]=[ids[j],ids[i]];}
  const testSize=Math.ceil(encoded.length*.2),trainIds=ids.slice(testSize);
  records=trainIds.map(i=>encoded[i]);
  const foldIds=records.map((_,i)=>i),mt=makeMT19937(43);
  for(let i=foldIds.length-1;i>0;i--){const j=mt.interval(i);[foldIds[i],foldIds[j]]=[foldIds[j],foldIds[i]];}
  folds=[];let start=0;for(let f=0;f<4;f++){const size=Math.floor(foldIds.length/4)+(f<foldIds.length%4?1:0);folds.push(foldIds.slice(start,start+size));start+=size;}
}

function makeMT19937(seed){const state=new Uint32Array(624);let index=624;state[0]=seed>>>0;for(let i=1;i<624;i++)state[i]=(Math.imul(1812433253,state[i-1]^(state[i-1]>>>30))+i)>>>0;function uint32(){if(index>=624){for(let k=0;k<624;k++){const y=(state[k]&0x80000000)|(state[(k+1)%624]&0x7fffffff);state[k]=state[(k+397)%624]^(y>>>1)^((y&1)?0x9908b0df:0);}index=0;}let y=state[index++];y^=y>>>11;y^=(y<<7)&0x9d2c5680;y^=(y<<15)&0xefc60000;y^=y>>>18;return y>>>0;}return {interval(max){let mask=max;mask|=mask>>>1;mask|=mask>>>2;mask|=mask>>>4;mask|=mask>>>8;mask|=mask>>>16;let value;do{value=uint32()&mask;}while(value>max);return value;}};}

function solve(a,b){
  const n=b.length,m=a.map((r,i)=>[...r,b[i]]);
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(m[r][c])>Math.abs(m[p][c]))p=r;if(Math.abs(m[p][c])<1e-12)throw Error('特徵間高度相關，矩陣無法穩定求解；請檢查資料欄位。');[m[c],m[p]]=[m[p],m[c]];const v=m[c][c];for(let j=c;j<=n;j++)m[c][j]/=v;for(let r=0;r<n;r++){if(r===c)continue;const f=m[r][c];for(let j=c;j<=n;j++)m[r][j]-=f*m[c][j];}}
  return m.map(r=>r[n]);
}
// Fit once and apply the standardized coefficients to validation rows.
function predictFold(train,valid,selected){
  const p=selected.length,means=[],scales=[];for(const j of selected){const mean=train.reduce((s,r)=>s+r.x[j],0)/train.length;const sd=Math.sqrt(train.reduce((s,r)=>s+(r.x[j]-mean)**2,0)/train.length)||1;means.push(mean);scales.push(sd);}
  const size=p+1,xtx=Array.from({length:size},()=>Array(size).fill(0)),xty=Array(size).fill(0);
  for(const r of train){const z=[1,...selected.map((j,k)=>(r.x[j]-means[k])/scales[k])];for(let a=0;a<size;a++){xty[a]+=z[a]*r.Profit;for(let b=0;b<size;b++)xtx[a][b]+=z[a]*z[b];}}
  for(let i=1;i<size;i++)xtx[i][i]+=1e-9;const beta=solve(xtx,xty);
  return valid.map(r=>[1,...selected.map((j,k)=>(r.x[j]-means[k])/scales[k])].reduce((s,v,i)=>s+v*beta[i],0));
}
function metrics(selected,source=records){
  const predictions=Array(source.length).fill(0),actual=source.map(r=>r.Profit);
  for(const valIds of folds){const valSet=new Set(valIds),train=source.filter((_,i)=>!valSet.has(i)),valid=source.filter((_,i)=>valSet.has(i));const pred=predictFold(train,valid,selected);valid.forEach((r,i)=>{const idx=source.indexOf(r);predictions[idx]=pred[i];});}
  const mean=actual.reduce((s,v)=>s+v,0)/actual.length;const mse=actual.reduce((s,v,i)=>s+(v-predictions[i])**2,0)/actual.length;const ss=actual.reduce((s,v)=>s+(v-mean)**2,0);return {rmse:Math.sqrt(mse),r2:1-actual.reduce((s,v,i)=>s+(v-predictions[i])**2,0)/ss};
}
function fullCoefficients(selected,source=records){
  const n=source.length,p=selected.length,means=[],scales=[];for(const j of selected){const mean=source.reduce((s,r)=>s+r.x[j],0)/n;const sd=Math.sqrt(source.reduce((s,r)=>s+(r.x[j]-mean)**2,0)/n)||1;means.push(mean);scales.push(sd);}
  // Center and standardize, then use Householder QR rather than normal equations.
  // This avoids squaring the condition number during RFE coefficient ranking.
  const yMean=source.reduce((s,r)=>s+r.Profit,0)/n,A=source.map(r=>selected.map((j,k)=>(r.x[j]-means[k])/scales[k])),b=source.map(r=>r.Profit-yMean);
  for(let k=0;k<p;k++){let norm=0;for(let i=k;i<n;i++)norm=Math.hypot(norm,A[i][k]);if(norm<1e-12)continue;const alpha=A[k][k]>=0?-norm:norm,v=[];for(let i=k;i<n;i++)v.push(A[i][k]);v[0]-=alpha;const vnorm=v.reduce((s,x)=>s+x*x,0)||1;for(let j=k;j<p;j++){let dot=0;for(let i=k;i<n;i++)dot+=v[i-k]*A[i][j];const factor=2*dot/vnorm;for(let i=k;i<n;i++)A[i][j]-=factor*v[i-k];}let dot=0;for(let i=k;i<n;i++)dot+=v[i-k]*b[i];const factor=2*dot/vnorm;for(let i=k;i<n;i++)b[i]-=factor*v[i-k];}
  const beta=Array(p).fill(0);for(let i=p-1;i>=0;i--){let rhs=b[i];for(let j=i+1;j<p;j++)rhs-=A[i][j]*beta[j];beta[i]=Math.abs(A[i][i])<1e-12?0:rhs/A[i][i];}return beta.map((coef,k)=>coef/scales[k]);
}
function weakestByTStatistic(selected){
  const n=records.length,p=selected.length,means=[],scales=[];for(const j of selected){const mean=records.reduce((s,r)=>s+r.x[j],0)/n;const sd=Math.sqrt(records.reduce((s,r)=>s+(r.x[j]-mean)**2,0)/n)||1;means.push(mean);scales.push(sd);}
  const size=p+1,a=Array.from({length:size},()=>Array(size).fill(0)),b=Array(size).fill(0),design=[];
  for(const r of records){const z=[1,...selected.map((j,k)=>(r.x[j]-means[k])/scales[k])];design.push(z);for(let i=0;i<size;i++){b[i]+=z[i]*r.Profit;for(let j=0;j<size;j++)a[i][j]+=z[i]*z[j];}}
  for(let i=1;i<size;i++)a[i][i]+=1e-9;const beta=solve(a,b);let rss=0;for(let i=0;i<n;i++){const error=records[i].Profit-design[i].reduce((s,v,j)=>s+v*beta[j],0);rss+=error*error;}const variance=rss/Math.max(1,n-size);
  let weakest=0,smallestT=Infinity;for(let j=0;j<p;j++){const unit=Array(size).fill(0);unit[j+1]=1;const invCol=solve(a,unit);const se=Math.sqrt(Math.max(1e-18,variance*invCol[j+1]));const t=Math.abs(beta[j+1]/se);if(t<smallestT){smallestT=t;weakest=j;}}return weakest;
}
function bestSubset(k,candidates){let best=null;for(const subset of candidates){const m=metrics(subset);if(!best||m.rmse<best.RMSE)best={Number:k,Features:subset.slice(),RMSE:m.rmse,R2:m.r2};}return best;}
function combinations(items,k,start=0,prefix=[],out=[]){if(prefix.length===k){out.push(prefix.slice());return out;}for(let i=start;i<=items.length-(k-prefix.length);i++){prefix.push(items[i]);combinations(items,k,i+1,prefix,out);prefix.pop();}return out;}
function rankRFE(source=records){let remaining=features.map((_,i)=>i),path={};while(remaining.length){path[remaining.length]=remaining.slice();if(remaining.length===1)break;const coefs=fullCoefficients(remaining,source);let weakest=0;for(let i=1;i<coefs.length;i++)if(Math.abs(coefs[i])<Math.abs(coefs[weakest]))weakest=i;remaining.splice(weakest,1);}return path;}
function rfeCV(k){const predictions=Array(records.length).fill(0);for(const validIds of folds){const set=new Set(validIds),train=records.filter((_,i)=>!set.has(i)),valid=records.filter((_,i)=>set.has(i)),selected=rankRFE(train)[k],pred=predictFold(train,valid,selected);valid.forEach((r,j)=>predictions[records.indexOf(r)]=pred[j]);}return summarizePredictions(predictions);}
function summarizePredictions(predictions){const actual=records.map(r=>r.Profit),mean=actual.reduce((s,v)=>s+v,0)/actual.length,sse=actual.reduce((s,v,i)=>s+(v-predictions[i])**2,0),sst=actual.reduce((s,v)=>s+(v-mean)**2,0);return {rmse:Math.sqrt(sse/actual.length),r2:1-sse/sst};}
function fitLasso(train,alpha){const p=features.length,n=train.length,means=Array(p).fill(0),scales=Array(p).fill(0),yMean=train.reduce((s,r)=>s+r.Profit,0)/n,x=train.map(r=>r.x.map((v,j)=>{means[j]+=v/n;return v;})),y=train.map(r=>r.Profit-yMean);
  for(let j=0;j<p;j++){scales[j]=Math.sqrt(train.reduce((s,r)=>s+(r.x[j]-means[j])**2,0)/n)||1;for(let i=0;i<n;i++)x[i][j]=(x[i][j]-means[j])/scales[j];}
  const coef=Array(p).fill(0),norm=x[0].map((_,j)=>x.reduce((s,row)=>s+row[j]**2,0)/n);
  for(let iter=0;iter<10000;iter++){let change=0;for(let j=0;j<p;j++){let rho=0;for(let i=0;i<n;i++){let residual=y[i];for(let q=0;q<p;q++)if(q!==j)residual-=x[i][q]*coef[q];rho+=x[i][j]*residual;}rho/=n;const next=Math.sign(rho)*Math.max(0,Math.abs(rho)-alpha)/(norm[j]||1);change=Math.max(change,Math.abs(next-coef[j]));coef[j]=next;}if(change<1e-8)break;}
  return {coef,predict:r=>yMean+coef.reduce((s,c,j)=>s+c*(r.x[j]-means[j])/scales[j],0)};
}
function lassoCV(alpha){const predictions=Array(records.length).fill(0);for(const validIds of folds){const set=new Set(validIds),train=records.filter((_,i)=>!set.has(i)),valid=records.filter((_,i)=>set.has(i)),model=fitLasso(train,alpha);valid.forEach((r,j)=>predictions[records.indexOf(r)]=model.predict(r));}return summarizePredictions(predictions);}
function lassoPath(){const best={},alphas=Array.from({length:160},(_,i)=>Math.pow(10,-4+i*10/159));for(const alpha of alphas){const score=lassoCV(alpha),model=fitLasso(records,alpha),chosen=model.coef.map((v,i)=>Math.abs(v)>1e-8?i:-1).filter(i=>i>=0);if(!chosen.length)continue;const k=chosen.length;if(!best[k]||score.rmse<best[k].RMSE)best[k]={Number:k,Features:chosen,RMSE:score.rmse,R2:score.r2};}return best;}
function runMethod(id){
  const n=features.length;if(n>10&&id==='EFS')throw Error('EFS 需比較 2^特徵數個組合。為避免瀏覽器卡住，請先減少 State 類別（目前特徵數 '+n+'）。');
  const out=[];
  if(id==='EFS'){for(let k=1;k<=n;k++)out.push(bestSubset(k,combinations(features.map((_,i)=>i),k)));}
  else if(id==='RFE'){for(let k=1;k<=n;k++){const cv=rfeCV(k);out.push({Number:k,Features:rankRFE(records)[k],RMSE:cv.rmse,R2:cv.r2});}}
  else if(id==='Lasso'){const path=lassoPath();for(const k of Object.keys(path).map(Number).sort((a,b)=>a-b))out.push(path[k]);}
  else if(id==='Backward'){let active=features.map((_,i)=>i);const path={};while(active.length){path[active.length]=active.slice();if(active.length===1)break;active.splice(weakestByTStatistic(active),1);}for(let k=1;k<=n;k++){const fs=path[k],m=metrics(fs);out.push({Number:k,Features:fs,RMSE:m.rmse,R2:m.r2});}}
  else if(id==='SFS'){let chosen=[];const remaining=features.map((_,i)=>i);for(let k=1;k<=n;k++){let best=null,bestIndex=-1;for(const j of remaining){const candidate=[...chosen,j],m=metrics(candidate);if(!best||m.rmse<best.RMSE){best={Number:k,Features:candidate,RMSE:m.rmse,R2:m.r2};bestIndex=j;}}chosen=best.Features.slice();remaining.splice(remaining.indexOf(bestIndex),1);out.push(best);}}
  return out;
}

function renderTabs(){ $('method-tabs').innerHTML=METHODS.map((m,i)=>`<button class="method-tab ${i===0?'active':''}" role="tab" aria-selected="${i===0}" data-method="${m.id}" type="button"><span class="method-number">方法 ${i+1}</span>${m.label}</button>`).join('');$('method-tabs').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectMethod(b.dataset.method))); }
function selectMethod(id){$('method-tabs').querySelectorAll('button').forEach(b=>{const active=b.dataset.method===id;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});const m=METHODS.find(x=>x.id===id);$('method-title').textContent=m.title;$('method-note').textContent=m.note;runAnalysis(id);}
function renderTable(rows){const best=rows.reduce((a,b)=>b.RMSE<a.RMSE?b:a);$('result-rows').innerHTML=rows.map(r=>`<tr class="${r===best?'optimal':''}"><td>${r.Number}${r===best?' <span class="optimal-badge">最佳 ★</span>':''}</td><td>[${r.Features.map(i=>esc(features[i])).join(', ')}]</td><td>${fmt(r.RMSE,6)}</td><td>${fmt(r.R2,6)}</td></tr>`).join('');$('best-summary').textContent=`交叉驗證 RMSE 最低：${best.Number} 個特徵 · ${fmt(best.RMSE,2)} · ${best.Features.map(i=>features[i]).join(', ')}`;}
function drawLineChart(canvasId,rows,key,color,lowerBetter){const c=$(canvasId),rect=c.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);const pad={l:58,r:18,t:18,b:44},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b,vals=rows.map(r=>r[key]);let min=Math.min(...vals),max=Math.max(...vals);let range=max-min;if(range===0)range=Math.max(1,Math.abs(max)*.1);if(key==='R2')range=Math.max(range,.05);min-=range*.12;max+=range*.12;const x=i=>pad.l+(rows.length===1?cw/2:i*cw/(rows.length-1)),y=v=>pad.t+(max-v)/(max-min)*ch;
  ctx.font='12px DM Sans, sans-serif';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillStyle='#94a0b5';ctx.strokeStyle='rgba(157,170,194,.14)';ctx.lineWidth=1;for(let i=0;i<=4;i++){const value=max-(max-min)*i/4,yy=y(value);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(key==='RMSE'?Math.round(value).toLocaleString('en-US'):value.toFixed(2),pad.l-10,yy);}
  ctx.beginPath();rows.forEach((r,i)=>i?ctx.lineTo(x(i),y(r[key])):ctx.moveTo(x(i),y(r[key])));ctx.strokeStyle=color;ctx.lineWidth=3;ctx.lineJoin='round';ctx.stroke();const bestIndex=vals.indexOf(lowerBetter?Math.min(...vals):Math.max(...vals));rows.forEach((r,i)=>{ctx.beginPath();ctx.arc(x(i),y(r[key]),i===bestIndex?7:5,0,Math.PI*2);ctx.fillStyle=i===bestIndex?'#f6a723':color;ctx.fill();ctx.lineWidth=2;ctx.strokeStyle='#f7f8fc';ctx.stroke();ctx.textAlign='center';ctx.textBaseline='top';ctx.fillStyle='#a2aec1';ctx.fillText(r.Number,x(i),h-pad.b+13);});ctx.textAlign='center';ctx.fillStyle='#8e9bb1';ctx.fillText('Number of Features',pad.l+cw/2,h-8);}
function drawResults(rows){currentResults=rows;$('results').hidden=false;$('download-chart').disabled=false;requestAnimationFrame(()=>{drawLineChart('rmse-chart',rows,'RMSE','#39b8ed',true);drawLineChart('r2-chart',rows,'R2','#19bd88',false);});renderTable(rows);}
function downloadCharts(){const a=$('rmse-chart'),b=$('r2-chart'),out=document.createElement('canvas'),scale=2,gap=28,header=64;out.width=(a.width+b.width+gap)*scale;out.height=Math.max(a.height,b.height)*scale+header*scale;const ctx=out.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#111722';ctx.fillRect(0,0,out.width/scale,out.height/scale);ctx.fillStyle='#eef2fa';ctx.font='bold 20px sans-serif';ctx.fillText($('method-title').textContent,20,34);ctx.drawImage(a,0,header,a.width/scale,a.height/scale);ctx.drawImage(b,a.width/scale+gap,header,b.width/scale,b.height/scale);const link=document.createElement('a');link.download=`${$('method-tabs').querySelector('.active').dataset.method}_feature_selection.png`;link.href=out.toDataURL('image/png');link.click();}
function runAnalysis(id){if(!records.length)return;const token=++analysisToken;$('results').hidden=true;$('analysis-status').textContent='分析中… 少量資料也需要逐一評估特徵組合。';$('analysis-status').classList.add('working');$('run-button').disabled=true;setTimeout(()=>{try{const result=runMethod(id);if(token!==analysisToken)return;drawResults(result);$('analysis-status').textContent=`分析完成 · ${records.length} 筆訓練資料 · 4-fold cross-validation`; }catch(e){$('analysis-status').textContent='分析無法完成';$('error').textContent=e.message;$('error').hidden=false;}finally{$('analysis-status').classList.remove('working');$('run-button').disabled=false;}},30);}
async function loadData(url){$('error').hidden=true;$('status').textContent='正在讀取 CSV…';$('status-dot').classList.add('loading');$('load-button').disabled=true;try{const res=await fetch(url);if(!res.ok)throw Error(`HTTP ${res.status}：無法讀取 CSV。`);records=parseCSV(await res.text());const total=records.length;prepareData();$('row-count').textContent=`${total} 筆資料 · ${records.length} 筆訓練資料 · ${features.length} 個特徵`;$('method-tabs').querySelectorAll('button').forEach(b=>b.disabled=false);$('run-button').disabled=false;$('status').textContent='資料載入完成';selectMethod($('method-tabs').querySelector('.active')?.dataset.method||'EFS');}catch(e){$('status').textContent='載入失敗';$('error').textContent=`${e.message} 確認網址公開且來源允許 CORS。`;$('error').hidden=false;}finally{$('status-dot').classList.remove('loading');$('load-button').disabled=false;}}

$('source-form').addEventListener('submit',e=>{e.preventDefault();loadData($('csv-url').value.trim());});$('run-button').addEventListener('click',()=>selectMethod($('method-tabs').querySelector('.active')?.dataset.method||'EFS'));$('download-chart').addEventListener('click',downloadCharts);renderTabs();$('method-tabs').querySelectorAll('button').forEach(b=>b.disabled=true);loadData(DEFAULT_URL);window.addEventListener('resize',()=>{if(currentResults.length){drawLineChart('rmse-chart',currentResults,'RMSE','#39b8ed',true);drawLineChart('r2-chart',currentResults,'R2','#19bd88',false);}});
