"""P2 v0.18 offline interactive Shape Explorer; no web server or network.

A browser-side 64-state exact assignment DP computes the same D4 plus optimal
centroid-translation six-point shape distance as v0.13 for any legal 6/49
query. Historical labels are display-only and never used for forecasting.
"""
from __future__ import annotations

from html import escape
import json
from math import comb
from pathlib import Path

import numpy as np

from .shape_taxonomy_v015 import classify_numbers,shape_key
from .shape_theory_v016 import RULE_VERSION,UNIVERSE,classification_keys,category_name


def _validate_inputs(ids,numbers,universe):
    a=np.asarray(numbers)
    d=np.asarray(ids)
    if a.shape!=(len(d),6) or len(d)<3 or len(set(map(int,d)))!=len(d):
        raise ValueError('need unambiguous historical six-number records')
    if (not universe.get('complete') or int(universe.get('total',0))!=UNIVERSE or
        universe.get('schema')!=RULE_VERSION):raise ValueError('need exact P1 theory')
    codes=classification_keys(classify_numbers(a))
    return np.sort(a.astype(np.uint8),axis=1),codes


def gallery_data(ids:np.ndarray,numbers:np.ndarray,universe:dict,*,
                 density:np.ndarray|None=None,mode_labels:np.ndarray|None=None)->dict:
    """Only validated historical metadata; supports arbitrary interactive query."""
    a,tags=_validate_inputs(ids,numbers,universe)
    n=len(a)
    dens=np.zeros(n,dtype=np.float64) if density is None else np.asarray(density,dtype=np.float64)
    modes=np.full(n,-1,dtype=np.int32) if mode_labels is None else np.asarray(mode_labels,dtype=np.int32)
    if dens.shape!=(n,) or modes.shape!=(n,) or not np.isfinite(dens).all():
        raise ValueError('P1 KDE / mode labels not aligned with history')
    records=[]
    for i in range(n):
        records.append([int(ids[i]),*map(int,a[i]),int(tags['component'][i]),
             int(tags['partition'][i]),int(tags['edge'][i]),round(float(dens[i]),7),int(modes[i])])
    return {'schema':'lotto7x7.offline_shape_explorer.v0.18','count':n,
      'universe':UNIVERSE,'records':records,'counts':universe['counts'],
      'graph_rule':RULE_VERSION,'columns':['draw_id','n1','n2','n3','n4','n5','n6',
        'component','partition','edge','historical_kde','historical_mode']}


def offline_shape_explorer(ids,numbers,universe,*,density=None,mode_labels=None)->str:
    """Single fully offline HTML with editable six numbers + exact nearest shapes."""
    data=gallery_data(ids,numbers,universe,density=density,mode_labels=mode_labels)
    record_json=json.dumps(data,ensure_ascii=True,separators=(',',':'))
    # Data comes from strict integers and fixed P1 class codes, not raw HTML.
    page=r'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Lotto649 v0.18 — Interactive 7×7 Shape Explorer</title>
<style>
:root{color-scheme:light;background:#f4f7f8;color:#243342;font:15px system-ui,-apple-system,'Segoe UI',sans-serif}
body{max-width:1300px;margin:1.5rem auto;padding:0 1rem}h1{font-size:1.7rem;margin:.5rem 0}h2{font-size:1.15rem;margin:.25rem 0 1rem}
small,.muted{color:#61727e}.panel{background:#fff;border:1px solid #e0e7eb;padding:1.2rem;border-radius:12px;margin:1rem 0;box-shadow:0 1px 2px #0001}
.flex{display:flex;flex-wrap:wrap;align-items:center;gap:.75rem}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:.8rem}
.item{padding:.75rem;border:1px solid #e1e7e9;border-radius:9px;background:#fcfdfd}.item svg{max-width:160px;width:100%}
label{font-weight:600}input,select{font:inherit;border:1px solid #b6c5cc;border-radius:7px;padding:.65rem;max-width:360px}
button{font:inherit;background:#177e74;color:#fff;border:0;border-radius:7px;padding:.7rem 1.05rem;cursor:pointer}
button:hover{background:#11655d}button:focus-visible,input:focus-visible{outline:3px solid #81b8ae}table{width:100%;border-collapse:collapse}
th,td{text-align:left;border-bottom:1px solid #e6ebed;padding:.52rem;font-size:.88rem}th{color:#53616c}
.metric{min-width:165px}.metric b{font-size:1.35rem}.stat{font-variant-numeric:tabular-nums}
.warning{padding:.7rem;background:#f0f4f5;border-radius:7px;color:#495967}
.dot{stroke:#fff;stroke-width:.3}a{color:#08776b}
</style></head><body>
<h1>7×7 Shape Explorer <small>v0.18</small></h1>
<p class="muted">輸入六個合法號碼，探索連通拓樸、精確等價形狀，以及旋轉／鏡射／質心平移後距離最近的歷史影像。全程離線，在瀏覽器本機運算。</p>
<div class="panel"><h2>任意六碼查詢</h2>
<div class="flex"><label for="query">六個號碼</label><input id="query" size="35" value="1, 8, 16, 22, 24, 30" aria-label="六個號碼"/>
<button id="run">搜尋完整形狀距離</button></div>
<p class="muted">可用空白或逗號分隔；程式會逐一計算所有 {{COUNT}} 期的六點精確指派距離。</p>
<p id="error" role="alert" style="color:#b33333"></p>
<div class="flex" id="metrics"></div><div class="cards" id="query-shape"></div></div>
<div class="panel"><h2>最相似的歷史 7×7 形狀</h2><div class="cards" id="neighbors"></div></div>
<div class="panel"><h2>固定分類查詢</h2>
<p class="muted">此區顯示固定的 L3 結構類別，不使用未來期別；並非中獎率排名。</p>
<div class="flex"><label for="edge-filter">L3 類別</label><select id="edge-filter"></select>
<button id="filter-btn">查詢歷史成員</button></div><p id="class-qty" class="muted"></p>
<div class="cards" id="class-images"></div></div>
<div class="panel"><h2>技術限制</h2><div class="warning">
此工具中，歷史 KDE 值使用完整歷史計算，僅供描述，不能做過去時點的回測特徵；固定圖形類別的理論比例則來自全組合精確計量。
歷史影像的接近程度、密度或稀有程度，均不改變公平 6/49 中任何特定六碼下一期的機率（每組 1／13,983,816）。
形狀距離允許連續質心平移，並不強制平移後仍位於 7×7 的整數格點。</div></div>
<script id="history-data" type="application/json">{{DATA}}</script>
<script>
'use strict';
const DATA=JSON.parse(document.getElementById('history-data').textContent);
const REC=DATA.records, WEIGHTS=DATA.counts;
const $=id=>document.getElementById(id);
function points(nums){return nums.map(v=>[Math.floor((v-1)/7),(v-1)%7]);}
function popcount(mask){let n=0;while(mask){mask&=mask-1;n++;}return n;}
function canonical(coords){let keys=[];
 for(let f=0;f<2;f++){
  let original=coords.map(([r,c])=>[r,f?-c:c]);
  for(let t=0;t<4;t++){
    let q=original.map(x=>x.slice());for(let k=0;k<t;k++)q=q.map(([r,c])=>[-c,r]);
    const r0=Math.min(...q.map(x=>x[0])),c0=Math.min(...q.map(x=>x[1]));
    q=q.map(([r,c])=>[r-r0,c-c0]);q.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    keys.push(q.map(p=>p.join(',')).join(';'));
  }
 }return keys.sort()[0];}
function orbitCount(coords){let seen=new Set(),total=0;
 for(let f=0;f<2;f++){
  let original=coords.map(([r,c])=>[r,f?-c:c]);
  for(let t=0;t<4;t++){
   let q=original.map(x=>x.slice());for(let k=0;k<t;k++)q=q.map(([r,c])=>[-c,r]);
   let loR=Math.min(...q.map(p=>p[0])),loC=Math.min(...q.map(p=>p[1]));
   q=q.map(([r,c])=>[r-loR,c-loC]);q.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
   let key=q.map(p=>p.join(',')).join(';');if(seen.has(key))continue;seen.add(key);
   let h=1+Math.max(...q.map(p=>p[0])),w=1+Math.max(...q.map(p=>p[1]));total+=(8-h)*(8-w);
  }
 }return total;}
function topology(nums){const p=points(nums);let conn=Array.from({length:6},()=>[]),o=0,d=0;
 for(let i=0;i<6;i++)for(let j=i+1;j<6;j++){
  let dr=Math.abs(p[i][0]-p[j][0]),dc=Math.abs(p[i][1]-p[j][1]);
  if(dr<=1&&dc<=1){conn[i].push(j);conn[j].push(i);}
  if(dr+dc===1)o++;if(dr===1&&dc===1)d++;
 }
 let seen=new Set(),sizes=[];for(let i=0;i<6;i++)if(!seen.has(i)){
  let stack=[i],size=0;seen.add(i);while(stack.length){let v=stack.pop();size++;
  for(const j of conn[v])if(!seen.has(j)){seen.add(j);stack.push(j);}}sizes.push(size);
 }sizes.sort((a,b)=>b-a);let shift=[15,12,9,6,3,0],partition=0;
 for(let i=0;i<sizes.length;i++)partition|=(sizes[i]<<shift[i]);
 return {component:sizes.length,partition,edge:partition*1024+o*32+d,
         partitionName:'P'+sizes.join('-'),o,d};}
function bitmap(nums){let chosen=new Set(nums),s='<svg viewBox="0 0 126 126" role="img" aria-label="7 by 7 grid">';
 for(let i=1;i<=49;i++){let row=Math.floor((i-1)/7),col=(i-1)%7;
 s+='<rect class="dot" x="'+(col*18+1)+'" y="'+(row*18+1)+'" width="16" height="16" rx="2" fill="'+(chosen.has(i)?'#17867e':'#e5eaed')+'"/>';
 }return s+'</svg>';}
function numsText(nums){return nums.map(v=>String(v).padStart(2,'0')).join(', ');}
function centered(coords){let r=coords.reduce((a,p)=>a+p[0],0)/6,c=coords.reduce((a,p)=>a+p[1],0)/6;
 return coords.map(([x,y])=>[x-r,y-c]);}
function transformed(coords){const out=[];
 for(let f=0;f<2;f++){
  let original=coords.map(([r,c])=>[r,f?-c:c]);
  for(let t=0;t<4;t++){
   let q=original.map(x=>x.slice());for(let j=0;j<t;j++)q=q.map(([r,c])=>[-c,r]);out.push(centered(q));
  }
 }return out;}
function shapeDistance(qVariants,record){
 const h=record.centered;let best=Infinity;
 for(let v=0;v<8;v++){
  const q=qVariants[v];let cost=Array.from({length:6},()=>new Float64Array(6));
  for(let i=0;i<6;i++)for(let j=0;j<6;j++){
   const dx=q[i][0]-h[j][0],dy=q[i][1]-h[j][1];cost[i][j]=dx*dx+dy*dy;
  }
  let dp=new Float64Array(64);dp.fill(Infinity);dp[0]=0;
  for(let mask=0;mask<63;mask++){
   let i=BITCOUNT[mask],base=dp[mask];if(!Number.isFinite(base)||i>=6)continue;
   for(let j=0;j<6;j++)if(!(mask&(1<<j))){let next=mask|(1<<j),candidate=base+cost[i][j];
    if(candidate<dp[next])dp[next]=candidate;}
  }if(dp[63]<best)best=dp[63];
 }return Math.sqrt(Math.max(0,best)/6);}
const BITCOUNT=Array.from({length:64},(_,i)=>popcount(i));
const history=REC.map(v=>({id:v[0],nums:v.slice(1,7),component:v[7],partition:v[8],edge:v[9],density:v[10],mode:v[11],
 centered:centered(points(v.slice(1,7)))}));
function renderHistory(rec,distance){let el=document.createElement('article');el.className='item';
 const dist=distance===undefined?'':'<p class="muted stat">形狀距離：'+distance.toFixed(4)+'</p>';
 el.innerHTML=bitmap(rec.nums)+'<b>第 '+rec.id+' 期</b><p>'+numsText(rec.nums)+'</p>'+dist+
 '<small>L3 '+nameEdge(rec.edge)+' · KDE '+rec.density.toFixed(4)+'</small>';
 return el;}
function nameEdge(key){let a=key>>10,o=(key>>5)&31,d=key&31,s=[];
 for(let shift of [15,12,9,6,3,0]){let v=(a>>shift)&7;if(v)s.push(v);}
 return 'P'+s.join('-')+'_O'+o+'_D'+d;}
function parseQuery(text){let parts=text.trim().split(/[\s,，、;；]+/).filter(Boolean);
 if(parts.length!==6)throw Error('請輸入恰好六個號碼。');
 let nums=parts.map(x=>{if(!/^\d+$/.test(x))throw Error('只接受整數。');return Number(x);});
 if(nums.some(v=>v<1||v>49)||new Set(nums).size!==6)throw Error('必須是 1–49 之間六個不重複的整數。');
 return nums.sort((a,b)=>a-b);}
function runQuery(){const start=performance.now();$('error').textContent='';
 try{const nums=parseQuery($('query').value),top=topology(nums),p=points(nums);
 const family=canonical(p),orbit=orbitCount(p),tw=WEIGHTS;
 const vals=[['L1 連通元件','C'+top.component,'component',top.component],
  ['L2 元件分區',top.partitionName,'partition',top.partition],
  ['L3 鄰接結構',top.partitionName+'_O'+top.o+'_D'+top.d,'edge',top.edge]];
 const m=$('metrics');m.replaceChildren();
 for(const [label,name,level,key] of vals){let node=document.createElement('div');node.className='metric';
  const mass=Number(tw[level][String(key)]||0);
  node.innerHTML='<small>'+label+'</small><br/><b>'+name+'</b><br/><small>公平組合 '+mass.toLocaleString()+'／'+DATA.universe.toLocaleString()+'</small>';m.append(node);}
 let self=document.createElement('article');self.className='item';
 const duplicates=history.filter(h=>canonical(points(h.nums))===family);
 self.innerHTML=bitmap(nums)+'<b>查詢形狀</b><p>'+numsText(nums)+'</p>'+
 '<p>精確旋轉／鏡射／平移等價組合：'+orbit.toLocaleString()+'（'+(100*orbit/DATA.universe).toFixed(6)+'%）</p>'+
 '<small>歷史相同形狀：'+duplicates.length+' 筆</small>';
 $('query-shape').replaceChildren(self);
 const variants=transformed(p);const results=history.map(h=>({h,d:shapeDistance(variants,h)}));
 results.sort((a,b)=>a.d-b.d||a.h.id-b.h.id);$('neighbors').replaceChildren(...results.slice(0,18).map(v=>renderHistory(v.h,v.d)));
 const elapsed=performance.now()-start;
 let note=document.createElement('small');note.className='muted';note.textContent='已對 '+history.length+' 期計算完整 8 種對稱與六點最佳指派距離（'+Math.round(elapsed)+' ms）。';
 $('neighbors').append(note);
 $('edge-filter').value=String(top.edge);showClass();
 }catch(err){$('error').textContent=err.message;}}
function setupFilters(){const counts={};for(const r of history)counts[r.edge]=(counts[r.edge]||0)+1;
 const ordered=Object.keys(WEIGHTS.edge).map(Number).sort((a,b)=>(counts[b]||0)-(counts[a]||0)||a-b);
 for(const edge of ordered){let opt=document.createElement('option');opt.value=String(edge);
 opt.textContent=nameEdge(edge)+'  歷史 '+(counts[edge]||0)+' / 全宇宙 '+Number(WEIGHTS.edge[edge]).toLocaleString();$('edge-filter').append(opt);}}
function showClass(){const key=Number($('edge-filter').value);const matches=history.filter(r=>r.edge===key);
 $('class-qty').textContent='類別 '+nameEdge(key)+'：歷史 '+matches.length+' 期；理論 '+Number(WEIGHTS.edge[String(key)]).toLocaleString()+' 種合法組合。';
 $('class-images').replaceChildren(...matches.slice(0,24).map(r=>renderHistory(r)));}
$('run').addEventListener('click',runQuery);
$('query').addEventListener('keydown',e=>{if(e.key==='Enter')runQuery();});
$('filter-btn').addEventListener('click',showClass);
setupFilters();runQuery();
</script></body></html>'''
    return page.replace('{{DATA}}',record_json).replace('{{COUNT}}',str(data['count']))
