// RealBench 页面脚本。数据来自 DATA（构建时内嵌）。
const $=(s,r)=>(r||document).querySelector(s);
function h(tag,attrs,kids){const e=document.createElement(tag);
  if(attrs)for(const k in attrs){const v=attrs[k];if(v==null||v===false)continue;
    if(k==='class')e.className=v;else if(k==='text')e.textContent=v;else if(k==='html')e.innerHTML=v;
    else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v===true?'':v)}
  (kids||[]).forEach(c=>{if(c==null||c===false)return;e.appendChild(typeof c==='string'?document.createTextNode(c):c)});return e}
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};

const RUNS={},EVALS={};
DATA.runs.forEach(r=>RUNS[r.id]=r);DATA.evals.forEach(e=>EVALS[e.id]=e);
const KIND={scratch:'从头训',pretrain:'预训练',finetune:'微调',posttrain:'后训练'};
const STATUS={finished:'已完成',running:'进行中',failed:'失败',stopped:'已停止',superseded:'已被取代'};
const RESULT={success:'成功',fail:'失败',partial:'部分',invalid:'无效',unlabeled:'未标注'};

function fmtLr(x){return x==null?'':x.toExponential(0).replace('e-0','e-').replace('e+0','e')}
function fmtNum(x,d){return x==null?'—':Number(x).toLocaleString('en-US',{maximumFractionDigits:d==null?0:d})}
function fmtLoss(x){return x==null?'—':x.toFixed(x<0.1?4:3)}
function dateOf(iso){return iso?iso.slice(0,10):'—'}
function basename(p){return (p||'').split('/').filter(Boolean).pop()||p||''}
function missing(t){return h('span',{class:'missing',text:t||'缺'})}

/* ---------- 迷你曲线 ---------- */
function spark(curve,w,hh){
  const ys=(curve&&curve.loss||[]).filter(v=>v!=null);
  if(ys.length<2)return h('span',{class:'sub',text:'无曲线'});
  const lo=Math.min(...ys),hi=Math.max(...ys),n=ys.length;
  // 取对数纵轴：loss 从 0.18 降到 0.008，线性轴上后半段是一条平线
  const L=v=>Math.log10(Math.max(v,1e-6)),a=L(lo),b=L(hi),span=(b-a)||1;
  const pts=ys.map((v,i)=>[2+(w-4)*i/(n-1),2+(hh-4)*(1-(L(v)-a)/span)]);
  const d=pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join('');
  const ns='http://www.w3.org/2000/svg';
  const s=document.createElementNS(ns,'svg');s.setAttribute('width',w);s.setAttribute('height',hh);s.setAttribute('viewBox','0 0 '+w+' '+hh);s.setAttribute('class','spark');
  s.setAttribute('role','img');s.setAttribute('aria-label','训练 loss 曲线，纵轴为对数，从 '+fmtLoss(ys[0])+' 到 '+fmtLoss(ys[n-1]));
  const p=document.createElementNS(ns,'path');p.setAttribute('d',d);p.setAttribute('fill','none');p.setAttribute('stroke','var(--accent)');p.setAttribute('stroke-width','1.4');s.appendChild(p);
  const e=document.createElementNS(ns,'circle');e.setAttribute('cx',pts[n-1][0]);e.setAttribute('cy',pts[n-1][1]);e.setAttribute('r','2.4');e.setAttribute('fill','var(--accent)');s.appendChild(e);
  return s}

/* ---------- 评测汇总 ---------- */
function closedEvalsOf(run){return (run.evals_closed||[]).map(i=>EVALS[i]).filter(Boolean)}
function openEvalsOf(run){return (run.evals_offline||[]).map(i=>EVALS[i]).filter(Boolean)}
function trialStats(ev){
  const t=(ev.closed_loop&&ev.closed_loop.trials)||[];
  const c={success:0,fail:0,partial:0,invalid:0,unlabeled:0};t.forEach(x=>c[x.result]++);
  const labeled=t.length-c.unlabeled;
  return {n:t.length,labeled,...c,videos:t.reduce((a,x)=>a+(x.videos||[]).length,0),
    rate:labeled?c.success/labeled:null,enough:t.length>=((ev.closed_loop&&ev.closed_loop.min_trials_required)||3)}}

/* ---------- 与同族基线的差 ---------- */
function flat(r){
  const o={};
  o['基座']=basename(r.base.model_path);
  o['数据配置']=r.data.config_name;
  o['训练集数']=r.data.train_episodes;
  o['步数']=r.hparams.max_steps;
  o['全局 batch']=r.hparams.global_batch;
  o['学习率']=fmtLr(r.hparams.learning_rate);
  o['调度']=r.hparams.scheduler;
  o['预热']=r.hparams.warmup_steps;
  o['种子']=r.hparams.seed;
  o['可训练']=(r.hparams.trainable||[]).join('+');
  Object.entries(r.hparams.extra||{}).forEach(([k,v])=>o[k]=String(v));
  return o}
function baselineOf(r,all){return r.baseline_id?(all.find(x=>x.id===r.baseline_id)||null):null}
function diffsOf(r,all){
  const b=baselineOf(r,all);if(!b)return null;
  const x=flat(r),y=flat(b),out=[];
  Object.keys({...x,...y}).forEach(k=>{if(String(x[k])!==String(y[k]))out.push([k,y[k],x[k]])});
  return {base:b,list:out}}

/* ---------- 路由 ---------- */
function route(){const x=location.hash.slice(1);
  if(x.startsWith('r-'))return {v:'run',id:'tr-'+x.slice(2)};
  if(x.startsWith('e-'))return {v:'eval',id:'ev-'+x.slice(2)};
  if(x==='compare')return {v:'compare'};
  return {v:'home'}}
function hashFor(id){return id.startsWith('tr-')?'#r-'+id.slice(3):'#e-'+id.slice(3)}
function renderTabs(cur){const t=$('#tabs');t.textContent='';
  [['home','全部训练','#'],['compare','对比','#compare']].forEach(([k,l,hr])=>t.appendChild(h('a',{href:hr,text:l,'aria-current':cur===k?'page':null})))}

/* ================= 总览：一行一次训练 ================= */
const flt={q:'',kind:'',family:'',task:'',real:''};
function tasksOf(r){return [...new Set(closedEvalsOf(r).concat(openEvalsOf(r)).map(e=>e.task))]}
function filtered(){return DATA.runs.filter(r=>{
  if(flt.kind&&r.kind!==flt.kind)return false;
  if(flt.family&&r.family!==flt.family)return false;
  if(flt.task&&!tasksOf(r).includes(flt.task))return false;
  if(flt.real==='yes'&&!closedEvalsOf(r).length)return false;
  if(flt.real==='no'&&closedEvalsOf(r).length)return false;
  if(flt.q){const s=[r.name,r.id,r.owner,r.data.config_name,r.base.model_path].join(' ').toLowerCase();if(!s.includes(flt.q.toLowerCase()))return false}
  return true}).sort((a,b)=>((b.time&&b.time.start)||'').localeCompare((a.time&&a.time.start)||''))}

function realCell(r){
  const evs=closedEvalsOf(r);
  if(!evs.length)return h('span',{class:'idle',text:'未真机'});
  const box=h('div',{style:'display:flex;flex-direction:column;gap:6px'});
  evs.forEach(e=>{
    const s=trialStats(e);
    const head=h('div',null,[h('a',{href:hashFor(e.id),text:e.task}),h('span',{class:'cell-sub',text:e.deployment&&e.deployment.robot?e.deployment.robot+' · '+e.date:e.date})]);
    let rate;
    if(!s.labeled)rate=h('span',{class:'warn num',text:s.n+' 次，成败未标注'});
    else{const pct=Math.round(s.rate*100);
      rate=h('span',{class:'num '+(s.enough?'':'warn'),text:s.success+'/'+s.labeled+' 成功（'+pct+'%）'+(s.enough?'':'，不足 '+((e.closed_loop&&e.closed_loop.min_trials_required)||3)+' 次')})}
    const vid=s.videos?h('a',{href:hashFor(e.id)+'',text:s.videos+' 段视频'}):h('span',{class:'missing',text:'无视频'});
    box.appendChild(h('div',null,[head,rate,h('span',{class:'cell-sub'},[vid])]))});
  return box}
function openCell(r){
  const evs=openEvalsOf(r);
  if(!evs.length)return h('span',{class:'idle',text:'未评'});
  return h('div',{style:'display:flex;flex-direction:column;gap:4px'},evs.map(e=>{
    const m=(e.open_loop&&e.open_loop.metrics)||{};const k=Object.keys(m)[0];
    return h('div',null,[h('a',{href:hashFor(e.id),text:e.protocol.name}),h('span',{class:'cell-sub num',text:k?k+' '+Number(m[k]).toFixed(4):'无指标'})])}))}
function dataCell(r){
  const d=r.data;
  return h('div',null,[
    h('div',null,[d.config_name||'—']),
    h('span',{class:'cell-sub num',text:d.train_episodes+' 集训练 / '+d.val_episodes+' 集验证'}),
    d.sources.length>1||(d.sources[0]&&d.sources[0].weight!=null)
      ?h('span',{class:'cell-sub',text:'配比 '+d.sources.map(s=>s.name+' '+(s.weight==null?'?':s.weight)).join(' : ')})
      :h('span',{class:'cell-sub',text:'单数据源，无配比'})])}
function paramCell(r){
  const p=r.hparams;
  return h('dl',{class:'kv'},[
    h('dt',{text:'步数'}),h('dd',{text:fmtNum(p.max_steps)}),
    h('dt',{text:'batch'}),h('dd',{text:fmtNum(p.global_batch)+(p.per_device_batch?'（'+p.per_device_batch+'×'+(p.grad_accum||1)+'×'+r.hardware.gpus+'）':'')}),
    h('dt',{text:'lr'}),h('dd',{text:fmtLr(p.learning_rate)+' · '+p.scheduler}),
    h('dt',{text:'训练'}),h('dd',{text:(p.trainable||[]).join(' ')||'—'})])}
function diffCell(r){
  if(!r.baseline_id)return h('span',{class:'idle',text:'未指定基线'});
  const d=diffsOf(r,DATA.runs);
  if(!d)return h('span',{class:'missing',text:'基线不存在'});
  if(!d.list.length)return h('span',{class:'idle',text:'与基线相同'});
  return h('div',{class:'diffs'},[h('span',{class:'cell-sub',text:'对 '+d.base.name}),...d.list.slice(0,4).map(([k,a,b])=>h('span',null,[k+' ',h('b',{text:a+' → '+b})])),d.list.length>4?h('span',{class:'idle',text:'另 '+(d.list.length-4)+' 项'}):null])}

function viewHome(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  const samples=DATA.runs.filter(r=>r._sample).length;
  if(samples)v.appendChild(h('div',{class:'banner',text:'本页含 '+samples+' 条“样本”记录，是用来测试导入器和页面的真实训练产物，数据是真的，但不是正式登记的实验。正式记录放在 records/ 下，样本放在 .local/。'}));
  const total=DATA.runs.length,real=DATA.runs.filter(r=>closedEvalsOf(r).length).length;
  v.appendChild(h('div',null,[h('h2',{text:'训练总览'}),
    h('p',{class:'sub',text:'一行一次训练。类型、基座、数据、参数、loss、开环、真机在同一行，点训练名进入详情，点评测进入评测页。共 '+total+' 次训练，其中 '+real+' 次有真机评测。'})]));
  const kinds=[...new Set(DATA.runs.map(r=>r.kind))],fams=[...new Set(DATA.runs.map(r=>r.family).filter(Boolean))],tasks=[...new Set(DATA.runs.flatMap(tasksOf))];
  const sel=(label,key,opts,map)=>{const s=h('select',{id:'f-'+key,onchange:e=>{flt[key]=e.target.value;renderTable()}},[h('option',{value:'',text:'全部'}),...opts.map(o=>h('option',{value:o,text:map?map(o):o}))]);s.value=flt[key];return h('label',{for:'f-'+key},[label,s])};
  v.appendChild(h('div',{class:'filters'},[
    h('label',{for:'f-q'},['搜索',h('input',{type:'search',id:'f-q',value:flt.q,placeholder:'名称、数据、基座',oninput:e=>{flt.q=e.target.value;renderTable()}})]),
    sel('类型','kind',kinds,k=>KIND[k]),sel('模型族','family',fams),sel('任务','task',tasks),
    sel('真机','real',['yes','no'],k=>k==='yes'?'已真机':'未真机')]));
  v.appendChild(h('div',{id:'tbl'}));
  return v}
function renderTable(){
  const box=$('#tbl');if(!box)return;box.textContent='';
  const rows=filtered();
  if(!rows.length){box.appendChild(h('div',{class:'empty',text:DATA.runs.length?'没有符合筛选条件的训练。':'还没有训练记录。用 tools/import_tau0_run.py 导入第一条。'}));return}
  const t=h('table',{class:'runs'});
  t.appendChild(h('thead',null,[h('tr',null,['训练','类型 / 基座','数据','关键参数','与基线的差','loss','开环','真机'].map(x=>h('th',{text:x})))]));
  const tb=h('tbody');
  rows.forEach(r=>{
    const flags=[];
    if(r._sample)flags.push(h('span',{class:'pill tag-sample',text:'样本'}));
    if(r.code.dirty)flags.push(h('span',{class:'pill warn',text:'代码未提交',title:'训练时工作区有未提交改动'}));
    tb.appendChild(h('tr',{class:'row'},[
      h('td',{class:'name'},[h('a',{href:hashFor(r.id),text:r.name}),
        h('span',{class:'cell-sub',text:r.owner?r.owner+' · '+dateOf(r.time&&r.time.start):dateOf(r.time&&r.time.start)}),
        h('span',{class:'cell-sub num',text:r.hardware.machine+' · '+r.hardware.gpus+' 卡'}),
        h('span',{class:'pill s-'+r.status,text:STATUS[r.status],style:'margin-top:4px;margin-right:4px'}),...flags]),
      h('td',null,[h('span',{class:'pill k-'+r.kind,text:KIND[r.kind]}),
        h('span',{class:'cell-sub',text:basename(r.base.model_path)}),
        r.kind_note?h('span',{class:'cell-sub',text:r.kind_note}):h('span',{class:'missing',text:'缺归类依据'})]),
      h('td',null,[dataCell(r)]),h('td',null,[paramCell(r)]),h('td',null,[diffCell(r)]),
      h('td',null,[spark(DATA.curves[r.id],110,34),h('span',{class:'cell-sub num',text:fmtLoss(r.loss.first)+' → '+fmtLoss(r.loss.final)}),
        r.loss.summary_bad?h('span',{class:'missing',text:'训练器汇总无效，已重算'}):null]),
      h('td',null,[openCell(r)]),h('td',null,[realCell(r)])]))});
  t.appendChild(tb);box.appendChild(h('div',{class:'scroll'},[t]))}

/* ================= 曲线图 ================= */
function lineChart(series,opts){
  // series: [{label,x:[],y:[],color}]，纵轴可选对数
  const W=480,H=210,m={l:50,r:12,t:22,b:28},ns='http://www.w3.org/2000/svg';
  const all=series.flatMap(s=>s.y.map((y,i)=>[s.x[i],y])).filter(p=>p[1]!=null&&isFinite(p[1])&&(!opts.log||p[1]>0));
  if(all.length<2)return h('div',{class:'empty',text:'没有可画的数据'});
  const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
  const x0=Math.min(...xs),x1=Math.max(...xs);
  let y0=Math.min(...ys),y1=Math.max(...ys);
  const T=opts.log?(v=>Math.log10(v)):(v=>v);
  if(!opts.log){const pad=(y1-y0)*0.06||1;y0=Math.min(0,y0);y1+=pad}
  const a=T(y0),b=T(y1),sy=v=>m.t+(H-m.t-m.b)*(1-(T(v)-a)/((b-a)||1)),sx=v=>m.l+(W-m.l-m.r)*(v-x0)/((x1-x0)||1);
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.setAttribute('role','img');svg.setAttribute('aria-label',opts.title||'曲线');
  const add=(tag,at,txt)=>{const e=document.createElementNS(ns,tag);for(const k in at)e.setAttribute(k,at[k]);if(txt!=null)e.textContent=txt;svg.appendChild(e);return e};
  // 纵轴刻度：对数轴取每个十倍，线性轴取 5 等分
  let ticks=[];
  if(opts.log){for(let e=Math.ceil(a);e<=Math.floor(b);e++)ticks.push(Math.pow(10,e));if(ticks.length<2){ticks=[y0,Math.sqrt(y0*y1),y1]}}
  else{for(let i=0;i<=4;i++)ticks.push(y0+(y1-y0)*i/4)}
  ticks.forEach(t=>{const y=sy(t);add('line',{x1:m.l,x2:W-m.r,y1:y,y2:y,stroke:'var(--line)','stroke-width':1});
    add('text',{x:m.l-6,y:y+4,'text-anchor':'end','font-size':12,fill:'var(--muted)','font-family':'var(--f-mono)'},t>=1000?(+(t/1000).toPrecision(3))+'k':(t<0.01?t.toExponential(0):+t.toPrecision(2)))});
  const xt=[];for(let i=0;i<=4;i++)xt.push(x0+(x1-x0)*i/4);
  xt.forEach(t=>add('text',{x:sx(t),y:H-10,'text-anchor':'middle','font-size':12,fill:'var(--muted)','font-family':'var(--f-mono)'},Math.round(t).toLocaleString('en-US')));
  add('text',{x:m.l,y:13,'font-size':12,fill:'var(--muted)'},opts.ylabel||'');
  series.forEach(s=>{
    const pts=s.x.map((x,i)=>[x,s.y[i]]).filter(p=>p[1]!=null&&isFinite(p[1])&&(!opts.log||p[1]>0));
    if(pts.length<2)return;
    add('path',{d:pts.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(''),fill:'none',stroke:s.color,'stroke-width':1.6,'stroke-linejoin':'round'});
    const l=pts[pts.length-1];add('circle',{cx:sx(l[0]),cy:sy(l[1]),r:3,fill:s.color})});
  return svg}
const PALETTE=['#1f4fd1','#b3261e','#1d7a3a','#9a5b00','#6a3fb5','#0b7a75'];
function colorFor(i){const dark=document.documentElement.getAttribute('data-theme')==='dark'||(!document.documentElement.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
  const D=['#7ea1ff','#ff8a80','#62d283','#ffb74d','#b79cff','#5fd3cd'];return (dark?D:PALETTE)[i%6]}
function chartBlock(title,series,opts){
  return h('div',{class:'chartbox'},[h('h3',{text:title}),lineChart(series,{...opts,title}),
    series.length>1?h('div',{class:'legend'},series.map(s=>h('span',null,[h('i',{style:'background:'+s.color}),s.label]))):null])}

/* ================= 训练详情 ================= */
function kvTable(rows){return h('table',{class:'t kvt'},[h('tbody',null,rows.map(([k,v])=>h('tr',null,[h('th',{text:k,style:'width:30%'}),h('td',{class:'n'},[typeof v==='string'||typeof v==='number'?document.createTextNode(String(v)):v])])))])}
function chainBar(r){
  const evC=closedEvalsOf(r),evO=openEvalsOf(r);
  const cell=(t,v,sec)=>h('a',{href:'#',onclick:e=>{e.preventDefault();jump(sec)}},[h('span',{class:'t',text:t}),h('span',{class:'v',text:v})]);
  const first=r.checkpoints.find(c=>c.sha256);
  return h('div',{class:'chain'},[
    cell('数据',r.data.config_name+' · '+r.data.train_episodes+' 集','sec-data'),
    cell('训练',KIND[r.kind]+' · '+fmtNum(r.hparams.max_steps)+' 步','sec-train'),
    cell('checkpoint',r.checkpoints.length+' 个'+(first?' · 已记哈希':' · 无哈希'),'sec-ckpt'),
    cell('部署',r.deployed_as.length?r.deployed_as.length+' 处':'未记录','sec-deploy'),
    cell('开环',evO.length?evO.length+' 条':'无','sec-open'),
    cell('真机',evC.length?evC.length+' 条':'无','sec-real')])}
function jump(id){const e=document.getElementById(id);if(e)e.scrollIntoView({behavior:'smooth',block:'start'})}

function viewRun(id){
  const r=RUNS[id];if(!r)return viewMissing('找不到训练 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(h('div',{class:'crumb'},[h('a',{href:'#',text:'← 全部训练'})]));
  v.appendChild(h('div',null,[h('h2',{text:r.name,style:'font-size:20px'}),
    h('div',{style:'display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center'},[
      h('span',{class:'pill k-'+r.kind,text:KIND[r.kind]}),h('span',{class:'pill s-'+r.status,text:STATUS[r.status]}),
      r._sample?h('span',{class:'pill tag-sample',text:'样本'}):null,
      h('span',{class:'sub num',text:r.id+' · '+(r.owner||'无负责人')+' · '+dateOf(r.time&&r.time.start)})]),
    r.kind_note?h('p',{class:'sub',style:'margin-top:6px',text:'归类：'+r.kind_note}):h('p',{class:'sub'},[missing('缺归类依据'),' 训练类型要人工确认，导入器只给建议（建议：'+KIND[r.kind_suggested||'']+'）。'])]));
  v.appendChild(chainBar(r));
  const warn=[];
  if(r.code.dirty)warn.push('训练时代码有未提交改动，不能靠 commit 复现这次训练。');
  if(r.code.git_hash==null)warn.push('没有 git 版本信息。');
  if(r.loss.summary_bad)warn.push('训练器写的汇总 loss 无效（续训后常见），页面上的 loss 是从逐步日志重算的。');
  (r.notes||'').split(/(?=\[导入器)/).filter(Boolean).forEach(n=>warn.push(n.trim()));
  if(warn.length)v.appendChild(h('div',{class:'banner'},warn.map(w=>h('div',{text:w}))));

  // 数据
  v.appendChild(h('section',{id:'sec-data',class:'sec'},[h('h2',{text:'数据'}),h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['数据配置',r.data.config_name||'—'],['训练集数',fmtNum(r.data.train_episodes)],['验证集数',fmtNum(r.data.val_episodes)],
      ['保留帧数',r.data.train_frames_kept?fmtNum(r.data.train_frames_kept):'未记录'],
      ['划分文件',r.data.split_file||'未记录'],['划分哈希',r.data.split_sha256?h('span',{class:'hash',text:r.data.split_sha256}):'未记录']])]),
    h('div',{class:'panel'},[h('h3',{text:'数据源与配比'}),h('table',{class:'t'},[h('thead',null,[h('tr',null,['来源','集数','权重','成败'].map(x=>h('th',{text:x})))]),
      h('tbody',null,r.data.sources.map(s=>h('tr',null,[h('td',{text:s.name}),h('td',{class:'n',text:fmtNum(s.episodes)}),h('td',{class:'n',text:s.weight==null?'未记录':s.weight}),h('td',{text:s.kind||'—'})])))]),
      h('p',{class:'sub',style:'margin-top:8px',text:'权重为“未记录”表示导入器读不到，单数据源时可忽略；混合数据要手填。'})])])]));

  // 训练
  const p=r.hparams;
  v.appendChild(h('section',{id:'sec-train',class:'sec'},[h('h2',{text:'训练'}),h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['基座',h('span',{class:'hash',text:r.base.model_path})],['步数',fmtNum(p.max_steps)],
      ['全局 batch',fmtNum(p.global_batch)+(p.per_device_batch?'（每卡 '+p.per_device_batch+' × 累积 '+(p.grad_accum||1)+' × '+r.hardware.gpus+' 卡）':'')],
      ['学习率',p.learning_rate+' · '+p.scheduler+' · 预热 '+p.warmup_steps],['权重衰减',p.weight_decay==null?'—':p.weight_decay],['种子',p.seed==null?'—':p.seed],
      ['可训练部分',(p.trainable||[]).join('、')||'—'],...Object.entries(p.extra||{}).map(([k,x])=>[k,String(x)])])]),
    h('div',{class:'panel'},[kvTable([['机器',r.hardware.machine],['卡数',r.hardware.gpus+(r.hardware.gpu_type?' × '+r.hardware.gpu_type:'')],
      ['GPU 序号',(r.hardware.gpu_indices||[]).length?r.hardware.gpu_indices.join(', '):'未记录'],
      ['开始',r.time&&r.time.start||'—'],['结束',r.time&&r.time.end||'—'],
      ['代码',(r.code.git_hash?r.code.git_hash.slice(0,8):'无版本')+(r.code.dirty?'（有未提交改动）':'')],
      ['命令',h('span',{class:'hash',text:r.code.command||'未记录'})],['配置哈希',h('span',{class:'hash',text:r.code.config_yaml_sha256||'未记录'})]])])]),
    (()=>{const c=DATA.curves[r.id];if(!c)return h('div',{class:'empty',text:'没有曲线文件'});
      const col=colorFor(0);
      return h('div',{class:'charts'},[
        chartBlock('loss（对数轴）',[{label:'loss',x:c.step,y:c.loss,color:col}],{log:true,ylabel:'loss'}),
        chartBlock('梯度范数',[{label:'grad_norm',x:c.step,y:c.grad_norm,color:colorFor(1)}],{log:true,ylabel:'grad_norm'}),
        chartBlock('学习率',[{label:'lr',x:c.step,y:c.learning_rate,color:colorFor(2)}],{log:false,ylabel:'learning rate'}),
        chartBlock('步耗时（ms）',[{label:'step_time_ms',x:c.step,y:c.step_time_ms,color:colorFor(3)}],{log:false,ylabel:'ms'})])})(),
    h('p',{class:'sub',text:'曲线降采样到约 240 点；loss 末值取最后 '+(r.loss.final_window||'?')+' 个记录点的均值。'})]));

  // checkpoint
  v.appendChild(h('section',{id:'sec-ckpt',class:'sec'},[h('h2',{text:'checkpoint'}),h('div',{class:'panel scroll',style:'border:1px solid var(--line)'},[
    h('table',{class:'t'},[h('thead',null,[h('tr',null,['步数','路径','sha256','HF 位置','保留'].map(x=>h('th',{text:x})))]),
      h('tbody',null,r.checkpoints.length?r.checkpoints.map(c=>h('tr',null,[h('td',{class:'n',text:fmtNum(c.step)}),h('td',{class:'hash',text:c.path}),
        h('td',null,[c.sha256?h('span',{class:'hash',text:c.sha256}):missing('无哈希')]),
        h('td',null,[c.hf_repo?h('span',{class:'hash',text:c.hf_repo+(c.hf_revision?'@'+c.hf_revision.slice(0,8):'')}):h('span',{class:'idle',text:'未上传'})]),
        h('td',{text:c.kept===false?'已删':'是'})])):[h('tr',null,[h('td',{colspan:5,text:'没有记录 checkpoint'})])])])])]));

  // 部署
  v.appendChild(h('section',{id:'sec-deploy',class:'sec'},[h('h2',{text:'部署'}),
    r.deployed_as.length?h('div',{class:'panel scroll'},[h('table',{class:'t'},[h('thead',null,[h('tr',null,['步数','机器','端口','服务启动','备注'].map(x=>h('th',{text:x})))]),
      h('tbody',null,r.deployed_as.map(d=>h('tr',null,[h('td',{class:'n',text:fmtNum(d.checkpoint_step)}),h('td',{text:d.host}),h('td',{class:'n',text:d.port}),h('td',{class:'n',text:d.service_started||'未记录'}),h('td',{text:d.note||''})])))])]):
    h('div',{class:'empty',text:'没有部署记录。只靠端口号分不出某次真机跑的是哪个模型，真机评测要记权重哈希和服务启动时间。'})]));

  // 开环
  v.appendChild(h('section',{id:'sec-open',class:'sec'},[h('h2',{text:'开环评测'}),
    openEvalsOf(r).length?h('div',{class:'panel'},[evalList(openEvalsOf(r))]):h('div',{class:'empty',text:'还没有开环评测。协议草案见 protocols/openloop-fold-v0.md，数值待组内确认。'})]));
  // 真机
  v.appendChild(h('section',{id:'sec-real',class:'sec'},[h('h2',{text:'真机评测'}),
    closedEvalsOf(r).length?h('div',{class:'panel'},[evalList(closedEvalsOf(r))]):h('div',{class:'empty',text:'还没有真机评测。'})]));
  return v}
function evalList(list){return h('table',{class:'t'},[h('thead',null,[h('tr',null,['日期','任务','协议','结果','视频'].map(x=>h('th',{text:x})))]),
  h('tbody',null,list.map(e=>{const s=e.type==='closed_loop'?trialStats(e):null;
    return h('tr',null,[h('td',{class:'n',text:e.date}),h('td',null,[h('a',{href:hashFor(e.id),text:e.task})]),h('td',{class:'n',text:e.protocol.name+' '+e.protocol.version}),
      h('td',{class:'n',text:s?(s.labeled?s.success+'/'+s.labeled+' 成功':s.n+' 次未标注'):(Object.entries((e.open_loop&&e.open_loop.metrics)||{}).slice(0,2).map(([k,x])=>k+' '+Number(x).toFixed(4)).join('，')||'无指标')}),
      h('td',{text:s?(s.videos?s.videos+' 段':'无'):'—'})])}))])}
function viewMissing(t){return h('div',null,[h('div',{class:'crumb'},[h('a',{href:'#',text:'← 全部训练'})]),h('div',{class:'empty',text:t})])}

/* ================= 评测详情 ================= */
function videoTile(t,v,ev){
  if(!v)return h('figure',null,[h('div',{class:'vph',text:'还没有视频。把视频上传到 HF 数据集后，在记录里填 url。'}),h('figcaption',{text:'第 '+t.n+' 次 · '+RESULT[t.result]})]);
  const vid=h('video',{controls:true,preload:'none',playsinline:true,src:v.url,'aria-label':'第 '+t.n+' 次 '+v.camera});
  return h('figure',null,[vid,h('figcaption',null,['第 '+t.n+' 次 · '+v.camera+' · '+RESULT[t.result]+' · ',h('a',{href:v.url,target:'_blank',rel:'noreferrer',text:'打开原文件'})])])}
function deployTable(e){
  const d=e.deployment||{},c=d.client||{},s=d.server||{};
  const rows=[['机器',d.robot||'—'],['驱动',d.driver||'—'],['服务端口',s.port==null?'—':String(s.port)],['服务协议',s.protocol||'—'],
    ['服务启动',s.started||missing('缺，无法排除端口换了别的模型')],['模型目录',s.model_dir?h('span',{class:'hash',text:s.model_dir}):'—'],
    ['客户端',(c.name||'—')+(c.git_hash?' @'+c.git_hash.slice(0,8):'')],['控制循环（实测）',c.loop_hz==null?'—':c.loop_hz+' Hz']];
  Object.entries(c.args||{}).forEach(([k,x])=>{if(x!=null)rows.push(['参数 '+k,String(x)])});
  return kvTable(rows)}
function viewEval(id){
  const e=EVALS[id];if(!e)return viewMissing('找不到评测 '+id);
  const run=RUNS[e.run_id];
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(h('div',{class:'crumb'},[h('a',{href:'#',text:'← 全部训练'}),run?h('span',null,[' / ',h('a',{href:hashFor(run.id),text:run.name})]):null]));
  v.appendChild(h('div',null,[h('h2',{text:e.task+' · '+(e.type==='closed_loop'?'真机评测':'开环评测'),style:'font-size:20px'}),
    h('div',{style:'display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center'},[e._sample?h('span',{class:'pill tag-sample',text:'样本'}):null,
      h('span',{class:'sub num',text:e.id+' · '+e.date+' · checkpoint '+fmtNum(e.checkpoint.step)+' · 协议 '+e.protocol.name+' '+e.protocol.version})])]));
  const hashOK=run&&(run.checkpoints.find(c=>c.step===e.checkpoint.step)||{}).sha256;
  v.appendChild(h('div',{class:'panel'},[kvTable([['训练',run?h('a',{href:hashFor(run.id),text:run.name}):e.run_id],
    ['权重哈希（实际加载）',e.checkpoint.sha256?h('span',{class:'hash',text:e.checkpoint.sha256}):missing('缺，无法确认跑的是哪个权重')],
    ['与训练记录一致',e.checkpoint.sha256&&hashOK?(e.checkpoint.sha256===hashOK?h('span',{class:'ok',text:'一致'}):h('span',{class:'bad',text:'不一致'})):'无法核对'],
    ['提示词',e.instruction||'—']])]));
  if(e.type==='closed_loop'){
    const s=trialStats(e);
    const w=[];
    if(!s.enough)w.push('只有 '+s.n+' 次试验，少于要求的 '+((e.closed_loop&&e.closed_loop.min_trials_required)||3)+' 次，不能当结论。');
    if(!e.prediction||!e.prediction.text)w.push('没有跑之前写的预测。');else if(e.prediction.after_the_fact)w.push('预测是事后补写的。');
    if(s.unlabeled)w.push(s.unlabeled+' 次试验还没人标注成败。');
    if(w.length)v.appendChild(h('div',{class:'banner'},w.map(x=>h('div',{text:x}))));
    v.appendChild(h('section',{class:'sec'},[h('h2',{text:'部署配置'}),h('p',{class:'sub',text:'同一个 checkpoint 换了下面任何一项，就是另一个系统。'}),h('div',{class:'panel'},[deployTable(e)])]));
    if(e.prediction&&e.prediction.text)v.appendChild(h('section',{class:'sec'},[h('h2',{text:'跑之前的预测'}),h('div',{class:'panel'},[h('p',{text:e.prediction.text}),e.prediction.falsified_if?h('p',{class:'sub',text:'判否条件：'+e.prediction.falsified_if}):null])]));
    v.appendChild(h('section',{class:'sec'},[h('h2',{text:'逐次试验'}),h('div',{class:'panel scroll'},[h('table',{class:'t'},[
      h('thead',null,[h('tr',null,['#','结果','时长(s)','失败方式','标注人','运行目录','备注'].map(x=>h('th',{text:x})))]),
      h('tbody',null,e.closed_loop.trials.map(t=>h('tr',null,[h('td',{class:'n',text:t.n}),h('td',null,[h('span',{class:t.result==='success'?'ok':t.result==='unlabeled'?'warn':'',text:RESULT[t.result]})]),
        h('td',{class:'n',text:t.duration_s==null?'—':t.duration_s}),h('td',{text:t.failure_mode||''}),h('td',{text:t.label_by||''}),h('td',{class:'hash',text:t.run_dir||''}),h('td',{text:t.note||''})])))])])]));
    const tiles=[];e.closed_loop.trials.forEach(t=>{if((t.videos||[]).length)t.videos.forEach(x=>tiles.push(videoTile(t,x,e)));else tiles.push(videoTile(t,null,e))});
    v.appendChild(h('section',{class:'sec'},[h('h2',{text:'视频'}),h('div',{class:'videos'},tiles)]));
  }else{
    const m=(e.open_loop&&e.open_loop.metrics)||{};
    v.appendChild(h('section',{class:'sec'},[h('h2',{text:'开环指标'}),h('div',{class:'panel'},[Object.keys(m).length?kvTable(Object.entries(m).map(([k,x])=>[k,Number(x).toFixed(5)])):h('p',{class:'sub',text:'没有指标'}),
      h('p',{class:'sub',text:'请求数 '+((e.open_loop&&e.open_loop.num_requests)||'—')+'，种子数 '+((e.open_loop&&e.open_loop.num_seeds)||'—')})])]));
  }
  if(e.conclusion)v.appendChild(h('section',{class:'sec'},[h('h2',{text:'结论'}),h('div',{class:'panel'},[h('p',{text:e.conclusion})])]));
  return v}

/* ================= 对比：沿链逐环显示不同 ================= */
const cmpSel=new Set();
function viewCompare(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  v.appendChild(h('div',null,[h('h2',{text:'对比'}),h('p',{class:'sub',text:'选 2 条以上训练，按 数据 → 训练 → checkpoint 逐环对齐，取值不同的行高亮。真机结果变好或变差时，先看哪一环变了。'})]));
  const list=h('div',{class:'panel',style:'display:flex;flex-wrap:wrap;gap:8px 18px'},DATA.runs.map(r=>{
    const cb=h('input',{type:'checkbox',id:'c-'+r.id,onchange:ev=>{ev.target.checked?cmpSel.add(r.id):cmpSel.delete(r.id);renderCmp()}});cb.checked=cmpSel.has(r.id);
    return h('label',{for:'c-'+r.id,style:'display:flex;gap:6px;align-items:center;font-size:13px'},[cb,r.name])}));
  v.appendChild(list);v.appendChild(h('div',{id:'cmpbox'}));return v}
function renderCmp(){
  const box=$('#cmpbox');if(!box)return;box.textContent='';
  const items=[...cmpSel].map(i=>RUNS[i]).filter(Boolean);
  if(items.length<2){box.appendChild(h('div',{class:'empty',text:'选 2 条以上。'}));return}
  const groups=[
    ['数据',[['数据配置',r=>r.data.config_name],['训练集数',r=>r.data.train_episodes],['验证集数',r=>r.data.val_episodes],['保留帧数',r=>r.data.train_frames_kept||'未记录'],['配比',r=>r.data.sources.map(s=>s.name+':'+(s.weight==null?'?':s.weight)).join(' ')]]],
    ['训练',[['类型',r=>KIND[r.kind]],['基座',r=>basename(r.base.model_path)],['步数',r=>r.hparams.max_steps],['全局 batch',r=>r.hparams.global_batch],['学习率',r=>r.hparams.learning_rate],['调度',r=>r.hparams.scheduler],['预热',r=>r.hparams.warmup_steps],['种子',r=>r.hparams.seed==null?'—':r.hparams.seed],['可训练',r=>(r.hparams.trainable||[]).join('+')],
      ...[...new Set(items.flatMap(r=>Object.keys(r.hparams.extra||{})))].map(k=>[k,r=>(r.hparams.extra||{})[k]==null?'—':String((r.hparams.extra||{})[k])]),
      ['机器',r=>r.hardware.machine+' × '+r.hardware.gpus],['代码',r=>(r.code.git_hash?r.code.git_hash.slice(0,8):'无版本')+(r.code.dirty?' 未提交':'')]]],
    ['checkpoint',[['权重哈希（末）',r=>{const c=r.checkpoints.filter(x=>x.sha256).slice(-1)[0];return c?c.sha256.slice(0,12)+'…':'无'}]]],
    ['结果',[['末 loss',r=>fmtLoss(r.loss.final)],['真机',r=>{const e=closedEvalsOf(r);if(!e.length)return '未真机';return e.map(x=>{const s=trialStats(x);return x.task+' '+(s.labeled?s.success+'/'+s.labeled:'未标注 '+s.n+' 次')}).join('；')}]]]];
  const t=h('table',{class:'cmp'});
  t.appendChild(h('thead',null,[h('tr',null,[h('th',{text:'项'}),...items.map(r=>h('th',{text:r.name}))])]));
  const tb=h('tbody');
  groups.forEach(([g,rows])=>{tb.appendChild(h('tr',{class:'grp'},[h('td',{colspan:items.length+1,text:g})]));
    rows.forEach(([k,f])=>{const vals=items.map(r=>String(f(r)));tb.appendChild(h('tr',{class:new Set(vals).size>1?'diff':''},[h('td',{class:'k',text:k}),...vals.map(x=>h('td',{class:'n',text:x}))]))})});
  t.appendChild(tb);box.appendChild(h('div',{class:'scroll'},[t]))}

/* ================= 启动 ================= */
function render(){
  const r=route(),main=$('#main');main.textContent='';
  renderTabs(r.v==='compare'?'compare':'home');
  let view;
  if(r.v==='run')view=viewRun(r.id);else if(r.v==='eval')view=viewEval(r.id);else if(r.v==='compare')view=viewCompare();else view=viewHome();
  main.appendChild(view);
  if(r.v==='home')renderTable();if(r.v==='compare')renderCmp();
  window.scrollTo(0,0)}
$('#built').textContent='构建 '+DATA.built+' · '+DATA.commit;
window.addEventListener('hashchange',render);
render();
