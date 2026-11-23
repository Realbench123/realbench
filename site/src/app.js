// RealBench 页面脚本。数据来自 DATA（构建时内嵌）。
const $=(s,r)=>(r||document).querySelector(s);
function h(tag,attrs,kids){const e=document.createElement(tag);
  if(attrs)for(const k in attrs){const v=attrs[k];if(v==null||v===false)continue;
    if(k==='class')e.className=v;else if(k==='text')e.textContent=v;else if(k==='html')e.innerHTML=v;
    else if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v===true?'':v)}
  (kids||[]).forEach(c=>{if(c==null||c===false)return;e.appendChild(typeof c==='string'?document.createTextNode(c):c)});return e}
const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};

const RUNS={},EVALS={},RECIPES={};
DATA.runs.forEach(r=>RUNS[r.id]=r);DATA.evals.forEach(e=>EVALS[e.id]=e);(DATA.recipes||[]).forEach(c=>RECIPES[c.id]=c);
const KIND={scratch:'从头训',pretrain:'预训练',finetune:'微调',posttrain:'后训练'};
const STATUS={finished:'已完成',running:'进行中',failed:'失败',stopped:'已停止',superseded:'已被取代'};
const RESULT={success:'成功',fail:'失败',partial:'部分',invalid:'无效',unlabeled:'未标注'};

function fmtLr(x){return x==null?'未记录':x.toExponential(0).replace('e-0','e-').replace('e+0','e')}
function fmtNum(x,d){return x==null?'—':Number(x).toLocaleString('en-US',{maximumFractionDigits:d==null?0:d})}
function fmtLoss(x){return x==null?'—':x.toFixed(x<0.1?4:3)}
function dateOf(iso){return iso?iso.slice(0,10):'—'}
function basename(p){return (p||'').split('/').filter(Boolean).pop()||p||''}
function missing(t){return h('span',{class:'missing',text:t||'缺'})}

const tagOf=x=>x._sample?'样本':x._pending?'待登记':null;
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

/* ---------- 管理模式：只有从本机管理服务打开时才有删除键 ---------- */
const ADMIN={on:false,token:null};
async function detectAdmin(){
  if(!/^https?:$/.test(location.protocol))return;
  try{const r=await fetch('/api/ping',{cache:'no-store'});if(!r.ok)return;const j=await r.json();
    if(j&&j.admin&&j.token){ADMIN.on=true;ADMIN.token=j.token;document.body.classList.add('admin');render()}}catch(e){}}
async function adminCall(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Admin-Token':ADMIN.token},body:JSON.stringify(body||{})});
  const j=await r.json().catch(()=>({error:'服务返回的不是 JSON'}));
  if(!r.ok||j.ok===false)throw new Error(j.error||('HTTP '+r.status));return j}
function confirmDelete(id){
  adminCall('/api/plan',{id}).then(pl=>{
    const d=document.createElement('dialog');
    const kind={run:'训练',eval:'评测',recipe:'配方'}[pl.type];
    const kids=[h('h3',{text:'删除这条'+kind+'？'}),h('p',{text:pl.name+'（'+pl.id+'）'})];
    if(pl.blockers.length){
      kids.push(h('div',{class:'warnline'},[h('b',{text:'现在不能删：'}),h('ul',null,pl.blockers.map(b=>h('li',{text:b})))]));
      kids.push(h('div',{class:'row'},[h('button',{class:'btn',text:'知道了',onclick:()=>d.close()})]))}
    else{
      kids.push(h('p',{class:'sub',text:'会一起删除：'}));
      kids.push(h('ul',null,pl.delete.map(x=>h('li',{text:{run:'训练',eval:'评测',recipe:'配方'}[x.type]+' · '+x.name+(x.public?'':'（只在本机的样本）')}))));
      if(pl.modify.length){kids.push(h('p',{class:'sub',text:'同时改动：'}));kids.push(h('ul',null,pl.modify.map(m=>h('li',{text:m.what+'（'+m.id+'）'}))))}
      if(pl.public)kids.push(h('div',{class:'warnline',text:'这条记录已经在公开仓库里。删除只影响本机文件和之后的发布，git 历史里仍然留着旧内容，已经发布的页面也要等你重新发布才会消失。'}));
      const inp=h('input',{type:'text',id:'del-reason',placeholder:'原因（可选，会写进公开的删除清单，别写敏感信息）',maxlength:'120'});
      kids.push(inp);
      const err=h('div',{class:'warnline',style:'display:none'});kids.push(err);
      kids.push(h('div',{class:'row'},[h('button',{class:'btn',text:'取消',onclick:()=>d.close()}),
        h('button',{class:'btn danger',text:'确认删除',onclick:async e=>{e.target.disabled=true;
          try{const r=await adminCall('/api/delete',{id,reason:inp.value});d.close();
            alert_line('已删除 '+r.deleted.join('、')+'。文件在回收站 '+r.trash+'，可恢复。'+(r.validate_ok===false?' 注意：删除后校验未通过。':''));
            setTimeout(()=>location.assign(location.pathname+'#'),0);setTimeout(()=>location.reload(),150)}
          catch(x){err.style.display='';err.textContent=x.message;e.target.disabled=false}}})]))}
    d.appendChild(h('div',null,kids));document.body.appendChild(d);d.addEventListener('close',()=>d.remove());d.showModal();
  }).catch(e=>alert_line('无法生成删除计划：'+e.message))}
function alert_line(t){const b=h('div',{class:'banner',text:t,style:'position:fixed;left:16px;right:16px;bottom:16px;z-index:50;max-width:720px;margin:auto'});document.body.appendChild(b);setTimeout(()=>b.remove(),6000)}
function delBtn(id,label){return h('button',{class:'btn danger sm adm',type:'button',text:label||'删除',onclick:e=>{e.preventDefault();e.stopPropagation();confirmDelete(id)}})}
function adminBar(){return h('div',{class:'adminbar'},[h('b',{text:'管理模式'}),h('span',{class:'sub',text:'这里能删除记录，删除的文件进回收站，可以恢复。公开页面没有这些按钮。'}),
  h('button',{class:'btn sm',type:'button',text:'回收站',onclick:showTrash})])}
async function showTrash(){
  const j=await adminCall('/api/trash',{});const d=document.createElement('dialog');
  d.appendChild(h('div',null,[h('h3',{text:'回收站'}),j.items.length?h('ul',null,j.items.map(it=>h('li',null,[(it.name||it.id)+' · '+it.at.slice(0,16)+' ',
    h('button',{class:'btn sm',type:'button',text:'恢复',onclick:async()=>{try{await adminCall('/api/restore',{id:it.id});d.close();location.reload()}catch(e){alert_line(e.message)}}})]))):h('p',{class:'sub',text:'空的。'}),
    h('div',{class:'row'},[h('button',{class:'btn',text:'关闭',onclick:()=>d.close()})])]));
  document.body.appendChild(d);d.addEventListener('close',()=>d.remove());d.showModal()}

/* ---------- 路由 ---------- */
function route(){const x=location.hash.slice(1);
  if(x==='recipes')return {v:'recipes'};
  if(x==='real')return {v:'real'};
  if(x==='datasets')return {v:'datasets'};
  if(x==='plans')return {v:'plans'};
  if(x==='stacks')return {v:'stacks'};
  if(x==='groups')return {v:'groups'};
  if(x.startsWith('d-'))return {v:'dataset',id:'ds-'+x.slice(2)};
  if(x.startsWith('p-'))return {v:'plan',id:'cp-'+x.slice(2)};
  if(x.startsWith('s-'))return {v:'stack',id:'st-'+x.slice(2)};
  if(x.startsWith('x-'))return {v:'group',id:'xg-'+x.slice(2)};
  if(x==='train')return {v:'home',group:'ft'};
  if(x==='pretrain')return {v:'home',group:'pre'};
  if(x.startsWith('c-'))return {v:'recipe',id:'rc-'+x.slice(2)};
  if(x.startsWith('r-'))return {v:'run',id:'tr-'+x.slice(2)};
  if(x.startsWith('e-'))return {v:'eval',id:'ev-'+x.slice(2)};
  if(x==='compare')return {v:'compare'};
  return {v:'home'}}
function hashFor(id){return id.startsWith('tr-')?'#r-'+id.slice(3):id.startsWith('rc-')?'#c-'+id.slice(3):'#e-'+id.slice(3)}
function renderTabs(cur){const t=$('#tabs');t.textContent='';
  const n=g=>DATA.runs.filter(r=>inGroup(r,g)).length;
  const L=[['home','总览（'+n('')+'）','#'],['ft','微调 / 后训练（'+n('ft')+'）','#train'],['pre','预训练 / 从头训（'+n('pre')+'）','#pretrain'],['real','真机测试（'+realEvals().length+'）','#real'],['groups','实验组（'+(DATA.groups||[]).length+'）','#groups'],
    ['datasets','数据集（'+(DATA.datasets||[]).length+'）','#datasets'],['recipes','数据配方（'+(DATA.recipes||[]).length+'）','#recipes'],['plans','采集方案（'+(DATA.plans||[]).length+'）','#plans'],['stacks','训练栈（'+(DATA.stacks||[]).length+'）','#stacks'],['compare','对比','#compare']];
  L.forEach(([k,l,hr])=>t.appendChild(h('a',{href:hr,text:l,'aria-current':cur===k?'page':null})))}

/* ================= 总览：一行一次训练 ================= */
const flt={q:'',kind:'',family:'',task:'',real:'',group:''};
const GROUP={ft:['finetune','posttrain'],pre:['pretrain','scratch']};
function inGroup(r,g){return !g||GROUP[g].includes(r.kind)}
function tasksOf(r){return [...new Set(closedEvalsOf(r).concat(openEvalsOf(r)).map(e=>e.task))]}
function filtered(){return DATA.runs.filter(r=>{
  if(!inGroup(r,flt.group))return false;
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
    const vid=s.videos?h('a',{href:hashFor(e.id),text:s.videos+' 段视频'}):h('span',{class:'missing',text:'无视频'});
    box.appendChild(h('div',null,[head,rate,h('span',{class:'cell-sub'},[vid])]))});
  return box}
function openCell(r){
  const evs=openEvalsOf(r);
  if(!evs.length)return h('span',{class:'idle',text:'未评'});
  return h('div',{style:'display:flex;flex-direction:column;gap:4px'},evs.map(e=>{
    const m=(e.open_loop&&e.open_loop.metrics)||{};const k=Object.keys(m)[0];
    return h('div',null,[h('a',{href:hashFor(e.id),text:e.protocol.name}),h('span',{class:'cell-sub num',text:k?k+' '+Number(m[k]).toFixed(4):'无指标'})])}))}
function dataCell(r){
  const d=r.data,rc=r.recipe_id&&RECIPES[r.recipe_id];
  return h('div',null,[
    rc?h('a',{href:hashFor(rc.id),text:rc.name}):h('div',null,[d.config_name||'—',h('span',{class:'missing',text:' 无配方',style:'margin-left:4px'})]),
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

function viewHome(group){
  group=group||'';
  if(flt.group!==group){flt.kind='';flt.group=group}
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  const pend=DATA.runs.filter(r=>r._pending||r._sample).length;
  if(pend)v.appendChild(h('div',{class:'banner',text:'有 '+pend+' 条记录还没正式登记（在本机 .local/ 里，标「待登记」或「样本」）。确认后移进 records/，才会出现在公开页面。'}));
  const pool=DATA.runs.filter(r=>inGroup(r,group));
  const real=pool.filter(r=>closedEvalsOf(r).length).length;
  const T={'':['训练总览','微调、后训练、预训练都在这一张表里。'],ft:['微调 / 后训练','从预训练基座出发，在目标任务数据上训练；后训练指微调之后再加额外目标（RTC、RL、偏好等）。'],pre:['预训练 / 从头训','大规模混合数据上的预训练，以及随机初始化的从头训练。类型靠人工确认，导入器只给建议。']}[group];
  v.appendChild(h('div',null,[h('h2',{text:T[0]}),h('p',{class:'sub'},[T[1]+' 一行一次训练，点训练名进入详情，点曲线可以放大。共 '+pool.length+' 次训练，其中 '+real+' 次有真机评测。',h('a',{href:'#real',text:' 看全部真机测试 →'})])]));
  const kinds=[...new Set(pool.map(r=>r.kind))],fams=[...new Set(pool.map(r=>r.family).filter(Boolean))],tasks=[...new Set(pool.flatMap(tasksOf))];
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
  if(!rows.length){const g=DATA.runs.filter(r=>inGroup(r,flt.group)).length;box.appendChild(h('div',{class:'empty',text:g?'没有符合筛选条件的训练。':(flt.group==='pre'?'还没有预训练或从头训的记录。这类记录要人工确认类型后才会登记；有了会出现在这里，也会出现在总览里。':'还没有训练记录。用 tools/ 下的导入器导入第一条。')}));return}
  const t=h('table',{class:'runs'});
  t.appendChild(h('thead',null,[h('tr',null,['训练','类型 / 基座','数据','关键参数','与基线的差','loss','开环','真机'].map(x=>h('th',{text:x})))]));
  const tb=h('tbody');
  rows.forEach(r=>{
    const flags=[];
    if(tagOf(r))flags.push(h('span',{class:'pill tag-sample',text:tagOf(r)}));
    if(r.code.dirty)flags.push(h('span',{class:'pill warn',text:'代码未提交',title:'训练时工作区有未提交改动'}));
    tb.appendChild(h('tr',{class:'row'},[
      h('td',{class:'name'},[h('a',{href:hashFor(r.id),text:r.name}),delBtn(r.id),
        h('span',{class:'cell-sub',text:r.owner?r.owner+' · '+dateOf(r.time&&r.time.start):dateOf(r.time&&r.time.start)}),
        h('span',{class:'cell-sub num',text:r.hardware.machine+' · '+r.hardware.gpus+' 卡'}),
        h('span',{class:'pill s-'+r.status,text:STATUS[r.status],style:'margin-top:4px;margin-right:4px'}),...flags]),
      h('td',null,[h('span',{class:'pill k-'+r.kind,text:KIND[r.kind]}),
        h('span',{class:'cell-sub',text:basename(r.base.model_path)}),
        r.kind_note?h('span',{class:'cell-sub',text:r.kind_note}):h('span',{class:'missing',text:'缺归类依据'})]),
      h('td',null,[dataCell(r)]),h('td',null,[paramCell(r)]),h('td',null,[diffCell(r)]),
      h('td',null,[sparkBtn(r),h('span',{class:'cell-sub num',text:fmtLoss(r.loss.first)+' → '+fmtLoss(r.loss.final)}),
        r.loss.summary_bad?h('span',{class:'missing',text:'训练器汇总无效，已重算'}):null]),
      h('td',null,[openCell(r)]),h('td',null,[realCell(r)])]))});
  t.appendChild(tb);box.appendChild(h('div',{class:'scroll'},[t]))}

/* ================= 曲线图 ================= */
function lineChart(series,opts){
  // series: [{label,x:[],y:[],color}]，纵轴可选对数
  const FS=opts.big?14:12;const W=opts.big?1120:480,H=opts.big?520:210,m=opts.big?{l:70,r:40,t:34,b:42}:{l:50,r:26,t:22,b:28},ns='http://www.w3.org/2000/svg';
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
    add('text',{x:m.l-6,y:y+4,'text-anchor':'end','font-size':FS,fill:'var(--muted)','font-family':'var(--f-mono)'},t>=1000?(+(t/1000).toPrecision(3))+'k':(t<0.01?t.toExponential(0):+t.toPrecision(2)))});
  const xt=[];for(let i=0;i<=4;i++)xt.push(x0+(x1-x0)*i/4);
  xt.forEach(t=>add('text',{x:sx(t),y:H-10,'text-anchor':'middle','font-size':FS,fill:'var(--muted)','font-family':'var(--f-mono)'},Math.round(t).toLocaleString('en-US')));
  add('text',{x:m.l,y:13,'font-size':FS,fill:'var(--muted)'},opts.ylabel||'');
  series.forEach(s=>{
    const pts=s.x.map((x,i)=>[x,s.y[i]]).filter(p=>p[1]!=null&&isFinite(p[1])&&(!opts.log||p[1]>0));
    if(pts.length<2)return;
    add('path',{d:pts.map((p,i)=>(i?'L':'M')+sx(p[0]).toFixed(1)+' '+sy(p[1]).toFixed(1)).join(''),fill:'none',stroke:s.color,'stroke-width':opts.big?2.2:1.6,'stroke-linejoin':'round'});
    const l=pts[pts.length-1];add('circle',{cx:sx(l[0]),cy:sy(l[1]),r:opts.big?4.5:3,fill:s.color})});
  svg.__meta={sx,sy,m,W,H,x0,x1,series,log:!!opts.log};
  return svg}
const PALETTE=['#1f4fd1','#b3261e','#1d7a3a','#9a5b00','#6a3fb5','#0b7a75'];
function colorFor(i){const dark=document.documentElement.getAttribute('data-theme')==='dark'||(!document.documentElement.getAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
  const D=['#7ea1ff','#ff8a80','#62d283','#ffb74d','#b79cff','#5fd3cd'];return (dark?D:PALETTE)[i%6]}
function chartBlock(title,series,opts){
  const open=()=>openZoom(title,series,opts);
  return h('div',{class:'chartbox zoomable',tabindex:'0',role:'button','aria-label':title+'，点击放大',onclick:open,
    onkeydown:e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open()}}},[
    h('span',{class:'zhint',text:'点击放大'}),h('h3',{text:title}),lineChart(series,{...opts,title}),
    series.length>1?h('div',{class:'legend'},series.map(s=>h('span',null,[h('i',{style:'background:'+s.color}),s.label]))):null])}
function fmtVal(v){const a=Math.abs(v);return a!==0&&(a<0.001||a>=10000)?v.toExponential(3):String(+v.toPrecision(5))}
function openZoom(title,series,opts){
  const d=document.createElement('dialog');d.className='zoom';
  const svg=lineChart(series,{...opts,title,big:true});
  const info=h('div',{class:'zinfo mono',text:'移动鼠标查看每个记录点的数值，Esc 关闭'});
  d.appendChild(h('div',null,[h('div',{class:'zhead'},[h('h3',{text:title}),h('button',{class:'btn sm',type:'button',text:'关闭',onclick:()=>d.close()})]),
    svg,info,series.length>1?h('div',{class:'legend'},series.map(s=>h('span',null,[h('i',{style:'background:'+s.color}),s.label]))):null]));
  document.body.appendChild(d);d.addEventListener('close',()=>d.remove());d.showModal();
  const M=svg.__meta;if(!M)return;
  const ns='http://www.w3.org/2000/svg';
  const vline=document.createElementNS(ns,'line');vline.setAttribute('y1',M.m.t);vline.setAttribute('y2',M.H-M.m.b);vline.setAttribute('stroke','var(--muted)');vline.setAttribute('stroke-dasharray','3 3');vline.style.display='none';svg.appendChild(vline);
  const dots=M.series.map(s=>{const c=document.createElementNS(ns,'circle');c.setAttribute('r','5');c.setAttribute('fill',s.color);c.setAttribute('stroke','var(--panel)');c.setAttribute('stroke-width','1.5');c.style.display='none';svg.appendChild(c);return c});
  svg.addEventListener('pointermove',ev=>{
    const r=svg.getBoundingClientRect();const px=(ev.clientX-r.left)*M.W/r.width;
    if(px<M.m.l||px>M.W-M.m.r)return;
    const xv=M.x0+(px-M.m.l)/(M.W-M.m.l-M.m.r)*(M.x1-M.x0);
    let X0=null;const parts=[];
    M.series.forEach((s,k)=>{
      let bi=-1,bd=Infinity;
      for(let i=0;i<s.x.length;i++){const y=s.y[i];if(y==null||!isFinite(y)||(M.log&&y<=0))continue;const dd=Math.abs(s.x[i]-xv);if(dd<bd){bd=dd;bi=i}}
      if(bi<0){dots[k].style.display='none';return}
      const X=M.sx(s.x[bi]),Y=M.sy(s.y[bi]);dots[k].setAttribute('cx',X);dots[k].setAttribute('cy',Y);dots[k].style.display='';
      if(X0==null)X0=X;parts.push((M.series.length>1?s.label+' ':'')+'第 '+Math.round(s.x[bi]).toLocaleString('en-US')+' 步：'+fmtVal(s.y[bi]))});
    if(X0!=null){vline.setAttribute('x1',X0);vline.setAttribute('x2',X0);vline.style.display=''}
    info.textContent=parts.join('    ')});
  svg.addEventListener('pointerleave',()=>{vline.style.display='none';dots.forEach(c=>c.style.display='none')})}
function sparkBtn(r){
  const c=DATA.curves[r.id];if(!c)return spark(c,110,34);
  return h('button',{class:'sparkbtn',type:'button',title:'点击放大','aria-label':'放大 '+r.name+' 的 loss 曲线',
    onclick:()=>openZoom(r.name+' · loss',[{label:'loss',x:c.step,y:c.loss,color:colorFor(0)}],{log:true,ylabel:'loss'})},[spark(c,110,34)])}


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
      tagOf(r)?h('span',{class:'pill tag-sample',text:tagOf(r)}):null,
      h('span',{class:'sub num',text:r.id+' · '+(r.owner||'无负责人')+' · '+dateOf(r.time&&r.time.start)}),delBtn(r.id,'删除这条训练')]),
    r.kind_note?h('p',{class:'sub',style:'margin-top:6px',text:'归类：'+r.kind_note}):h('p',{class:'sub'},[missing('缺归类依据'),' 训练类型要人工确认，导入器只给建议（建议：'+KIND[r.kind_suggested||'']+'）。'])]));
  v.appendChild(chainBar(r));
  const warn=[];
  if(r.code.dirty)warn.push('训练时代码有未提交改动，不能靠 commit 复现这次训练。');
  if(r.code.git_hash==null)warn.push('没有 git 版本信息。');
  if(r.loss.summary_bad)warn.push('训练器写的汇总 loss 无效（续训后常见），页面上的 loss 是从逐步日志重算的。');
  (r.notes||'').split(/(?=\[导入器)/).filter(Boolean).forEach(n=>warn.push(n.trim()));
  if(warn.length)v.appendChild(h('div',{class:'banner'},warn.map(w=>h('div',{text:w}))));

  // 数据
  const rcLink=r.recipe_id&&RECIPES[r.recipe_id];
  v.appendChild(h('section',{id:'sec-data',class:'sec'},[h('h2',{text:'数据'}),
    rcLink?h('p',null,['数据配方：',h('a',{href:hashFor(rcLink.id),text:rcLink.name}),h('span',{class:'sub',text:'（'+rcLink.id+'），来源、筛选、截取、划分和输入输出定义都在配方页。'})]):h('div',{class:'banner',text:'这次训练没有引用数据配方，下面只有训练日志里读到的粗略数字。'}),
    h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['数据配置',r.data.config_name||'—'],['训练集数',fmtNum(r.data.train_episodes)],['验证集数',fmtNum(r.data.val_episodes)],
      ['训练锚点数',r.data.train_anchors?fmtNum(r.data.train_anchors):'未记录'],
      ['划分文件',r.data.split_file||'未记录'],['划分哈希',r.data.split_sha256?h('span',{class:'hash',text:r.data.split_sha256}):'未记录']])]),
    h('div',{class:'panel'},[h('h3',{text:'数据源与配比'}),h('table',{class:'t'},[h('thead',null,[h('tr',null,['来源','集数','权重','成败'].map(x=>h('th',{text:x})))]),
      h('tbody',null,r.data.sources.map(s=>h('tr',null,[h('td',{text:s.name}),h('td',{class:'n',text:fmtNum(s.episodes)}),h('td',{class:'n',text:s.weight==null?'未记录':s.weight}),h('td',{text:s.kind||'—'})])))]),
      h('p',{class:'sub',style:'margin-top:8px',text:'权重为“未记录”表示导入器读不到，单数据源时可忽略；混合数据要手填。'})])])]));

  // 训练
  const p=r.hparams;
  v.appendChild(h('section',{id:'sec-train',class:'sec'},[h('h2',{text:'训练'}),h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['基座',h('span',{class:'hash',text:r.base.model_path})],['步数',fmtNum(p.max_steps)],
      ['全局 batch',fmtNum(p.global_batch)+(p.per_device_batch?'（每卡 '+p.per_device_batch+' × 累积 '+(p.grad_accum||1)+' × '+r.hardware.gpus+' 卡）':'')],
      ['学习率',(p.learning_rate==null?'未记录':p.learning_rate)+' · '+p.scheduler+' · 预热 '+p.warmup_steps],['权重衰减',p.weight_decay==null?'—':p.weight_decay],['种子',p.seed==null?'—':p.seed],
      ['可训练部分',(p.trainable||[]).join('、')||'—'],...Object.entries(p.extra||{}).map(([k,x])=>[k,String(x)])])]),
    h('div',{class:'panel'},[kvTable([['机器',r.hardware.machine],['卡数',r.hardware.gpus+(r.hardware.gpu_type?' × '+r.hardware.gpu_type:'')],
      ['GPU 序号',(r.hardware.gpu_indices||[]).length?r.hardware.gpu_indices.join(', '):'未记录'],
      ['开始',r.time&&r.time.start||'—'],['结束',r.time&&r.time.end||'—'],
      ['代码',(r.code.git_hash?r.code.git_hash.slice(0,8):'无版本')+(r.code.dirty?'（有未提交改动）':'')],
      ['命令',h('span',{class:'hash',text:r.code.command||'未记录'})],['配置哈希',h('span',{class:'hash',text:r.code.config_yaml_sha256||'未记录'})]])])]),
    (()=>{const c=DATA.curves[r.id];if(!c)return h('div',{class:'empty',text:'没有曲线文件'});
      const col=colorFor(0);
      const has=a=>a&&a.filter(v=>v!=null).length>=2;const out=[];
      if(has(c.loss))out.push(chartBlock('loss（对数轴）',[{label:'loss',x:c.step,y:c.loss,color:col}],{log:true,ylabel:'loss'}));
      if(has(c.video)&&has(c.action))out.push(chartBlock('视频 / 动作分项 loss（对数轴）',[{label:'video',x:c.step,y:c.video,color:colorFor(0)},{label:'action',x:c.step,y:c.action,color:colorFor(1)}],{log:true,ylabel:'loss'}));
      if(has(c.grad_norm))out.push(chartBlock('梯度范数',[{label:'grad_norm',x:c.step,y:c.grad_norm,color:colorFor(1)}],{log:true,ylabel:'grad_norm'}));
      if(has(c.learning_rate))out.push(chartBlock('学习率',[{label:'lr',x:c.step,y:c.learning_rate,color:colorFor(2)}],{log:false,ylabel:'learning rate'}));
      if(has(c.step_time_ms))out.push(chartBlock('步耗时（ms）',[{label:'step_time_ms',x:c.step,y:c.step_time_ms,color:colorFor(3)}],{log:false,ylabel:'ms'}));
      return h('div',{class:'charts'},out)})(),
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
  v.appendChild(h('section',{id:'sec-real',class:'sec'},[h('h2',null,['真机评测 ',h('a',{href:'#real',class:'sub',text:'（全部真机测试 →）'})]),
    closedEvalsOf(r).length?h('div',{class:'panel'},[evalList(closedEvalsOf(r))]):h('div',{class:'empty',text:'还没有真机评测。'})]));
  return v}
function evalList(list){return h('table',{class:'t'},[h('thead',null,[h('tr',null,['日期','任务','协议','结果','视频',''].map(x=>h('th',{text:x})))]),
  h('tbody',null,list.map(e=>{const s=e.type==='closed_loop'?trialStats(e):null;
    return h('tr',null,[h('td',{class:'n',text:e.date}),h('td',null,[h('a',{href:hashFor(e.id),text:e.task})]),h('td',{class:'n',text:e.protocol.name+' '+e.protocol.version}),
      h('td',{class:'n',text:s?(s.labeled?s.success+'/'+s.labeled+' 成功':s.n+' 次未标注'):(Object.entries((e.open_loop&&e.open_loop.metrics)||{}).slice(0,2).map(([k,x])=>k+' '+Number(x).toFixed(4)).join('，')||'无指标')}),
      h('td',{text:s?(s.videos?s.videos+' 段':'无'):'—'}),h('td',null,[delBtn(e.id)])])}))])}
/* ================= 真机测试（单独一页） ================= */
function realEvals(){return (DATA.evals||[]).filter(e=>e.type==='closed_loop')}
function rateText(s){
  if(!s.n)return h('span',{class:'idle',text:'0 次试验'});
  if(!s.labeled)return h('span',{class:'warn num',text:s.n+' 次，成败未标注'});
  const pct=Math.round(s.rate*100);
  return h('span',{class:'num '+(s.enough?'':'warn'),text:s.success+'/'+s.labeled+' 成功（'+pct+'%）'+(s.enough?'':'，不足 '+3+' 次')})}
function realFlt(){return window.__rf||(window.__rf={task:'',robot:'',q:''})}
function viewReal(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  const all=realEvals();
  const tot=all.reduce((a,e)=>{const s=trialStats(e);a.n+=s.n;a.lab+=s.labeled;a.vid+=s.videos;a.unl+=s.unlabeled;return a},{n:0,lab:0,vid:0,unl:0});
  v.appendChild(h('div',null,[h('h2',{text:'真机测试'}),h('p',{class:'sub',text:'每一行是一次真机评测（一个 checkpoint、一个任务、一套部署配置下的若干次试验）。成功率只统计已有人标注成败的试验，未标注的单独列出，不算失败也不算成功。共 '+all.length+' 次评测，'+tot.n+' 次试验，其中 '+tot.lab+' 次已标注，'+tot.vid+' 段视频。'})]));
  if(tot.unl)v.appendChild(h('div',{class:'banner',text:'有 '+tot.unl+' 次试验还没人看视频标注成败。导入器只会给“未标注”，成败要人来判。'}));
  if(!all.length){
    v.appendChild(h('div',{class:'panel'},[h('h3',{text:'还没有真机评测记录'}),
      h('p',{class:'sub',text:'真机评测记录由 tools/import_arx63_eval.py 从 63 号机的运行目录导入，成败留空待标注；视频用 tools/upload_video.py 转成 H.264 传到 HF 后写进记录。'}),
      h('p',{class:'sub',text:'4090 上持续录制的摄像头视频要和某次评测、某次试验对上，才能出现在这里。接入方式还没定，位置先留好：下面“未归属的录像”一栏。'})]));
    v.appendChild(unassignedBox());return v}
  const tasks=[...new Set(all.map(e=>e.task))],robots=[...new Set(all.map(e=>(e.deployment&&e.deployment.robot)||'').filter(Boolean))];
  const f=realFlt();
  const sel=(label,key,opts)=>{const s=h('select',{onchange:e=>{f[key]=e.target.value;renderRealTable()}},[h('option',{value:'',text:'全部'}),...opts.map(o=>h('option',{value:o,text:o}))]);s.value=f[key];return h('label',null,[label,s])};
  v.appendChild(h('div',{class:'filters'},[h('label',null,['搜索',h('input',{type:'search',value:f.q,placeholder:'任务、训练名、检查点',oninput:e=>{f.q=e.target.value;renderRealTable()}})]),sel('任务','task',tasks),sel('机器人','robot',robots)]));
  v.appendChild(h('div',{id:'real-tbl',class:'scroll'}));
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'最近的视频'}),h('div',{id:'real-vids',class:'videos'})]));
  v.appendChild(unassignedBox());
  setTimeout(renderRealTable,0);
  return v}
function unassignedBox(){
  return h('section',{class:'sec'},[h('h2',{text:'未归属的录像'}),h('div',{class:'panel'},[
    h('p',{class:'sub',text:'摄像头持续录制的视频先放在这里，还没有对应到哪一次评测、哪一次试验。对应上之后，才会出现在上面的评测里。'}),
    h('p',{class:'sub',text:'现在是空的：4090 上的摄像头还没接入。成败不会根据录像自动判定。'})])])}
function renderRealTable(){
  const t=document.getElementById('real-tbl');if(!t)return;
  const f=realFlt(),q=f.q.trim().toLowerCase();
  const rows=realEvals().filter(e=>{
    const r=RUNS[e.run_id];
    if(f.task&&e.task!==f.task)return false;
    if(f.robot&&((e.deployment&&e.deployment.robot)||'')!==f.robot)return false;
    if(q&&!((e.task+' '+(r?r.name:'')+' '+e.id+' '+(e.instruction||'')).toLowerCase().includes(q)))return false;
    return true}).sort((a,b)=>(b.date||'').localeCompare(a.date||''));
  t.textContent='';
  if(!rows.length){t.appendChild(h('div',{class:'empty',text:'没有符合条件的真机评测。'}));}
  else{
    const tb=h('table',{class:'runs'});
    tb.appendChild(h('thead',null,[h('tr',null,['日期','模型（训练）','checkpoint','任务','机器人','成功率','试验','视频',''].map(x=>h('th',{text:x})))]));
    tb.appendChild(h('tbody',null,rows.map(e=>{
      const r=RUNS[e.run_id],s=trialStats(e);
      return h('tr',{class:'row'},[h('td',{class:'num',text:e.date||'—'}),
        h('td',{class:'name'},r?[h('a',{href:hashFor(r.id),text:r.name}),r._pending?h('span',{class:'cell-sub',text:'待登记'}):null,h('span',{class:'cell-sub',text:KIND[r.kind]||r.kind})]:[document.createTextNode(e.run_id)]),
        h('td',{class:'num',text:fmtNum(e.checkpoint&&e.checkpoint.step)}),h('td',{text:e.task}),h('td',{text:(e.deployment&&e.deployment.robot)||'—'}),
        h('td',null,[rateText(s)]),
        h('td',{class:'num',text:s.n+' 次'+(s.unlabeled&&s.labeled?'（'+s.unlabeled+' 未标注）':'')}),
        h('td',null,[s.videos?h('span',{class:'num',text:s.videos+' 段'}):h('span',{class:'missing',text:'无'})]),
        h('td',null,[h('a',{href:hashFor(e.id),text:'详情'})])])})));
    t.appendChild(tb)}
  const vids=document.getElementById('real-vids');if(!vids)return;vids.textContent='';
  const tiles=[];
  rows.forEach(e=>((e.closed_loop&&e.closed_loop.trials)||[]).forEach(tr=>(tr.videos||[]).forEach(x=>{if(tiles.length<12)tiles.push(videoTile(tr,x,e))})));
  if(tiles.length)tiles.forEach(x=>vids.appendChild(x));else vids.appendChild(h('div',{class:'empty',text:'还没有真机视频。'}))}

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
    h('div',{style:'display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center'},[tagOf(e)?h('span',{class:'pill tag-sample',text:tagOf(e)}):null,
      h('span',{class:'sub num',text:e.id+' · '+e.date+' · checkpoint '+fmtNum(e.checkpoint.step)+' · 协议 '+e.protocol.name+' '+e.protocol.version}),delBtn(e.id,'删除这条评测')])]));
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
    ['数据',[['数据配置',r=>r.data.config_name],['训练集数',r=>r.data.train_episodes],['验证集数',r=>r.data.val_episodes],['训练锚点数',r=>r.data.train_anchors||'未记录'],['配比',r=>r.data.sources.map(s=>s.name+':'+(s.weight==null?'?':s.weight)).join(' ')]]],
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
  renderTabs(r.v==='compare'?'compare':r.v==='real'?'real':(r.v==='datasets'||r.v==='dataset')?'datasets':(r.v==='plans'||r.v==='plan')?'plans':(r.v==='stacks'||r.v==='stack')?'stacks':(r.v==='groups'||r.v==='group')?'groups':(r.v==='recipes'||r.v==='recipe')?'recipes':(r.v==='run'&&RUNS[r.id])?(GROUP.pre.includes(RUNS[r.id].kind)?'pre':'ft'):(r.v==='home'&&r.group)?r.group:'home');
  let view;
  if(r.v==='run')view=viewRun(r.id);else if(r.v==='eval')view=viewEval(r.id);else if(r.v==='compare')view=viewCompare();else if(r.v==='recipes')view=viewRecipes();else if(r.v==='real')view=viewReal();else if(r.v==='datasets')view=viewDatasets();else if(r.v==='dataset')view=viewDataset(r.id);else if(r.v==='plans')view=viewPlans();else if(r.v==='plan')view=viewPlan(r.id);else if(r.v==='stacks')view=viewStacks();else if(r.v==='stack')view=viewStack(r.id);else if(r.v==='groups')view=viewGroups();else if(r.v==='group')view=viewGroup(r.id);else if(r.v==='recipe')view=viewRecipe(r.id);else view=viewHome(r.group||'');
  main.appendChild(adminBar());
  main.appendChild(view);
  if(r.v==='home')renderTable();if(r.v==='compare')renderCmp();
  window.scrollTo(0,0)}
/* ================= 数据集 / 采集方案 / 训练栈 / 实验组 ================= */
const DS={},CP={},ST={},XG={};
(DATA.datasets||[]).forEach(x=>DS[x.id]=x);(DATA.plans||[]).forEach(x=>CP[x.id]=x);(DATA.stacks||[]).forEach(x=>ST[x.id]=x);(DATA.groups||[]).forEach(x=>XG[x.id]=x);
function hashAny(id){const m={'ds-':'#d-','cp-':'#p-','st-':'#s-','xg-':'#x-','rc-':'#c-','tr-':'#r-','ev-':'#e-'};for(const k in m)if(id.startsWith(k))return m[k]+id.slice(3);return '#'}
function linkTo(id,label){return h('a',{href:hashAny(id),text:label||id})}
function sec(title,kids,id){return h('section',{class:'sec',id:id||null},[h('h2',{text:title}),...kids])}
function crumb(href,text){return h('div',{class:'crumb'},[h('a',{href,text:'← '+text})])}
function pendTag(x){return x._pending?h('span',{class:'pill tag-sample',text:'待登记'}):null}
function listOf(items,fn){return items&&items.length?h('ul',{style:'margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:4px'},items.map(fn)):h('span',{class:'idle',text:'无'})}
const VSTAT={verified:'已验证',unverified:'未验证',partial:'部分验证',unknown:'未知'};
const WHERE={inspire:'启智',hf:'Hugging Face',local:'本机',machine:'机器'};

/* ---- 预览视频 ---- */
const VIEWN={head:'头部相机',left_wrist:'左腕相机',right_wrist:'右腕相机'};
function previewSrc(d){const pv=d.preview;if(!pv)return null;if(pv.url)return pv.url;if(ADMIN.on)return '/preview/'+pv.file;return null}
function previewBox(d){
  const pv=d.preview;
  if(!pv)return h('div',{class:'empty',text:'这个数据集还没有预览视频。'});
  const src=previewSrc(d);
  const labels=h('div',{class:'pvlabels',style:'grid-template-columns:repeat('+pv.views.length+',1fr)'},pv.views.map(v=>h('span',{text:VIEWN[v]||v})));
  const box=h('div',{class:'pvbox'},[labels,
    src?h('video',{class:'pv',controls:true,muted:true,loop:true,playsinline:true,preload:'metadata',src:src,'aria-label':d.name+' 预览'}):h('div',{class:'empty',text:'预览已生成，但还没有公开地址，只有在本机管理页里能播。'})]);
  const rows=pv.episodes.map(e=>h('tr',null,[h('td',{class:'num',text:'第 '+e.episode_index+' 集'}),h('td',{class:'num',text:fmtNum(e.frames)+' 帧'}),h('td',{class:'num',text:e.seconds+' 秒'}),
    h('td',{class:'num',text:e.speed>1?e.speed+' 倍速':'原速'})]));
  return h('div',{class:'two'},[h('div',null,[box,h('p',{class:'sub',text:pv.duration_s+' 秒 · '+pv.width+'×'+pv.height+' · '+(pv.bytes/1e6).toFixed(1)+' MB · '+pv.codec})]),
    h('div',{class:'panel'},[h('h3',{text:pv.episodes.length>1?'这条视频由哪几集拼成':'这条视频取自哪一集'}),h('p',{class:'sub',text:pv.selection+'。数据集共 '+fmtNum(pv.episodes_total)+' 集，预览只代表这一小部分。'}),
      h('table',{class:'t'},[h('thead',null,[h('tr',null,['集','长度','时长','播放速度'].map(x=>h('th',{text:x})))]),h('tbody',null,rows)]),
      (pv.shared_with||[]).length?h('p',{class:'sub',text:'同一份画面也属于 '+pv.shared_with.length+' 个数据集变体（只是动作定义不同）：'+pv.shared_with.map(i=>DS[i]?DS[i].variant||DS[i].name:i).join('、')}):null])])}
function openPreview(d){
  const dl=document.createElement('dialog');dl.className='zoom';
  dl.appendChild(h('div',null,[h('div',{class:'zhead'},[h('h3',{text:d.name+' · 预览'}),h('button',{class:'btn sm',type:'button',text:'关闭',onclick:()=>dl.close()})]),previewBox(d),
    h('p',{class:'sub'},[linkTo(d.id,'打开这个数据集的详情页')])]));
  document.body.appendChild(dl);dl.addEventListener('close',()=>{dl.querySelectorAll('video').forEach(v=>v.pause());dl.remove()});dl.showModal()}

/* ---- 数据集总览：按采集批次分组，批次下列版本 ---- */
function viewDatasets(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  const all=DATA.datasets||[];
  const batches=new Map();all.forEach(d=>{if(!batches.has(d.batch_id))batches.set(d.batch_id,[]);batches.get(d.batch_id).push(d)});
  const eps=all.filter(d=>!d.derived_from).reduce((a,d)=>a+d.episodes,0);
  v.appendChild(h('div',null,[h('h2',{text:'数据集'}),h('p',{class:'sub',text:'按采集批次分组，同一批的不同动作定义、不同子集放在一起。数据本体不进仓库，这里只记统计、校验结果和位置。共 '+all.length+' 个版本，'+batches.size+' 个批次，其中 '+all.filter(d=>d.preview).length+' 个有预览视频（头部相机，取正中间一集，不挑选）。'})]));
  if(!all.length){v.appendChild(h('div',{class:'empty',text:'还没有数据集记录。'}));return v}
  v.appendChild(h('div',{class:'panel'},[h('h3',{text:'采集数据集的命名'}),
    h('p',{class:'sub'},['数据目录里没有成文的命名规范。现有 23 个目录名里 17 个以 MMDD 日期开头，日期之后的部分没有固定结构；最新的叠杯子（arx_stackcups）和叠毛巾（fold_blue）反而没有日期。',
      '草案建议 ',h('code',{text:'<采集日期MMDD>_<任务>[_<批次序号>]'}),'，人名缩写、格式、动作语义不放进名字，写进记录字段。草案待组内确认，不改任何已有目录名。',' ',
      h('a',{href:'https://github.com/Realbench123/realbench/blob/main/protocols/dataset-naming.md',target:'_blank',rel:'noreferrer',text:'完整说明 →'})])]));
  const rows=[];
  batches.forEach((xs,b)=>{
    xs.sort((a,c)=>(a.derived_from?1:0)-(c.derived_from?1:0)||a.name.localeCompare(c.name));
    xs.forEach((d,i)=>rows.push(h('tr',{class:'row'},[
      h('td',{class:'name'},i===0?[h('b',{text:b}),h('span',{class:'cell-sub',text:(xs[0].batch_note||'')+' · '+xs.length+' 个版本'})]:[]),
      h('td',null,[linkTo(d.id,d.variant||d.name),pendTag(d)?h('span',{style:'margin-left:6px'},[pendTag(d)]):null,d.derived_from?h('span',{class:'cell-sub',text:'派生自 '+(DS[d.derived_from]?DS[d.derived_from].variant:d.derived_from)}):null]),
      h('td',null,[d.preview?h('button',{class:'btn sm',type:'button',text:'▶ 看',title:'播放这个数据集的预览',onclick:()=>openPreview(d)}):h('span',{class:'idle',text:'无'})]),
      h('td',{class:'num',text:fmtNum(d.episodes)+' 集'}),h('td',{class:'num',text:fmtNum(d.frames)+' 帧'}),
      h('td',null,[h('span',{class:'mono',text:d.action_semantics.length>46?d.action_semantics.slice(0,46)+'…':d.action_semantics,title:d.action_semantics})]),
      h('td',null,[d.collection_plan_id&&CP[d.collection_plan_id]?linkTo(d.collection_plan_id,CP[d.collection_plan_id].name.split('（')[0]):h('span',{class:'missing',text:'未关联'})]),
      h('td',null,[h('span',{class:'pill '+(d.validation.status==='verified'?'ok':d.validation.status==='unverified'?'warn':'idle'),text:VSTAT[d.validation.status]})]),
      h('td',null,(d.locations||[]).map(l=>h('span',{class:'pill '+(l.verified?'':'warn'),style:'margin-right:4px',title:l.label+(l.verified?'':'（未核实）'),text:WHERE[l.where]+(l.verified?'':'?')}))),
      h('td',null,[(d.used_by_recipes||[]).length?h('span',{class:'num',text:d.used_by_recipes.length+' 个配方'}):h('span',{class:'idle',text:'未被引用'})])])))});
  const t=h('table',{class:'runs'});
  t.appendChild(h('thead',null,[h('tr',null,['采集批次','版本','预览','集数','帧数','动作含义','采集方案','校验','位置','被引用'].map(x=>h('th',{text:x})))]));
  t.appendChild(h('tbody',null,rows));
  v.appendChild(h('div',{class:'scroll'},[t]));
  v.appendChild(h('p',{class:'sub',text:'位置标签后带“?”表示只是文档声明、我没有核实到存在。'}));
  return v}

function viewDataset(id){
  const d=DS[id];if(!d)return viewMissing('找不到数据集 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(crumb('#datasets','全部数据集'));
  v.appendChild(h('div',null,[h('h2',{text:d.name,style:'font-size:20px'}),h('div',{style:'display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center'},[pendTag(d),
    h('span',{class:'pill '+(d.validation.status==='verified'?'ok':'warn'),text:VSTAT[d.validation.status]}),h('span',{class:'sub num',text:d.id+' · 批次 '+d.batch_id}),delBtn(d.id,'删除这个数据集')])]));
  const sib=(DATA.datasets||[]).filter(x=>x.batch_id===d.batch_id&&x.id!==d.id);
  if(sib.length)v.appendChild(h('div',{class:'panel'},[h('h3',{text:'同一批次的其它版本'}),h('div',{class:'relset',style:'display:flex;flex-wrap:wrap;gap:6px 14px'},sib.map(x=>linkTo(x.id,x.variant||x.name)))]));
  v.appendChild(sec('长什么样',[previewBox(d)]));
  v.appendChild(sec('规模',[h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['集数',fmtNum(d.episodes)],['帧数',fmtNum(d.frames)],['帧率',d.fps+' Hz'],['折合时长','约 '+(d.frames/d.fps/60).toFixed(1)+' 分钟（名义，不是逐条时间戳之和）'],
      ['单集帧数',d.episode_frames&&d.episode_frames.median?d.episode_frames.min+' / '+d.episode_frames.median+' / '+d.episode_frames.max+'（最小/中位/最大）':'未记录'],['格式',d.format||'未记录'],['机器人',d.robot],['相机',(d.cameras||[]).join('、')||'未记录']])]),
    h('div',{class:'panel'},[h('h3',{text:'任务文本'}),listOf(d.instructions,t=>h('span',{class:'mono',text:t})),h('h3',{text:'成败标签',style:'margin-top:12px'}),
      kvTable([['成功',d.success_labels.success==null?'未记录':d.success_labels.success],['失败',d.success_labels.failed==null?'未记录':d.success_labels.failed],['依据',d.success_labels.basis||'—']])])])]));
  v.appendChild(sec('动作含义与采集',[h('div',{class:'panel'},[kvTable([['action 的含义',h('span',{class:'mono',text:d.action_semantics})],
    ['采集方案',d.collection_plan_id&&CP[d.collection_plan_id]?linkTo(d.collection_plan_id,CP[d.collection_plan_id].name):missing('未关联')],
    ['采集人',d.collected.by||'未记录'],['采集日期',(d.collected.date_start||d.collected.date_end)?(d.collected.date_start||'?')+' 至 '+(d.collected.date_end||'?'):missing('未记录')],['日期依据',d.collected.date_basis||'—'],
    ['遥操作员人数',d.collected.operators==null?'未记录':d.collected.operators],['采集命令',d.collected.command?h('span',{class:'hash',text:d.collected.command}):'未记录']])])]));
  v.appendChild(sec('校验',[h('div',{class:'panel'},[kvTable([['状态',VSTAT[d.validation.status]],['校验方',d.validation.by||'未记录'],['校验时间',d.validation.checked_at||'未记录'],
    ['视频完整解码',d.validation.video_full_decode==null?'未记录':(d.validation.video_full_decode?'是':'否')],['源 episode 已核对 / 不可用',(d.validation.source_episodes_verified==null?'—':d.validation.source_episodes_verified)+' / '+(d.validation.source_episodes_unavailable==null?'—':d.validation.source_episodes_unavailable)],['说明',d.validation.note||'—']])])]));
  v.appendChild(sec('数据在哪',[h('div',{class:'panel scroll'},[h('table',{class:'t'},[h('thead',null,[h('tr',null,['位置','类型','名称','已核实'].map(x=>h('th',{text:x})))]),
    h('tbody',null,d.locations.map(l=>h('tr',null,[h('td',{text:WHERE[l.where]}),h('td',{text:({raw:'原始',converted:'转换版',mirror:'镜像'})[l.kind]}),
      h('td',null,[l.url?h('a',{href:l.url,target:'_blank',rel:'noreferrer',text:l.label}):document.createTextNode(l.label)]),
      h('td',null,[h('span',{class:l.verified?'ok':'warn',text:l.verified?'是':'否，只是文档声明'})])])))])])]));
  v.appendChild(sec('谁在用',[h('div',{class:'panel'},[(d.used_by_recipes||[]).length?listOf(d.used_by_recipes,i=>RECIPES[i]?linkTo(i,RECIPES[i].name):document.createTextNode(i)):h('p',{class:'sub',text:'还没有配方引用这个数据集。'})])]));
  if(d.notes)v.appendChild(sec('备注',[h('div',{class:'panel'},[h('p',{text:d.notes})])]));
  return v}

/* ---- 采集方案 ---- */
function viewPlans(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  v.appendChild(h('div',null,[h('h2',{text:'采集方案'}),h('p',{class:'sub',text:'数据怎么采的。不同方案记录的 action 含义不同，混合训练前先看这里。'})]));
  const ps=DATA.plans||[];if(!ps.length){v.appendChild(h('div',{class:'empty',text:'还没有采集方案记录。'}));return v}
  const t=h('table',{class:'runs'});t.appendChild(h('thead',null,[h('tr',null,['方案','action 记录的是','state 记录的是','采样','客户端提交','数据集'].map(x=>h('th',{text:x})))]));
  t.appendChild(h('tbody',null,ps.map(p=>h('tr',{class:'row'},[h('td',{class:'name'},[linkTo(p.id,p.name),pendTag(p)?h('span',{style:'margin-left:6px'},[pendTag(p)]):null]),h('td',{text:p.action_recorded}),h('td',{text:p.state_recorded}),
    h('td',{class:'num',text:(p.sampling.fps||'?')+' Hz'}),h('td',null,[p.client&&p.client.commit?h('span',{class:'hash',text:p.client.commit.slice(0,10)}):h('span',{class:'missing',text:'未读到'})]),
    h('td',{class:'num',text:(p.used_by_datasets||[]).length+' 个'})]))));
  v.appendChild(h('div',{class:'scroll'},[t]));return v}
function viewPlan(id){
  const p=CP[id];if(!p)return viewMissing('找不到采集方案 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(crumb('#plans','全部采集方案'));
  v.appendChild(h('div',null,[h('h2',{text:p.name,style:'font-size:20px'}),h('div',{style:'display:flex;gap:8px;align-items:center'},[pendTag(p),h('span',{class:'sub num',text:p.id}),delBtn(p.id,'删除这个方案')])]));
  v.appendChild(sec('记录的内容',[h('div',{class:'panel'},[kvTable([['遥操作',p.teleop],['action',p.action_recorded],['state',p.state_recorded],['采样频率',(p.sampling.fps||'未记录')+' Hz'],['时间戳',p.sampling.timestamps||'未记录'],['图像',p.sampling.image_format||'未记录'],['说明',p.sampling.note||'—']])])]));
  if((p.safety||[]).length)v.appendChild(sec('安全机制',[h('div',{class:'panel'},[listOf(p.safety,s=>h('span',{text:s}))])]));
  const c=p.client||{};
  v.appendChild(sec('客户端与代码版本',[h('div',{class:'panel'},[kvTable([['客户端',c.name||'未记录'],['仓库',c.repo||'未记录'],['分支',c.branch||'未记录'],['提交',c.commit?h('span',{class:'hash',text:c.commit}):missing('未读到')],['入口命令',c.entry?h('span',{class:'hash',text:c.entry}):'未记录']])])]));
  if((p.differs_from||[]).length)v.appendChild(sec('和别的方案的区别',[h('div',{class:'panel'},[listOf(p.differs_from,x=>h('span',null,[CP[x.plan_id]?linkTo(x.plan_id,CP[x.plan_id].name):x.plan_id,'：'+x.what]))])]));
  if(p.mixing_note)v.appendChild(h('div',{class:'banner',text:'混合使用：'+p.mixing_note}));
  v.appendChild(sec('用了这个方案的数据集',[h('div',{class:'panel'},[(p.used_by_datasets||[]).length?listOf(p.used_by_datasets,i=>DS[i]?linkTo(i,DS[i].name):document.createTextNode(i)):h('p',{class:'sub',text:'还没有数据集关联这个方案。'})])]));
  if(p.notes)v.appendChild(sec('备注',[h('div',{class:'panel'},[h('p',{text:p.notes})])]));
  return v}

/* ---- 训练栈 ---- */
function viewStacks(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  v.appendChild(h('div',null,[h('h2',{text:'训练栈'}),h('p',{class:'sub',text:'怎么复现一次训练：解释器、源码快照、镜像、提交入口、配置变体、已知问题。不放凭据。'})]));
  const ss=DATA.stacks||[];if(!ss.length){v.appendChild(h('div',{class:'empty',text:'还没有训练栈记录。'}));return v}
  const t=h('table',{class:'runs'});t.appendChild(h('thead',null,[h('tr',null,['训练栈','框架','基座','典型配置','配置变体','已知问题','训练'].map(x=>h('th',{text:x})))]));
  t.appendChild(h('tbody',null,ss.map(s=>h('tr',{class:'row'},[h('td',{class:'name'},[linkTo(s.id,s.name),pendTag(s)?h('span',{style:'margin-left:6px'},[pendTag(s)]):null]),h('td',{text:s.framework||''}),
    h('td',{text:(s.base_models||[]).map(b=>b.name).join('、')||'—'}),h('td',{text:(s.compute&&s.compute.typical)||'—'}),h('td',{class:'num',text:(s.config_variants||[]).length+' 个'}),h('td',{class:'num',text:(s.known_issues||[]).length+' 条'}),h('td',{class:'num',text:(s.used_by_runs||[]).length+' 次'})]))));
  v.appendChild(h('div',{class:'scroll'},[t]));return v}
function viewStack(id){
  const s=ST[id];if(!s)return viewMissing('找不到训练栈 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(crumb('#stacks','全部训练栈'));
  v.appendChild(h('div',null,[h('h2',{text:s.name,style:'font-size:20px'}),h('div',{style:'display:flex;gap:8px;align-items:center'},[pendTag(s),h('span',{class:'sub num',text:s.id}),delBtn(s.id,'删除这个训练栈')])]));
  const r=s.runtime||{},c=s.compute||{},e=s.entrypoints||{},m=s.monitoring||{};
  v.appendChild(sec('基座与运行环境',[h('div',{class:'two'},[
    h('div',{class:'panel'},[h('h3',{text:'基座权重'}),h('table',{class:'t'},[h('thead',null,[h('tr',null,['名称','格式','位置'].map(x=>h('th',{text:x})))]),h('tbody',null,(s.base_models||[]).map(b=>h('tr',null,[h('td',{text:b.name}),h('td',{text:b.format||'—'}),h('td',{text:b.location_label||'—'})])))])]),
    h('div',{class:'panel'},[kvTable([['镜像',r.image||'未记录'],['解释器',r.interpreter||'未记录'],['源码快照',r.source_snapshot||'未记录'],['Python',r.python||'未记录'],['说明',r.note||'—']])])])]));
  v.appendChild(sec('算力与入口',[h('div',{class:'panel'},[kvTable([['平台',c.platform||'未记录'],['计算组',c.group||'未记录'],['典型配置',c.typical||'未记录'],['提交入口',e.submit||'未记录'],['节点入口',e.node||'未记录'],['CLI',e.cli||'未记录'],['smoke',e.smoke||'未记录'],['TensorBoard',m.tensorboard||'未记录'],['W&B',m.wandb||'未记录'],['状态文件',m.status_file||'未记录']])])]));
  v.appendChild(sec('配置变体',[h('div',{class:'panel scroll'},[h('table',{class:'t'},[h('thead',null,[h('tr',null,['变体','任务','动作模式','和基础配置的差别','配方'].map(x=>h('th',{text:x})))]),
    h('tbody',null,(s.config_variants||[]).map(x=>h('tr',null,[h('td',{text:x.name}),h('td',{text:x.task||'—'}),h('td',{class:'n',text:x.action_mode||'—'}),h('td',{text:x.differs||''}),h('td',null,[x.recipe_id&&RECIPES[x.recipe_id]?linkTo(x.recipe_id,RECIPES[x.recipe_id].name):document.createTextNode('—')])])))])])]));
  v.appendChild(sec('已知问题',[(s.known_issues||[]).length?h('div',{class:'panel scroll'},[h('table',{class:'t'},[h('thead',null,[h('tr',null,['问题','怎么处理','出现在'].map(x=>h('th',{text:x})))]),
    h('tbody',null,s.known_issues.map(x=>h('tr',null,[h('td',{text:x.what}),h('td',{text:x.fix||'没有记录'}),h('td',{text:x.seen_in||'—'})])))])]):h('div',{class:'empty',text:'没有记录。'})]));
  v.appendChild(sec('用了这个训练栈的训练',[h('div',{class:'panel'},[(s.used_by_runs||[]).length?listOf(s.used_by_runs,i=>RUNS[i]?linkTo(i,RUNS[i].name):document.createTextNode(i)):h('p',{class:'sub',text:'还没有训练关联这个训练栈。'})])]));
  if(s.notes)v.appendChild(sec('备注',[h('div',{class:'panel'},[h('p',{text:s.notes})])]));
  return v}

/* ---- 实验组 ---- */
function viewGroups(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  v.appendChild(h('div',null,[h('h2',{text:'实验组'}),h('p',{class:'sub',text:'共用基础配置、只在少数处不同的一组训练（ablation）。基础配置写一次，各次运行只记差异。'})]));
  const gs=DATA.groups||[];
  if(!gs.length){v.appendChild(h('div',{class:'panel'},[h('h3',{text:'这里将放预训练的数据配比实验（ablation）'}),
    h('p',{class:'sub',text:'每个实验组写一次共用的基础配置，各次运行只记差异：数据配比、每个数据源的轮数和降采样、对应的训练记录和评测。'}),
    h('p',{class:'sub',text:'位置已经留好，还没有内容。需要实验矩阵、各运行的配比、过滤规则这几张表，目前只拿到了基础配置的文字描述。'}),
    h('p',{class:'sub',text:'数据清洗那一段的位置也还没定，等要求给出后再决定它放在数据集之前的哪一环、记哪些字段。'})]));return v}
  gs.forEach(g=>v.appendChild(h('div',{class:'panel'},[linkTo(g.id,g.name),h('p',{class:'sub',text:g.question}),h('span',{class:'num',text:g.runs.length+' 次运行'})])));
  return v}
function viewGroup(id){
  const g=XG[id];if(!g)return viewMissing('找不到实验组 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(crumb('#groups','全部实验组'));
  v.appendChild(h('div',null,[h('h2',{text:g.name,style:'font-size:20px'}),h('p',{class:'sub',text:g.question}),h('div',{style:'display:flex;gap:8px;align-items:center'},[pendTag(g),h('span',{class:'sub num',text:g.id}),delBtn(g.id,'删除这个实验组')])]));
  if((g.missing||[]).length)v.appendChild(h('div',{class:'banner'},[h('b',{text:'这份记录还缺： '}),...g.missing.map(m=>h('div',{text:'· '+m}))]));
  const b=g.base_config||{};
  const rows=[['初始化',b.init],['可训练',(b.trainable||[]).join('、')],['动作头',b.action_head],['规模',b.scale],['步数',b.max_steps],['保存间隔',b.save_every],['精度',b.precision],['分布式',b.distributed],
    ['优化器',b.optimizer&&Object.keys(b.optimizer).length?Object.entries(b.optimizer).map(([k,x])=>k+'='+x).join('，'):null],['batch',b.batch?Object.entries(b.batch).filter(([,x])=>x!=null).map(([k,x])=>k+'='+x).join('，'):null],
    ['动作表示',b.action_repr],['采样',b.sampling],['图像',b.images],['数据分片',b.shards]].filter(r=>r[1]!=null&&r[1]!=='');
  v.appendChild(sec('所有运行共用的配置',[h('div',{class:'panel'},[kvTable(rows)])]));
  v.appendChild(sec('各次运行',[h('div',{class:'panel scroll'},[h('table',{class:'t'},[h('thead',null,[h('tr',null,['运行','状态','和基础配置的差异','数据配比','理由'].map(x=>h('th',{text:x})))]),
    h('tbody',null,g.runs.map(r=>h('tr',null,[h('td',null,[r.run_id&&RUNS[r.run_id]?linkTo(r.run_id,r.label):document.createTextNode(r.label)]),h('td',{text:r.status||'—'}),h('td',{text:r.differs||'—'}),
      h('td',null,[(r.mix||[]).length?listOf(r.mix,m=>h('span',{class:'num',text:m.source+' '+(m.weight==null?'?':m.weight)+(m.epochs!=null?' · '+m.epochs+' 轮':'')})):h('span',{class:'missing',text:'未记录'})]),h('td',{text:r.rationale||'—'})])))])])]));
  return v}


$('#built').textContent='构建 '+DATA.built+' · '+DATA.commit;
window.addEventListener('hashchange',render);
render();
detectAdmin();

/* ================= 数据配方 ================= */
const BAR_COLORS=['fill','fill minor','fill bad'];
function funnelRow(label,value,max,cls,sub){
  return h('div',{class:'bar'},[h('span',{text:label}),
    h('div',{class:'track',role:'img','aria-label':label+' '+fmtNum(value)},[h('div',{class:cls||'fill',style:'width:'+Math.max(0.5,100*value/max)+'%'})]),
    h('span',{class:'num',text:fmtNum(value)+(sub?' '+sub:''),style:'text-align:right'})])}
function recipeFunnel(c){
  const fr=c.frames,sel=c.selection,sp=c.split;
  const srcN=c.sources.reduce((a,x)=>a+x.episodes,0);
  const rows=[['源 episode',srcN,srcN,'fill','集'],['筛选后',sel.included,srcN,'fill','集（排除 '+sel.excluded.length+'）'],
    ['训练 / 验证',sp.train_ids.length,srcN,'fill','集 / 验证 '+sp.val_ids.length+' 集']];
  const f=[];
  rows.forEach(r=>f.push(funnelRow(...r)));
  if(fr.source_frames_total){
    const m=fr.source_frames_total;
    f.push(h('div',{class:'sub',style:'margin-top:6px',text:'帧（进入配方的 '+fmtNum(sel.included)+' 集）'}));
    f.push(funnelRow('原始帧',fr.source_frames_total,m,'fill',''));
    if(fr.kept_frames_total)f.push(funnelRow('截取后保留',fr.kept_frames_total,m,'fill minor','（'+(100*fr.kept_frames_total/m).toFixed(1)+'%）'));
    if(fr.anchors_train!=null)f.push(funnelRow('训练锚点',fr.anchors_train,m,'fill','（整块 '+fr.horizon+' 步落在保留区内）'));
    if(fr.anchors_val!=null)f.push(funnelRow('验证锚点',fr.anchors_val,m,'fill','')) }
  return h('div',{class:'funnel'},f)}
function idGrid(c){
  const sp=c.split,ex=new Map(c.selection.excluded.map(x=>[x.id,x.reason]));
  const tr=new Set(sp.train_ids),va=new Set(sp.val_ids),vf=new Set(sp.val_failed_ids||[]);
  const all=[...new Set([...sp.train_ids,...sp.val_ids])].sort((a,b)=>a-b);
  const cells=all.map(i=>h('span',{class:vf.has(i)?'vf':va.has(i)?'va':'tr',title:(vf.has(i)?'验证集·失败示教':va.has(i)?'验证集':'训练集')+' #'+i,text:String(i)}));
  return h('div',null,[h('div',{class:'idgrid'},cells),
    h('div',{class:'keys'},[h('span',null,[h('i',{style:'background:var(--accent-soft)'}),'训练 '+tr.size]),h('span',null,[h('i',{style:'border-color:var(--ok)'}),'验证 '+(va.size-vf.size)]),h('span',null,[h('i',{style:'border-color:var(--bad)'}),'验证·失败示教 '+vf.size])]),
    h('p',{class:'sub',style:'margin-top:6px',text:'编号用的是“'+(sp.id_space||'未说明')+'”。'+(sp.id_note||'')})])}
function histogram(vals,unit,fps){
  const W=480,H=150,m={l:40,r:10,t:10,b:26},ns='http://www.w3.org/2000/svg';
  const lo=Math.min(...vals),hi=Math.max(...vals),B=14,bw=(hi-lo)/B||1,bins=Array(B).fill(0);
  vals.forEach(v=>{bins[Math.min(B-1,Math.floor((v-lo)/bw))]++});
  const mx=Math.max(...bins);
  const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.setAttribute('role','img');svg.setAttribute('aria-label','单集帧数分布，最短 '+lo+'，最长 '+hi);
  const add=(t,at,tx)=>{const e=document.createElementNS(ns,t);for(const k in at)e.setAttribute(k,at[k]);if(tx!=null)e.textContent=tx;svg.appendChild(e)};
  const pw=(W-m.l-m.r)/B;
  bins.forEach((n,i)=>{const hh=(H-m.t-m.b)*n/mx;add('rect',{x:m.l+i*pw+1,y:H-m.b-hh,width:pw-2,height:hh,fill:'var(--accent)',opacity:.85})});
  add('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:'var(--line)'});
  [0,0.5,1].forEach(t=>{const v=lo+(hi-lo)*t;add('text',{x:m.l+(W-m.l-m.r)*t,y:H-8,'text-anchor':t===0?'start':t===1?'end':'middle','font-size':12,fill:'var(--muted)','font-family':'var(--f-mono)'},Math.round(v).toLocaleString('en-US')+(fps?' ('+(v/fps).toFixed(0)+'s)':''))});
  add('text',{x:m.l-6,y:m.t+10,'text-anchor':'end','font-size':12,fill:'var(--muted)','font-family':'var(--f-mono)'},String(mx));
  add('text',{x:m.l-6,y:H-m.b,'text-anchor':'end','font-size':12,fill:'var(--muted)','font-family':'var(--f-mono)'},'0');
  return svg}

function viewRecipe(id){
  const c=RECIPES[id];if(!c)return viewMissing('找不到配方 '+id);
  const v=h('div',{style:'display:flex;flex-direction:column;gap:16px'});
  v.appendChild(h('div',{class:'crumb'},[h('a',{href:'#recipes',text:'← 全部配方'})]));
  v.appendChild(h('div',null,[h('h2',{text:c.name,style:'font-size:20px'}),
    h('div',{style:'display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center'},[tagOf(c)?h('span',{class:'pill tag-sample',text:tagOf(c)}):null,
      h('span',{class:'sub num',text:c.id+' · '+c.task+' · '+c.robot+(c.owner?' · '+c.owner:'')}),delBtn(c.id,'删除这个配方')]),
    c.parent_id?h('p',{class:'sub',style:'margin-top:6px'},['改自 ',RECIPES[c.parent_id]?h('a',{href:hashFor(c.parent_id),text:RECIPES[c.parent_id].name}):c.parent_id,'：'+(c.changes_from_parent||[]).join('；')]):null]));
  const w=[];
  if(c.notes)v.appendChild(h('p',{class:'sub',text:c.notes}));
  if(c.sources.length>1&&c.sources.some(s=>s.weight==null))w.push('有多个数据源但部分没有采样权重。');
  c.sources.forEach(s=>{if(!s.sha256)w.push('数据源 '+s.name+' 没有记哈希，别人无法确认拿到的是同一份数据。')});
  if(w.length)v.appendChild(h('div',{class:'banner'},w.map(x=>h('div',{text:x}))));

  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'从源数据到训练样本'}),h('div',{class:'panel'},[recipeFunnel(c)])]));
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'数据来源'}),h('div',{class:'panel scroll'},[h('table',{class:'t'},[
    h('thead',null,[h('tr',null,['名称','类型','集数','采样权重','格式','哈希','备注'].map(x=>h('th',{text:x})))]),
    h('tbody',null,c.sources.map(s=>h('tr',null,[h('td',{text:s.name}),h('td',{text:s.kind}),h('td',{class:'n',text:fmtNum(s.episodes)}),
      h('td',{class:'n',text:s.weight==null?(c.sources.length>1?'未记录':'单源'):s.weight}),h('td',{text:s.format||'—'}),
      h('td',null,[s.sha256?h('span',{class:'hash',text:s.sha256.slice(0,16)+'…'}):missing('无')]),h('td',{text:s.note||''})])))])])]));
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'筛选'}),h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['规则',c.selection.rule||'未记录'],['进入配方',fmtNum(c.selection.included)+' 集'],
      ['成功示教',c.selection.success_count==null?'未记录':c.selection.success_count],['失败示教',c.selection.failed_count==null?'未记录':c.selection.failed_count]])]),
    h('div',{class:'panel'},[h('h3',{text:'被排除的 episode'}),c.selection.excluded.length?h('table',{class:'t'},[h('thead',null,[h('tr',null,['源编号','原因'].map(x=>h('th',{text:x})))]),
      h('tbody',null,c.selection.excluded.map(x=>h('tr',null,[h('td',{class:'n',text:x.id}),h('td',{text:x.reason})])))]):h('p',{class:'sub',text:'没有排除任何集。'})])])]));
  const fr=c.frames,tm=fr.trim||{},det=tm.detector||{};
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'截取与采样'}),h('div',{class:'two'},[
    h('div',{class:'panel'},[kvTable([['频率',fr.fps+' Hz'],['动作块长度',fr.horizon==null?'未记录':fr.horizon+' 步（'+(fr.horizon/fr.fps).toFixed(2)+' 秒）'],['方式',tm.method||'未记录'],
      ['前留白',tm.pad_before_s==null?'—':tm.pad_before_s+' 秒'],['后留白',tm.pad_after_s==null?'—':tm.pad_after_s+' 秒'],['裁集内部',tm.interior_trim?'是':'否'],
      ...Object.entries(det).filter(([k])=>!k.startsWith('padding')).map(([k,x])=>['检测器 '+k,String(x)]),
      ['保留文件',fr.keep_file?h('span',{class:'hash',text:fr.keep_file+(fr.keep_file_sha256?'  '+fr.keep_file_sha256.slice(0,12)+'…':'')}):'未记录']])]),
    h('div',{class:'chartbox'},[h('h3',{text:'单集帧数分布'}),fr.episode_frames&&fr.episode_frames.values.length?histogram(fr.episode_frames.values,'帧',fr.fps):h('p',{class:'sub',text:'没有记录'})])])]));
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'训练 / 验证划分'}),c.split.no_per_episode_split?h('div',{class:'banner',text:'这个配方没有逐集的训练/验证划分：'+(c.split.id_note||'全部集都用于训练，没有验证集。')}):null,h('div',{class:'panel'},[kvTable([['划分单位',c.split.unit||'未记录'],['种子',c.split.seed==null?'—':c.split.seed],
    ['只用成功示教训练',c.split.train_success_only?'是':'否'],['划分文件',c.split.file?h('span',{class:'hash',text:c.split.file+(c.split.file_sha256?'  '+c.split.file_sha256.slice(0,12)+'…':'')}):'未记录']]),
    h('div',{style:'margin-top:12px'},[idGrid(c)])])]));
  const io=c.io;
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'输入输出定义'}),h('div',{class:'panel'},[kvTable([['状态',io.state||'未记录'],['动作',io.action||'未记录'],
    ['动作语义',io.action_semantics||'—'],['动作参数化',({absolute:'绝对',relative_to_anchor:'相对锚点状态',delta:'相邻差分',other:'其他'})[io.action_param]||io.action_param],
    ['参数化说明',io.action_param_note||'—'],['相机',(io.cameras||[]).join('、')],['图像尺寸',io.image_size?io.image_size.join('×'):'—'],
    ['归一化',io.normalization||missing('未记录')],['任务文本',c.instruction||'—']])])]));
  const used=(c.used_by||[]).map(i=>RUNS[i]).filter(Boolean);
  v.appendChild(h('section',{class:'sec'},[h('h2',{text:'用了这个配方的训练'}),used.length?h('div',{class:'panel'},[h('ul',{style:'margin:0;padding-left:1.2em'},used.map(r=>h('li',null,[h('a',{href:hashFor(r.id),text:r.name}),h('span',{class:'sub',text:' · '+KIND[r.kind]+' · '+fmtNum(r.hparams.max_steps)+' 步'})])))]):h('div',{class:'empty',text:'还没有训练引用这个配方。'})]));
  return v}

function viewRecipes(){
  const v=h('div',{style:'display:flex;flex-direction:column;gap:14px'});
  v.appendChild(h('div',null,[h('h2',{text:'数据配方'}),h('p',{class:'sub',text:'一个配方说清楚训练数据怎么来：哪些源、筛了什么、怎么截、怎么划分、输入输出怎么定义。训练记录引用配方，同一配方可被多次训练共用。数据本体不进仓库，只记标识和哈希。'})]));
  const rows=DATA.recipes||[];
  if(!rows.length){v.appendChild(h('div',{class:'empty',text:'还没有配方。用 tools/import_tau0_recipe.py 导入。'}));return v}
  const t=h('table',{class:'runs rcs'});
  t.appendChild(h('thead',null,[h('tr',null,['配方','任务 / 本体','数据来源','筛选','划分','帧与锚点','动作参数化','被引用'].map(x=>h('th',{text:x})))]));
  const tb=h('tbody');
  rows.forEach(c=>{const fr=c.frames;
    tb.appendChild(h('tr',{class:'row'},[
      h('td',{class:'name'},[h('a',{href:hashFor(c.id),text:c.name}),h('span',{class:'cell-sub num',text:c.id}),tagOf(c)?h('span',{class:'pill tag-sample',text:tagOf(c),style:'margin-top:4px'}):null]),
      h('td',null,[h('div',{text:c.task}),h('span',{class:'cell-sub',text:c.robot})]),
      h('td',null,c.sources.map(s=>h('div',null,[s.name,h('span',{class:'cell-sub num',text:s.kind+' · '+fmtNum(s.episodes)+' 集'+(c.sources.length>1?' · 权重 '+(s.weight==null?'?':s.weight):'')})]))),
      h('td',null,[h('div',{class:'num',text:fmtNum(c.selection.included)+' / '+fmtNum(c.sources.reduce((a,x)=>a+x.episodes,0))+' 集'}),h('span',{class:'cell-sub',text:'排除 '+c.selection.excluded.length+'；'+(c.selection.rule||'')})]),
      h('td',null,[h('div',{class:'num',text:c.split.train_ids.length+' 训 / '+c.split.val_ids.length+' 验'}),h('span',{class:'cell-sub',text:'seed '+(c.split.seed==null?'—':c.split.seed)+(c.split.val_failed_ids&&c.split.val_failed_ids.length?'；验证含 '+c.split.val_failed_ids.length+' 个失败示教':'')})]),
      h('td',null,[h('div',{class:'num',text:fr.kept_frames_total?fmtNum(fr.kept_frames_total)+' 帧':'—'}),h('span',{class:'cell-sub num',text:fr.anchors_train!=null?'训练锚点 '+fmtNum(fr.anchors_train):''}),h('span',{class:'cell-sub',text:fr.fps+' Hz · 块 '+fr.horizon+' 步'})]),
      h('td',null,[({absolute:'绝对',relative_to_anchor:'相对锚点',delta:'差分',other:'其他'})[c.io.action_param]||c.io.action_param]),
      h('td',null,[(c.used_by||[]).length?h('div',null,(c.used_by||[]).map(i=>h('div',null,[h('a',{href:hashFor(i),text:RUNS[i]?RUNS[i].name:i})]))):h('span',{class:'idle',text:'未被引用'})])]))});
  t.appendChild(tb);v.appendChild(h('div',{class:'scroll'},[t]));return v}
