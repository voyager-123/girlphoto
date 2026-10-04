"use strict";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
let state = null, activeTab = "single", currentId = null, currentPair = null, dirty = false, busy = false, editingId = null;
let pairRequest = 0, currentPairStyle = null, previousQueue = "pending";
let reviewer = localStorage.getItem("girlphoto-reviewer") || "我";
$("#reviewer").value = reviewer;
const escapeHTML = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const notice = (message, error=false) => { $("#status").textContent = message; $("#status").classList.toggle("error", error); };
const safely = (fn) => (...args) => { try { return Promise.resolve(fn(...args)).catch(e => notice(e.message, true)); } catch(e) { notice(e.message, true); } };

async function api(path, payload) {
  const options = payload === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-Girlphoto-Token":state?.token || ""}, body:JSON.stringify(payload)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败，请重试。");
  return data;
}
function canLeave() { return !dirty || confirm("当前修改还没有保存。确定离开吗？"); }
function setBusy(value) { busy=value; $$("#save,#skip,[data-vote],#undo,#pairStyle,#retryPairs,#refreshPair,#switchReviewer,#queue").forEach(el => el.disabled=value); }
function queue() {
  if (!state) return [];
  const filter=$("#queue").value;
  return state.photos.filter(p => filter==="all" || (filter==="pending" ? !p.label : p.label?.status===filter));
}
async function refresh() {
  const selection=$("#pairStyle").value;
  state=await api("/api/state?reviewer="+encodeURIComponent(reviewer));
  const labeled=state.photos.filter(p=>p.label?.status==="labeled").length;
  const compared=Object.entries(state.pair_counts).filter(([key])=>key!=="skip").reduce((sum,[,n])=>sum+n,0);
  $("#stats").innerHTML = [[state.photos.length,"照片"],[labeled,"已标注"],[compared,"有效比较"]].map(([n,label])=>`<div class="stat"><b>${n}</b><span>${label}</span></div>`).join("");
  $("#pairStyle").innerHTML=state.styles.map(s=>`<option value="${escapeHTML(s.id)}">${escapeHTML(s.name)}</option>`).join("");
  if (state.styles.some(s=>s.id===selection)) $("#pairStyle").value=selection;
  $("#dataPath").textContent="当前数据目录："+state.data_dir;
  renderLibrary();
}
function renderSingle(id=currentId) {
  const photos=queue();
  const photo=photos.find(p=>p.id===id)||photos[0];
  currentId=photo?.id||null; dirty=false;
  $("#singleEmpty").hidden=!!photo; $("#singleWork").hidden=!photo;
  $("#queueInfo").textContent=photo ? `${photos.findIndex(p=>p.id===photo.id)+1} / ${photos.length}` : "队列为空";
  $("#previous").disabled=!photo; $("#next").disabled=!photo;
  if (!photo) {
    $("#singleEmpty h2").textContent=state.photos.length ? "这个队列已经处理完了" : "从第一组照片开始";
    $("#singleEmpty>p").textContent=state.photos.length ? "可以切换到全部照片，或进入版本比较。" : "导入你的原图与调色版本，或载入合成演示图体验流程。";
    return;
  }
  const label=photo.label;
  $("#mainPhoto").src=photo.url;
  $("#demoBadge").hidden=!photo.is_demo;
  $("#photoInfo").textContent=`${photo.name} · 原图组 ${photo.group_id} · ${photo.width} × ${photo.height} · ${photo.split}`;
  $("#colorNote").textContent=photo.color_note;
  $("#styleFields").innerHTML=state.styles.map(s=>{
    const selected=label?.styles[s.id]||"";
    return `<div class="style-row"><label><span>${escapeHTML(s.name)}</span><select data-style="${escapeHTML(s.id)}" aria-label="${escapeHTML(s.name)}风格">${[["","尚未判断"],["yes","符合"],["no","不符合"],["unknown","不确定"]].map(([value,text])=>`<option value="${value}" ${value===selected?"selected":""}>${text}</option>`).join("")}</select></label><p>${escapeHTML(s.description)}</p></div>`;
  }).join("");
  $$('input[name="quality"]').forEach(el=>el.checked=el.value===String(label?.quality??""));
  $("#defectFields").innerHTML=state.defects.map(d=>`<label><input type="checkbox" value="${escapeHTML(d)}" ${label?.defects.includes(d)?"checked":""}>${escapeHTML(d)}</label>`).join("");
  $("#note").value=label?.note||"";
}
async function switchTab(tab) {
  if (busy || !canLeave()) return;
  dirty=false; activeTab=tab;
  $$(".panel").forEach(el=>el.hidden=el.id!==tab);
  $$(".tab").forEach(el=>{el.classList.toggle("active",el.dataset.tab===tab);el.setAttribute("aria-pressed",String(el.dataset.tab===tab));});
  if(tab==="single") renderSingle();
  if(tab==="compare") await nextPair();
  if(tab==="library") renderLibrary();
}
$$(".tab").forEach(el=>el.addEventListener("click",safely(()=>switchTab(el.dataset.tab))));
$("#labelForm").addEventListener("input",()=>dirty=true);
$("#labelForm").addEventListener("change",()=>dirty=true);
$("#queue").addEventListener("change",()=>{if(canLeave()){previousQueue=$("#queue").value;renderSingle(null);}else{$("#queue").value=previousQueue;}});
function navigate(delta) {if(busy||!canLeave())return;const photos=queue();const index=photos.findIndex(p=>p.id===currentId);renderSingle(photos[(index+delta+photos.length)%photos.length]?.id);}
$("#previous").onclick=()=>navigate(-1); $("#next").onclick=()=>navigate(1);
async function saveLabel(status="labeled") {
  if(busy||!currentId)return;
  const photos=queue(), index=photos.findIndex(p=>p.id===currentId), nextId=photos[index+1]?.id;
  const styles={}; $$('[data-style]').forEach(el=>{if(el.value) styles[el.dataset.style]=el.value;});
  const value=$('input[name="quality"]:checked')?.value;
  const payload={photo_id:currentId,reviewer,quality:value?Number(value):null,styles,
    defects:$$('#defectFields input:checked').map(el=>el.value),note:$("#note").value,status};
  setBusy(true);
  try { await api("/api/label",payload); dirty=false;await refresh();renderSingle(nextId||currentId);notice(status==="skipped"?"已记录跳过，可在已跳过队列继续。":"已保存到本机。"); }
  finally {setBusy(false);}
}
$("#labelForm").onsubmit=safely(event=>{event.preventDefault();return saveLabel();});
$("#skip").onclick=safely(()=>saveLabel("skipped"));
async function nextPair(){
  const request=++pairRequest, style=$("#pairStyle").value;
  currentPair=null;$("#pairWork").hidden=true;$("#pairEmpty").hidden=true;
  const result=await api(`/api/pair?reviewer=${encodeURIComponent(reviewer)}&style=${encodeURIComponent(style)}&retry=${$("#retryPairs").checked?1:0}`);
  if(request!==pairRequest)return;
  currentPairStyle=style;
  currentPair=result.pair;
  $("#pairInfo").textContent=`当前风格剩余 ${result.remaining} 组比较`;
  $("#pairEmpty").hidden=!!currentPair;$("#pairWork").hidden=!currentPair;
  if(currentPair){$("#leftPhoto").src=currentPair[0].url;$("#rightPhoto").src=currentPair[1].url;}
}
$("#pairStyle").onchange=safely(nextPair);$("#retryPairs").onchange=safely(nextPair);$("#refreshPair").onclick=safely(nextPair);
async function vote(outcome){
  if(busy||!currentPair)return;setBusy(true);
  try{await api("/api/pair",{reviewer,left:currentPair[0].id,right:currentPair[1].id,style_id:currentPairStyle,outcome});await refresh();await nextPair();notice("比较已保存。左右顺序会随机变化。");}
  finally{setBusy(false);}
}
$$('[data-vote]').forEach(el=>el.onclick=safely(()=>vote(el.dataset.vote)));
$("#undo").onclick=safely(async()=>{
  if(busy||!canLeave())return;
  const result=await api("/api/undo",{reviewer});await refresh();
  if(activeTab==="single")renderSingle(result.kind==="label"?result.key[0]:currentId);
  if(activeTab==="compare")await nextPair();
  notice(result.undone?"已撤销上次标注；先前记录已恢复。":"没有可撤销的标注。");
});
$("#switchReviewer").onclick=safely(async()=>{
  if(busy||!canLeave())return;
  const next=$("#reviewer").value.trim();if(!next)throw new Error("请填写标注者名称。");
  reviewer=next;localStorage.setItem("girlphoto-reviewer",reviewer);dirty=false;await refresh();renderSingle(null);if(activeTab==="compare")await nextPair();notice("已切换标注者。每个人的记录独立保存。");
});
function showZoom(url){if(!url)return;$("#zoomPhoto").src=url;$("#zoomDialog").showModal();}
$("#openPhoto").onclick=()=>showZoom($("#mainPhoto").src);
$("#zoomLeft").onclick=()=>showZoom(currentPair?.[0].url);$("#zoomRight").onclick=()=>showZoom(currentPair?.[1].url);
$("#closeZoom").onclick=()=>$("#zoomDialog").close();
$("#emptyImport").onclick=safely(()=>switchTab("library"));$("#goLibrary").onclick=safely(()=>switchTab("library"));
$("#demo").onclick=safely(async()=>{if(busy)return;setBusy(true);try{const r=await api("/api/demo",{});await refresh();renderSingle(null);notice(`新增 ${r.created} 张合成演示图。请先标注，再试版本比较。`);}finally{setBusy(false);}});
$("#pickFiles").onclick=()=>$("#files").click();$("#pickFolder").onclick=()=>$("#folder").click();
async function importFiles(files){
  if(busy)return;
  const selected=[...files].filter(f=>/\.(jpe?g|png|webp)$/i.test(f.name));
  if(!selected.length)throw new Error("没有可导入的 JPEG / PNG / WebP 图片。");
  const group=$("#importGroup").value.trim(),cohort=$("#importCohort").value.trim();
  let added=0,duplicates=0,errors=[];setBusy(true);$("#pickFiles").disabled=true;$("#pickFolder").disabled=true;
  try{
    for(let i=0;i<selected.length;i++){
      const file=selected[i];$("#importProgress").textContent=`${i+1} / ${selected.length} · ${file.name}`;
      try{
        if(file.size>25*1024*1024)throw new Error("大于 25 MB");
        const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(",")[1]);reader.onerror=()=>reject(new Error("文件读取失败"));reader.readAsDataURL(file);});
        const result=await api("/api/import",{name:file.webkitRelativePath||file.name,data,group_id:group||null,cohort_id:cohort||null});
        if(result.duplicate)duplicates++;else added++;
      }catch(e){errors.push(`${file.name}：${e.message}`);}
    }
    await refresh();renderSingle();
    $("#importProgress").textContent=`已导入 ${added} 张，重复 ${duplicates} 张，失败 ${errors.length} 张。`;
    notice(errors.length?errors.join("；"):`导入完成。请检查原图组与拍摄批次，再开始标注。`,errors.length>0);
  }finally{setBusy(false);$("#pickFiles").disabled=false;$("#pickFolder").disabled=false;$("#files").value="";$("#folder").value="";}
}
$("#files").onchange=safely(e=>importFiles(e.target.files));$("#folder").onchange=safely(e=>importFiles(e.target.files));
function renderLibrary(){
  if(!state)return;
  const needle=$("#librarySearch").value.toLowerCase();const photos=state.photos.filter(p=>(p.name+" "+p.group_id+" "+p.cohort_id).toLowerCase().includes(needle));
  if(!photos.length){$("#photoTable").innerHTML='<p class="hint">暂无匹配照片。</p>';return;}
  $("#photoTable").innerHTML=`<div class="table-wrap"><table><thead><tr><th>照片</th><th>原图组 / 拍摄批次</th><th>数据集划分</th><th>当前标注者</th><th>操作</th></tr></thead><tbody>${photos.map(p=>`<tr><td class="name"><div class="name-cell"><img class="thumb" src="${p.url}" alt="" loading="lazy"><span>${escapeHTML(p.name)}${p.is_demo?'<br><span class="hint">合成演示</span>':''}</span></div></td><td>${escapeHTML(p.group_id)}<br><span class="hint">${escapeHTML(p.cohort_id)}</span></td><td><select class="split-select" data-cohort="${escapeHTML(p.cohort_id)}" aria-label="${escapeHTML(p.name)}数据集划分">${["train","validation","test"].map(s=>`<option ${s===p.split?'selected':''}>${s}</option>`).join("")}</select></td><td>${p.label?(p.label.status==='skipped'?'已跳过':`已标注 · 质量 ${p.label.quality??'未评'}`):'未标注'}</td><td><div class="table-actions"><button data-open="${p.id}">标注</button><button data-edit="${p.id}">分组</button></div></td></tr>`).join("")}</tbody></table></div>`;
  $$('[data-open]').forEach(el=>el.onclick=safely(async()=>{if(!canLeave())return;dirty=false;$("#queue").value="all";previousQueue="all";currentId=el.dataset.open;await switchTab("single");}));
  $$('[data-edit]').forEach(el=>el.onclick=()=>{const p=state.photos.find(p=>p.id===el.dataset.edit);editingId=p.id;$("#editGroup").value=p.group_id;$("#editCohort").value=p.cohort_id;$("#editSplit").value=p.split;$("#groupDialog").showModal();});
  $$('[data-cohort]').forEach(el=>el.onchange=safely(async()=>{
    if(!confirm(`整个拍摄批次“${el.dataset.cohort}”都将改为 ${el.value}。继续吗？`)){renderLibrary();return;}
    await api("/api/split",{cohort_id:el.dataset.cohort,split:el.value});await refresh();notice("已修改整个批次的划分。");
  }));
}
$("#librarySearch").oninput=renderLibrary;
$("#groupForm").onsubmit=safely(async e=>{e.preventDefault();await api("/api/group",{photo_id:editingId,group_id:$("#editGroup").value,cohort_id:$("#editCohort").value,split:$("#editSplit").value});$("#groupDialog").close();await refresh();notice("分组已保存。");});
$("#cancelGroup").onclick=()=>$("#groupDialog").close();
$("#addStyle").onclick=safely(async()=>{await api("/api/style",{name:$("#newStyleName").value,description:$("#newStyleDescription").value});$("#newStyleName").value="";$("#newStyleDescription").value="";await refresh();notice("新风格已添加，旧标注中的该类别保持未判断。");});
document.addEventListener("keydown",safely(async e=>{
  if(busy||e.ctrlKey||e.metaKey||e.altKey||$("dialog[open]"))return;
  if(["INPUT","SELECT","TEXTAREA"].includes(e.target.tagName))return;
  if(activeTab==="single"&&currentId){
    if(["1","2","3"].includes(e.key)){$(`input[name="quality"][value="${e.key}"]`).checked=true;dirty=true;e.preventDefault();}
    if(e.key==="Enter"){e.preventDefault();await saveLabel();}
  }
  if(activeTab==="compare"&&currentPair){
    const keys={ArrowLeft:"left",ArrowRight:"right","=":"tie",s:"skip"};if(keys[e.key]){e.preventDefault();await vote(keys[e.key]);}
  }
}));
window.addEventListener("beforeunload",e=>{if(dirty){e.preventDefault();e.returnValue="";}});
safely(async()=>{await refresh();renderSingle();})();
