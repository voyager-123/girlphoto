"use strict";
const el = id => document.getElementById(id);
let catalog = [], token = "", file = null, processing = false, result = null;
const blobs = new Map();
const status = (message, error=false) => {el("tryStatus").textContent=message;el("tryStatus").classList.toggle("error",error);};
function replaceBlob(key, blob) {if(blobs.has(key))URL.revokeObjectURL(blobs.get(key));const url=URL.createObjectURL(blob);blobs.set(key,url);return url;}
function clearResult() {
  result=null;el("afterPhoto").hidden=true;el("afterPhoto").removeAttribute("src");el("afterEmpty").hidden=false;
  el("resultTag").textContent="等待生成";el("parameterGrid").replaceChildren();el("parameterHint").hidden=false;
  el("parameterBadge").textContent="生成后显示";el("zoomResult").disabled=true;
  for(const id of ["downloadImage","downloadParams"]){el(id).hidden=true;el(id).removeAttribute("href");}
  el("resultInfo").textContent="预览最长边 1600 像素；参数由模型读取上传的原图后预测。";
  for(const key of ["after","params"]){if(blobs.has(key)){URL.revokeObjectURL(blobs.get(key));blobs.delete(key);}}
}
function selectedModel(){return catalog.find(m=>m.id===el("tryModel").value);}
function buttons() {
  for(const id of ["choosePhoto","tryModel","tryStyle","tryStrength","refreshModels"]){el(id).disabled=processing;}
  if(!catalog.length){el("tryModel").disabled=true;el("tryStyle").disabled=true;}
  el("generate").disabled=processing||!file||!selectedModel();
  el("generate").textContent=processing?"正在处理照片…":"生成调色结果";
}
function modelChanged() {
  const model=selectedModel();el("tryStyle").replaceChildren();
  if(model){for(const style of model.styles){el("tryStyle").add(new Option(style.name,style.id));}
    el("modelNotice").textContent=model.demo_only?"演示模型：仅用合成图片训练，尚未学会真实人像审美。适合检查流程，不适合判断最终效果。":"非演示权重：请用独立照片评估效果，训练完成不等于效果已验证。";
    el("modelNotice").classList.toggle("real",!model.demo_only);el("trainingInfo").textContent=model.training;
  }
  clearResult();buttons();
}
async function refreshModels() {
  if(processing)return;processing=true;buttons();status("正在读取本机模型…");
  const previous=el("tryModel").value;
  try{
    const response=await fetch("/api/try/models"), data=await response.json();
    if(!response.ok)throw new Error(data.error||"模型读取失败，请确认使用 start_try.bat 启动。");
    catalog=data.models;token=data.token;el("tryModel").replaceChildren();
    for(const model of catalog){el("tryModel").add(new Option(`${model.name}${model.demo_only?" · 演示":""}`,model.id));}
    if(catalog.some(m=>m.id===previous))el("tryModel").value=previous;
    modelChanged();
    if(!data.available){el("modelNotice").textContent=data.message;status(data.message,true);}
    else status(file?"模型已就绪，点击生成调色结果。":"模型已就绪，请选择照片。");
  }catch(error){catalog=[];clearResult();el("tryModel").replaceChildren();el("tryStyle").replaceChildren();status(error.message,true);el("modelNotice").textContent="模型读取失败。请确认依赖与权重可用，再点击刷新模型。";}
  finally{processing=false;buttons();}
}
el("refreshModels").onclick=refreshModels;
el("choosePhoto").onclick=()=>el("tryFile").click();
el("tryFile").onchange=()=>{
  const selected=el("tryFile").files[0];el("tryFile").value="";
  if(!selected||processing)return;
  if(!/\.(jpe?g|png|webp)$/i.test(selected.name)||selected.size>25*1024*1024){status("请选择小于 25 MB 的 JPEG、PNG 或 WebP 图片。",true);return;}
  file=selected;clearResult();el("beforePhoto").src=replaceBlob("before",file);el("beforePhoto").hidden=false;el("beforeEmpty").hidden=true;
  el("fileInfo").textContent=`${file.name} · ${(file.size/1024/1024).toFixed(2)} MB · 不加入训练集`;
  status("照片已选择。设置风格与强度后，点击生成。");buttons();
};
el("tryModel").onchange=()=>{modelChanged();status("模型已切换，请重新生成结果。");};
el("tryStyle").onchange=()=>{clearResult();status("风格已修改，请重新生成结果。");};
el("tryStrength").oninput=()=>{el("strengthValue").textContent=el("tryStrength").value+"%";clearResult();status("强度已修改，请重新生成结果。");};
const readBase64 = file => new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result).split(",")[1]);reader.onerror=()=>reject(new Error("照片读取失败。"));reader.readAsDataURL(file);});
function pngBlob(base64){const binary=atob(base64);const bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);return new Blob([bytes],{type:"image/png"});}
function showParameters(parameters){
  const values=parameters.effective;
  const rows=[["曝光",values.exposure_stops,"EV","正值提亮，负值变暗"],["对比度",values.contrast_multiplier,"×","1 为中性"],["冷暖",values.warmth_rgb_offset,"","正值偏暖，负值偏冷"],["绿 / 洋红",values.tint_rgb_offset,"","正值偏绿，负值偏洋红"],["饱和度",values.saturation_multiplier,"×","1 为中性"],["暗部调整",values.shadow_offset,"","正值抬升暗部"]];
  el("parameterGrid").replaceChildren();
  for(const [name,value,unit,hint] of rows){const card=document.createElement("div");card.className="parameter";
    const label=document.createElement("span"),number=document.createElement("b"),note=document.createElement("small");
    label.textContent=name;number.textContent=`${value.toFixed(3)}${unit?" "+unit:""}`;note.textContent=hint;card.append(label,number,note);el("parameterGrid").append(card);}
  el("parameterHint").hidden=true;el("parameterBadge").textContent=parameters.engine;
}
el("generate").onclick=async()=>{
  if(processing||!file||!selectedModel())return;
  processing=true;buttons();clearResult();status("正在生成…首次加载模型可能稍慢。");
  try{
    const response=await fetch("/api/try/predict",{method:"POST",headers:{"Content-Type":"application/json","X-Girlphoto-Token":token},
      body:JSON.stringify({model_id:el("tryModel").value,style:el("tryStyle").value,strength:Number(el("tryStrength").value)/100,image:await readBase64(file)})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||"处理失败，请换一张照片重试。");
    result=data;el("beforePhoto").src=replaceBlob("before",pngBlob(data.before_png));el("afterPhoto").src=replaceBlob("after",pngBlob(data.after_png));
    el("beforePhoto").hidden=false;el("afterPhoto").hidden=false;el("beforeEmpty").hidden=true;el("afterEmpty").hidden=true;
    el("resultTag").textContent=`${data.model.demo_only?"演示 · ":""}${el("tryStyle").selectedOptions[0].text} · ${Math.round(data.parameters.strength*100)}%`;
    el("downloadImage").href=blobs.get("after");el("downloadImage").hidden=false;
    el("downloadParams").href=replaceBlob("params",new Blob([JSON.stringify(data.parameters,null,2)],{type:"application/json"}));el("downloadParams").hidden=false;
    el("zoomResult").disabled=false;showParameters(data.parameters);
    el("resultInfo").textContent=`处理 ${data.elapsed_ms} ms · 原图 ${data.parameters.source_size.join(" × ")} · 预览 ${data.parameters.preview_size.join(" × ")}`;
    status(data.model.demo_only?"已生成演示结果。请重点检查流程；此模型尚未用真实人像训练。":"结果已生成。请对照原图观察肤色与明暗变化。");
  }catch(error){status(error.message,true);}
  finally{processing=false;buttons();}
};
el("zoomResult").onclick=()=>{if(!result)return;el("largeBefore").src=blobs.get("before");el("largeAfter").src=blobs.get("after");el("compareDialog").showModal();};
el("closeCompare").onclick=()=>el("compareDialog").close();
refreshModels();
