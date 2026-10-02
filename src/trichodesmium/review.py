"""Self-contained offline mask correction; no installation or web service."""
import base64
import html
import io
import json

import numpy as np
from PIL import Image


def png_data(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode("ascii")


def write_editor(path, rgb, labels, rows, photo_name):
    # Only reported objects enter the editor, renumbered 1..N.
    if len(rows) > 255:
        return
    seed = np.zeros(labels.shape, np.uint8)
    for index, row in enumerate(rows, 1):
        seed[labels == row["source_label"]] = index
        row["review_instance_id"] = index
    payload = {"source": png_data(rgb), "seed": png_data(seed),
               "filename": photo_name.rsplit("/", 1)[-1].rsplit(".", 1)[0]+".png"}
    data = json.dumps(payload).replace("<", "\\u003c")
    page = TEMPLATE.replace("__DATA__", data).replace("__TITLE__", html.escape(photo_name))
    path.write_text(page, encoding="utf-8")


TEMPLATE = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Исправление маски</title>
<style>body{font:16px system-ui;margin:20px;color:#20323a;background:#f4f6f7}h1{font-size:22px}
button,input,select{font:inherit;margin:4px;padding:7px}button{cursor:pointer}#viewport{overflow:auto;max-height:72vh;background:#182126}
canvas{display:block;touch-action:none;cursor:crosshair}#toolbar{display:flex;align-items:center;flex-wrap:wrap}
#status{font-weight:600}p{max-width:1000px;line-height:1.5}.note{color:#875500}</style>
<h1>__TITLE__</h1>
<p>Это кандидаты, а не подтверждённые нити. Каждый цвет — отдельный ID. Выберите ID и проведите кистью по видимой нити.
Толщина кисти задаётся в пикселях исходного фото. Ластик удаляет маску. Проверяйте границы на увеличении;
слабые участки и отдельные кандидаты могут относиться к одной нити.</p>
<p class="note">Рисуйте только видимую часть. Программа не измерит единую длину разорванной или разветвлённой маски.
При перекрытии двумя ID один пиксель принадлежит последнему нарисованному объекту.</p>
<div id="toolbar"><label>ID <input id="object" type="number" min="1" max="255" value="1" style="width:65px"></label>
<label>Кисть <input id="brush" type="number" min="1" max="100" value="12" style="width:65px"></label>
<label><input type="checkbox" id="erase">Ластик</label>
<label>Масштаб <select id="zoom"><option value=".5">50%</option><option value="1" selected>100%</option><option value="2">200%</option></select></label>
<label><input type="checkbox" id="show" checked>Показать маску</label>
<button id="undo">Отменить штрих</button><button id="clear">Очистить маску</button>
<button id="restore">Вернуть кандидатов</button><button id="save" disabled>Скачать маску PNG</button></div>
<p id="status" role="status">Загрузка…</p><div id="viewport"><canvas id="view"></canvas></div>
<p>Сложите скачанные PNG в папку <code>corrected_masks</code> и повторите обработку с
<code>--mask-dir ./corrected_masks --mask-format instances</code> в новую папку результатов.
Для вложенных исходных папок восстановите такую же структуру в <code>corrected_masks</code>.
Скачанная маска хранит ID, поэтому сама по себе выглядит почти чёрной. Фото и цветная разметка видны здесь и в галерее.
Изменения сохраняются только после нажатия «Скачать маску PNG».</p>
<script>
"use strict";
const data=__DATA__, view=document.querySelector('#view'), ctx=view.getContext('2d');
const mask=document.createElement('canvas'), mctx=mask.getContext('2d',{willReadFrequently:true});
const tint=document.createElement('canvas'), tctx=tint.getContext('2d');
const status=document.querySelector('#status'), source=new Image(), seed=new Image(), history=[];
let ready=false, down=false, previous=null, changed=false;
const palette=[[255,60,80],[25,195,255],[40,230,100],[255,200,30],[225,80,255],[255,140,40]];
function idValue(){return Math.round(Math.max(1,Math.min(255,Number(document.querySelector('#object').value)||1)));}
function message(text){status.textContent=text;}
function snapshot(){history.push(mctx.getImageData(0,0,mask.width,mask.height));if(history.length>12)history.shift();}
function draw(){
 ctx.drawImage(source,0,0);
 if(document.querySelector('#show').checked){
  const original=mctx.getImageData(0,0,mask.width,mask.height), coloured=tctx.createImageData(mask.width,mask.height);
  for(let i=0;i<original.data.length;i+=4){const id=original.data[i];if(id){const c=palette[(id-1)%palette.length];coloured.data.set([c[0],c[1],c[2],115],i);}}
  tctx.putImageData(coloured,0,0);ctx.drawImage(tint,0,0);
 }
}
function position(event){const r=view.getBoundingClientRect();return[(event.clientX-r.left)*view.width/r.width,(event.clientY-r.top)*view.height/r.height];}
function paint(a,b){
 const id=document.querySelector('#erase').checked?0:idValue();
 const width=Math.max(1,Math.min(100,Number(document.querySelector('#brush').value)||12));
 // Rasterize numeric labels directly. Canvas strokes would antialias IDs.
 const x0=Math.max(0,Math.floor(Math.min(a[0],b[0])-width)),y0=Math.max(0,Math.floor(Math.min(a[1],b[1])-width));
 const x1=Math.min(mask.width,Math.ceil(Math.max(a[0],b[0])+width)),y1=Math.min(mask.height,Math.ceil(Math.max(a[1],b[1])+width));
 if(x1>x0&&y1>y0){
  const patch=mctx.getImageData(x0,y0,x1-x0,y1-y0);
  for(let y=0;y<y1-y0;y++)for(let x=0;x<x1-x0;x++){
   const px=x0+x+.5,py=y0+y+.5,dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy;
   const f=den?Math.max(0,Math.min(1,((px-a[0])*dx+(py-a[1])*dy)/den)):0;
   const index=4*(y*(x1-x0)+x),inside=Math.hypot(px-a[0]-f*dx,py-a[1]-f*dy)<=width/2;
   if(inside)patch.data.set([id,id,id,255],index);
  }
  mctx.putImageData(patch,x0,y0);
 }
 changed=true;message('Есть изменения — скачайте маску, чтобы сохранить их.');draw();
}
view.addEventListener('pointerdown',e=>{if(!ready)return;e.preventDefault();snapshot();down=true;view.setPointerCapture(e.pointerId);previous=position(e);paint(previous,previous);});
view.addEventListener('pointermove',e=>{if(!down)return;const p=position(e);paint(previous,p);previous=p;});
view.addEventListener('pointerup',()=>{down=false;previous=null;});
view.addEventListener('pointercancel',()=>{down=false;previous=null;});
document.querySelector('#undo').onclick=()=>{if(history.length){mctx.putImageData(history.pop(),0,0);changed=true;draw();message('Штрих отменён. Скачайте маску для сохранения.');}};
document.querySelector('#clear').onclick=()=>{if(!ready)return;snapshot();mctx.fillStyle='black';mctx.fillRect(0,0,mask.width,mask.height);changed=true;draw();message('Маска очищена.');};
document.querySelector('#restore').onclick=()=>{if(!ready)return;snapshot();mctx.drawImage(seed,0,0);changed=true;draw();message('Исходные кандидаты восстановлены.');};
document.querySelector('#show').onchange=()=>{if(ready)draw();};
document.querySelector('#zoom').onchange=()=>{const z=Number(document.querySelector('#zoom').value);view.style.width=view.width*z+'px';view.style.height=view.height*z+'px';};
document.querySelector('#save').onclick=()=>{mask.toBlob(blob=>{if(!blob){message('Не удалось сформировать PNG.');return;}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=data.filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);changed=false;message('Маска скачана: '+data.filename+'. Повторите обработку с --mask-format instances.');},'image/png');};
window.addEventListener('beforeunload',e=>{if(changed){e.preventDefault();e.returnValue='';}});
Promise.all([new Promise((resolve,reject)=>{source.onload=resolve;source.onerror=reject;source.src=data.source;}),new Promise((resolve,reject)=>{seed.onload=resolve;seed.onerror=reject;seed.src=data.seed;})]).then(()=>{
 view.width=mask.width=tint.width=source.naturalWidth;view.height=mask.height=tint.height=source.naturalHeight;
 mctx.drawImage(seed,0,0);ready=true;document.querySelector('#save').disabled=false;draw();message('Готово. Исходные кандидаты загружены; проверьте всю фотографию.');
}).catch(()=>message('Не удалось загрузить изображения редактора.'));
</script></html>'''
