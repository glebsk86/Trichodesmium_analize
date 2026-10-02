// Optional developer check: npm install @napi-rs/canvas; node tools/check_editor.cjs editor.html export.png
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const {createCanvas,Image}=require('@napi-rs/canvas');
const page=fs.readFileSync(process.argv[2],'utf8'),script=page.match(/<script>([\s\S]*?)<\/script>/)[1],nodes=new Map();
let blobSaved=null;
function canvas(){const c=createCanvas(1,1);c.style={};c.listeners={};c.addEventListener=(n,f)=>c.listeners[n]=f;c.setPointerCapture=()=>{};c.getBoundingClientRect=()=>({left:0,top:0,width:c.width,height:c.height});c.toBlob=(fn)=>{blobSaved=c.toBuffer('image/png');fn(new Blob([blobSaved],{type:'image/png'}));};return c;}
const view=canvas();nodes.set('#view',view);
for(const n of ['object','brush','erase','zoom','show','undo','clear','restore','save','status'])nodes.set('#'+n,{value:n==='object'?'1':n==='brush'?'12':'1',checked:n==='show',style:{},textContent:''});
const context=vm.createContext({console,Image,Blob,setTimeout,Promise,Uint8ClampedArray,
 URL:{createObjectURL:()=> 'blob:test',revokeObjectURL:()=>{}},
 window:{addEventListener:()=>{}},document:{querySelector:s=>nodes.get(s),createElement:t=>t==='canvas'?canvas():{click(){}}}});
vm.runInContext(script,context);
(async()=>{
 for(let i=0;i<50&&!vm.runInContext('ready',context);i++)await new Promise(r=>setTimeout(r,20));
 assert(vm.runInContext('ready',context));assert.equal(view.width,160);assert.equal(view.height,100);
 nodes.get('#object').value='7';nodes.get('#brush').value='8';
 view.listeners.pointerdown({preventDefault(){},clientX:50,clientY:20,pointerId:1});
 view.listeners.pointermove({clientX:110,clientY:20,pointerId:1});view.listeners.pointerup();
 let pixels=vm.runInContext('mctx.getImageData(0,0,mask.width,mask.height).data',context);
 let ids=new Set();for(let i=0;i<pixels.length;i+=4){ids.add(pixels[i]);assert.equal(pixels[i],pixels[i+1]);assert.equal(pixels[i],pixels[i+2]);assert.equal(pixels[i+3],255);}assert.deepEqual([...ids].sort((a,b)=>a-b),[0,1,7]);
 assert.equal(pixels[4*(20*160+80)],7);assert.equal(pixels[4*(45*160+80)],1);
 nodes.get('#undo').onclick();pixels=vm.runInContext('mctx.getImageData(0,0,mask.width,mask.height).data',context);assert.equal(pixels[4*(20*160+80)],0);
 nodes.get('#erase').checked=true;nodes.get('#brush').value='20';view.listeners.pointerdown({preventDefault(){},clientX:80,clientY:45,pointerId:2});view.listeners.pointerup();
 pixels=vm.runInContext('mctx.getImageData(0,0,mask.width,mask.height).data',context);assert.equal(pixels[4*(45*160+80)],0);
 nodes.get('#save').onclick();assert(blobSaved.length>0);fs.writeFileSync(process.argv[3],blobSaved);
 nodes.get('#clear').onclick();pixels=vm.runInContext('mctx.getImageData(0,0,mask.width,mask.height).data',context);assert.equal(Math.max(...pixels.filter((_,i)=>i%4===0)),0);
 nodes.get('#restore').onclick();pixels=vm.runInContext('mctx.getImageData(0,0,mask.width,mask.height).data',context);assert.equal(pixels[4*(45*160+80)],1);
 console.log('Editor smoke checks passed: load, drawing IDs, no antialias IDs, undo, eraser, PNG export, clear, restore.');
})().catch(e=>{console.error(e);process.exitCode=1;});
