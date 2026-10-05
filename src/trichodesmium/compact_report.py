"""Self-contained summary HTML; no PNG/mask/JSON files in limited mode."""
import base64
import hashlib
import html
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from .imaging import field_mask
from .tubular import compare_proposal
from .tube_compare import global_overlaps,contour
from .tube_guides import measurement_guides,draw_guides,guide_label,with_header


def embedded(image):
    if not isinstance(image,Image.Image):image=Image.fromarray(image)
    buffer=BytesIO();image.save(buffer,format='PNG',compress_level=1)
    return '<img src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'


def write_result(output,source,relative,ordinal,original,rgb,decoding,labels,obstruction,calibration,settings,segmentation,links):
    field=field_mask(rgb)
    proposals=[dict(source_label=int(i),result=compare_proposal(rgb,labels==i,field,obstruction,settings)) for i in np.unique(labels) if i>0]
    events=global_overlaps(proposals,settings)
    row=dict(photo=relative,decoded_rgb_sha256=hashlib.sha256(original.tobytes()+str(original.shape).encode()).hexdigest(),
             calibration=calibration,decoding=decoding,segmentation=segmentation,fragment_associations=links,
             cross_proposal_overlaps=events,proposals=[])
    parts=[f'<h2>{html.escape(relative)}</h2><h3>Исходное фото · рабочий обзор</h3>',embedded(rgb)]
    objects=0
    for proposal in proposals:
        ident=proposal['source_label'];method=proposal['result']['methods']['ensemble'];records=[]
        for j,fit in enumerate(method['fits'],1):
            guides=measurement_guides(fit,calibration=calibration)
            records.append({'reasons':fit['reasons'],'measurement_guides':guides,'status':fit['status']})
            if fit['reasons']:continue
            objects+=1;annotation=rgb.copy();contour(annotation,fit['template'],(225,35,15))
            measured=annotation.copy();draw_guides(measured,fit,guides)
            yy,xx=np.nonzero(fit['template']);pad=30
            x0,y0=max(0,int(xx.min())-pad),max(0,int(yy.min())-pad)
            x1,y1=min(rgb.shape[1],int(xx.max())+pad+1),min(rgb.shape[0],int(yy.max())+pad+1)
            parts.extend([f'<h3>Объект {ident}.{j} · {html.escape(method["status"])}</h3><p>Разметка</p>',
                          embedded(annotation[y0:y1,x0:x1]),'<p>Измерение</p>',
                          embedded(with_header(measured[y0:y1,x0:x1],[guide_label(f'{ident}.{j}',guides)])[0])])
        row['proposals'].append({'source_label':ident,'methods':{'ensemble':{'status':method['status'],'fits':records}}})
    if not objects:parts.append('<p>Ничего не найдено: нет объектов, прошедших фильтр B.</p>')
    row['compact_html']='\n'.join(parts)
    return row


def build_report(output,manifest):
    parts=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Находки B</title>',
           '<style>body{font:18px system-ui;margin:24px;background:#eee}section{background:white;padding:20px;margin:24px 0}img{max-width:100%;max-height:1100px}p{line-height:1.5}</style>',
           '<h1>Находки B: фото по порядку</h1>',
           '<p>Ограниченный вывод. Непроверенные кандидаты. Красный — контур; голубой — поддержанная часть оси; штрих — оценка в разрыве. '
           'L оси включает всю прослеженную линию, L видимого — только наблюдаемую часть. W — ширина маски. Автоматический масштаб требует проверки.</p>']
    for row in manifest['images']:
        factors=row['calibration']['working_um_per_px_xy']
        caption='Масштаб не определён; доступны только px.' if factors is None else f'Масштаб: {factors[0]:.4f} × {factors[1]:.4f} мкм/рабочий px; требует проверки.'
        check=row['calibration']['consistency']
        if check['status']=='outlier_requires_review':caption+=' Отклонение масштаба в группе разрешения — проверить шкалу.'
        parts.extend(['<section>',f'<p>{html.escape(caption)}</p>',row.pop('compact_html'),'</section>'])
    for error in manifest['errors']:
        parts.append(f'<section><h2>{html.escape(error["photo"])}</h2><p>Ошибка обработки: {html.escape(error["error"])}. Это не отсутствие объектов.</p></section>')
    parts.append('</html>');(output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
