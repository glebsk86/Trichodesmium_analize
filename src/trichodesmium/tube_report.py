"""Ordered B report: original, annotation, separate measurement, next photo."""
import html
import re
import textwrap
from PIL import Image, ImageDraw
from .fonts import report_font


def natural_key(name):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)',name)]


def report_objects(row):
    for proposal in row['proposals']:
        method = proposal['methods']['ensemble']
        for j,fit in enumerate(method['fits'],1):
            if fit['reasons']:
                continue
            yield dict(id=f"{proposal['source_label']}.{j}",
                       status=method['status'],
                       path=f"proposal-{proposal['source_label']:02d}/ensemble/spine-{j:02d}")


def build_report(output, manifest):
    sheets = output/'png-report';sheets.mkdir()
    font = report_font(20)
    parts = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>Находки B</title>",
             "<style>body{font:18px system-ui;background:#eee;margin:24px}section{background:white;padding:20px;margin:24px 0}img{max-width:100%;max-height:1100px}p{line-height:1.5}</style>",
             "<h1>Находки B: фото по порядку</h1>",
             "<p>Непроверенные кандидаты. Красный — контур гипотезы; голубая ось — измеренные поддержанные участки; "
             "жёлтые сечения — ширина маски. L — сумма показанных отрезков, W — среднее сечений. "
             "Шкала 3 мкм на малый интервал; автоматическая калибровка требует проверки. "
             "Перекрытия и скрытые части не измеряются. Число осей не равно числу трихомов.</p>",
             "<p><a href='comparison.html'>Подробное сравнение A/B и отклонённые гипотезы</a></p>"]
    for row in manifest['images']:
        directory = output/'images'/row['image_id']
        objects = list(report_objects(row))
        images = [(row['photo'],None)]
        def add(label,path,max_size=(960,1100)):
            with Image.open(path) as source:
                im = source.convert('RGB');im.thumbnail(max_size)
            images.append((label,im))
        add('Исходное фото · обзор',directory/'working-original.png')
        prefix = f"images/{row['image_id']}"
        original = row['original_file']
        preview = row.get('original_preview_file',original)
        parts.extend([f"<section><h2>{html.escape(row['photo'])}</h2>",
                      f"<p><a href='png-report/{row['image_id']}.png'>PNG отчёт этого фото</a></p>",
                      f"<h3>Исходное фото</h3><a href='{prefix}/{preview}'><img src='{prefix}/{preview}'></a>"])
        if preview!=original:
            parts.append(f"<p><a href='{prefix}/{original}' download>Скачать исходный HEIC/HEIF</a></p>")
        scale = row['calibration']['working_um_per_px_xy']
        caption = ('Масштаб не определён; доступны только px.' if scale is None else
                   f'Масштаб: {scale[0]:.4f} × {scale[1]:.4f} мкм/рабочий px. Требует проверки.')
        check = row['calibration']['consistency']
        if check['status']=='outlier_requires_review':
            caption += f" ВНИМАНИЕ: отклонение {check['relative_deviation']:.1%}; проверить шкалу и обрезку."
        elif check['status']=='consistent':
            caption += f" Отклонение в группе разрешения: {check['relative_deviation']:.1%}."
        images.extend((part,None) for part in textwrap.wrap(caption,width=80));parts.append(f'<p>{caption}</p>')
        if not objects:
            message = 'Ничего не найдено: нет объектов, прошедших фильтр B.'
            images.append((message,None));parts.append(f'<p>{message}</p>')
        for obj in objects:
            title = f"Объект {obj['id']} · кандидат"
            if obj['status']!='accepted_candidate':
                title += ' · требуется проверка фрагментов/узла'
            images.append((title,None))
            add('Разметка',directory/obj['path']/'annotation.png',(960,1100))
            add('Измерение',directory/obj['path']/'measurement.png',(960,1100))
            parts.extend([f'<h3>{title}</h3>',
                          f"<p>Разметка</p><a href='{prefix}/{obj['path']}/annotation.png'><img src='{prefix}/{obj['path']}/annotation.png'></a>",
                          f"<p>Измерение с подписями</p><a href='{prefix}/{obj['path']}/measurement.png'><img src='{prefix}/{obj['path']}/measurement.png'></a>"])
        # Vertical sequence deliberately preserves the same order as HTML.
        height = sum(58+(im.height if im else 0) for _,im in images)+20
        sheet = Image.new('RGB',(1000,height),'white');draw = ImageDraw.Draw(sheet);y=10
        for label,im in images:
            draw.text((20,y),label,font=font,fill='black');y+=48
            if im:
                sheet.paste(im,((1000-im.width)//2,y));y+=im.height
            y+=10
        file = sheets/f"{row['image_id']}.png";sheet.save(file)
        row['png_report'] = str(file.relative_to(output))
        parts.append('</section>')
    for error in manifest['errors']:
        parts.append(f"<section><h2>{html.escape(error['photo'] or '')}</h2><p>Ошибка обработки: {html.escape(error['error'])}. Это не отсутствие объектов.</p></section>")
    parts.append('</html>')
    (output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
