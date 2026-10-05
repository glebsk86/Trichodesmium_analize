"""One local command: photo directory -> proposals -> primary B report."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

from openpyxl import Workbook
from openpyxl.styles import Font
from . import __version__
from . import cli, tube_compare
from .reporting import write_json

COLUMNS = ['Фото','ID кандидата','Статус проверки','Длина поддержанных участков, мкм',
           'Средняя ширина маски, мкм','Длина поддержанных участков, рабочие px',
           'Средняя ширина маски, рабочие px','Сечений ширины','Масштаб исходника, мкм/px',
           'Масштаб рабочего растра X, мкм/px','Масштаб рабочего растра Y, мкм/px',
           'Вид','Полная длина цепи, мкм','Число клеток']


def tables(output, manifest):
    rows = []
    for photo in manifest['images']:
        scale = photo['calibration'];factors = scale['working_um_per_px_xy'] or [None,None]
        for proposal in photo['proposals']:
            method = proposal['methods']['ensemble']
            for i,fit in enumerate(method['fits'],1):
                if fit['reasons']:continue
                g = fit['measurement_guides']
                rows.append([photo['photo'],f"{proposal['source_label']}.{i}",
                             tube_compare.STATUS.get(method['status'],method['status']),
                             g['length_um'],g['width_um'],g['visible_axis_length_px'],g['mean_fitted_width_px'],
                             len(g['width_samples_px']),scale['source_calibration']['um_per_px'],
                             *factors,'не определён',None,None])
    with (output/'B-measurements.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.writer(stream,delimiter=';');writer.writerow(COLUMNS);writer.writerows(rows)
    wb = Workbook();ws = wb.active;ws.title='Кандидаты B';ws.append(COLUMNS)
    for row in rows:ws.append(row)
    for cell in ws[1]:cell.font = Font(bold=True)
    ws.freeze_panes = 'A2';ws.auto_filter.ref = ws.dimensions
    for column in ws.columns:ws.column_dimensions[column[0].column_letter].width = min(48,max(16,len(column[0].value or '')+2))
    notes = wb.create_sheet('Определения')
    for line in ["Объекты — непроверенные кандидаты; ID оси не равен доказанному трихому.",
                 "L — сумма поддержанных отрезков оси; пропуски/шкала/общие узлы исключены.",
                 "W — средняя ширина подобранной маски по нарисованным сечениям.",
                 "Вид, полная скрытая длина и число клеток не определяются.",
                 "Строки общих маршрутов в узле нельзя суммировать для биомассы."]:
        notes.append([line])
    notes.column_dimensions['A'].width = 105;wb.save(output/'B-measurements.xlsx')
    return len(rows)


def parser():
    p = argparse.ArgumentParser(description='Фото → отчёт B, PNG и таблица измерений (CPU).')
    p.add_argument('input',type=Path,nargs='?',help='Папка JPG/PNG/HEIC/HEIF/TIFF/BMP')
    p.add_argument('-o','--output',type=Path,help='Новая папка результата; по умолчанию results/run-дата-время')
    p.add_argument('--recursive',action='store_true',help='Включать подпапки')
    p.add_argument('--scale-mode',choices=['auto','manual','reference'],default='auto')
    p.add_argument('--um-per-pixel',type=float,help='Ручные мкм на исходный пиксель')
    p.add_argument('--reference',type=Path)
    p.add_argument('--reference-points',type=float,nargs=4)
    p.add_argument('--reference-distance-um',type=float)
    p.add_argument('--mask-dir',type=Path,help='Папка проверенных PNG масок вместо детектора')
    p.add_argument('--mask-format',choices=['auto','binary','instances'],default='auto')
    p.add_argument('--working-width',type=int,default=960,help='Ширина рабочего B-растра; по умолчанию 960 px')
    p.add_argument('--version',action='version',version=__version__)
    return p


def run(args):
    source = args.input.expanduser().resolve()
    output = (args.output or Path('results')/datetime.now().strftime('run-%Y%m%d-%H%M%S-%f')).expanduser().resolve()
    if output.exists():raise ValueError('Папка результата уже существует; выберите новое имя.')
    if args.working_width<64:raise ValueError('Рабочая ширина должна быть не меньше 64 px.')
    seed_args = [str(source),'-o',str(output/'seed-report'),'--scale-mode',args.scale_mode]
    for flag,value in [('--um-per-pixel',args.um_per_pixel),('--reference',args.reference),
                       ('--reference-distance-um',args.reference_distance_um),('--mask-dir',args.mask_dir)]:
        if value is not None:seed_args += [flag,str(value)]
    if args.reference_points:seed_args += ['--reference-points',*[str(v) for v in args.reference_points]]
    if args.recursive:seed_args += ['--recursive']
    seed_args += ['--mask-format',args.mask_format]
    # Validate before creating anything; output inside input must never recurse.
    seed_options = cli.parser().parse_args(seed_args);cli.validate(seed_options)
    print('Этап 1/2: выделение кандидатов и шкалы',flush=True)
    seed_code = cli.main(seed_args)
    if not (output/'seed-report/manifest.json').exists():return 2
    print('Этап 2/2: уточнение B, измерения и PNG',flush=True)
    compare_args = argparse.Namespace(seed_report=[output/'seed-report'],source_dir=[source],
                                     output=output/'report',working_width=args.working_width,
                                     min_width=4.,max_width=30.,patch_side=2,
                                     smoothness_weight=3.,scale_tolerance=.10)
    compare_code = tube_compare.run(compare_args)
    manifest = json.loads((output/'report/manifest.json').read_text(encoding='utf-8'))
    count = tables(output,manifest)
    index = output/'report/index.html'
    text = index.read_text(encoding='utf-8').replace('<h1>Находки B: фото по порядку</h1>',
            '<h1>Находки B: фото по порядку</h1><p><a href="../B-measurements.xlsx">Таблица XLSX</a> · '
            '<a href="../B-measurements.csv">Таблица CSV</a></p>')
    index.write_text(text,encoding='utf-8')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=report/index.html">'
                                   '<a href="report/index.html">Открыть отчёт B</a>',encoding='utf-8')
    info = dict(version=__version__,input=str(source),output=str(output),seed_exit_code=seed_code,
                report_exit_code=compare_code,rows=count,summary=manifest['summary'],errors=manifest['errors'])
    write_json(output/'run-info.json',info)
    print(f'Готово: {len(manifest["images"])} уникальных фото, {count} кандидатных осей.\n'
          f'Откройте: {output/"index.html"}\nPNG: {output/"report/png-report"}\n'
          f'Таблица: {output/"B-measurements.xlsx"}',flush=True)
    if manifest['errors']:print('Есть ошибки отдельных файлов; см. run-info.json.',file=sys.stderr)
    return 1 if compare_code or not manifest['images'] else 0


def main(argv=None):
    p = parser();args = p.parse_args(argv)
    if args.input is None:
        if not sys.stdin.isatty():p.error('Укажите папку фотографий первым аргументом.')
        try:
            value = input('Путь к папке с фотографиями: ').strip()
        except (EOFError,KeyboardInterrupt):return 2
        if not value:return 2
        args.input = Path(value)
    try:return run(args)
    except (OSError,ValueError) as exc:
        print(f'Ошибка: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
