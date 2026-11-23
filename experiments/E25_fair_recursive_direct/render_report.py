from pathlib import Path
import markdown
from weasyprint import HTML
p=Path(__file__).parent
css='''@font-face {font-family: Chinese;src:url(fonts/NotoSansCJKsc-Regular.otf)} @page {size:A4;margin:18mm;@bottom-center {content:counter(page);font-size:9pt}} body {font-family:Chinese,sans-serif;font-size:10pt;line-height:1.65;color:#22313b} h1 {font-size:21pt} h2 {font-size:15pt;page-break-after:avoid} img {width:100%} table {border-collapse:collapse;width:100%;font-size:9pt} td,th {border:1px solid #bbb;padding:5px;text-align:left} pre,code {font-size:8pt;overflow-wrap:anywhere} a {color:#126b66}'''
body=markdown.markdown((p/'REPORT.md').read_text(),extensions=['tables','fenced_code'])
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>'+css+'</style><body>'+body+'</body></html>'
(p/'REPORT.html').write_text(html)
HTML(string=html,base_url=str(p)).write_pdf(p/'REPORT.pdf')
print('report rendered')
