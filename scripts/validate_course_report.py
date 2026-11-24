"""Check the delivered report using a Python with pypdf installed."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/course_report_v1'

def main():
    manifest=json.loads((REPORT/'document_build_manifest.json').read_text(encoding='utf-8'))
    reader=PdfReader(REPORT/'2026秋-课程报告-说明书_完成稿.pdf')
    texts=[p.extract_text() for p in reader.pages]
    full='\n'.join(texts)
    required=['技术领域','背景技术','发明内容','附图说明','具体实施方式','权利要求书',
              'S01','S08','0.6862','6.5428','1.0675','1.9754','1.7384',
              '17/21','240','207','202611241','CloudCompare','参考文献']
    missing=[t for t in required if t not in full]
    assert not missing,missing
    assert len(texts)==manifest['pdf_pages']
    assert min(map(len,texts))>=100,'Unexpected nearly empty page'
    plan=json.loads((REPORT/'work/report_page_plan.json').read_text(encoding='utf-8'))
    for page in plan:
        for e in page['elements']:
            if e['kind']=='text':bottom=e['y']+e['leading']*len(e['lines'])
            elif e['kind']=='table':bottom=e['y']+sum(r['height'] for r in e['rows'])
            else:bottom=e['y']+e.get('height',0)
            assert bottom<790,(e['kind'],bottom)
    with zipfile.ZipFile(REPORT/'2026秋-课程报告-说明书_完成稿.docx') as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith('.xml'):ET.fromstring(z.read(name))
        root=ET.fromstring(z.read('word/document.xml'))
        math_count=len(root.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))
        assert math_count==19
        hashes={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
        figures=list((ROOT/'reports/paper_figures_v1').glob('Fig*.png'))
        assert len(figures)==5
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() in hashes for p in figures)
    log=(ROOT/'logs/course_report_build.log').read_text(encoding='utf-8')
    assert 'Glyph ' not in log
    result={'passed':True,'pdf_pages':len(texts),'required_text_checks':len(required),
            'minimum_pdf_text_per_page':min(map(len,texts)),'docx_native_math':math_count,
            'figure_original_bytes_preserved':len(figures),'page_plan_bounds':'passed',
            'word_native_render':False,'font_glyph_warnings':0}
    (REPORT/'document_content_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
