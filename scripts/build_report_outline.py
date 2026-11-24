"""Build the course outline's numbered flowchart and editable Word reading copy.

Uses standard-library OOXML packaging; does not modify algorithms or results.
Run from the project directory with .venv/Scripts/python.exe.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import zipfile
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SOURCES = [REPORTS / "报告大纲_v1.md", REPORTS / "主方案说明_v1.md"]
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
for prefix, uri in [("w", W), ("r", R), ("wp", WP), ("a", A), ("pic", PIC)]:
    ET.register_namespace(prefix, uri)


def sub(parent, tag, attrs=None, text=None):
    ns, local = tag.split(":")
    uris = {"w": W, "r": R, "wp": WP, "a": A, "pic": PIC}
    mapped = {}
    for k, v in (attrs or {}).items():
        if ":" in k:
            pre, name = k.split(":")
            mapped[f"{{{uris[pre]}}}{name}"] = str(v)
        else:
            mapped[k] = str(v)
    node = ET.SubElement(parent, f"{{{uris[ns]}}}{local}", mapped)
    node.text = text
    return node


def xml(node):
    return ET.tostring(node, encoding="utf-8", xml_declaration=True)


def flowchart():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"],
                         "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8.3, 11.0), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 10), ylim=(-.2, 14.3))
    ax.axis("off")
    ax.text(5, 13.95, "法向残差筛选配准与体积估计", ha="center", fontsize=17, weight="bold")
    ax.text(5, 13.51, "主方案：pointwise_gate  ·  筛选仅用于拟合，变化点保留在完整输出中",
            ha="center", fontsize=9.3, color="#475569")

    def box(x, y, text, width=7.2, height=.83, color="#eef5fd"):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                     boxstyle="round,pad=0.04,rounding_size=0.10", linewidth=1.1,
                     edgecolor="#385675", facecolor=color))
        ax.text(x, y, text, ha="center", va="center", fontsize=10.1, linespacing=1.45)

    def arrow(start, end, label=None):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.1, color="#334155"))
        if label:
            ax.text((start[0]+end[0])/2+.15, (start[1]+end[1])/2+.08,
                    label, fontsize=9, color="#334155")

    ys = [12.65, 11.42, 10.19, 8.96, 7.73]
    texts = ["S01  两期坐标与局部坐标系\n输入参考Q、待测P；统一单位和变换方向",
             "S02  提取尺度与参考法向\n最近邻间距中位数s；24点PCA法向，统一+Z",
             "S03  普通ICP初步配准\n最多50次，得到初步变换T0",
             "S04  双向法向残差与稳健阈值\n有效对应d≤3s；中位数μ、MAD尺度σ、阈值τ=3σ",
             "S05  分别筛选两期拟合点\n有效对应且 |r−μ|≤τ；仅筛选一次"]
    for y, text in zip(ys, texts):
        box(5, y, text)
    for upper, lower in zip(ys, ys[1:]):
        arrow((5, upper-.46), (5, lower+.46))
    diamond_y = 6.32
    ax.add_patch(Polygon([(5,7.00),(8.4,diamond_y),(5,5.64),(1.6,diamond_y)],
                         facecolor="#fff7df", edgecolor="#8c681c", linewidth=1.1))
    ax.text(5, diamond_y, "两期各保留≥50%\n且各≥30点？", ha="center", va="center", fontsize=10)
    arrow((5,7.28),(5,7.02))
    box(2.65,4.7,"S06a  筛选点ICP精配准\n以T0为初值，最多100次",width=4.05,color="#eaf6ed")
    box(7.35,4.7,"S06b  返回初步配准\nT=T0；恢复全点掩膜并记录回退",width=4.05,color="#fff0e9")
    arrow((3.45,5.99),(2.65,5.16),"是")
    arrow((6.55,5.99),(7.35,5.16),"否")
    box(5,3.17,"S07  对完整待测点云应用最终变换T\n输出完整点云、矩阵、筛选掩膜及日志",color="#eaf6ed")
    arrow((2.65,4.24),(4.10,3.63))
    arrow((7.35,4.24),(5.90,3.63))
    box(5,1.83,"S08  高差与体积估计\n两期完整点云TIN插值；0.25 m网格；无支持单元不填零",height=.93)
    arrow((5,2.71),(5,2.34))
    ax.text(5,.72,"实际应用：两期共同有效支持  |  实验比较：所有方法统一有效支持交集",
            ha="center", fontsize=9, color="#475569")
    ax.text(5,.25,"前提：初始对齐足够接近、存在重叠稳定地形、满足2.5D高程假设。",
            ha="center", fontsize=9, color="#475569")
    fig.savefig(FIGURES / "main_method_flow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


class WordCopy:
    def __init__(self):
        self.document = ET.Element(f"{{{W}}}document")
        self.body = sub(self.document, "w:body")
        self.media = {}

    def paragraph(self, text="", style=None, code=False, parent=None, size=None):
        p = sub(parent if parent is not None else self.body, "w:p")
        pp = sub(p, "w:pPr")
        if style:
            sub(pp, "w:pStyle", {"w:val": style})
        if code:
            sub(pp, "w:shd", {"w:fill": "F1F5F9", "w:val": "clear"})
            sub(pp, "w:spacing", {"w:after": "25", "w:line": "260", "w:lineRule": "auto"})
        # Web references retain their URLs; local links retain the readable label.
        def link(match):
            label, target = match.groups()
            return f"{label}（{target}）" if target.startswith("http") else label
        text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", link, text)
        tokens = re.split(r"(\*\*.*?\*\*|`[^`]+`)", text)
        for token in tokens:
            if not token:
                continue
            bold = token.startswith("**") and token.endswith("**")
            mono = token.startswith("`") and token.endswith("`")
            value = token[2:-2] if bold else token[1:-1] if mono else token
            run = sub(p, "w:r")
            rp = sub(run, "w:rPr")
            if bold:
                sub(rp, "w:b")
            if mono or code:
                sub(rp, "w:rFonts", {"w:ascii": "Consolas", "w:hAnsi": "Consolas", "w:eastAsia": "等线"})
            if size or code:
                sub(rp, "w:sz", {"w:val": str(size or 19)})
            t = sub(run, "w:t", text=value)
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p

    def page_break(self):
        sub(sub(sub(self.body,"w:p"),"w:r"),"w:br",{"w:type":"page"})

    def table(self, lines):
        rows = [[x.strip() for x in line.strip().strip("|").split("|")] for line in lines]
        rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", x) for x in row)]
        table = sub(self.body, "w:tbl")
        prop = sub(table, "w:tblPr")
        sub(prop, "w:tblW", {"w:w":"9360", "w:type":"dxa"})
        borders = sub(prop, "w:tblBorders")
        for edge in ["top","left","bottom","right","insideH","insideV"]:
            sub(borders,f"w:{edge}",{"w:val":"single","w:sz":"4","w:color":"CBD5E1"})
        grid = sub(table,"w:tblGrid")
        width = 9360 // len(rows[0])
        for _ in rows[0]:
            sub(grid,"w:gridCol",{"w:w":str(width)})
        for i, row in enumerate(rows):
            tr = sub(table, "w:tr")
            trp = sub(tr,"w:trPr")
            sub(trp,"w:cantSplit")
            if i == 0:
                sub(trp,"w:tblHeader")
            for cell in row:
                tc = sub(tr, "w:tc")
                tcp = sub(tc,"w:tcPr")
                sub(tcp,"w:tcW",{"w:w":str(width),"w:type":"dxa"})
                if i == 0:
                    sub(tcp,"w:shd",{"w:fill":"E8EFF7"})
                self.paragraph("**"+cell+"**" if i==0 else cell,parent=tc,size=18)
        self.paragraph()

    def image(self, path, caption):
        raw = path.read_bytes()
        width, height = struct.unpack(">II",raw[16:24])
        inch_w = min(6.3, 6.7*width/height)
        cx, cy = int(inch_w*914400), int(inch_w*914400*height/width)
        index = len(self.media)+1
        name, rid = f"figure{index}.png",f"rIdImage{index}"
        self.media[name] = raw
        p=sub(self.body,"w:p"); sub(sub(p,"w:pPr"),"w:jc",{"w:val":"center"})
        drawing=sub(sub(p,"w:r"),"w:drawing")
        inline=sub(drawing,"wp:inline",{"distT":"0","distB":"0","distL":"0","distR":"0"})
        sub(inline,"wp:extent",{"cx":cx,"cy":cy})
        sub(inline,"wp:docPr",{"id":index,"name":caption,"descr":caption})
        graphic=sub(inline,"a:graphic"); data=sub(graphic,"a:graphicData",{"uri":PIC})
        pic=sub(data,"pic:pic"); nv=sub(pic,"pic:nvPicPr")
        sub(nv,"pic:cNvPr",{"id":0,"name":name}); sub(nv,"pic:cNvPicPr")
        fill=sub(pic,"pic:blipFill"); sub(fill,"a:blip",{"r:embed":rid})
        sub(sub(fill,"a:stretch"),"a:fillRect")
        sp=sub(pic,"pic:spPr"); xfrm=sub(sp,"a:xfrm")
        sub(xfrm,"a:off",{"x":0,"y":0}); sub(xfrm,"a:ext",{"cx":cx,"cy":cy})
        sub(sub(sp,"a:prstGeom",{"prst":"rect"}),"a:avLst")
        self.paragraph(caption,style="Caption")

    def markdown(self, source):
        lines=source.read_text(encoding="utf-8").splitlines(); i=0; code=False
        while i<len(lines):
            line=lines[i]; i+=1
            if line.startswith("```"):
                code=not code; continue
            if code:
                self.paragraph(line,code=True); continue
            if not line.strip():
                continue
            if line.startswith("|"):
                group=[line]
                while i<len(lines) and lines[i].startswith("|"):
                    group.append(lines[i]);i+=1
                self.table(group);continue
            image_match=re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)",line)
            if image_match:
                self.image(source.parent/image_match[2],image_match[1]);continue
            heading=re.match(r"^(#{1,4})\s+(.*)",line)
            if heading:
                self.paragraph(heading[2],style=f"Heading{min(len(heading[1]),3)}");continue
            self.paragraph(line)

    def save(self, path):
        sec=sub(self.body,"w:sectPr")
        sub(sec,"w:footerReference",{"w:type":"default","r:id":"rIdFooter"})
        sub(sec,"w:pgSz",{"w:w":"11906","w:h":"16838"})
        sub(sec,"w:pgMar",{"w:top":"1134","w:right":"1273","w:bottom":"1134",
                              "w:left":"1273","w:header":"500","w:footer":"500","w:gutter":"0"})
        styles=ET.Element(f"{{{W}}}styles")
        defaults=sub(styles,"w:docDefaults"); rp=sub(sub(defaults,"w:rPrDefault"),"w:rPr")
        sub(rp,"w:rFonts",{"w:ascii":"Calibri","w:hAnsi":"Calibri","w:eastAsia":"宋体"})
        sub(rp,"w:sz",{"w:val":"22"});sub(rp,"w:lang",{"w:val":"zh-CN","w:eastAsia":"zh-CN"})
        pp=sub(sub(defaults,"w:pPrDefault"),"w:pPr")
        sub(pp,"w:spacing",{"w:after":"100","w:line":"300","w:lineRule":"auto"})
        normal=sub(styles,"w:style",{"w:type":"paragraph","w:default":"1","w:styleId":"Normal"})
        sub(normal,"w:name",{"w:val":"Normal"})
        for name,size in [("Title",38),("Heading1",32),("Heading2",27),("Heading3",24),("Caption",19)]:
            st=sub(styles,"w:style",{"w:type":"paragraph","w:styleId":name})
            sub(st,"w:name",{"w:val":name}); sub(st,"w:basedOn",{"w:val":"Normal"})
            pp=sub(st,"w:pPr");sub(pp,"w:spacing",{"w:before":"220","w:after":"120"})
            if name.startswith("Heading"):
                sub(pp,"w:keepNext");sub(pp,"w:outlineLvl",{"w:val":int(name[-1])-1})
            if name in ("Title","Caption"):
                sub(pp,"w:jc",{"w:val":"center"})
            rp=sub(st,"w:rPr");sub(rp,"w:rFonts",{"w:eastAsia":"微软雅黑"})
            sub(rp,"w:sz",{"w:val":size})
            if name!="Caption": sub(rp,"w:b")
        footer=ET.Element(f"{{{W}}}ftr");p=sub(footer,"w:p");sub(sub(p,"w:pPr"),"w:jc",{"w:val":"center"})
        sub(sub(p,"w:r"),"w:t",text="报告大纲与主方案说明 v1  ·  ")
        sub(p,"w:fldSimple",{"w:instr":"PAGE"})
        rel_ns="http://schemas.openxmlformats.org/package/2006/relationships"
        rels=ET.Element("Relationships",{"xmlns":rel_ns})
        for rid,typ,target in [("rIdStyles","styles","styles.xml"),("rIdFooter","footer","footer1.xml")]:
            ET.SubElement(rels,"Relationship",{"Id":rid,"Type":R+"/"+typ,"Target":target})
        for i,name in enumerate(self.media,1):
            ET.SubElement(rels,"Relationship",{"Id":f"rIdImage{i}","Type":R+"/image","Target":"media/"+name})
        rootrels=ET.Element("Relationships",{"xmlns":rel_ns})
        ET.SubElement(rootrels,"Relationship",{"Id":"rId1","Type":R+"/officeDocument","Target":"word/document.xml"})
        cts=ET.Element("Types",{"xmlns":"http://schemas.openxmlformats.org/package/2006/content-types"})
        for ext,typ in [("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml"),("png","image/png")]:
            ET.SubElement(cts,"Default",{"Extension":ext,"ContentType":typ})
        for name,typ in [("document","document.main"),("styles","styles"),("footer1","footer")]:
            ET.SubElement(cts,"Override",{"PartName":f"/word/{name}.xml",
                "ContentType":f"application/vnd.openxmlformats-officedocument.wordprocessingml.{typ}+xml"})
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
            for name,node in [("[Content_Types].xml",cts),("_rels/.rels",rootrels),("word/document.xml",self.document),
                              ("word/styles.xml",styles),("word/footer1.xml",footer),("word/_rels/document.xml.rels",rels)]:
                z.writestr(name,xml(node))
            for name,raw in self.media.items():
                z.writestr("word/media/"+name,raw)


def main():
    flowchart()
    word=WordCopy()
    word.paragraph("报告大纲与主方案说明",style="Title")
    word.paragraph("数据分析与特征提取课程  |  2026-11-24  |  v1",style="Caption")
    word.paragraph("撰写准备稿。主方案采用逐点法向残差筛选；区域聚合作为扩展对照。本文不是最终提交的专利说明书。")
    for i,source in enumerate(SOURCES):
        if i: word.page_break()
        word.markdown(source)
    output=REPORTS/"报告大纲与主方案说明_v1.docx"
    word.save(output)
    # Structural validation and retained source/result fingerprints.
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(z.read(name))
        doc=ET.fromstring(z.read("word/document.xml"))
        text="".join(doc.itertext())
        for keyword in ["S01","S08","pointwise_gate","0.6862","6.5428","CN122384746A","尚未完成"]:
            assert keyword in text,keyword
        assert "word/media/figure1.png" in z.namelist()
    summary=ROOT/"results/robustness_volume_v1/overall_summary.csv"
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SOURCES+[output,FIGURES/"main_method_flow.png",summary]}
    manifest={"date":"2026-11-24","status":"outline_and_method_preparation",
              "main_method":"pointwise_gate","algorithm_or_result_modified":False,
              "docx_xml_and_zip_validation":"passed","word_layout_rendered":False,"sha256":hashes}
    (REPORTS/"report_outline_manifest_v1.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(output),"structural_validation":"passed","files":list(hashes)},ensure_ascii=False))


if __name__=="__main__":
    main()
