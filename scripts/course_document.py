"""Offline, editable OOXML and PDF renderer for the course specification.

The same measured page plan drives both deliverables. Formula ASTs become
native Office Math rather than pictures or visible TeX in Word.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
import struct
import zipfile
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.patches import Rectangle

PW, PH = 595.276, 841.89
LEFT, RIGHT, TOP, BOTTOM = 66.0, 66.0, 62.0, 56.0
WIDTH = PW-LEFT-RIGHT
FONTS={"body":FontProperties(family=["SimSun","DejaVu Sans"]),
       "heading":FontProperties(family=["SimHei","DejaVu Sans"]),
       "latin":FontProperties(family=["Arial","DejaVu Sans"])}
mpl.rcParams.update({"pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False,
                    "font.sans-serif":["SimSun","DejaVu Sans"],"mathtext.fontset":"dejavuserif","text.hinting":"none"})
TEXT_PATH=TextToPath(); CHAR_WIDTH={}


def width(text,size=12,font="body"):
    total=0
    for ch in text:
        key=(ch,size,font)
        if key not in CHAR_WIDTH:
            prop=FONTS[font].copy();prop.set_size(size)
            CHAR_WIDTH[key]=TEXT_PATH.get_text_width_height_descent(ch,prop,False)[0]
        total+=CHAR_WIDTH[key]
    return total


def wrap(text,size=12,available=WIDTH,font="body",indent=True):
    # Measured CJK wrapping; preserve every character and keep closing marks
    # off the start of a line. Full-width spaces implement the paragraph indent.
    text=("\u3000\u3000" if indent else "")+str(text)
    lines=[];line=""; closing="，。；：！？、）】》”’％,.!?;:)]}"
    tokens=re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:@%+^=\-]*|[\s\S]",text)
    expanded=[]
    for token in tokens:
        expanded.extend(list(token) if width(token,size,font)>available-4 else [token])
    for ch in expanded:
        if ch=="\n":lines.append(line);line="";continue
        if line and width(line+ch,size,font)>available-4:
            if ch in closing:line+=ch;lines.append(line);line=""
            elif line[-1] in "（【《“‘":
                carry=line[-1];lines.append(line[:-1]);line=carry+ch
            else:lines.append(line);line=ch
        else:line+=ch
    if line:lines.append(line)
    return lines or [""]


class Pages:
    def __init__(self,title):
        self.title=title;self.pages=[];self.current=[];self.y=TOP;self.headings=[]

    def new(self,style="normal"):
        if self.current:self.pages.append({"style":getattr(self,"style","normal"),"elements":self.current})
        self.current=[];self.y=TOP;self.style=style

    def need(self,height):
        if self.y+height>PH-BOTTOM:self.new()

    def text(self,text,size=12,leading=19.5,font="body",indent=True,align="left",after=5,before=0,
             heading=None,bookmark=None):
        lines=wrap(text,size,WIDTH,font,indent)
        if heading:
            self.need(before+leading*len(lines)+after+42)
            self.headings.append({"title":text,"page":len(self.pages)+1,"level":heading})
        else:self.need(before+leading+after)
        self.y+=before
        while lines:
            capacity=int((PH-BOTTOM-self.y-after)//leading)
            if capacity<=0:self.new();continue
            take=min(len(lines),capacity)
            if len(lines)-take==1 and take>1:take-=1
            used=lines[:take];lines=lines[take:]
            self.current.append({"kind":"text","text":text,"lines":used,"x":LEFT,"y":self.y,
                "size":size,"leading":leading,"font":font,"align":align,"heading":heading,"bookmark":bookmark})
            self.y+=len(used)*leading+after
            if lines:self.new()

    def h(self,text,level=1):
        self.text(text,size=15 if level==1 else 12.5,leading=24 if level==1 else 21,font="heading",
                  indent=False,before=10 if level==1 else 6,after=7,heading=level)

    def equation(self,eq):
        lines=eq.get("latex_lines",[eq["latex"]]);size=11.0
        # Formulae use a separate, measured block and native editable math.
        for line in lines:
            prop=FONTS["latin"].copy();prop.set_size(size)
            measured=TEXT_PATH.get_text_width_height_descent("$"+line+"$",prop,True)[0]
            if measured>WIDTH-40:size=min(size,11.0*(WIDTH-40)/measured)
        if size<9:raise ValueError(("equation too wide",eq["number"],size))
        height=len(lines)*29+8
        self.need(height+36)
        self.current.append({"kind":"equation","equation":eq,"x":LEFT,"y":self.y,"height":height,"size":size})
        self.y+=height

    def table(self,title,headers,rows,widths=None,size=9.5):
        widths=widths or [WIDTH/len(headers)]*len(headers)
        widths=[x*WIDTH/sum(widths) for x in widths]
        self.text(title,size=10.5,leading=16,font="heading",indent=False,before=5,after=5)
        def make(row,header=False):
            cells=[wrap(str(t),size,w-10,"heading" if header else "body",False) for t,w in zip(row,widths)]
            height=max(map(len,cells))*14+10
            return {"cells":cells,"height":height,"header":header}
        head=make(headers,True);prepared=[make(r) for r in rows]
        while prepared:
            self.need(head["height"]+prepared[0]["height"])
            chunk=[head]; used=head["height"]
            while prepared and self.y+used+prepared[0]["height"]<PH-BOTTOM-6:
                row=prepared.pop(0);chunk.append(row);used+=row["height"]
            if len(chunk)==1:raise ValueError("oversized table row")
            self.current.append({"kind":"table","x":LEFT,"y":self.y,"widths":widths,"size":size,"rows":chunk,"height":used})
            self.y+=used+9
            if prepared:
                self.new()
                self.text(title+'（续）',size=10.5,leading=16,font='heading',indent=False,after=5)

    def picture(self,path,caption=None,caption_paragraphs=None,width_pt=WIDTH):
        raw=Path(path).read_bytes();w,h=struct.unpack(">II",raw[16:24]);height=width_pt*h/w
        captions=caption_paragraphs or []
        estimate=sum(len(wrap(t,10.5,WIDTH,indent=False))*15+4 for t in captions)
        if caption:estimate+=24
        self.need(height+estimate+12)
        self.current.append({"kind":"image","path":str(path),"x":LEFT+(WIDTH-width_pt)/2,"y":self.y,
                             "width":width_pt,"height":height})
        self.y+=height+6
        if caption:self.text(caption,size=10.5,leading=17,font="heading",align="center",indent=False,after=5)
        for t in captions:self.text(t,size=10.5,leading=15,indent=False,after=5)

    def cover(self,logo,title):
        self.new("cover")
        self.current.append({"kind":"image","path":str(logo),"x":132,"y":75,"width":331,"height":81.2})
        self.y=171;self.text("Harbin Institute of Technology",size=14,leading=22,font="latin",align="center",indent=False,after=22)
        self.text("课程设计说明书",size=26,leading=38,font="heading",align="center",indent=False,after=33)
        self.text("课程名称：数据分析与特征提取",size=16,leading=25,align="center",indent=False,after=2)
        self.text("（模拟专利撰写）",size=14,leading=23,align="center",indent=False,after=30)
        self.text("设计题目：",size=14,leading=22,align="center",indent=False,after=4)
        self.text(title,size=16,leading=26,font="heading",align="center",indent=False,after=28)
        for t in ["院    系：航天学院","控制科学与工程","设 计 者：________________","学    号：________________","指导教师：张淼","设计时间：2026年秋季学期"]:
            self.text(t,size=14,leading=24,align="center",indent=False,after=4)
        self.y=max(self.y+10,728)
        self.text("哈尔滨工业大学",size=16,leading=24,align="center",indent=False)
        self.new()

    def finish(self):
        if self.current:self.pages.append({"style":getattr(self,"style","normal"),"elements":self.current});self.current=[]
        return self.pages


def draw_pdf(pages,path,title,preview_dir=None):
    if preview_dir:Path(preview_dir).mkdir(parents=True,exist_ok=True)
    with PdfPages(path,metadata={"Title":title,"Author":"课程项目","Subject":"数据分析与特征提取课程设计说明书"}) as pdf:
        for index,page in enumerate(pages,1):
            fig=plt.figure(figsize=(PW/72,PH/72));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,PW),ylim=(PH,0));ax.axis("off")
            if page["style"]!="cover":
                ax.text(LEFT,32,"哈尔滨工业大学  数据分析与特征提取课程报告",fontproperties=FONTS["body"],fontsize=9,color="#606060",va="top")
                ax.plot([LEFT,PW-RIGHT],[47,47],color="#888888",lw=.5)
                ax.text(PW/2,PH-28,str(index),fontproperties=FONTS["latin"],fontsize=9,ha="center",va="top")
            for item in page["elements"]:
                kind=item["kind"]
                if kind=="text":
                    for j,line in enumerate(item["lines"]):
                        x=item["x"] if item["align"]=="left" else PW/2
                        ax.text(x,item["y"]+j*item["leading"],line,fontproperties=FONTS[item["font"]],fontsize=item["size"],
                                ha=item["align"],va="top",linespacing=1)
                elif kind=="equation":
                    eq=item["equation"];lines=eq.get("latex_lines",[eq["latex"]])
                    for j,line in enumerate(lines):
                        ax.text(PW/2,item["y"]+3+j*29,"$"+line+"$",fontsize=item["size"],ha="center",va="top")
                    ax.text(PW-RIGHT,item["y"]+6,"("+str(eq["number"])+")",fontproperties=FONTS["latin"],fontsize=11,ha="right",va="top")
                elif kind=="image":
                    image=plt.imread(item["path"])
                    ax.imshow(image,extent=(item["x"],item["x"]+item["width"],item["y"]+item["height"],item["y"]),aspect="auto",interpolation="lanczos")
                elif kind=="table":
                    y=item["y"]
                    for row in item["rows"]:
                        x=item["x"]
                        for lines,w in zip(row["cells"],item["widths"]):
                            ax.add_patch(Rectangle((x,y),w,row["height"],facecolor="#F0F2F5" if row["header"] else "white",edgecolor="#777777",lw=.45))
                            for j,line in enumerate(lines):
                                ax.text(x+5,y+5+j*14,line,fontproperties=FONTS["heading" if row["header"] else "body"],fontsize=item["size"],va="top")
                            x+=w
                        y+=row["height"]
            pdf.savefig(fig,dpi=220)
            if preview_dir:fig.savefig(Path(preview_dir)/f"page_{index:02d}.png",dpi=180,facecolor="white")
            plt.close(fig)


NS={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
    "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships","wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a":"http://schemas.openxmlformats.org/drawingml/2006/main","pic":"http://schemas.openxmlformats.org/drawingml/2006/picture"}
for pre,uri in NS.items():ET.register_namespace(pre,uri)


def el(parent,tag,attrs=None,text=None):
    pre,name=tag.split(":");attributes={}
    for k,v in (attrs or {}).items():
        if ":" in k:p,n=k.split(":");attributes[f"{{{NS[p]}}}{n}"]=str(v)
        else:attributes[k]=str(v)
    n=ET.SubElement(parent,f"{{{NS[pre]}}}{name}",attributes);n.text=text;return n


def math_ast(parent,node):
    if isinstance(node,(str,int,float)):
        if isinstance(node,str):
            match=re.search(r"([A-Za-zα-ωΑ-Ω][\u0300-\u036f]*)_([A-Za-z0-9]+)",node)
            if match:
                if match.start():math_ast(parent,node[:match.start()])
                math_ast(parent,["sub",match[1],match[2]])
                if match.end()<len(node):math_ast(parent,node[match.end():])
                return
        run=el(parent,"m:r");el(el(run,"m:rPr"),"m:sty",{"m:val":"p"});el(run,"m:t",text=str(node));return
    if not node:return
    if node[0] in ("sub","sup","subsup"):
        kind={"sub":"sSub","sup":"sSup","subsup":"sSubSup"}[node[0]];n=el(parent,"m:"+kind)
        math_ast(el(n,"m:e"),node[1]);math_ast(el(n,"m:sub" if node[0]=="sub" else "m:sup"),node[2])
        if node[0]=="subsup":
            # The simultaneous form requires sub before sup in OOXML.
            n.clear();math_ast(el(n,"m:e"),node[1]);math_ast(el(n,"m:sub"),node[2]);math_ast(el(n,"m:sup"),node[3])
    elif node[0]=="frac":
        n=el(parent,"m:f");math_ast(el(n,"m:num"),node[1]);math_ast(el(n,"m:den"),node[2])
    elif node[0]=="sqrt":
        n=el(parent,"m:rad");el(el(n,"m:radPr"),"m:degHide",{"m:val":"1"});el(n,"m:deg");math_ast(el(n,"m:e"),node[1])
    else:
        for part in node:math_ast(parent,part)


def xml(node):return ET.tostring(node,encoding="utf-8",xml_declaration=True)


def write_docx(pages,path,title):
    document=ET.Element(f"{{{NS['w']}}}document");body=el(document,"w:body");media={};image_ids={}
    def text_run(p,text,size=12,font="宋体",bold=False):
        run=el(p,"w:r");pr=el(run,"w:rPr");el(pr,"w:rFonts",{"w:ascii":font,"w:hAnsi":font,"w:eastAsia":font});el(pr,"w:sz",{"w:val":round(size*2)});el(pr,"w:szCs",{"w:val":round(size*2)})
        if bold:el(pr,"w:b")
        n=el(run,"w:t",text=text);n.set("{http://www.w3.org/XML/1998/namespace}space","preserve");return run
    def paragraph(parent,y_before=0,line=20,align="left",break_before=False,heading=None):
        p=el(parent,"w:p");pr=el(p,"w:pPr");el(pr,"w:spacing",{"w:before":max(0,round(y_before*20)),"w:after":0,"w:line":round(line*20),"w:lineRule":"exact"})
        el(pr,"w:jc",{"w:val":align})
        if break_before:el(pr,"w:pageBreakBefore")
        if heading:el(pr,"w:outlineLvl",{"w:val":heading-1});el(pr,"w:keepNext")
        return p
    for page_index,page in enumerate(pages):
        cursor=TOP;first=True
        for item in page["elements"]:
            before=item["y"]-cursor;page_break=first and page_index>0;first=False
            if item["kind"]=="text":
                p=paragraph(body,before,item["leading"],item["align"],page_break,item.get("heading"))
                for i,line in enumerate(item["lines"]):
                    if i:el(el(p,"w:r"),"w:br")
                    text_run(p,line,item["size"],{"body":"宋体","heading":"黑体","latin":"Arial"}[item["font"]],item["font"]=="heading")
                cursor=item["y"]+len(item["lines"])*item["leading"]
            elif item["kind"]=="equation":
                eq=item["equation"];asts=eq.get("ast_lines",[eq["ast"]])
                for i,ast in enumerate(asts):
                    p=paragraph(body,before if i==0 else 0,29,"center",page_break and i==0)
                    m=el(p,"m:oMath");math_ast(m,ast)
                    if i==len(asts)-1:text_run(p,"    ("+str(eq["number"])+")",11,"Cambria Math")
                cursor=item["y"]+item["height"]-8
            elif item["kind"]=="image":
                p=paragraph(body,before,item["height"]+1,"center",page_break)
                pr=p.find(f"{{{NS['w']}}}pPr");sp=pr.find(f"{{{NS['w']}}}spacing");sp.set(f"{{{NS['w']}}}lineRule","atLeast")
                raw=Path(item["path"]).read_bytes();digest=__import__('hashlib').sha256(raw).hexdigest()
                if digest not in image_ids:
                    index=len(media)+1;name=f"image{index}.png";rid=f"rIdImage{index}";media[name]=raw;image_ids[digest]=(index,name,rid)
                index,name,rid=image_ids[digest];cx=round(item["width"]*12700);cy=round(item["height"]*12700)
                inline=el(el(el(p,"w:r"),"w:drawing"),"wp:inline",{"distT":0,"distB":0,"distL":0,"distR":0})
                el(inline,"wp:extent",{"cx":cx,"cy":cy});el(inline,"wp:docPr",{"id":len(image_ids)+page_index*100,"name":name})
                pic=el(el(el(inline,"a:graphic"),"a:graphicData",{"uri":NS["pic"]}),"pic:pic");nv=el(pic,"pic:nvPicPr");el(nv,"pic:cNvPr",{"id":0,"name":name});el(nv,"pic:cNvPicPr")
                fill=el(pic,"pic:blipFill");el(fill,"a:blip",{"r:embed":rid});el(el(fill,"a:stretch"),"a:fillRect")
                trans=el(el(pic,"pic:spPr"),"a:xfrm");el(trans,"a:off",{"x":0,"y":0});el(trans,"a:ext",{"cx":cx,"cy":cy})
                el(el(pic.find(f"{{{NS['pic']}}}spPr"),"a:prstGeom",{"prst":"rect"}),"a:avLst")
                cursor=item["y"]+item["height"]+1
            elif item["kind"]=="table":
                # A spacer carries any page break before the native Word table.
                paragraph(body,before,1,break_before=page_break)
                table=el(body,"w:tbl");pr=el(table,"w:tblPr");el(pr,"w:tblW",{"w:w":round(WIDTH*20),"w:type":"dxa"});el(pr,"w:tblLayout",{"w:type":"fixed"})
                borders=el(pr,"w:tblBorders")
                for edge in ["top","left","bottom","right","insideH","insideV"]:el(borders,"w:"+edge,{"w:val":"single","w:sz":4,"w:color":"777777"})
                grid=el(table,"w:tblGrid")
                for w in item["widths"]:el(grid,"w:gridCol",{"w:w":round(w*20)})
                for row in item["rows"]:
                    tr=el(table,"w:tr");trp=el(tr,"w:trPr");el(trp,"w:cantSplit");el(trp,"w:trHeight",{"w:val":round(row["height"]*20),"w:hRule":"exact"})
                    if row["header"]:el(trp,"w:tblHeader")
                    for lines,w in zip(row["cells"],item["widths"]):
                        tc=el(tr,"w:tc");tcp=el(tc,"w:tcPr");el(tcp,"w:tcW",{"w:w":round(w*20),"w:type":"dxa"})
                        if row["header"]:el(tcp,"w:shd",{"w:fill":"F0F2F5"})
                        margin=el(tcp,"w:tcMar")
                        for side in ["top","bottom","left","right"]:el(margin,"w:"+side,{"w:w":90,"w:type":"dxa"})
                        p=paragraph(tc,line=14)
                        for j,line in enumerate(lines):
                            if j:el(el(p,"w:r"),"w:br")
                            text_run(p,line,item["size"],bold=row["header"])
                cursor=item["y"]+item["height"]
    sec=el(body,"w:sectPr");el(sec,"w:footerReference",{"w:type":"default","r:id":"rIdFooter"});el(sec,"w:headerReference",{"w:type":"default","r:id":"rIdHeader"})
    el(sec,"w:titlePg")
    el(sec,"w:pgSz",{"w:w":round(PW*20),"w:h":round(PH*20)})
    el(sec,"w:pgMar",{"w:top":round(TOP*20),"w:right":round(RIGHT*20),"w:bottom":round(BOTTOM*20),"w:left":round(LEFT*20),"w:header":500,"w:footer":500,"w:gutter":0})
    footer=ET.Element(f"{{{NS['w']}}}ftr");p=paragraph(footer,line=12,align="center");el(p,"w:fldSimple",{"w:instr":" PAGE "})
    header=ET.Element(f"{{{NS['w']}}}hdr");p=paragraph(header,line=12);text_run(p,"哈尔滨工业大学  数据分析与特征提取课程报告",9)
    styles=ET.Element(f"{{{NS['w']}}}styles");normal=el(styles,"w:style",{"w:type":"paragraph","w:default":1,"w:styleId":"Normal"});el(normal,"w:name",{"w:val":"Normal"})
    settings=ET.Element(f"{{{NS['w']}}}settings");el(settings,"w:defaultTabStop",{"w:val":420});el(settings,"w:autoHyphenation",{"w:val":0})
    mathpr=el(settings,"m:mathPr");el(mathpr,"m:mathFont",{"m:val":"Cambria Math"})
    REL="http://schemas.openxmlformats.org/package/2006/relationships";rels=ET.Element("Relationships",{"xmlns":REL})
    for rid,typ,target in [("rIdStyles","styles","styles.xml"),("rIdSettings","settings","settings.xml"),("rIdFooter","footer","footer1.xml"),("rIdHeader","header","header1.xml")]:ET.SubElement(rels,"Relationship",{"Id":rid,"Type":NS["r"]+"/"+typ,"Target":target})
    for i,name in enumerate(media,1):ET.SubElement(rels,"Relationship",{"Id":f"rIdImage{i}","Type":NS["r"]+"/image","Target":"media/"+name})
    rootrels=ET.Element("Relationships",{"xmlns":REL});ET.SubElement(rootrels,"Relationship",{"Id":"rId1","Type":NS["r"]+"/officeDocument","Target":"word/document.xml"})
    cts=ET.Element("Types",{"xmlns":"http://schemas.openxmlformats.org/package/2006/content-types"})
    for ext,typ in [("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml"),("png","image/png")]:ET.SubElement(cts,"Default",{"Extension":ext,"ContentType":typ})
    for name,typ in [("document","document.main"),("styles","styles"),("settings","settings"),("footer1","footer"),("header1","header")]:ET.SubElement(cts,"Override",{"PartName":f"/word/{name}.xml","ContentType":f"application/vnd.openxmlformats-officedocument.wordprocessingml.{typ}+xml"})
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
        for name,node in [("[Content_Types].xml",cts),("_rels/.rels",rootrels),("word/document.xml",document),("word/_rels/document.xml.rels",rels),("word/styles.xml",styles),("word/settings.xml",settings),("word/footer1.xml",footer),("word/header1.xml",header)]:z.writestr(name,xml(node))
        for name,raw in media.items():z.writestr("word/media/"+name,raw)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith((".xml",".rels")):ET.fromstring(z.read(name))


def markdown(pages,path,title):
    chunks=["# "+title,"\n（课程模拟专利说明书；个人信息待填。）\n"]
    for page in pages:
        for e in page["elements"]:
            if e["kind"]=="text":
                prefix="#"*min(4,(e.get("heading") or 0)+1)+" " if e.get("heading") else ""
                chunks.append(prefix+"\n".join(e["lines"]).replace("\u3000", ""))
            elif e["kind"]=="equation":chunks.append("$$\n"+e["equation"]["latex"]+"\\qquad ("+str(e["equation"]["number"])+")\n$$")
            elif e["kind"]=="image":chunks.append("![配图]("+Path(e["path"]).name+")")
            elif e["kind"]=="table":
                for i,row in enumerate(e["rows"]):
                    chunks.append("| "+" | ".join("".join(x) for x in row["cells"])+" |")
                    if i==0:chunks.append("| "+" | ".join("---" for _ in row["cells"])+" |")
                chunks.append("")
    path.write_text("\n\n".join(chunks),encoding="utf-8")
