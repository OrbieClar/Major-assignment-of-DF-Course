"""Course-template report builder from frozen code, measurements and figures."""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, shutil, sys, zipfile
import xml.etree.ElementTree as ET
from course_document import Pages, draw_pdf, write_docx, markdown, NS
from course_report_content import *
from course_report_math import equations

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/course_report_v1';WORK=OUT/'work';FIG=ROOT/'reports/paper_figures_v1'
PREFIX='2026秋-课程报告-说明书_完成稿'

def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def structured_draft():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(exist_ok=True)
    files=['scripts/automatic_stable_icp.py','scripts/terrain_volume.py','scripts/build_terrain_benchmark.py',
        'scripts/diagnose_stable_region.py','scripts/run_automatic_stable_experiment.py','scripts/run_robustness_volume.py',
        'configs/automatic_stable_v1.json','configs/robustness_volume_v1.json',
        'results/robustness_volume_v1/all_results.csv','results/robustness_volume_v1/scenario_summary.csv',
        'results/robustness_volume_v1/overall_summary.csv','results/stable_region_diagnosis_v1/summary.csv',
        'results/robustness_volume_v1/validation.json','results/robustness_volume_v1/review_validation.json',
        'reports/主方案说明_v1.md','reports/paper_figures_v1/图注与使用说明.md']
    sources=[]
    for i,p in enumerate(files,1):
        text=(ROOT/p).read_text(encoding='utf-8-sig'); functions=[]
        if p.endswith('.py'):
            functions=[{'name':n.name,'start':n.lineno,'end':n.end_lineno} for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef)]
        sources.append({'id':f'C{i:03d}','type':'project-evidence','locator':p,'summary':p.split('/')[-1],
            'confidence':'high','functions':functions,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()})
    sources.extend([{'id':'P001','type':'teacher-template','locator':'封面、任务书、四项评分标准与五部分示例正文','summary':'课程与格式要求','confidence':'high'},
        {'id':'P002','type':'project-method','locator':'主方案说明，第3—7节','summary':'固定方法、公式和证据边界','confidence':'high'}])
    eqs=equations()
    for e in eqs:sources.append({'id':e['source_ids'][0],'type':'equation','locator':e['source_location'],'summary':e['technical_role'],'confidence':'high'})
    chain=[('共用局部坐标和完整点云输出',['C001','C003','E001','E011']),
        ('输入尺度和PCA参考法向',['C001','E002','E003']),('初步ICP及反射校正刚性解',['C001','E004','E005']),
        ('双向同号参考法向残差',['C001','E006','E007']),('稳健尺度与双期保留掩膜',['C001','C007','E008','E009']),
        ('最小保留条件及回退',['C001','C007','E010']),('完整点云高差和有效支持体积积分',['C002','C006','C008','E011','E012','E013'])]
    ledger=[{'id':f'L{i}','feature':feature,'source_ids':ids,'support_status':'explicit','technical_role':feature,
             'effect':'对应操作输入输出有代码支持；不单独推断新颖性'} for i,(feature,ids) in enumerate(chain,1)]
    claim_map=[{'claim_number':1,'feature':x['feature'],'evidence_ids':[x['id']],'specification_locations':['发明内容S01—S08及实施例三']} for x in ledger]
    for n,ids in [(2,['L2']),(3,['L4','L5']),(4,['L5']),(5,['L3','L6']),(6,['L1','L3']),(7,['L7'])]:
        claim_map.append({'claim_number':n,'feature':CLAIMS[n-1],'evidence_ids':ids,'specification_locations':['对应步骤、参数表及实施例三']})
    figs=[]
    nodes=[('S01','统一坐标'),('S02','点间距与参考法向'),('S03','初步配准'),('S04','双向法向残差及阈值'),
           ('S05','两期逐点筛选'),('S06a','保留充分时精配准'),('S06b','不足时回退'),('S07','完整点云变换'),('S08','输出高差及净变化体积')]
    for i,stem in enumerate(FIG_STEMS,1):
        sources.append({'id':f'F{i:03d}','type':'project-figure','locator':f'reports/paper_figures_v1/{stem}.png','summary':FIG_DESCRIPTIONS[i-1],'confidence':'high'})
        f={'number':i,'title':FIG_DESCRIPTIONS[i-1],'type':'flowchart' if i==1 else 'result-chart','source_ids':[f'F{i:03d}'],'asset':stem}
        if i==1:f.update(orientation='vertical',claim_number=1,complete_claim_flow=True,
            nodes=[{'id':n,'label':label,'claim_step':n} for n,label in nodes],
            edges=[{'from':a,'to':b} for a,b in [('S01','S02'),('S02','S03'),('S03','S04'),('S04','S05'),('S05','S06a'),('S05','S06b'),('S06a','S07'),('S06b','S07'),('S07','S08')]])
        figs.append(f)
    scores={k:{'score':s,'evidence':e} for k,s,e in [
        ('evidence_support',4,'核心特征均有代码、配置和公式支持，未加入未来机制。'),
        ('claim_architecture',4,'一条封闭方法主链及六项从属限定，均与实现对应。'),
        ('terminology_consistency',4,'候选稳定点、保留掩膜、有效网格等用词统一。'),
        ('enablement_detail',4,'双向残差、阈值、回退、支持判断和数据条件明确。'),
        ('technical_effect_reasoning',3,'受控诊断和完整对照支持有限效果，但部分参数缺少单项消融。'),
        ('formula_coverage',4,'17组公式列出符号、来源及作用，并输出原生Office Math。'),
        ('figure_alignment',4,'复用五幅配图，流程节点与步骤相符，对齐已核查；碰撞检查另记未完成。')]}
    d={'schema_version':'2.0','title':TITLE,'metadata':{'source':'课程模板、冻结代码、结果及配图','target':'课程模拟发明专利说明书','draft_status':'供课程审阅，不是正式申请就绪声明'},
       'source_analysis':{'contains_core_formulas':True,'formula_count_in_source':len(eqs),'contains_methodology_figures':False},
       'source_map':sources,'terminology_ledger':[{'concept':s,'canonical_zh':s,'source_terms':[s],'forbidden_aliases':[]} for s in ['参考点云','待测点云','候选稳定参考点','法向残差','有效对应','保留掩膜','公共有效网格']],
       'formula_inventory':[{'source_id':e['source_ids'][0],'source_number':str(e['number']),'technical_role':e['technical_role'],'disposition':'specification-equation-'+str(e['number'])} for e in eqs],
       'figure_inventory':[{'source_id':f'F{i:03d}','source_number':f'图{i}','type':'flowchart' if i==1 else 'result-chart','disposition':'reuse-existing-figure-'+str(i)} for i in range(1,6)],
       'abstract_figure_number':1,'assumptions':['课程报告优先于正式申请文件的分件组织要求'],
       'invention_concept':{'technical_problem':'局部真实变化与整体错位共同影响配准和体积','technical_means':'双向法向残差筛选、数量回退、筛选拟合与完整输出分离、有效支持积分','technical_effect':'在限定实验条件下降低净体积误差'},
       'evidence_ledger':ledger,'claims':[{'number':i,'text':t} for i,t in enumerate(CLAIMS,1)],'claim_feature_map':claim_map,'figures':figs,
       'specification':{'technical_field':FIELD,'background':BACKGROUND,'invention_content':{'problem':PROBLEM,'solution':SOLUTION,'beneficial_effects':EFFECTS},'figure_descriptions':FIG_DESCRIPTIONS,'equations':eqs,'embodiments':EMBODIMENTS},
       'abstract':ABSTRACT,'audit':{'support_findings':['无未实现的条件数、空间覆盖或全局初始化机制进入权利要求'],'consistency_findings':['逐点为主方案；区域为对照；图3下排不为总真值误差']},
       'quality_assessment':{'status':'review-draft','scores':scores},'inventor_questions':['[TO CONFIRM: 姓名及学号]','[TO CONFIRM: 如正式申请，另核实发明人、权属、公开时间及接近技术]']}
    save(WORK/'00-intake.json',{'scope':'课程报告与配套模拟专利材料','source_files':files,'unknown_personal_fields':['姓名','学号'],'publication_status':'unknown','ownership':'unknown','inventorship':'unknown','native_office_conversion':'Word/WPS E_ACCESSDENIED; reconstruct from template text and original wordmark'})
    save(WORK/'01-source-map.json',sources)
    save(WORK/'02-technical-inventory.json',{'terms':d['terminology_ledger'],'equations':d['formula_inventory'],'figures':d['figure_inventory'],'input_operation_output':chain,'gaps':['无全局初始化','无条件数/空间覆盖约束','无可靠同基准超体素复现','无独立地形']})
    save(WORK/'03-evidence-ledger.json',ledger);save(WORK/'04-claim-strategy.json',d['invention_concept'])
    (WORK/'05-claims.txt').write_text('\n\n'.join(f'{i}. {t}' for i,t in enumerate(CLAIMS,1)),encoding='utf-8');save(WORK/'05-claim-map.json',claim_map)
    save(WORK/'06-draft.json',d);save(OUT/'结构化草稿.json',d)
    return d

def task_page(doc):
    doc.h('数据处理与特征提取——任务书')
    doc.text('设计内容：基于课程内容模拟专利撰写',size=11,leading=17,indent=False)
    sections=[('1．课程报告要求',['结合课程所讲述的数据约减、数据分析、特征选择、特征提取、聚类分类等方面内容以及专利撰写规范，主要结合数据分析或其他智能算法，题目自拟，撰写一份发明专利申请材料，重点涉及专利文件的说明书（含附图）撰写。']),
       ('2．任务工作量',['（1）技术背景介绍；（2）主要涉及方案；（3）发明内容的有益效果分析；（4）设计方案的执行实施例；（5）给出必要的说明书附图，以增强说明书的可读性。']),
       ('3．工作计划（考试后1周完成并提交）',['（1）掌握课程所讲内容，研读专利撰写PPT，自拟题目并明确说明书背景。','（2）结合深度学习或其他智能算法，自拟仿真题目，通过计算机仿真，完成发明内容的主要步骤。','（3）撰写发明实施例，完善有益效果分析、说明书附图等环节，同时写明如何应用发明的区别特征来解决实际问题或困难的。','（4）按照模板样例，校验报告格式，生成word和pdf文件，与数据、程序源码一并发送至：zm1840@163.com，如数据过大也可加QQ（14812267）后传送。'])]
    for h,paragraphs in sections:
        doc.text(h,size=10.5,leading=15,font='heading',indent=False,before=3,after=2)
        for p in paragraphs:doc.text(p,size=10.5,leading=15,after=2)
    doc.table('4．评分标准（以下4方面分数占比相同）',['评分项','对应环节','打分'],[
       ['（1）格式规范性：数据能体现出合理整理工作、程序源码规范、报告格式正确，且不缺少发明专利所要求各个环节','数据、程序、报告整体',''],
       ['（2）内容完整性：各个公式、特征点、权利要求点是否公开充分，发明专利务必包含流程图，且必须标注步骤序号','背景技术、流程图',''],
       ['（3）技术特征是否清晰：步骤化陈述专利的技术环节，区别特征明显，命名时体现出对最接近现有技术的升级改造','发明内容',''],
       ['（4）分析如何解决实际困难：实际困难是否被明确指出，实施例体现出如何应用本发明特征技术来解决实际困难的','具体实施方式、附图',''],
       ['评语：','编号：','总分：']],[335,90,38],size=9)
    doc.text('教师签字：____________   （本页无需填写，所提交课程报告保留本页）',size=9.5,leading=15,indent=False)
    doc.new()

def data_tables(doc,index):
    if index==1:
        config=load('configs/robustness_volume_v1.json');labels=['纯刚性','标准堆积','15 mm噪声','待测点云变稀','低幅堆积','挖除','大范围堆积']
        values=[[label,f"{s['noise_sigma_m']*1000:g}",f"{s['moving_keep']:.2f}",f"{s['radius_m']:g}",f"{s['height_m']:g}"] for label,s in zip(labels,config['scenarios'])]
        doc.table('表1  新增固定参数实验条件（每条件三个种子）',['条件','每轴噪声/mm','待测保留概率','半径/m','峰值/m'],values,[125,90,100,70,70])
        doc.text('两期抽样概率作用于留出检查点后的候选顶点；参考保留概率固定0.90。表中保留概率是数据生成的随机抽样，与算法拟合掩膜不同。',size=10.5,leading=16)
    elif index==2:
        diag=[r for r in rows('results/stable_region_diagnosis_v1/summary.csv') if r['case']=='02_local_deposition']
        labels={'all_points':'全点拟合','oracle_stable':'真值稳定区（诊断）','random_matched':'等数量随机点'}
        values=[[labels[r['method']],r['runs'],f"{float(r['translation_error_mm_mean']):.4f}",f"{float(r['stable_holdout_pose_rmse_mm_mean']):.4f}"] for r in diag]
        doc.table('表2  开发堆积案例的参考区诊断',['方式','重复数','平移误差/mm','稳定点残差/mm'],values,[180,55,110,120])
    elif index==3:
        values=[['法向近邻数','24（含查询点）'],['有效对应距离','≤3s'],['尺度下限与筛选系数','0.05s；τ=3σ'],['初步/精配准上限','50/100次'],['两期最小保留','各≥50%，且各≥30点'],['网格/XY支持/最大三角形边长','0.25/0.75/2.0 m']]
        doc.table('表3  主方案固定参数',['参数','实现值'],values,[240,220])
    elif index==4:
        overall=rows('results/robustness_volume_v1/overall_summary.csv');allrows=rows('results/robustness_volume_v1/all_results.csv')
        labels={'ordinary':'普通ICP','trimmed_90':'固定90%截尾','huber':'Huber点到点加权','normal_space':'简化法向采样','pointwise_gate':'逐点筛选（主方案）','patch_no_dilation':'区域筛选、不扩张','patch_candidate':'区域筛选、扩张','oracle_pose':'正确位姿体积对照'}
        values=[]
        for r in overall:
            rotation=sum(float(x['rotation_error_deg']) for x in allrows if x['split']=='new_conditions' and x['method']==r['method'])/21
            values.append([labels[r['method']],f'{rotation:.6f}',f"{float(r['mean_translation_error_mm']):.4f}",f"{float(r['mean_stable_pose_rmse_mm']):.4f}",f"{float(r['mean_net_volume_absolute_error_m3']):.4f}"])
        doc.table('表4  新增21例的总体误差均值',['方法','旋转/°','平移/mm','稳定点/mm','净体积/m³'],values,[145,80,80,80,80])
        scen=rows('results/robustness_volume_v1/scenario_summary.csv');config=load('configs/robustness_volume_v1.json');values=[]
        for label,s in zip(['纯刚性','标准堆积','15 mm噪声','待测变稀','低幅堆积','挖除','大范围堆积'],config['scenarios']):
            cells=[]
            for method in ['huber','pointwise_gate','patch_candidate','oracle_pose']:
                r=next(x for x in scen if x['scenario']==s['name'] and x['method']==method)
                cells.append(f"{float(r['net_absolute_error_m3_mean']):.4f}±{float(r['net_absolute_error_m3_std']):.4f}")
            values.append([label,*cells])
        doc.table('表5  分条件净体积绝对误差（均值±样本标准差，m³）',['条件','Huber','主方案','区域扩张','正确位姿对照'],values,[80,96,96,96,96],size=9)

def specification(doc,draft):
    eqs={e['number']:e for e in draft['specification']['equations']}
    doc.h(TITLE)
    doc.text('本报告按照课程模板采用模拟发明专利说明书形式。方法与效果以已完成实验为依据，技术名称不构成新颖性或授权前景的判断。',size=10.5,leading=17,indent=False,after=10)
    doc.h('（一）技术领域')
    for p in FIELD:doc.text(p)
    doc.h('（二）背景技术')
    for p in BACKGROUND:doc.text(p)
    doc.h('（三）发明内容')
    for p in PROBLEM+SOLUTION:doc.text(p)
    for title,numbers,intro in STEPS:
        doc.h(title,2);doc.text(intro)
        for n in numbers:doc.equation(eqs[n]);doc.text(eqs[n]['description'],size=11,leading=18)
    doc.h('有益效果及证据边界',2)
    for p in EFFECTS:doc.text(p)
    doc.h('（四）附图说明')
    for p in FIG_DESCRIPTIONS:doc.text(p)
    doc.text('五幅附图集中附于说明书后，采用已完成的同一套图形，编号、颜色及比较口径一致。图2真值、图3正确位姿及图4、图5误差均只用于评价。')
    doc.h('（五）具体实施方式')
    for i,item in enumerate(EMBODIMENTS,1):
        doc.h(item['heading'],2)
        for j,p in enumerate(item['paragraphs']):
            doc.text(p)
            nums=[14,15] if i==1 and j==3 else [16,17] if i==4 and j==2 else []
            for n in nums:doc.equation(eqs[n]);doc.text(eqs[n]['description'],size=11,leading=18)
        data_tables(doc,i)

def figures(doc):
    for i,stem in enumerate(FIG_STEMS):
        doc.new();doc.picture(FIG/(stem+'.png'),caption_paragraphs=CAPTIONS[i])

def references(doc):
    doc.new();doc.h('参考文献与技术资料')
    for p in REFERENCES:doc.text(p,size=10.5,leading=17,indent=False,after=8)
    doc.text('课程依据：教师提供的2026秋课程报告模板及《课程设计说明书——数据存储预处理技巧》。保留封面字段、任务要求和评分内容，用本项目正文替换模板中的无人机示例。',size=10.5,leading=17)

def appendix(doc):
    doc.new();doc.h('附录  实验复现与提交说明')
    doc.h('A．项目组织与关键入口',2)
    doc.table('表A1  可复现材料索引',['材料','内容'],[
        ['scripts/automatic_stable_icp.py','输入坐标算法、成熟思路对照及筛选分支'],
        ['scripts/terrain_volume.py','高程插值、有效支持及体积积分'],
        ['scripts/run_robustness_volume.py','21例扩展及评价入口'],
        ['configs/automatic_stable_v1.json','冻结算法参数'],
        ['configs/robustness_volume_v1.json','条件、种子及网格规则'],
        ['results/robustness_volume_v1','输入、真值、矩阵、网格及结果表'],
        ['reports/paper_figures_v1','五幅图、源数据、图注及图形QA']],[235,225],size=9)
    doc.text('在项目目录使用独立环境运行：',size=11,leading=18)
    doc.text('.venv/Scripts/python.exe scripts/run_robustness_volume.py',size=10,leading=17,font='latin',indent=False)
    doc.text('已有结果可直接阅读和回读，无须为了写报告重跑所有实验。提交包保留源码、配置、依赖版本、关键结果及代表案例；大型原始示例包保留在项目中，不将虚拟环境和重复大文件作为报告附件。',size=11,leading=18)
    doc.h('B．配图与文档检查边界',2)
    doc.text('五幅配图的面板对齐、字号及目视检查已完成。自动PDF碰撞审计依赖未安装，其安装此前因非必要遥测连接被自动审批拒绝，故不声称该项通过。本文另检查公式、附图和表格引用、数值来源、OOXML结构及PDF分页。',size=11,leading=18)
    doc.text('当前环境的Word/WPS自动转换返回访问被拒绝，因此按原模板文字、学校字标和章节结构重建可编辑DOCX，并由同一分页源生成PDF阅读版。文字和公式可编辑，但尚未完成本机Word的实际重新分页渲染；正式提交前应在Word打开后复核页码。',size=11,leading=18)
    doc.h('C．个人信息与模拟申请边界',2)
    doc.text('姓名、学号未提供，封面留空，不推测作者身份。评分页保留且不自行填写。本文为课程模拟专利材料；后续正式申请仍需确认发明人、权属及公开时间，并经相应专业人员核对接近技术与权利要求，不将结构检查视作专利性意见。',size=11,leading=18)

def main_report(draft):
    doc=Pages(TITLE);doc.cover(WORK/'template_wordmark.png',TITLE);task_page(doc)
    doc.h('说明书摘要');doc.text(ABSTRACT)
    doc.text('关键词：地形点云；法向残差；候选稳定参考点；ICP；体积变化。',size=11,leading=18,indent=False)
    doc.text('实验定位：真实地形几何上的半合成变化及已知位姿扰动，不将真实两期变化当作已知真值。',size=10.5,leading=17,indent=False,before=12)
    doc.new();doc.h('权利要求书（课程模拟）')
    for i,t in enumerate(CLAIMS,1):doc.text(f'{i}．{t}',size=11,leading=18,indent=False,after=10)
    doc.new();specification(doc,draft);figures(doc);references(doc);appendix(doc)
    pages=doc.finish();save(WORK/'report_page_plan.json',pages)
    draw_pdf(pages,OUT/(PREFIX+'.pdf'),TITLE,OUT/'qa_pages')
    write_docx(pages,OUT/(PREFIX+'.docx'),TITLE)
    markdown(pages,OUT/(PREFIX+'.md'),TITLE)
    for stem in FIG_STEMS:shutil.copy2(FIG/(stem+'.png'),OUT/(stem+'.png'))
    shutil.copy2(WORK/'template_wordmark.png',OUT/'template_wordmark.png');save(OUT/'目录与页码.json',doc.headings)
    return pages

def separate_parts(draft):
    folder=OUT/'模拟专利分件';folder.mkdir(exist_ok=True)
    for name in ['权利要求书','说明书','说明书摘要','摘要附图']:
        doc=Pages(TITLE)
        if name=='权利要求书':
            doc.h(name)
            for i,t in enumerate(CLAIMS,1):doc.text(f'{i}．{t}',size=11,leading=18,indent=False,after=10)
        elif name=='说明书':specification(doc,draft);figures(doc)
        elif name=='说明书摘要':doc.h(name);doc.text(ABSTRACT)
        else:doc.h(name);doc.picture(FIG/'Fig01_method_flow.png',caption='图1  主方案流程图（与说明书图1相同）')
        write_docx(doc.finish(),folder/(name+'.docx'),TITLE)
    shutil.copy2(OUT/(PREFIX+'.docx'),folder/'完整审阅稿.docx')

def validate_draft(draft):
    base=Path(__file__).resolve().parent/'patent_helpers'
    if not base.exists():base=Path.home()/'.codex/skills/nature-paper-to-patent/scripts'
    outputs=[]
    for module,file in [('patent_validator','validate_patent_draft.py'),('claim_auditor','audit_claims.py'),('package_validator','build_patent_package.py')]:
        spec=importlib.util.spec_from_file_location(module,base/file);obj=importlib.util.module_from_spec(spec);sys.modules[module]=obj;spec.loader.exec_module(obj);outputs.append(obj)
    validator,auditor,package=outputs;findings=validator.validate(draft);report=validator.format_report(findings)
    (OUT/'草稿验证报告.txt').write_text(report,encoding='utf-8')
    assert not any(x.level=='ERROR' for x in findings),report
    package.validate(draft)
    checks=auditor.audit((WORK/'05-claims.txt').read_text(encoding='utf-8'))
    lines=[f'{x.level}\t权利要求{x.claim}\t{x.code}\t{x.message}' for x in checks] or ['PASS：未发现权利要求结构性问题。']
    (OUT/'权利要求检查.txt').write_text('\n'.join(lines),encoding='utf-8')
    assert not any(x.level=='ERROR' for x in checks),lines
    return {'draft_errors':sum(x.level=='ERROR' for x in findings),'draft_warnings':sum(x.level=='WARNING' for x in findings),
            'claim_errors':sum(x.level=='ERROR' for x in checks),'claim_warnings':sum(x.level=='WARNING' for x in checks)}

def main():
    draft=structured_draft();checks=validate_draft(draft);print('DRAFT_VALIDATION',checks,flush=True)
    pages=main_report(draft);print('REPORT_RENDERED',len(pages),flush=True);separate_parts(draft)
    with zipfile.ZipFile(OUT/(PREFIX+'.docx')) as z:
        xml=ET.fromstring(z.read('word/document.xml'));omml=len(xml.findall('.//{'+NS['m']+'}oMath'));images=len(xml.findall('.//{'+NS['w']+'}drawing'))
        assert omml>=17 and images>=6
    template_file=ROOT.parent/'【姓名】2026秋-课程报告-说明书.doc'
    prior_manifest=load('reports/course_report_v1/document_build_manifest.json') if (OUT/'document_build_manifest.json').exists() else {}
    template_hash=hashlib.sha256(template_file.read_bytes()).hexdigest() if template_file.exists() else prior_manifest.get('template_original_sha256','not-available')
    save(OUT/'document_build_manifest.json',{'title':TITLE,'date':'2026-11-24','pdf_pages':len(pages),'native_office_math_objects':omml,
        'embedded_figures_and_wordmark':images,'docx_zip_xml_validation':'passed','word_native_render_performed':False,
        'pdf_render':'same measured page plan; Python matplotlib offline','unfilled_personal_fields':['姓名','学号'],
        'template_original_sha256':template_hash,'template_file_available_in_this_build':template_file.exists(),'validation':checks,
        'report_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob(PREFIX+'.*')}})
    print(json.dumps({'output':str(OUT),'pages':len(pages),'native_math':omml,'images':images},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
