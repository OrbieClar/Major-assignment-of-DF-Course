"""Core mathematical operations with PDF TeX and native Office Math ASTs."""
def sub(a,b):return ["sub",a,b]
def sup(a,b):return ["sup",a,b]
def frac(a,b):return ["frac",a,b]
def equations():
    result=[]
    def add(tex,expression,tree,description,role,symbols,source,lines=None,asts=None):
        n=len(result)+1
        x={"number":n,"latex":tex,"expression":expression,"ast":tree,"description":description,
           "technical_role":role,"symbols":[{"symbol":a,"meaning":b} for a,b in symbols],
           "source_ids":[f"E{n:03d}"],"source_location":source}
        if lines:x.update(latex_lines=lines,ast_lines=asts)
        result.append(x)
    add(r"p^l=p^g-o,\qquad q^l=q^g-o","p局部=p原始−o，q局部=q原始−o",
        [sup("p","l"),"=",sup("p","g"),"−o，",sup("q","l"),"=",sup("q","g"),"−o"],
        "p、q分别为待测与参考点三维坐标，g、l表示原坐标与局部坐标，o为共用原点。后续省略局部上标；两期不能分别居中后丢弃原点差。",
        "统一两期运算坐标",[("p,q","两期点坐标"),("o","共用原点"),("g,l","原与局部坐标")],"数据构建脚本：local=source−origin")
    add(r"s=\mathrm{median}_i\left(\min_{j\ne i}\|q_i-q_j\|_2\right)","s=参考点最近其他点距离中位数",
        ["s=",sub("median","i"),"(",sub("min","j≠i"),"‖",sub("q","i"),"−",sub("q","j"),sub("‖","2"),")"],
        "s为代表点间距，i、j为参考点索引，欧氏范数描述坐标距离。最近邻排除查询点自身；该尺度来自实际输入，不直接等于下采样参数。",
        "估计输入长度尺度",[("s","代表点间距"),("i,j","点索引")],"automatic_stable_icp.py：run_methods")
    add(r"C_i=\frac{1}{k}\sum_{v\in\mathcal{N}_i}(v-\bar q_i)(v-\bar q_i)^T","C_i=近邻去均值后的协方差",
        [sub("C","i"),"=",frac("1","k"),sub("Σ","v∈Nᵢ"),"(v−",sub("q̄","i"),")",sup(["(v−",sub("q̄","i"),")"],"T")],
        "式中，邻域包含参考点的k个近邻，带横线的坐标为邻域均值，C为3×3协方差矩阵，T表示转置。取最小特征值对应的单位向量为该点法向，并使Z分量非负。本例k=24，查询包括点自身。",
        "提取地形局部法向",[("N_i,k","邻域和点数"),("C_i","协方差"),("q̄_i","邻域均值"),("n_i","单位法向")],"automatic_stable_icp.py：estimate_normals")
    add(r"E(R,t)=\sum_i\|Rp_i+t-q_{j(i)}\|_2^2","E=对应点的点到点平方距离和",
        ["E(R,t)=",sub("Σ","i"),sup(["‖R",sub("p","i"),"+t−",sub("q","j(i)"),"‖"],"2")],
        "R为满足正交约束且行列式为1的旋转，t为平移，j(i)是第i个待测点在当前位姿下对应的参考最近邻索引。重复匹配并求刚性增量，得到下标为0的初步变换。",
        "普通ICP刚性拟合目标",[("R,t","旋转和平移"),("j(i)","当前最近邻索引"),("E","拟合目标")],"automatic_stable_icp.py：icp")
    add(r"R=VDU^T,\quad D=\mathrm{diag}(1,1,\det(VU^T)),\quad t=\bar b-R\bar a","R=反射校正后的Kabsch旋转，t=b均值−Ra均值",
        ["R=VD",sup("U","T"),"，D=diag(1,1,det(V",sup("U","T"),"))，t=b̄−Rā"],
        "对对应点去均值得互协方差H，作奇异值分解得到U、Σ和V。带横线的a、b为源、目标均值，D为反射校正矩阵。先完成旋转校正，再使用校正后的R计算平移；Σ为奇异值对角阵。",
        "反射校正后的刚性增量",[("U,Σ,V","H的奇异值分解"),("D","反射校正"),("ā,b̄","对应点均值")],"automatic_stable_icp.py：fit_rigid")
    add(r"r_i^P=(T_0p_i-q_{j(i)})\cdot n_{j(i)}","正向残差=(待测初步变换点−参考最近邻)·参考法向",
        [["subsup","r","i","P"],"=(",sub("T","0"),sub("p","i"),"−",sub("q","j(i)"),")·",sub("n","j(i)")],
        "式中，初步变换后的待测点减去其参考最近邻，再与该参考点的法向点乘，得到上标为P的正向残差。符号统一为待测减参考。",
        "待测点的法向差异",[("r_i^P","正向残差"),("T_0","初步变换"),("n_j(i)","参考法向")],"automatic_stable_icp.py：geometry_evidence")
    add(r"r_j^Q=(T_0p_{i(j)}-q_j)\cdot n_j","反向残差仍使用参考法向和待测减参考的符号",
        [["subsup","r","j","Q"],"=(",sub("T","0"),sub("p","i(j)"),"−",sub("q","j"),")·",sub("n","j")],
        "i(j)为第j个参考点在初步配准待测点云中的最近邻索引。上标Q表示反向对应残差；两方向的匹配距离d使用相应上下标区分。双向对应不要求互为最近邻。",
        "参考点的筛选证据",[("r_j^Q","反向残差"),("i(j)","反向索引"),("d_i^P,d_j^Q","匹配距离")],"automatic_stable_icp.py：geometry_evidence")
    lines=[r"\mu=\mathrm{median}(\mathcal{D}),\qquad\tau=3\sigma",r"\sigma=\max\left(1.4826\,\mathrm{median}_{r\in\mathcal{D}}|r-\mu|,\ 0.05s\right)"]
    add(lines[0],"μ=median(D)，σ=max(1.4826MAD,0.05s)，τ=3σ",["μ=median(D)，τ=3σ"],
        "D为两方向d≤3s的有效残差合并集合，μ为分布中心，σ为稳健尺度，τ为筛选阈值。1.4826为MAD尺度因子，0.05s避免尺度退化。该阈值是工程规则，不是已标定的形变检测限或概率置信区间。",
        "稳健中心和阈值",[("D","有效双向残差"),("μ,σ,τ","中心、尺度和阈值")],"automatic_stable_icp.py：geometry_evidence",lines,[["μ=median(D)，τ=3σ"],["σ=max(1.4826 ",sub("median","r∈D"),"|r−μ|,0.05s)"]])
    lines=[r"m_i^P=[d_i^P\leq3s]\wedge[|r_i^P-\mu|\leq\tau]",r"m_j^Q=[d_j^Q\leq3s]\wedge[|r_j^Q-\mu|\leq\tau]"]
    asts=[[["subsup","m","i","P"],"=[",["subsup","d","i","P"],"≤3s]∧[|",["subsup","r","i","P"],"−μ|≤τ]"],[["subsup","m","j","Q"],"=[",["subsup","d","j","Q"],"≤3s]∧[|",["subsup","r","j","Q"],"−μ|≤τ]"]]
    add(lines[0],"两期掩膜=有效距离且中心化残差不超过τ",asts[0],
        "方括号为条件真值，逻辑与表示同时满足。两期布尔掩膜m为真的点组成下标为s的待测与参考拟合集。掩膜只计算一次，精配准期间不更新；法向用于筛选，后续目标仍是点到点。",
        "两期逐点筛选",[("m^P,m^Q","两期保留掩膜"),("P_s,Q_s","筛选后的拟合集")],"automatic_stable_icp.py：pointwise_gate分支",lines,asts)
    add(r"\min\left(\frac{|P_s|}{|P|},\frac{|Q_s|}{|Q|}\right)\geq0.5,\quad\min(|P_s|,|Q_s|)\geq30","两期各保留≥50%且各≥30点",
        ["min(",frac("|P_s|","|P|"),",",frac("|Q_s|","|Q|"),")≥0.5，min(|P_s|,|Q_s|)≥30"],
        "竖线表示点数。两个条件均满足时S06a以初步变换为初值精配准最多100次，否则S06b返回初步变换、恢复全点掩膜并记录回退。保留数量不能保证空间分布和几何可观测性，当前未实现额外条件数判别。",
        "防止筛选不足后继续拟合",[("|P|,|Q|","输入点数"),("|P_s|,|Q_s|","拟合点数")],"automatic_stable_icp.py：run_methods")
    add(r"p_{\mathrm{aligned}}=Rp+t,\qquad t_g=t+o-Ro","完整待测点全部变换；原坐标平移项=t+o−Ro",
        [sub("p","aligned"),"=Rp+t，",sub("t","g"),"=t+o−Ro"],
        "最终R、t作用于完整待测点云P，包含未参与精配准拟合的点。下标g表示恢复到同一原坐标系时的平移项；矩阵与原点须一起保存，避免局部矩阵误用于大地坐标。",
        "保留真实变化的完整输出",[("p_aligned","配准后完整点"),("t_g","原坐标系平移")],"transform函数及数据坐标约定")
    add(r"V_{\mathrm{net}}=\sum_{k\in\mathcal{K}}\Delta z_k a_g^2","净体积=共同有效网格高差乘面积之和",
        [sub("V","net"),"=",sub("Σ","k∈K"),sub("Δz","k"),sup(sub("a","g"),"2")],
        "K为两期共同有效网格集合，下标g表示网格边长，第k个网格高差为待测配准高程减参考高程。本例比较进一步取所有算法及正确位姿对照的有效交集，使各方法同区域积分；无效单元不赋零。",
        "公共支持范围的净体积",[("K","有效网格集合"),("a_g","边长"),("Δz_k","高差"),("V_net","净体积")],"terrain_volume.py：integrate")
    add(r"V_+=\sum_{k\in\mathcal{K}}\max(\Delta z_k,0)a_g^2,\quad V_-=\sum_{k\in\mathcal{K}}\max(-\Delta z_k,0)a_g^2","V正=Σ正高差面积，V负=Σ负高差绝对值面积",
        [sub("V","+"),"=",sub("Σ","k∈K"),"max(",sub("Δz","k"),",0)",sup(sub("a","g"),"2"),"，",sub("V","−"),"=",sub("Σ","k∈K"),"max(−",sub("Δz","k"),",0)",sup(sub("a","g"),"2")],
        "下标正、负号分别标记正负高差分量，净体积为两者之差。未额外施加检测限，分量会积累噪声及插值起伏，不能直接当作可靠检出的堆积量或挖除量；主评价采用净体积绝对误差。",
        "保存正负分量以供复核",[("V_+,V_-","正负高差积分")],"terrain_volume.py：integrate")
    add(r"\delta z=h\max\left(1-\frac{(x-c_x)^2+(y-c_y)^2}{a^2},0\right)^2","人工变化=h·max(1−距离平方/半径平方,0)²",
        ["δz=h·",sup(["max(1−",frac([sup("(x−c_x)","2"),"+",sup("(y−c_y)","2")],sup("a","2")),",0)"],"2")],
        "此式只用于生成半合成变化。c的两个分量为平面中心坐标，a为半径，h为峰值，h正表示堆积、负表示挖除，圆域外变化为零。中心和参数仅供生成、评价，不传入自动筛选算法。",
        "生成已知局部变化",[("c_x,c_y","人工中心"),("a,h","半径和峰值")],"地形基准及扩展数据生成脚本")
    add(r"V_{\mathrm{full}}=\frac{\pi a^2h}{3}","完整圆域解析体积=πa²h/3",
        [sub("V","full"),"=",frac(["π",sup("a","2"),"h"],"3")],
        "下标full表示由规定高差函数对完整圆域面积积分得到的体积。公共支持不完整时，应与覆盖内离散真值比较，不能直接与完整圆域解析值混比。",
        "校验规定变化体积",[("V_full","全圆解析体积")],"规定高差函数的面积积分及生成脚本")
    add(r"e_t=\|\hat t-t^*\|_2,\quad e_R=\arccos\left(\frac{\mathrm{tr}(\hat R{R^*}^T)-1}{2}\right)","平移误差为平移差范数；旋转误差由相对旋转迹计算",
        [sub("e","t"),"=‖t̂−t*‖₂，",sub("e","R"),"=arccos(",frac(["tr(R̂",sup("R*","T"),")−1"],"2"),")"],
        "带帽量为估计，星号为正确逆变换，tr为矩阵迹。平移在相同局部原点下计算后换算毫米，旋转换算度；反余弦输入截断到[−1,1]避免浮点越界。",
        "位姿误差评价",[("e_t,e_R","平移和旋转误差"),("估计与真值","估计及正确逆变换")],"扩展实验评价脚本")
    add(r"e_s=\sqrt{\frac{1}{M}\sum_{k=1}^{M}\|\hat T u_k-T^*u_k\|_2^2}","独立稳定检查点的变换作用误差均方根",
        [sub("e","s"),"=",["sqrt",[frac("1","M"),sub("Σ","k=1…M"),sup("‖T̂u_k−T*u_k‖₂","2")]]],
        "u为未参与拟合的干净稳定检查点，下标k为其索引，M为其数量。该值测量估计变换与正确变换对同一检查点的作用差，区别于含噪点云最近邻拟合RMSE。",
        "独立稳定点验证",[("u_k,M","独立检查点和数量"),("e_s","稳定点位姿残差")],"扩展实验独立检查点评价")
    return result
