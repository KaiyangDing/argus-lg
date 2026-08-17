# S1–S3 报告：LangChain 原生链从脚手架到混合检索

> 2026-08-17 · argus-lg（LangChain + LangGraph 全量重写实验仓）
> 一句话：**通用件全链（pypdf 解析 + 递归字符切块 + BM25/向量/RRF 混合）在金标口径上做到
> 语料在场 18/18、混合检索召回 18/18(@16)**；代价与差距均已量化入档。

## 0. 路线与对照物

本仓从零重写 argus 引擎，一切能用 LangChain 生态原生件的地方不写自研；对照物是 argus v3
（MinerU 结构感知解析 + 自研词频检索，端到端基线 7/18）。**两仓指标属不同层，不直接互比**：
v3 的 7/18 是端到端报告覆盖率，本仓 S2/S3 是语料层与检索层的独立仪表；同口径对比等 S5 评测。

## S1 脚手架（实花 ≈¥0.0003）

uv + py3.14 + langgraph 1.2.11。首日撞墙即定型模型接入：`ChatTongyi`（community）实测
不回填标准 `usage_metadata`，且 langchain-community 整包宣布日落——拍板换
**`langchain-openai` `ChatOpenAI` + 阿里官方 OpenAI 兼容端点**，标准用量字段回填，
控费走「运行末尾 usage 折 ¥ 打印 + 花钱命令人工触发」。测试策略：无 cassette，
单测全部 LC Fake 系列（`FakeListChatModel` / `DeterministicFakeEmbedding`），零网络零成本。

## S2 语料摄取：LC 解析与切块的真实成色（¥0）

**用了什么**：`PyPDFLoader`（页粒度，底层 pypdf）+ `RecursiveCharacterTextSplitter`
（**LC 原生切块器，零自研**；定制仅中文分隔符优先级 `段落>换行>。；，>空格>硬切` 与
参数 500/50、`keep_separator="end"`）。chunk 契约：source_id / company / page / seq / chunk_id。

**效果读数**：

| 指标 | 读数 | 对照 |
|---|---|---|
| 解析成功 | 23/23 文档，5134 块 / 2.03M 字符 / **58.7s（纯 CPU）** | v3 MinerU：5770 块 / 2.01M，GPU 60.7 分钟 |
| **金标在场率** | **18/18** | v3 = 18/18（打平）；v2 pdfplumber = 18/18 |
| 数字劈半探针（`\d,\n\d`） | 88/5134 = **1.7%** | v2 = 57/1995 = 2.9%；v3 经表格重排在样张上修复此形态 |

在场率口径：18 条 must 要点各配人工锚点（`eval/presence_anchors.json`，自 v3 已知好语料
挖掘的真实表面形态，如全精度 `7,159,201,563.03`）；组内 OR / 组间 AND / 归一化子串匹配
（小写+去空白+去逗号，对 PDF 空格噪声免疫）。锚点本身成为新数据资产。

**诚实边界**：①在场率是必要条件非充分条件——v2 时代同样 18/18 在场却另有灾难面；
②劈半病在 pypdf 中同型存在（深层合并表：数字劈半、公司名竖切），只是恰未伤及金标锚点；
③纯线性文本，无 v3 的章节面包屑与表格竖线重排。**结论：通用件用 1/60 的时间与零模型
依赖买到「金标可及」这一关的全票，质量差距被局部化到深层表格，伤害量 S5 计量。**

## S3 检索研究：三算法对比与召回曲线（实花 ≈¥0.77）

**对比对象**（全 LC 原生）：

| 路线 | 实现 | 关键定制 |
|---|---|---|
| BM25 | `BM25Retriever`（rank_bm25 内核） | `preprocess_func` 挂 **jieba**——默认空白切词对中文整句成一 token，直接退化 |
| 稠密向量 | `OpenAIEmbeddings`(text-embedding-v4, 1024 维, 兼容端点) + `InMemoryVectorStore`(numpy 余弦) | `check_embedding_ctx_length=False` 必设（tiktoken 预编码 token 数组，非 OpenAI 端点不认）；5134 块全量 ≈¥0.76，分段落盘可续跑 |
| 混合 | `EnsembleRetriever`（加权 RRF，0.5/0.5） | argus 停在门口的混合检索，生态开箱 |

**仪器**：recall@k 探针（要点原文为查询、公司域内 top-k、锚点组全中=召回）＋
`--diag`（miss 要点的锚点组在各路完整排序中的真实名次）＋
`--sweep`（深池 200 检索一次、免费切出全 k 曲线）。

**实验序列（四轮）**：

1. 三路 @4：bm25 12 / vector 10 / hybrid 12——混合无净增益，且 `yh/m1` 向量单路命中被
   RRF 融合稀释出局。
2. sub_k 扩池（4→12）：总分不动，另添反例 `lpz/m3`——两条单路各自 ✓、混合 ✗
   （两路命中的是不同含锚块，单路分被两路共现的平庸块压下）。**扩池不是杠杆。**
3. 名次诊断：5 条顽固 miss（lpz/m1,m2、szss/m1,m2,m6，全部财务数字类——问法是四舍五入的
   「71.6 亿」，正文是全精度数）在混合深池排序中全部落在 **rank 6~14**：排位差一口气，
   不是嵌入够不着。单路结构性盲区实证：BM25 对 `szss/m2` 主锚 rank 79（语义盲），
   向量对 `szss/m5`（人物词条）rank 84（词面盲）——互补性有名次背书。
4. 召回曲线（深池 200，融合后切 top-k）：

| k | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| bm25 | 12 | 13 | **13（平台）** | 13 |
| vector | 10 | 11 | 12 | 14 |
| **hybrid** | 12 | 14 | 16 | **18/18** |

单条要点最深需求 k=14（lpz/m1、lpz/m2 双锚对）。

**结论**：①混合检索价值在本语料被完整量化——≥@8 严格双优于两条单路，@16 全收；
②**证据预算 k 是主杠杆**（不是融合池宽度）；③本探针以要点原文为单查询，是真实系统的
**下界代理**（图内查询由 LLM 生成、每方面多条取并集）。
**S4 检索配置由此钉定：hybrid + 深池 200 + 每查询 k=12 + researcher 每方面 2~3 查询并集。**

## 花费总账

S1 ≈¥0.0003 ＋ S2 ¥0 ＋ S3 ≈¥0.77（embedding 0.76 + query 零头）＝ **≈¥0.77**（精确以百炼控制台为准）。

## 撞墙与替换总账（截至 S3）

| # | 墙 | 裁决 |
|---|---|---|
| 1 | ChatTongyi 无标准 usage_metadata；community 整包日落 | 换 ChatOpenAI + 阿里兼容端点；dashscope/community 依赖摘除（loader/BM25 仍暂用 community，冻结可用有直读退路） |
| 2 | BM25Retriever 默认空白切词毁中文 | preprocess_func 挂 jieba |
| 3 | OpenAIEmbeddings tiktoken 预编码不容非 OpenAI 端点 | check_embedding_ctx_length=False |
| 4 | RRF 稀释单路命中（yh/m1、lpz/m3 两展品） | 深池+加大 k 洗掉；权重旋钮留档未动 |
| 5 | pypdf 深层表劈半病（1.7%） | 未伤金标锚点，S5 忠实度口径计量后再议 |
