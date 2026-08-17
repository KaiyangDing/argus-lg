# argus-lg 开发计划（一页纸）

> 2026-08-17 拍板：LangChain + LangGraph 全量重写 argus 引擎；core/eval（cassette / BudgetGate / 金标四闸）不带。
> 理念：先用 LangChain 原生件，发现问题再逐个讨论替换——撞墙本身是实验数据。
> 协议：代码全部 AI 写与落盘；步进式（每步 报告→检查→通过→commit）；代码零复制、数据资产（PDF/金标）复用；
> 数据路线 = **A 全 LC 链**（从 23 份原始 PDF 起全链走 LC，重 embed；v3 chunks 仅作 S2 对照基准）。

## 步骤与状态

| 步 | 内容 | 验收 | 预估花费 | 状态 |
|---|---|---|---|---|
| S1 | 脚手架：uv+py3.14 / ChatOpenAI(兼容端点) 冒烟 / Fake 测试骨架 | 冒烟真调用通；pytest / ruff 绿 | 实花 ≈¥0.0003（两次冒烟） | ✅ 2026-08-17 |
| S2 | 语料摄取：PyPDFLoader → 切块 → documents.jsonl + sha256 清单 | **18/18 在场**（23/23 文档、5134 块/2.03M 字符；劈半探针 88 块=1.7% 已量化，伤害待 S5 读数） | ¥0 | ✅ 2026-08-17 |
| S3 | 检索：BM25(jieba) → text-embedding-v4(兼容端点) → InMemoryVectorStore → EnsembleRetriever | **hybrid 召回曲线 12/14/16/18(@4/8/12/16)，@16 全收**；bm25 平台 13 / vector 14；仪器 --diag/--sweep 入仓；报告 reports/s1_s3_lc_pipeline.md | 实花 ≈¥0.77 | ✅ 2026-08-17 |
| S4 | 研究图：supervisor → Send 并行 researcher(动态 Send) → merge → write；MemorySaver；struct_factory 注入缝；单测全 Fake | 真跑 lpz×2（各 ≈¥0.073/30s）：4 方面×16 证据、全局去重编号现形；**财务事实手数 ≈5/6（v3 同司 0/6）**；引用句末化修复实证；聚合句残留转 S5 计量 | 实花 ≈¥0.146 | ✅ 2026-08-17 |
| S5 | 评测：金标 cases.jsonl + structured-output judge（覆盖/陷阱/引用质标含聚合句计数/忠实度抽样）；防作弊承 argus | **三轮：覆盖 7→11→9（均值 9，全 ≥ v3 7）**；陷阱 0×3；抓获幻觉+装饰引用一例；分节生成后引用 81.8~85.7%/聚合 0/忠实 18/18 三家池满；judge ~120 调用零崩；轮间方差 ±2 入档（录放层价格实测）；报告 reports/s5_e2e_eval.md | 实花 ≈¥0.88 | ✅ 2026-08-17 |
| S6 | langgraph dev（Studio）demo + README | Studio 起服探活实测通过（12s/200/杀树干净）；README=60 秒图+三层仪表+撞墙总账六条+数据点名制 v0.2 | ¥0 | ✅ 2026-08-17 |

## 替代设计（砍 core/eval 后）

- **控费**：qwen-flash 默认；每次运行汇总 usage_metadata 折 ¥ 打印；花钱命令（embedding 全量/整图真跑/评测轮）用户手动触发，AI 只跑零成本验证；单测永远 Fake 零真调。
- **测试**：LC Fake 系列（FakeListChatModel / DeterministicFakeEmbedding）验编排与解析逻辑；真实模型行为靠人工冒烟 + S5 评测读数兜底，不做回放回归。

## 自研 → LC 原生对照

MinerU→PyPDFLoader ｜ build_chunks→RecursiveCharacterTextSplitter ｜ 薄客户端→ChatOpenAI(阿里兼容端点) / embeddings S3 定 ｜
词频检索→BM25Retriever(jieba)+InMemoryVectorStore+EnsembleRetriever ｜ cassette→Fake 系列 ｜
行协议 judge→with_structured_output ｜ Streamlit→langgraph dev ｜ AsyncSqliteSaver→MemorySaver 起步 ｜ BudgetGate→用量读数+人工触发

## 撞墙与替换记录

1. **2026-08-17 S1 冒烟**：①`ChatTongyi` 不回填标准 `usage_metadata`（数据滞留 `response_metadata.token_usage`，生态通用控费件看不见）；②`langchain-community` 整包宣布日落（迁移方向=独立集成包+OpenAI 兼容端点）。**拍板**：聊天模型换 `langchain-openai` `ChatOpenAI` + 阿里官方兼容端点；`dashscope`/`langchain-community` 依赖摘除；S3 embeddings 候选同路（`OpenAIEmbeddings`+兼容端点）；S2 loader / S3 BM25 暂仍需 community（冻结可用，坏则 pypdf/rank_bm25 直读退路），到步再议。
2. **2026-08-17 S3 检索层**：①`BM25Retriever` 默认空白切词，中文整句一 token 直接退化——`preprocess_func` 挂 jieba；②`OpenAIEmbeddings` 默认 tiktoken 预编码 token 数组送非 OpenAI 端点不认——`check_embedding_ctx_length=False` 必设；③RRF 稀释单路命中两件实证（yh/m1 @sub4、lpz/m3 @sub12），深池下消失；④召回曲线（深池200）：bm25 12→13 平台 / vector 10→14 / **hybrid 12/14/16/18(@4/8/12/16)**，最深需求 k=14——**证据预算 k 是主杠杆**；S4 检索配置钉定 hybrid+深池200+k=12+多查询并集。
3. **2026-08-17 S4 真跑**：qwen-flash writer 引用聚堆两形态——①章节标题聚合罗列（`render_sections` 把编号放进标题自种的病，已修：候选编号独立行+句末硬约束，二跑实证句末引用回归）；②段尾总结句全量编号倾倒（残留）——不追 prompt，S5 计量口径处理（>3 引用聚合句单独计数；忠实度抽样只取 1~3 引用句）。正面数据点：`with_structured_output` 过兼容端点 5 调用零失败，预告的墙没出现。
4. **2026-08-17 S5 一轮**：①**时间盲区**——yh 0/6 机检=报告通篇 2025、金标 2024 事实不在文本（检索层 S3 已证可及，病灶在 LLM 查询生成无年份维度）；处方一行（查询强制 2024/2025 各至少一条）→ 二轮 yh 0→3、总分 7→11 兑现。②**writer 幻觉+装饰性引用**——「宁波汉意被监管警示 [17][18]」全语料无此事实、被引块=退市样板页（预训练记忆掏事实+字面配引用）；二轮消失=非确定性，指标每轮必跑的理由。
5. **2026-08-17 S5 二轮**：**引用纪律 prompt-only 不稳定**——同一 prompt 三形态振荡（标题聚合→句末合规 84%→lpz 段落倾倒：正文全裸+节末 16 连发、率 16.7%、忠实度抽样池空转），yh/szss 同轮却守住；处方=结构性「write 分节生成」，用户拍板当轮执行。**三轮兑现：三家 81.8/82.4/85.7%、聚合句 0、忠实度 18/18 池满——结构性处方全指标达标**。正面：judge structured output 三轮 ~120 调用零崩、quote 纠错 3→0→0——旧仓行协议战役在 LC 工具调用通道未复发。
6. **2026-08-17 S5 三轮**：**覆盖轮间方差 ±2（7→11→9，逐点翻转）**——temperature=0 不救（兼容端点服务面非确定+计划/查询措辞漂移级联）；含义=无录放层则每轮评的是系统随机样本，prompt 回归验证需多轮采样；cassette 的价格首次实测标出（「core 不带」裁决不变）。对外口径钉死：三轮 7/11/9 均值 9，不摘单轮最高。v0.2 候选：计划/查询钉板 或 多轮采样评测。
