# AI 对话安全防护系统

> 面向中文 LLM 对话场景的内容安全防护系统，支持输入/输出双侧检测与风险处置

- **仓库地址**：https://github.com/JasmineWA/ai-chat-safety
- **作者**：JasmineWA
- **邮箱**：wanglixia@nuaa.edu.cn

## 项目简介

本项目是一个面向中文大语言模型（LLM）对话场景的内容安全系统原型，解决"LLM 聊天系统在真实运行时的内容安全问题"。

系统以聊天应用为入口，对**用户输入**和**模型输出**分别进行安全检测，并对风险内容执行放行、提示、脱敏、替换或拦截等动作，同时将聊天记录与风险日志持久化到本地数据库中。

**核心功能：**

- **输入侧检测**：规则匹配 + 本地分类模型两级检测，拦截 prompt injection、违法内容、仇恨言论、隐私风险等 7 类风险
- **输出侧检测**：二分类 ONNX 模型对大模型回复做二次审查（safe / unsafe）
- **风险处置体系**：`pass`（放行）→ `warn`（提示）→ `mask`（脱敏）→ `replace`（替换）→ `block`（拦截）
- **后台管理**：风险日志查看、统计分析、规则浏览
- **会话管理**：新建/删除/搜索会话，聊天记录持久化
- **IP 访问控制**：黑白名单机制

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 10 / macOS 12+ / Ubuntu 20.04 | 开发与测试所用系统 |
| Python | 3.12.4 | 核心语言版本 |
| GPU | 无（CPU 推理） | 本系统使用 ONNX Runtime CPU 推理，无需 GPU |

### 开源程序与第三方依赖

> 体积较大或需单独安装的程序，请在此列出下载链接与版本。

| 依赖名称 | 使用版本 | 下载链接 | 安装方式 | 说明 |
|----------|----------|----------|----------|------|
| DeepSeek API | - | https://platform.deepseek.com/ | 注册获取 API Key | 大模型对话接口（也可使用 SiliconFlow 代理） |

> **注意**：本系统通过 OpenAI 兼容接口调用 DeepSeek，也支持通过 SiliconFlow 代理访问。无需额外安装本地大模型。

### Python 依赖

依赖清单文件：
- 运行期 → `requirements.txt`（已包含在本仓库中）
- 训练期 → `FineTuning/requirements-finetune.txt`（已包含在本仓库中）

安装命令：
```bash
# 运行期依赖
pip install -r requirements.txt

# 训练期依赖（如需微调模型）
pip install -r FineTuning/requirements-finetune.txt
```

运行期 `requirements.txt` 内容：
```
flask>=2.0
openai>=1.0
onnxruntime>=1.15.0
numpy>=1.24
transformers>=4.30.0
```

## 配置说明

### 大模型接口配置

运行前需设置大模型接口的环境变量：

```powershell
# PowerShell 示例
$env:DEEPSEEK_API_KEY="你的 API Key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

> **安全提示**：敏感配置（API Key）请使用环境变量或 `.env` 文件，`.env` 已加入 `.gitignore`，请勿提交真实密钥到仓库。

### 其他关键配置

| 配置项 | 默认值 | 说明 | 配置方式 |
|--------|--------|------|----------|
| `DEEPSEEK_API_KEY` | （无） | 大模型 API 密钥 | 环境变量 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 接口地址 | 环境变量 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 使用的模型名称 | 环境变量 |
| `DB_FILE` | `ai_chat_safety.db` | SQLite 数据库文件路径 | 环境变量 |
| `IP_BLACKLIST` | （空） | IP 黑名单，逗号分隔 | 环境变量 |
| `IP_WHITELIST` | （空） | IP 白名单，逗号分隔 | 环境变量 |
| 服务端口 | `5000` | Flask 默认监听端口 | `app.py` 内修改 |
| 日志级别 | `INFO` | 控制台与文件日志级别 | `logger.py` 内修改 |

### 数据文件配置

系统启动时会自动从 `data/` 目录加载以下配置文件：

| 配置文件 | 路径 | 说明 |
|----------|------|------|
| 输入检测规则 | `data/rules_input.json` | 正则表达式规则（风险类别、等级、动作） |
| 风险关键词库 | `data/risk_keywords.json` | 中文关键词 → 风险类别/分数映射 |
| 规则类别映射 | `data/rule_category_map.json` | 规则 ID → 风险类别映射 |
| 安全替换模板 | `data/safety_templates.json` | 6 种场景的替换话术 |
| 语义样例库 | `data/semantic_examples.json` | 语义匹配用的典型风险样例 |

## 数据集

### 数据集说明

| 数据集名称 | 来源 | 大小 | 格式 | 说明 |
|-----------|------|------|------|------|
| Safety-Prompts（输入侧） | [THUDM/Safety-Prompts](https://huggingface.co/datasets/thu-coai/Safety-Prompts) | ~73MB（训练集）+ ~13MB（验证集） | JSON/JSONL | 输入侧 7 类安全分类数据 |
| PKU-SafeRLHF（输出侧） | [PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ~163MB（训练集）+ ~14MB（验证集） | JSON/JSONL | 输出侧 safe/unsafe 二分类数据 |
| 中文 RoBERTa 底模 | [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | ~412MB | PyTorch | 微调基础模型 |

> **体积较大的数据集不纳入 Git 仓库**，请通过外部链接下载后放置到指定目录。
>
> **小部分数据示例已提交到 Git 仓库中**（放置于 `data/samples/` 目录），用于：
> - 让其他开发者无需下载完整数据集即可快速了解数据格式与字段含义
> - 支撑单元测试和本地调试的最小可运行数据
> - 作为数据处理流程的输入示例，方便 Code Review 时对照理解逻辑
>
> 示例数据要求：
> - 条数控制在 5~20 条，文件大小不超过 100KB
> - 已脱敏处理，不包含真实用户隐私信息

### 数据集下载与放置

```bash
# 1. 安装 Git LFS（用于下载大模型权重文件）
git lfs install

# 2. 克隆底模（可选，仅训练时需要）
git clone https://huggingface.co/hfl/chinese-roberta-wwm-ext FineTuning/models/chinese-roberta-wwm-ext

# 3. 数据集通过训练脚本自动构建
python FineTuning/build_safety_prompts_dataset.py       # 构建输入侧数据集
python FineTuning/build_output_pku_saferlhf_dataset.py   # 构建输出侧数据集
```

数据集目录结构：
```
data/
├── samples/                          # ✅ 小部分数据示例（已提交到 Git 仓库）
│   ├── sample_rules_input.json       #    输入检测规则示例
│   ├── sample_risk_keywords.json     #    风险关键词示例
│   ├── sample_safety_templates.json  #    安全替换模板
│   ├── sample_semantic_examples.json #    语义样例示例
│   ├── sample_label_config_input.json#    输入侧标签配置
│   └── sample_label_config_output.json#   输出侧标签配置
├── rules_input.json                  # 输入检测规则（完整版）
├── risk_keywords.json                # 风险关键词库（完整版）
├── safety_templates.json             # 安全替换模板
├── semantic_examples.json            # 语义样例库
├── rule_category_map.json            # 规则类别映射
├── train.jsonl                       # 原始训练数据（不提交）
├── test.jsonl                        # 原始测试数据（不提交）
├── Safety-Prompts/                   # Safety-Prompts 原始数据集（不提交）
├── safety_prompts_eval_report.json   # 输入侧评测报告
└── output_pku_zh_eval_report.json    # 输出侧评测报告
```

> `data/` 目录已加入 `.gitignore`，但 `data/samples/` 通过 `!data/samples/` 规则**强制纳入版本控制**。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/HouYaohui0603/ai-chat-safety.git
cd ai-chat-safety

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量（PowerShell）
$env:DEEPSEEK_API_KEY="你的 API Key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"

# 4. 启动系统
python app.py
```

启动后访问：
- 聊天页面：http://127.0.0.1:5000/

## 项目结构

```
ai-chat-safety/
├── app.py                      # Flask 主入口，路由、会话管理、检测调用
├── db_sqlite.py                # SQLite 数据库层，表结构与 CRUD
├── detection_core.py           # 检测核心模块：规则结构、匹配引擎、脱敏
├── input_detector.py           # 输入侧检测器（规则 + 模型两级检测）
├── output_checker_v3.py        # 输出侧检测器（二分类模型）
├── local_model.py              # ONNX 本地模型封装层
├── logger.py                   # 统一日志模块
├── evaluate_safety_prompts.py  # 输入侧 ONNX 模型评测脚本
├── evaluate_output_pku_zh.py   # 输出侧 ONNX 模型评测脚本
├── requirements.txt            # 运行期依赖清单
├── .gitignore                  # Git 忽略规则
├── README.md                   # 本文件
│
├── data/                       # 运行期配置数据 + 数据集
│   ├── samples/                #   ✅ 示例数据（已提交到 Git）
│   ├── rules_input.json        #   输入检测规则
│   ├── risk_keywords.json      #   风险关键词库
│   ├── safety_templates.json   #   安全替换模板
│   └── semantic_examples.json  #   语义样例库
│
├── FineTuning/                 # 模型训练/导出/数据集构建
│   ├── requirements-finetune.txt  # 训练期依赖
│   ├── build_safety_prompts_dataset.py
│   ├── build_output_pku_saferlhf_dataset.py
│   ├── fine_tune_lora.py           # 输入侧 LoRA 微调
│   ├── fine_tune_output_pku_zh_lora.py  # 输出侧 LoRA 微调
│   ├── export_onnx.py              # 导出 ONNX 模型
│   ├── datafiles/                  # 训练/评测数据集（不提交）
│   └── models/                     # 模型文件
│       ├── chinese-roberta-wwm-ext/          # 底模（不提交）
│       ├── risk_classifier_input_safety_prompts/  # 输入侧分类模型
│       └── risk_classifier_output_pku_zh/    # 输出侧二分类模型
│
├── templates/                  # Flask HTML 模板
│   ├── chat.html               #   聊天页面
│   └── admin.html              #   后台管理页面
│
└── static/                     # 静态资源
    ├── css/app.css             #   统一样式
    └── js/
        ├── chat.js             #   聊天页前端逻辑
        └── admin.js            #   后台页前端逻辑
```
