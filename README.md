# ModelRouter + Recursive Self-Learning (RSL)

本项目基于课程提供的 [ModelRouterCourse](https://github.com/NewYearEve-Godsheep/ModelRouterCourse) 进行扩展，在原有多模型路由框架基础上加入了一个简单的 **Recursive Self-Learning（RSL，递归自学习）** 模块。

RSL 会根据不同类型任务中各模型过去的实际表现，不断更新模型选择策略，从而让 ModelRouter 逐渐学习：

> 对于不同类型的问题，应该选择哪个模型。

---

## 1. 项目目标

原始 ModelRouter 已经支持多模型路由、模型调用、Cascade、Judge、Telemetry 等功能。

本项目在此基础上增加一个持续学习闭环：

```text
User Prompt
     ↓
Task Classification
     ↓
RSL Policy
     ↓
Model Selection
     ↓
ModelRouter
     ↓
LLM Response
     ↓
Judge Evaluation
     ↓
Reward
     ↓
Update Q-value
     ↓
Next Request
```

每次请求都会产生新的经验。

模型回答的质量和运行延迟会被转换成 Reward，并用于更新当前模型在对应任务类型下的 Q-value。

新的 Q-value 又会影响下一次模型选择，从而形成 Recursive Self-Learning。

---

## 2. 本项目的主要修改

在原始 ModelRouterCourse 基础上新增：

```text
app/core/rsl.py
```

负责：

- Task Classification
- RSL Policy
- ε-greedy Model Selection
- Judge Evaluation
- Reward Calculation
- Q-value Update
- Experience Persistence

新增：

```text
app/api/rsl.py
```

提供新的 RSL API：

```text
POST /v1/rsl/chat
```

修改：

```text
app/main.py
```

用于：

- 初始化 RSLEngine
- 注册 RSL API
- 将 RSLEngine 挂载到 FastAPI app state

增加：

```text
config/providers.yaml
```

用于将本地 Ollama 模型接入 ModelRouter。

---

## 3. 本地模型

本实验使用两个本地语言模型：

```text
qwen-small
    ↓
Qwen3-1.7B
```

以及：

```text
qwen-strong
    ↓
Qwen3-4B-Instruct
```

两个模型均通过 Ollama 部署在本地。

这样可以形成模型能力和推理开销之间的差异。

一般来说：

```text
Qwen3-1.7B
```

特点：

- 显存占用较低
- 推理速度较快
- 简单任务成本较低

而：

```text
Qwen3-4B-Instruct
```

特点：

- 模型能力更强
- 更适合 Coding / Reasoning 等复杂任务
- 推理时间和显存占用相对更高

RSL 的目标就是学习：

> 什么情况下小模型已经足够，什么情况下需要使用更强模型。

---

## 4. 系统架构

```text
                    ┌─────────────────┐
                    │   User Prompt   │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Task Classifier │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   RSL Policy    │
                    │    Q-values     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │   ModelRouter   │
                    └───────┬─────────┘
                           / \
                          /   \
                         ↓     ↓
                qwen-small   qwen-strong
                    ↓             ↓
               Qwen3-1.7B    Qwen3-4B
                         \   /
                          \ /
                           ↓
                         Answer
                           ↓
                    ┌──────────────┐
                    │  Judge Model │
                    │ Qwen3-4B     │
                    └──────┬───────┘
                           ↓
                         Reward
                           ↓
                      Update Q
                           ↓
                     Next Request
```

---

## 5. Task Classification

当前实现将请求分为四类：

```text
coding
math
translation
general
```

当前使用关键词进行简单分类。

例如：

```text
请用 Python 实现快速排序算法
```

会分类为：

```text
coding
```

而：

```text
把“人工智能正在改变世界”翻译成英文
```

会分类为：

```text
translation
```

每种任务类型拥有独立的模型策略。

例如：

```text
coding:
    qwen-small
    qwen-strong

math:
    qwen-small
    qwen-strong

translation:
    qwen-small
    qwen-strong

general:
    qwen-small
    qwen-strong
```

---

## 6. Model Selection

模型选择使用 **ε-greedy** 策略。

当前设置：

```text
epsilon = 0.2
```

因此：

```text
80%
→ Exploitation
→ 使用当前历史 Q-value 更高的模型
```

```text
20%
→ Exploration
→ 随机尝试候选模型
```

这样可以避免系统永久锁定在当前最优模型上。

系统仍然会偶尔尝试其它模型，以获得新的运行经验。

---

## 7. Initial Q-value

每个新的：

```text
Task Type + Model
```

组合初始 Q-value 为：

```text
Q = 0.5
```

例如第一次出现 Coding 请求：

```text
coding

qwen-small   Q = 0.50
qwen-strong  Q = 0.50
```

之后随着执行次数增加，Q-value 会不断变化。

---

## 8. Judge Model

本实验使用：

```text
qwen-strong
```

即：

```text
Qwen3-4B-Instruct
```

作为 Judge。

Judge 根据以下指标评价回答：

1. Correctness
2. Relevance
3. Completeness

Judge 输出格式：

```json
{
  "score": 0.85,
  "reason": "The answer is mostly correct."
}
```

其中：

```text
score ∈ [0, 1]
```

---

## 9. Reward Function

本实验的 Reward 同时考虑：

- 回答质量
- 推理延迟

计算方式：

```text
Reward
=
0.85 × JudgeScore
+
0.15 × LatencyScore
```

其中：

```text
LatencyScore
=
1 / (1 + latency_ms / 5000)
```

因此：

```text
回答质量越高
+
推理速度越快
↓
Reward 越高
```

---

## 10. Recursive Q-value Update

每次模型完成回答并获得 Reward 后，系统更新该模型的 Q-value。

更新公式：

```text
Q_new
=
(1 - alpha) × Q_old
+
alpha × Reward
```

当前：

```text
alpha = 0.2
```

例如：

```text
Q_old = 0.50
Reward = 0.90
alpha = 0.20
```

则：

```text
Q_new
=
0.8 × 0.50
+
0.2 × 0.90

=
0.58
```

于是：

```text
Q = 0.50
↓
Q = 0.58
```

新的 Q-value 会影响下一次模型选择。

整个过程形成：

```text
Experience
    ↓
Evaluation
    ↓
Reward
    ↓
Policy Update
    ↓
New Decision
    ↓
New Experience
```

---

## 11. RSL 数据持久化

RSL 学习结果保存在原项目的 SQLite 数据库中。

新增两个表。

### rsl_policy

记录不同任务类型下各模型当前的学习状态：

```text
task_type
model
q_value
sample_count
```

例如：

```text
coding | qwen-small  | 0.68 | 10
coding | qwen-strong | 0.84 | 12
```

表示当前系统认为：

```text
对于 coding 任务：

qwen-strong 的历史表现更好
```

---

### rsl_experiences

记录每一次模型执行产生的经验：

```text
task_type
model
judge_score
latency_ms
reward
old_q
new_q
judge_reason
```

因此可以观察 RSL 策略如何随着请求不断更新。

数据库文件属于本地运行数据，因此：

```text
router.db
```

不会上传到 GitHub。

---

## 12. 环境要求

本项目测试环境：

```text
Windows 11
Python 3.12
Ollama
```

本地 GPU：

```text
NVIDIA GeForce RTX 5060
8 GB VRAM
```

本实验中的 Qwen3-4B-Instruct 可以完全运行在 GPU 上。

---

## 13. 安装 Ollama 模型

首先安装 Ollama。

然后下载两个模型：

```bash
ollama pull qwen3:1.7b
```

```bash
ollama pull qwen3:4b-instruct
```

查看本地模型：

```bash
ollama list
```

应该可以看到：

```text
qwen3:1.7b
qwen3:4b-instruct
```

Ollama 默认服务地址：

```text
http://127.0.0.1:11434
```

本项目使用其 OpenAI-compatible API：

```text
http://127.0.0.1:11434/v1
```

---

## 14. Python 环境

进入项目目录：

```powershell
cd ModelRouterCourse
```

创建 Python 虚拟环境：

```powershell
python -m venv .venv
```

Windows PowerShell 中如果禁止运行脚本，可以执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install fastapi uvicorn pydantic PyYAML python-dotenv aiosqlite httpx
```

---

## 15. Environment Variables

真正的 `.env` 不应该上传到 GitHub。

可以根据：

```text
.env.example
```

创建自己的：

```text
.env
```

示例：

```env
ROUTER_RUNTIME_MODE=standalone

ROUTER_ENABLE_STANDALONE_KEYS=true
ROUTER_ENABLE_ADMIN_API=true
ROUTER_ENABLE_USAGE_API=true

ADMIN_API_KEY=admin-local-key
ROUTER_API_KEY=router-local-key

ROUTER_DB=router.db

AUTO_SYNC_PROVIDER_MODELS=false

OPENAI_BASE_URL=http://127.0.0.1:11434/v1
OPENAI_KEY_1=ollama
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
```

---

## 16. Provider Configuration

`config/providers.yaml` 中定义两个模型。

核心配置类似：

```yaml
providers:
  openai:

    base_url: ${OPENAI_BASE_URL}

    keys:

      - key: ${OPENAI_KEY_1}

    models:

      - name: qwen3:1.7b
        routing_keys:
          - qwen-small

      - name: qwen3:4b-instruct
        routing_keys:
          - qwen-strong
```

ModelRouter 对外使用：

```text
qwen-small
qwen-strong
```

实际对应：

```text
qwen-small
→ Qwen3-1.7B
```

```text
qwen-strong
→ Qwen3-4B-Instruct
```

---

## 17. 启动 ModelRouter

确保 Ollama 正在运行。

然后在 Python 虚拟环境中运行：

```powershell
python -m app.main
```

成功后应该看到：

```text
Application startup complete.

Uvicorn running on http://127.0.0.1:8000
```

---

## 18. Health Check

测试：

```powershell
curl.exe http://127.0.0.1:8000/health
```

正常情况下：

```json
{
  "status": "healthy",
  "runtime_mode": "standalone",
  "providers": 1,
  "models": 2,
  "database": "connected"
}
```

---

## 19. 原 ModelRouter API

原接口保持不变：

```text
POST /v1/chat/completions
```

例如：

```text
model = qwen-small
```

实际调用：

```text
Qwen3-1.7B
```

而：

```text
model = qwen-strong
```

实际调用：

```text
Qwen3-4B-Instruct
```

---

## 20. RSL API

本项目新增：

```text
POST /v1/rsl/chat
```

Request 示例：

```json
{
  "prompt": "Please implement quick sort in Python."
}
```

PowerShell：

```powershell
$body = @{
    prompt = "Please implement quick sort in Python."
} | ConvertTo-Json
```

发送请求：

```powershell
$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/v1/rsl/chat" `
    -Method Post `
    -Headers @{
        Authorization="Bearer router-local-key"
    } `
    -ContentType "application/json" `
    -Body $body
```

查看完整结果：

```powershell
$response | ConvertTo-Json -Depth 10
```

---

## 21. RSL 返回结果

RSL API 会返回：

```text
task_type
selected_model
actual_model
selection_mode
answer
judge
latency_ms
reward
q_update
current_policy
```

示例：

```json
{
  "task_type": "coding",

  "selected_model": "qwen-strong",

  "actual_model": "qwen3:4b-instruct",

  "selection_mode": "exploitation",

  "judge": {
    "model": "qwen-strong",
    "score": 0.90,
    "reason": "The answer is correct and complete."
  },

  "latency_ms": 3200,

  "reward": 0.86,

  "q_update": {
    "old": 0.72,
    "new": 0.748
  }
}
```

其中最关键的是：

```text
old Q
  ↓
Reward
  ↓
new Q
```

这表示系统已经根据当前模型执行结果完成了一次学习。

---

## 22. 示例学习过程

初始：

```text
coding

qwen-small   Q = 0.50
qwen-strong  Q = 0.50
```

某次选择：

```text
qwen-small
```

Judge：

```text
Score = 0.65
```

Q-value 更新。

另外一次：

```text
qwen-strong
```

Judge：

```text
Score = 0.92
```

随着历史请求增加，可能得到：

```text
coding

qwen-small   Q = 0.68
qwen-strong  Q = 0.84
```

以后 Coding 请求在 exploitation 模式下就会更倾向：

```text
qwen-strong
```

这不是写死的：

```text
if coding:
    use strong model
```

而是由历史 Reward 逐渐学习得到。

---

## 23. 项目目录

主要相关结构：

```text
ModelRouterCourse
│
├── app
│   │
│   ├── api
│   │   ├── chat.py
│   │   ├── rsl.py
│   │   └── ...
│   │
│   ├── core
│   │   ├── router.py
│   │   ├── cascade.py
│   │   ├── db.py
│   │   ├── rsl.py
│   │   └── ...
│   │
│   └── main.py
│
├── config
│   └── providers.yaml
│
├── .env.example
├── .gitignore
└── README.md
```

---

## 24. 当前实现的特点

### 不需要 Fine-tuning

本项目不会修改 Qwen 模型参数。

学习发生在：

```text
Model Routing Policy
```

层。

因此实现简单、运行成本低。

---

### Online Learning

每一次请求都可以产生新经验：

```text
Request
→ Response
→ Judge
→ Reward
→ Update
```

---

### Persistent Learning

RSL Q-value 保存到 SQLite。

程序退出后重新运行：

```text
learned policy
```

仍然保留。

---

### Exploration

ε-greedy 可以避免系统过早固定在一个模型上。

系统会继续探索其它候选模型。

---

## 25. 当前限制

本项目主要用于课程实验，因此仍存在一些限制。

### Task Classification

目前使用关键词进行：

```text
coding
math
translation
general
```

分类。

未来可以使用：

- Embedding
- Semantic Routing
- LLM Classifier

改进。

### Judge Bias

目前使用：

```text
Qwen3-4B-Instruct
```

作为 Judge。

Judge 自身也可能产生错误评价。

未来可以采用：

- Multiple Judges
- Human Feedback
- External Benchmarks

### Model Pool 较小

目前只有：

```text
Qwen3-1.7B
Qwen3-4B-Instruct
```

未来可以继续加入其它本地模型或云端模型。

### Reward Function 较简单

当前 Reward 只考虑：

```text
Quality
Latency
```

未来还可以加入：

- API Cost
- Token Usage
- Success Rate
- Cascade Rate
- User Feedback

---

## 26. Future Work

未来可以继续尝试：

- Semantic Task Classification
- Multi-provider Routing
- Cost-aware Routing
- Adaptive Epsilon
- UCB
- Thompson Sampling
- Contextual Bandit
- Reinforcement Learning
- Multi-Judge Evaluation
- Automatic Benchmark
- Visualization Dashboard

---

## 27. Original Project

本项目基于课程提供的：

**ModelRouterCourse**

原始仓库：

https://github.com/NewYearEve-Godsheep/ModelRouterCourse

本仓库中的 RSL 功能是在原 ModelRouter 基础上的课程作业扩展。

---

## 28. Summary

本项目实现了一个简单的 Recursive Self-Learning Model Router。

核心过程：

```text
Task Classification
        ↓
Model Selection
        ↓
LLM Execution
        ↓
Judge Evaluation
        ↓
Reward
        ↓
Q-value Update
```

系统通过持续积累历史执行经验，让 Router 逐渐学习：

> 对于不同类型的任务，应该选择质量、速度和资源消耗更加合适的模型。