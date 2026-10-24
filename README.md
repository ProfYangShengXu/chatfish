# chatfish

**一套完整的 QQ 聊天机器人框架** —— 把一个大模型变成一个在群里待得住、记得住人、说得出人话的群友。

> 本项目脱胎于 **[QQbridge](https://github.com/Derpyu520/qq-bridge)**，在此郑重致敬。
> QQbridge 解决了「消息怎么进来、怎么出去」这一半；chatfish 在它的基础上补完了另一半：
> **社交仿真、分层记忆、人设工程、跨会话上下文** —— 让它不只是个能收发消息的壳，而是一套能长期运转的框架。

**English**: [README.en.md](README.en.md)

---

## 它解决什么问题

把一个 LLM 接进群聊，五分钟就能跑通。但真扔进一个活跃的群，一周之内一定会遇到这四件事：

| 问题 | 一般做法 | chatfish 的做法 |
|---|---|---|
| **太快** —— 每条都回，像客服 | 加个概率 | 防抖攒批 + 频率闭环 + 沉默权 |
| **记不住** —— 三天前的梗全忘 | 塞进 system prompt | 三层记忆（常驻 / 情境 / 长期向量） |
| **不像人** —— 句号、总结、面面俱到 | 写「不要像 AI」 | 黑名单 + 反例示例 + 允许犯错条款 |
| **割裂** —— 群里聊的和私聊聊的对不上 | 无 | 定时跨会话同步 + 带来源标签注入 |

这四件都不靠提示词硬凑，靠的是**机制**。

---

## 组成

```
chatfish/
├── bridge/      QQ ↔ Agent 桥接（Node.js）
│                 · 社交状态机：观望 / 活跃 / 潜水
│                 · 防抖攒批、频率闭环、沉默权、注意力漂移
│                 · MCP 工具层：发送、记忆、表情、联网
│                 · 人设装配：system 恒定前缀 + 唤醒增量
├── memory/      长期记忆服务（Python + PostgreSQL + pgvector）
│                 · 向量检索 + 关键词混合召回
│                 · 会话结束 → LLM 提炼事实 → 入库
│                 · 常驻区自动维护（写回 AGENTS.md）
├── roles/       角色卡系统
│                 · 人格 / 行为 / 说话方式分三段
│                 · 工具手册与人格分离，按需读取
└── docs/        架构与设计文档
```

### bridge/

桥接层。上游接 [SnowLuma](https://github.com/SnowLuma-Project)（OneBot v11），下游接任意支持 ACP / HTTP 的 agent runtime（默认 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)）。

核心不是协议转换，是**「什么时候该说话」的判断**：

- **防抖攒批**：同一波消息合并成一回合读，而不是逐条追
- **频率闭环**：按一小时滑动窗口的实际发言数做负反馈，避免概率掷骰的抖动
- **沉默权**：不发言是一等公民，不是异常
- **注意力漂移**：拆成「能跑多远」×「多久拉回」两个正交旋钮，比写「不要太死板」有效

### memory/

长期记忆。三层：

| 层 | 载体 | 特点 |
|---|---|---|
| 常驻 | `AGENTS.md` | 每次会话开头注入，内容稳定 |
| 情境 | 会话状态 JSON | 进行中的话题、想说的话、对某人的印象 |
| 长期 | PostgreSQL + pgvector | 语义召回，按需查 |

写入走「会话结束 → 提炼 → 去重 → 入库」，不是每条消息都记。

### roles/

角色卡和工具手册分开：

- **人格**进系统提示词（跨轮不变，缓存友好）
- **工具用法**留在角色卡，由 agent 按需读取

这样人设写得再长，也不会把每一轮的工具 schema 挤爆。

---

## 架构

```
QQ ←→ SnowLuma (OneBot v11)
          ↕
      bridge/  ──── 社交状态机、防抖、工具层
          ↕  ──── MCP / ACP
      agent runtime（默认 DSH）
          ↕
      memory/  ──── PostgreSQL + pgvector
```

细节见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 快速开始

> 目前仍在快速演进，接口会变。建议先通读 `docs/`，再动手接。

```bash
git clone https://github.com/ProfYangShengXu/chatfish
cd chatfish

# 桥接层
cd bridge && npm install && cp config.example.json config.json
# 填入 SnowLuma 的监听地址、agent runtime 的 endpoint 与 token

# 记忆服务
cd ../memory && uv sync && cp .env.example .env
# 填入数据库连接与模型 key
```

具体配置项见各目录下的 README。

---

## 设计原则

1. **机制优先于提示词** —— 能用状态机解决的，不写进 prompt
2. **可变量的放最后** —— 系统提示词零改写，动态内容只追加在尾部（前缀缓存友好）
3. **判据必须能响** —— 每条新规则都要有负向测试，不做「看起来对」的修改
4. **允许犯错** —— 允许答错、答一半、说不知道。完美的机器比不完美的人更像 AI
5. **人在环** —— 人管抽象决策，agent 干活

---

## 致敬

- **[QQbridge](https://github.com/Derpyu520/qq-bridge)** —— chatfish 的直接源头。消息桥接、MCP 工具层、表情包链路都来自它
- **[MaiBot](https://github.com/Mai-with-u/MaiBot)** —— 人性化机制的参考：防抖控制器、沉默权、表达学习、注意力漂移
- **[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)** —— 默认的 agent runtime
- **[SnowLuma](https://github.com/SnowLuma-Project)** —— OneBot v11 实现

---

## 许可证

**AGPL-3.0** —— 见 [LICENSE](LICENSE)。

简单说：你可以自由使用、修改、部署，但**如果你把它作为服务提供给他人使用，你必须公开你的修改**。这是为了防止有人拿这套框架做闭源商用。

如果你想闭源商用，请联系作者谈商业授权。
