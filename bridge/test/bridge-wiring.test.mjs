// 接线检查（静态）：新模块写了但没接进主流程，是最容易被漏掉的一类问题。
// 这里不启动 bridge（缺 node_modules / 需要 DSH），只断言关键接线点还在。
// 这是对「结构」的回归保护，不替代真实运行测试。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { normalizeDebounceConfig } from '../src/social-debounce.js';
import { normalizeGateConfig } from '../src/social-gate.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const bridgeSrc = fs.readFileSync(path.join(root, 'src/bridge.js'), 'utf8');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config.json'), 'utf8'));

test('bridge.js 导入了两个人性化模块', () => {
  assert.match(bridgeSrc, /from '\.\/social-debounce\.js'/);
  assert.match(bridgeSrc, /from '\.\/social-gate\.js'/);
  assert.match(bridgeSrc, /new DebounceRegistry\(/);
  assert.match(bridgeSrc, /new DynamicReplyGate\(/);
});

test('reserved2 唤醒走防抖批次，而不是固定 batchMs 定时器', () => {
  assert.match(bridgeSrc, /function armPendingWakeTimerV2/);
  assert.match(bridgeSrc, /function noteIncomingBatchV2/);
  assert.match(bridgeSrc, /armPendingWakeTimerV2\(key, st, meta, reason\)/);
  // handleIncoming 里必须真的调用 noteIncomingBatchV2（否则续发不会延长批次）
  assert.match(bridgeSrc, /noteIncomingBatchV2\(key, st, batchMeta\)/);
});

test('一代 reserved 观望期有防抖缓冲，并且 enterActive 会取消未投递批次', () => {
  assert.match(bridgeSrc, /function enqueueSocialBatch/);
  assert.match(bridgeSrc, /function flushSocialBatch/);
  assert.match(bridgeSrc, /enqueueSocialBatch\(key, \{/);
  assert.match(bridgeSrc, /function enterActive\(key\) \{\s*\n\s*cancelSocialBatch\(key\)/);
});

test('概率触发都走闭环门控（旧 triggerProbability / 新 probability / proactive）', () => {
  assert.ok((bridgeSrc.match(/evaluateTriggerGate\(/g) || []).length >= 4,
    '至少应有：定义 1 处 + 旧观望 / reserved2 概率 / proactive 三处调用');
  assert.match(bridgeSrc, /socialGate\.noteReply\(/);
  // reserved2 逐条消息路径必须用虚拟轮次，避免「3 条消息 = 3 份预计 vs 1 次实际」误判超发
  assert.match(bridgeSrc, /evaluateTriggerGate\(key, Number\(tr\.probability\), \{ virtualRound: true \}\)/);
  assert.match(bridgeSrc, /const gateRound = new Map\(\)/);
});

test('发送失败与主动沉默在日志里可区分', () => {
  // 三个显式标签必须都存在
  assert.ok(bridgeSrc.includes("'[silent-by-choice]'"));
  assert.ok(bridgeSrc.includes("'[send-failed]'"));
  assert.ok(bridgeSrc.includes("'[send-missing]'"));
  // sendToQQ 必须返回真实结果，不能只 return sendChain（否则失败会被当成功）
  assert.match(bridgeSrc, /return sendChain\.then\(\(\) => \(\{ ok: failed\.length === 0/);
  // 只发贴纸/拍一拍也算“发出去了”，不能误报 send-missing
  assert.match(bridgeSrc, /SEND_OUTCOME_TOOL_RE[\s\S]{0,120}send_sticker|send_poke/);
  assert.match(bridgeSrc, /sendOutcomeSucceededSessions\.add\(/);
});

test('有限睡眠有硬上界，防止被注入成“永眠”', () => {
  assert.ok(config.socialV2?.wake?.maxFiniteSleepMs > 0, 'config 缺少 maxFiniteSleepMs');
  assert.match(bridgeSrc, /MAX_FINITE_SLEEP_MS/);
  assert.ok((bridgeSrc.match(/MAX_FINITE_SLEEP_MS/g) || []).length >= 4,
    '硬上界应同时用于 sleepMs、sleepUntil、默认唤醒配置');
});

test('skipProbability 与 triggerProbability 是两层判定', () => {
  // 观望触发成功后还要再掷一次 skipProbability（文档语义：可能“没看见”）
  assert.ok((bridgeSrc.match(/skipProbability/g) || []).length >= 3,
    'skipProbability 应同时出现在活跃期沉默与观望触发后两层');
});

test('config.json 有 debounce / gate / persona / silence，且能规范化', () => {
  assert.ok(config.debounce, '缺少 debounce 段');
  assert.ok(config.gate, '缺少 gate 段');
  assert.ok(config.socialV2?.persona, '缺少 socialV2.persona');
  assert.ok(config.socialV2?.silence, '缺少 socialV2.silence');
  const d = normalizeDebounceConfig(config.debounce);
  assert.equal(d.enabled, true);
  assert.equal(d.batchLimitMs, 8000);
  const g = normalizeGateConfig(config.gate);
  assert.equal(g.enabled, true);
  assert.equal(g.frequency, 1.0); // 默认 1.0：不改变现有回复频率，调低才更安静
});

test('人设文件包含漂移档位 × 回钩策略，且带安全带', () => {
  for (const preset of ['qq-chat', 'qq-chat-v2']) {
    const p = path.join(root, 'dsh/agent-presets', preset, 'agent.cordis.yml');
    const text = fs.readFileSync(p, 'utf8');
    assert.match(text, /漂移档位/, `${preset} 缺少漂移档位`);
    assert.match(text, /回钩策略/, `${preset} 缺少回钩策略`);
    assert.match(text, /不能无视明确提问/, `${preset} 缺少 wild 的安全带`);
    assert.match(text, /不能凭空换话题/, `${preset} 缺少 loose 的安全带`);
  }
});
