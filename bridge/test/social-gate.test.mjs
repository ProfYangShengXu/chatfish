// 发言频率闭环（AIMD）的纯逻辑测试：不依赖 node_modules，直接 `node --test test/` 就能跑。
import test from 'node:test';
import assert from 'node:assert/strict';

import { DynamicReplyGate, normalizeGateConfig, DEFAULT_GATE_CONFIG } from '../src/social-gate.js';

// 固定时钟 + 可切换随机数（默认 0 = 判定必过）。
function makeGate(overrides = {}, rngValue = 0) {
  let t = 1000000;
  const cfg = {
    windowMs: 3600000,
    frequency: 0.2,
    minSamples: 3,
    deadband: 0,
    confirmWindows: 2,
    decreaseFactor: 0.5,
    increaseStep: 0.1,
    minScale: 0.1,
    settleWindowMs: 3600000,
    ...overrides
  };
  const gate = new DynamicReplyGate(cfg, { rng: () => rngValue, now: () => t });
  return { gate, cfg, tick: () => { t += 1000; return t; }, now: () => t };
}

test('配置规范化：频率/系数 clamp，非法值回落默认', () => {
  const cfg = normalizeGateConfig({ frequency: 9, decreaseFactor: -1, minScale: 5, windowMs: 1 });
  assert.equal(cfg.frequency, 1);
  assert.equal(cfg.decreaseFactor, 0);
  assert.equal(cfg.minScale, 1);
  assert.equal(cfg.windowMs, 60000);
  assert.equal(normalizeGateConfig(null).frequency, DEFAULT_GATE_CONFIG.frequency);
});

test('样本不足时不调整 scale（冷启动防抖）', () => {
  const { gate, tick } = makeGate();
  let r;
  for (let i = 0; i < 3; i += 1) {
    r = gate.evaluate('group:1', 1, tick());
    gate.noteReply('group:1', tick());
  }
  assert.equal(r.scale, 1, 'samples < minSamples 时不应调整');
  assert.equal(r.adjusted, false);
});

test('持续超发：confirmWindows 个窗口后乘性减（scale 变小）', () => {
  const { gate, tick } = makeGate();
  let r;
  // 每次放行且都真的回了 → actual 追平 predicted，但 frequency=0.2 意味着 target 只有 0.2*predicted
  for (let i = 0; i < 8; i += 1) {
    r = gate.evaluate('group:1', 1, tick());
    gate.noteReply('group:1', tick());
  }
  assert.ok(r.scale < 1, `超发后 scale 应变小，实际=${r.scale}`);
  assert.equal(r.scale, 0.25); // 两轮 confirm ×0.5
  assert.ok(r.effectiveProbability < 1, '有效概率应被压低');
  assert.ok(r.adjusted || r.scale < 1);
});

test('predicted 用基准概率累加，与 scale 无关（否则永远收敛不了）', () => {
  const { gate, tick } = makeGate();
  for (let i = 0; i < 8; i += 1) {
    gate.evaluate('group:1', 1, tick());
    gate.noteReply('group:1', tick());
  }
  const before = gate.stats('group:1', tick());
  const n = before.samples;
  assert.equal(before.predicted, n, 'predicted 应等于判定条数 × 基准概率 1');
  assert.ok(before.actual >= 1);
  // 再判定一次，predicted 仍按 base=1 增长，不因 scale 变小而缩水
  const r = gate.evaluate('group:1', 1, tick());
  assert.equal(r.predicted, n);
});

test('欠发：加性增把 scale 慢慢放宽回 1', () => {
  const { gate, cfg, tick } = makeGate({ windowMs: 60000, settleWindowMs: 60000 });
  // 先制造超发把 scale 压下去
  for (let i = 0; i < 8; i += 1) {
    gate.evaluate('group:1', 1, tick());
    gate.noteReply('group:1', tick());
  }
  const shrunk = gate.stats('group:1', tick()).scale;
  assert.ok(shrunk < 1);
  // 跳过整个统计窗口，让旧的 actual 全部过期 → 之后只有 predicted、没有 actual
  let far = tick() + cfg.windowMs + 10000;
  for (let i = 0; i < 8; i += 1) gate.evaluate('group:1', 1, (far += 1000));
  const relaxed = gate.stats('group:1', far).scale;
  assert.ok(relaxed > shrunk, `欠发后 scale 应回升：${shrunk} -> ${relaxed}`);
  assert.ok(relaxed <= 1);
});

test('noteReply 只标记「已放行」的判定，且只标记一次', () => {
  // rng=0.99 且 base=0.5 → effective 0.5 → 不放行
  const { gate, tick } = makeGate({ frequency: 1 }, 0.99);
  gate.evaluate('group:1', 0.5, tick());
  assert.equal(gate.noteReply('group:1', tick()), false, '没放行的判定不应被记回复');
  // 改成放行
  const g2 = new DynamicReplyGate({ frequency: 1 }, { rng: () => 0, now: () => 1000 });
  g2.evaluate('group:1', 1, 1000);
  assert.equal(g2.noteReply('group:1', 1001), true);
  assert.equal(g2.noteReply('group:1', 1002), false, '同一条判定不能重复计数');
});

test('settleWindow 之外的判定不会被记成回复', () => {
  const { gate, tick } = makeGate({ settleWindowMs: 1000 });
  gate.evaluate('group:1', 1, tick());
  const later = tick() + 5000;
  assert.equal(gate.noteReply('group:1', later), false);
});

test('窗口外的判定被 prune，不无限增长', () => {
  const { gate, tick } = makeGate({ windowMs: 60000 });
  for (let i = 0; i < 100; i += 1) gate.evaluate('group:1', 0.1, tick());
  const s = gate.stats('group:1', tick());
  assert.ok(s.samples <= 100);
  // 时间跳过窗口后，旧判定全部淘汰
  const far = tick() + 120000;
  const s2 = gate.stats('group:1', far);
  assert.equal(s2.samples, 0);
  assert.equal(s2.predicted, 0);
});

test('disabled 时不做 AIMD，按基准概率放行', () => {
  const { gate, tick } = makeGate({ enabled: false });
  let r;
  for (let i = 0; i < 10; i += 1) {
    r = gate.evaluate('group:1', 0.5, tick());
    gate.noteReply('group:1', tick());
  }
  assert.equal(r.scale, 1);
  assert.equal(r.adjustReason, 'gate-disabled');
  assert.equal(r.effectiveProbability, 0.5);
});

test('空闲衰减：长时间无活动后 scale 向 1 靠拢', () => {
  const { gate, tick } = makeGate({ idleResetMs: 60000, windowMs: 60000 });
  for (let i = 0; i < 8; i += 1) {
    gate.evaluate('group:1', 1, tick());
    gate.noteReply('group:1', tick());
  }
  const shrunk = gate.stats('group:1', tick()).scale;
  assert.ok(shrunk < 1);
  // 空闲超过 idleResetMs（规范化下限 60s），旧 actual 也出了窗口
  const far = tick() + 120000;
  gate.evaluate('group:1', 1, far);
  assert.ok(gate.stats('group:1', far).scale > shrunk, '空闲后应衰减回中性');
});
