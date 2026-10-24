// 防抖攒批的纯逻辑测试：不依赖 node_modules，直接 `node --test test/` 就能跑。
// 对应 MaiBot src/chat/debounce/controller.py 的每条规则。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MessageDebouncer,
  DebounceRegistry,
  classifyMessageShape,
  normalizeDebounceConfig
} from '../src/social-debounce.js';

// 固定时钟：方便断言精确毫秒数。
function fixedClock(start = 100000) {
  let t = start;
  return {
    now: () => t,
    set: (v) => { t = v; },
    add: (ms) => { t += ms; }
  };
}

function make(overrides = {}) {
  const clock = fixedClock();
  const deb = new MessageDebouncer({ ...overrides }, { now: clock.now });
  return { deb, clock };
}

test('安静期开批：等 quietMs（natural=2s）', () => {
  const { deb, clock } = make();
  const r = deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.newBatch, true);
  assert.equal(r.deadlineAt, 100000 + 2000);
  assert.match(r.reason, /安静开批/);
});

test('活跃期开批：距上条在 30s 内，等 activeMs（5s）', () => {
  const { deb, clock } = make();
  deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  deb.consume(); // 上一批消费掉，但保留 lastArrival
  clock.add(10000); // 距上条 10s < 30s
  const r = deb.observe({ author: 'b', receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.newBatch, true);
  assert.equal(r.deadlineAt, 110000 + 5000);
  assert.match(r.reason, /活跃开批/);
});

test('同人续发等 2s，换人等 0.5s', () => {
  const same = make();
  same.deb.observe({ author: 'a', receivedAt: same.clock.now(), now: same.clock.now() });
  same.clock.add(500);
  const r1 = same.deb.observe({ author: 'a', receivedAt: same.clock.now(), now: same.clock.now() });
  assert.equal(r1.deadlineAt, 100500 + 2000);
  assert.match(r1.reason, /同人续发/);

  const diff = make();
  diff.deb.observe({ author: 'a', receivedAt: diff.clock.now(), now: diff.clock.now() });
  diff.clock.add(500);
  const r2 = diff.deb.observe({ author: 'b', receivedAt: diff.clock.now(), now: diff.clock.now() });
  assert.equal(r2.deadlineAt, 100500 + 500);
  assert.match(r2.reason, /换人/);
});

test('@ 你：1s 上限，哪怕同人续发等 2s 也被压到 1s', () => {
  const { deb, clock } = make();
  const t0 = clock.now();
  deb.observe({ author: 'a', receivedAt: t0, now: t0 });
  clock.add(300);
  const r = deb.observe({ author: 'a', atBot: true, receivedAt: clock.now(), now: clock.now() });
  // 到达 100300 + mention 1000 = 101300；同人续发 102300 → 取 @ 上限
  assert.equal(r.deadlineAt, 101300);
  assert.match(r.reason, /@上限/);
});

test('只发图：开批至少等 4s', () => {
  const { deb, clock } = make();
  const r = deb.observe({ author: 'a', isBareImage: true, receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.deadlineAt, 100000 + 4000);
  assert.match(r.reason, /裸图/);
});

test('纯表情包不延长等待（截止只会变早或不变）', () => {
  const { deb, clock } = make();
  deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  const before = deb.deadline;
  clock.add(500);
  const r = deb.observe({ author: 'a', isPureEmoji: true, receivedAt: clock.now(), now: clock.now() });
  assert.ok(r.deadlineAt <= before, `表情包不应延长：before=${before} after=${r.deadlineAt}`);
  assert.match(r.reason, /纯表情包不延长等待/);
});

test('整批 8s 上限：连续同人续发也超不过 first+8s', () => {
  const { deb, clock } = make();
  const first = clock.now();
  for (let i = 0; i < 10; i += 1) {
    deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
    clock.add(900);
  }
  assert.ok(deb.deadline <= first + 8000, `deadline=${deb.deadline} 应 <= ${first + 8000}`);
  assert.equal(deb.deadline, first + 8000);
});

test('★ 已到期的批次不允许被新消息延长（死锁防护）', () => {
  const { deb, clock } = make();
  deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  const expiredAt = deb.deadline;
  clock.set(expiredAt + 1); // 已过期但还没 consume（模型/事件循环忙）
  const r = deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.expired, true);
  assert.equal(r.deadlineAt, expiredAt, '过期后的新消息不能把截止时间往后推');
  assert.match(r.reason, /已到期/);
});

test('consume 保留上一条到达时间/发言人，用于下一批安静/活跃判定', () => {
  const { deb, clock } = make();
  deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  deb.consume();
  assert.equal(deb.pending, false);
  clock.add(5000);
  const r = deb.observe({ author: 'b', receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.newBatch, true);
  assert.match(r.reason, /活跃开批/); // 5s < 30s 安静阈值
});

test('消息形状：纯图 / 纯表情 / 图文混合', () => {
  assert.deepEqual(classifyMessageShape('[图片]', [{ kind: 'image' }]), { isPureEmoji: false, isBareImage: true });
  assert.deepEqual(classifyMessageShape('[表情1]', [{ kind: 'face', faceId: '1' }]), { isPureEmoji: true, isBareImage: false });
  assert.deepEqual(classifyMessageShape('[图片]看这个', [{ kind: 'image' }]), { isPureEmoji: false, isBareImage: false });
  assert.deepEqual(classifyMessageShape('hello', []), { isPureEmoji: false, isBareImage: false });
});

test('配置规范化：非法档位回落 natural，时长被 clamp', () => {
  const cfg = normalizeDebounceConfig({ profile: 'nope', batchLimitMs: -5, mentionLimitMs: 'abc' });
  assert.equal(cfg.profile, 'natural');
  assert.equal(cfg.batchLimitMs, 1000); // min 1000
  assert.equal(cfg.mentionLimitMs, 1000); // fallback
});

test('DebounceRegistry：过期空闲批次会被回收，活跃批次不回收', () => {
  const clock = fixedClock();
  const reg = new DebounceRegistry({ idleTtlMs: 10000, maxBatches: 2 }, { now: clock.now });
  const d1 = reg.get('group:1');
  d1.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  d1.consume(); // 不再 pending
  clock.add(20000);
  reg.sweep();
  assert.equal(reg.peek('group:1'), null, '空闲超过 TTL 应被回收');

  const d2 = reg.get('group:2');
  d2.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() }); // 仍 pending
  clock.add(20000);
  reg.sweep();
  assert.ok(reg.peek('group:2'), '仍有截止时间的批次不回收');
});

test('enabled=false 是真正的总开关：立即到期，不攒批', () => {
  const { deb, clock } = make({ enabled: false });
  const r = deb.observe({ author: 'a', receivedAt: clock.now(), now: clock.now() });
  assert.equal(r.waitMs, 0);
  assert.equal(r.deadlineAt, clock.now());
  assert.match(r.reason, /防抖已关闭/);
});

test('时钟混用防护：误把 Date.now() 当 receivedAt 不会让截止时间溢出', () => {
  const { deb, clock } = make();
  const r = deb.observe({ author: 'a', receivedAt: Date.now(), now: clock.now() });
  // 收到的是 performance 时钟域，不应出现 1.7e12 量级的 deadline
  assert.ok(r.deadlineAt < 1e9, `deadline=${r.deadlineAt} 不应是 Date.now 量级`);
  assert.equal(r.deadlineAt, 100000 + 2000);
});
