// 发言频率闭环（dynamic gate）—— 从 MaiBot `src/maisaka/turn_trigger/dynamic_gate.py` 的思路移植。
//
// MaiBot 的原设计：预计回复数来自 reply_likelihood 的拟合概率，实际回复数按 Planner 成功 reply 统计，
// 让窗口内实际次数贴近「不设门控时预计次数 × 回复频率」。
//
// 我们不拟合（任务书：不一定真做拟合，把反馈控制做出来即可），只保留闭环骨架：
//   predicted = 窗口内所有「概率判定」的基准概率之和（不设门控时预计会回几次）
//   target    = frequency × predicted（按目标频率折算后应该回几次）
//   actual    = 窗口内真正发出消息的次数（只统计由概率判定放行、且随后确实发了的那批）
// 然后对 scale ∈ [minScale, 1] 做 AIMD（加性增、乘性减）：
//   超发（actual > target + deadband）持续 confirmWindows 个窗口 → scale *= decreaseFactor（收紧）
//   欠发（actual < target - deadband）→ scale += increaseStep（慢慢放宽，最多 1）
// 长时间没有活动 → scale 向 1 衰减（RFC 5681 §4.1「重启空闲连接不能信任过期估计」同理）。
//
// 为什么不是 PID：每个回合只有一个带噪的 one-bit 信号（发了/没发），PID 的微分项会被噪声放大；
// AIMD 在有噪声、目标会漂的场合更稳，也更容易解释。
//
// 本模块不 import 任何东西，方便用 `node --test` 直接验证。

export const DEFAULT_GATE_CONFIG = Object.freeze({
  enabled: true,
  windowMs: 3600000, // 统计窗口：1 小时（与 MaiBot REPLY_WINDOW_SECONDS 一致）
  frequency: 1.0, // 目标回复频率：默认 1.0 = 不改变现有频率；调低才更安静（MaiBot talk_value 默认也是 1）
  minSamples: 20, // 窗口内样本不足时不做调整，避免冷启动抖动（对齐 MaiBot MIN_WINDOW_SCORE_COUNT）
  deadband: 0.5, // 超发/欠发的死区（次数），滤掉噪声
  confirmWindows: 2, // 连续几个窗口超发才收紧
  decreaseFactor: 0.7, // 乘性减
  increaseStep: 0.05, // 加性增
  minScale: 0.15, // 收紧下限，别把闸门关死
  settleWindowMs: 300000, // 一次放行后，多久内发出去才算「这次判定真的回了」
  idleResetMs: 7200000, // 空闲这么久后，scale 向 1 衰减
  roundMs: 40000, // 虚拟轮次：同一轮次内的多条消息只算一次「预计回复」（MaiBot VIRTUAL_ROUND_SECONDS）
  maxDecisions: 2000 // 窗口内最多保留的判定条数（硬上界，防内存增长）
});

function num(value, fallback, { min = -Infinity, max = Infinity } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

export function normalizeGateConfig(raw) {
  const src = raw && typeof raw === 'object' ? raw : {};
  const minScale = num(src.minScale, DEFAULT_GATE_CONFIG.minScale, { min: 0, max: 1 });
  const decreaseFactor = num(src.decreaseFactor, DEFAULT_GATE_CONFIG.decreaseFactor, { min: 0, max: 1 });
  return {
    enabled: src.enabled !== false,
    windowMs: num(src.windowMs, DEFAULT_GATE_CONFIG.windowMs, { min: 60000, max: 86400000 }),
    frequency: num(src.frequency, DEFAULT_GATE_CONFIG.frequency, { min: 0, max: 1 }),
    minSamples: Math.round(num(src.minSamples, DEFAULT_GATE_CONFIG.minSamples, { min: 1, max: 1000 })),
    deadband: num(src.deadband, DEFAULT_GATE_CONFIG.deadband, { min: 0, max: 1000 }),
    confirmWindows: Math.round(num(src.confirmWindows, DEFAULT_GATE_CONFIG.confirmWindows, { min: 1, max: 100 })),
    decreaseFactor,
    increaseStep: num(src.increaseStep, DEFAULT_GATE_CONFIG.increaseStep, { min: 0, max: 1 }),
    minScale: Math.min(minScale, 1),
    settleWindowMs: num(src.settleWindowMs, DEFAULT_GATE_CONFIG.settleWindowMs, { min: 1000, max: 3600000 }),
    idleResetMs: num(src.idleResetMs, DEFAULT_GATE_CONFIG.idleResetMs, { min: 60000, max: 86400000 }),
    roundMs: num(src.roundMs, DEFAULT_GATE_CONFIG.roundMs, { min: 1000, max: 3600000 }),
    maxDecisions: Math.round(num(src.maxDecisions, DEFAULT_GATE_CONFIG.maxDecisions, { min: 10, max: 100000 }))
  };
}

// 一个聊天会话的闭环控制器。多会话各自独立（每个群/私聊一套 scale）。
export class DynamicReplyGate {
  constructor(config, { rng = Math.random, now = () => Date.now() } = {}) {
    this.cfg = normalizeGateConfig(config);
    this.rng = rng;
    this.now = now;
    this.streams = new Map(); // key -> stream state
    this.adjustCount = 0;
  }

  _stream(key) {
    let s = this.streams.get(key);
    if (!s) {
      s = {
        scale: 1,
        decisions: [], // { ts, p, passed, replied }
        lastAdjustAt: 0,
        lastReplyAt: 0,
        lastActivityAt: this.now(),
        overshootStreak: 0,
        underStreak: 0,
        lastDecision: null
      };
      this.streams.set(key, s);
    }
    return s;
  }

  _prune(s, now) {
    const cutoff = now - this.cfg.windowMs;
    let firstKept = 0;
    while (firstKept < s.decisions.length && s.decisions[firstKept].ts < cutoff) firstKept += 1;
    if (firstKept > 0) s.decisions.splice(0, firstKept);
    if (s.decisions.length > this.cfg.maxDecisions) {
      s.decisions.splice(0, s.decisions.length - this.cfg.maxDecisions);
    }
  }

  // 窗口统计：predicted 用「基准概率」累加，与 scale 无关——否则收紧后目标也跟着降，永远收敛不了。
  stats(key, now = this.now()) {
    const s = this._stream(key);
    this._prune(s, now);
    let predicted = 0;
    let actual = 0;
    for (const d of s.decisions) {
      predicted += d.p;
      if (d.replied) actual += 1;
    }
    const target = this.cfg.frequency * predicted;
    return {
      key,
      scale: s.scale,
      predicted,
      target,
      actual,
      samples: s.decisions.length,
      overshootStreak: s.overshootStreak,
      underStreak: s.underStreak,
      lastAdjustAt: s.lastAdjustAt,
      lastReplyAt: s.lastReplyAt
    };
  }

  // 对一个「概率判定」求值：先做 AIMD 调整，再按 base × scale 掷骰子。
  // 无论放行与否都会记一条 decision（因为「不设门控时预计会回」与是否放行无关）。
  evaluate(key, baseProbability, now = this.now()) {
    const cfg = this.cfg;
    const s = this._stream(key);
    const base = Math.min(1, Math.max(0, Number(baseProbability) || 0));
    this._prune(s, now);
    const before = this.stats(key, now);
    let adjusted = false;
    let adjustReason = '';

    if (!cfg.enabled) {
      const allowed = this.rng() < base;
      s.lastActivityAt = now;
      s.decisions.push({ ts: now, p: base, passed: allowed, replied: false });
      this._prune(s, now);
      return {
        ...before,
        allowed,
        baseProbability: base,
        effectiveProbability: base,
        scale: 1,
        adjusted: false,
        adjustReason: 'gate-disabled'
      };
    }

    // 空闲衰减：太久没活动，向中性 1 靠拢（过期的估计不可信）。
    if (now - s.lastActivityAt > cfg.idleResetMs && s.scale < 1) {
      s.scale = Math.min(1, s.scale + (1 - s.scale) * 0.5);
      adjusted = true;
      adjustReason = 'idle-decay';
    }

    // AIMD 只在样本足够时调整。
    if (before.samples >= cfg.minSamples) {
      const error = before.actual - before.target; // >0 超发，<0 欠发
      if (error > cfg.deadband) {
        s.underStreak = 0;
        s.overshootStreak += 1;
        if (s.overshootStreak >= cfg.confirmWindows) {
          const next = Math.max(cfg.minScale, s.scale * cfg.decreaseFactor);
          if (next < s.scale - 1e-9) {
            s.scale = next;
            adjusted = true;
            adjustReason = `overshoot actual=${before.actual.toFixed(1)} target=${before.target.toFixed(1)} -> scale=${next.toFixed(2)}`;
          }
          s.overshootStreak = 0;
        }
      } else if (error < -cfg.deadband) {
        s.overshootStreak = 0;
        s.underStreak += 1;
        const next = Math.min(1, s.scale + cfg.increaseStep);
        if (next > s.scale + 1e-9) {
          s.scale = next;
          adjusted = true;
          adjustReason = `underflow actual=${before.actual.toFixed(1)} target=${before.target.toFixed(1)} -> scale=${next.toFixed(2)}`;
        }
      } else {
        s.overshootStreak = 0;
        s.underStreak = 0;
      }
    }

    const effective = Math.min(1, Math.max(0, base * s.scale));
    const allowed = this.rng() < effective;
    s.lastActivityAt = now;
    s.lastDecision = { ts: now, base, effective, allowed };
    s.decisions.push({ ts: now, p: base, passed: allowed, replied: false });
    this._prune(s, now);
    if (adjusted) {
      s.lastAdjustAt = now;
      this.adjustCount += 1;
    }
    return {
      ...before,
      allowed,
      baseProbability: base,
      effectiveProbability: effective,
      scale: s.scale,
      adjusted,
      adjustReason
    };
  }

  // 真的发出了一条消息：把窗口内最早一条「已放行但还没记回复」的判定标记为已回。
  // 只认 settleWindowMs 内的判定；被 @ / 私聊等强制回复没有判定记录，不会误记。
  noteReply(key, now = this.now()) {
    const s = this._stream(key);
    this._prune(s, now);
    for (const d of s.decisions) {
      if (d.passed && !d.replied && now - d.ts <= this.cfg.settleWindowMs) {
        d.replied = true;
        s.lastReplyAt = now;
        s.lastActivityAt = now;
        return true;
      }
    }
    return false;
  }

  // 供控制台/日志读取：单会话或全部会话。
  snapshot(key = null) {
    const now = this.now();
    if (key) return this.stats(key, now);
    const out = { config: this.cfg, streams: {} };
    for (const k of this.streams.keys()) out.streams[k] = this.stats(k, now);
    return out;
  }

  reset(key = null) {
    if (key) this.streams.delete(key);
    else this.streams.clear();
  }
}
