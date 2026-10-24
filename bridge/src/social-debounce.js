// 消息防抖攒批（debounce）—— 从 MaiBot `src/chat/debounce/controller.py` 移植。
//
// 要解决的问题：用户连发三条补充说明，桥接现在会触发三次独立回合、回三次半截话。
// 目标：等用户说完，把同一批消息当成一次输入，只回一次。
//
// 与 MaiBot 原版的差异（这是 Node 单进程事件循环，不是 asyncio）：
// - 不在这里起后台任务，只维护纯状态；定时器由 bridge.js 持有。
// - 时钟用 performance.now()（单调，不受系统改时间影响），不用 Date.now()。
// - 批次状态拆成两个字段：
//     * idleDeadlineAt：可重置的「安静期截止」= 最后一次到达 + 本次等待时长；
//     * hardDeadlineAt：不可变的「整批上限」= 首批到达 + batchLimitMs（@ 时还会更早）。
//   有效截止 = min(idle, hard)。这样「已到期批次被新消息延长」在结构上不可能发生，
//   而不是靠一条 if 政策去防——MaiBot 注释里那条「防模型越慢批次越长」的死锁防护就是它。
//
// 规则（与 MaiBot 对齐）：
//   安静期(距上条 > quietThresholdMs)开批等 quietMs / 活跃期开批等 activeMs；
//   同人续发等 sameAuthorMs / 换人等 differentAuthorMs / @ 你等 mentionLimitMs 上限；
//   只发图至少等 bareImageMinWaitMs / 纯表情包不延长等待；
//   整批上限 batchLimitMs。
//
// 本模块不 import 任何东西，方便用 `node --test` 直接验证。

// 三档等待配置（毫秒）。MaiBot 的 fast/natural/steady 原值 ×1000。
export const DEBOUNCE_PROFILES = Object.freeze({
  fast: Object.freeze({ quietMs: 1000, activeMs: 3000, sameAuthorMs: 1000, differentAuthorMs: 500 }),
  natural: Object.freeze({ quietMs: 2000, activeMs: 5000, sameAuthorMs: 2000, differentAuthorMs: 500 }),
  steady: Object.freeze({ quietMs: 4000, activeMs: 7000, sameAuthorMs: 4000, differentAuthorMs: 2000 })
});

export const DEBOUNCE_PROFILE_NAMES = Object.freeze(Object.keys(DEBOUNCE_PROFILES));

// 默认值尽量贴近 MaiBot natural 档 + 它写死的全局常量。
export const DEFAULT_DEBOUNCE_CONFIG = Object.freeze({
  enabled: true,
  profile: 'natural',
  quietThresholdMs: 30000, // 距上一条超过这个时长 = 安静期；否则算活跃期
  bareImageMinWaitMs: 4000, // 只发图时至少等这么久（对方可能在补字）
  batchLimitMs: 8000, // 一整批最多攒这么久
  mentionLimitMs: 1000, // @ 你时 1 秒内必须响应
  idleTtlMs: 300000, // 没有任何活动的批次状态回收时限（防 Map 泄漏）
  maxBatches: 64 // 同时保留的会话批次数上限（防 Map 泄漏；正常群数远小于它）
});

function num(value, fallback, { min = -Infinity, max = Infinity, round = false } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  let out = Math.min(max, Math.max(min, n));
  if (round) out = Math.round(out);
  return out;
}

// 把 config.json 里的原始值规范化成可直接使用的配置：
// 非法档位名回落 natural；所有时长 clamp 到合理区间；enabled 只认 !== false。
export function normalizeDebounceConfig(raw) {
  const src = raw && typeof raw === 'object' ? raw : {};
  const profiles = {};
  for (const name of DEBOUNCE_PROFILE_NAMES) {
    const base = DEBOUNCE_PROFILES[name];
    const override = src.profiles?.[name] && typeof src.profiles[name] === 'object' ? src.profiles[name] : {};
    profiles[name] = {
      quietMs: num(override.quietMs, base.quietMs, { min: 0, max: 600000 }),
      activeMs: num(override.activeMs, base.activeMs, { min: 0, max: 600000 }),
      sameAuthorMs: num(override.sameAuthorMs, base.sameAuthorMs, { min: 0, max: 600000 }),
      differentAuthorMs: num(override.differentAuthorMs, base.differentAuthorMs, { min: 0, max: 600000 })
    };
  }
  const profile = DEBOUNCE_PROFILE_NAMES.includes(src.profile) ? src.profile : DEFAULT_DEBOUNCE_CONFIG.profile;
  return {
    enabled: src.enabled !== false,
    profile,
    profiles,
    quietThresholdMs: num(src.quietThresholdMs, DEFAULT_DEBOUNCE_CONFIG.quietThresholdMs, { min: 0, max: 3600000 }),
    bareImageMinWaitMs: num(src.bareImageMinWaitMs, DEFAULT_DEBOUNCE_CONFIG.bareImageMinWaitMs, { min: 0, max: 60000 }),
    batchLimitMs: num(src.batchLimitMs, DEFAULT_DEBOUNCE_CONFIG.batchLimitMs, { min: 1000, max: 600000 }),
    mentionLimitMs: num(src.mentionLimitMs, DEFAULT_DEBOUNCE_CONFIG.mentionLimitMs, { min: 0, max: 60000 }),
    idleTtlMs: num(src.idleTtlMs, DEFAULT_DEBOUNCE_CONFIG.idleTtlMs, { min: 10000, max: 86400000 }),
    maxBatches: num(src.maxBatches, DEFAULT_DEBOUNCE_CONFIG.maxBatches, { min: 1, max: 4096, round: true })
  };
}

// 从 OneBot 消息段提取出来的 media[] + 纯文本判断消息形状。
// - isBareImage：只有图片、没有文字（对方很可能马上补文字）；
// - isPureEmoji：只有 face/表情包、没有文字（表情包不是新话题，不延长等待）。
const MEDIA_PLACEHOLDER_RE = /\[(?:图片|表情\d*|语音|视频|文件[^\]]*|转发消息[^\]]*|卡片消息|引用消息)\]/g;

export function classifyMessageShape(plainContent, media) {
  const textual = String(plainContent ?? '').replace(MEDIA_PLACEHOLDER_RE, '').trim();
  const list = Array.isArray(media) ? media : [];
  const hasImage = list.some((m) => m && m.kind === 'image');
  const hasFace = list.some((m) => m && m.kind === 'face');
  const isPureEmoji = list.length > 0 && list.every((m) => m && m.kind === 'face') && !textual;
  const isBareImage = hasImage && !hasFace && !textual;
  return { isPureEmoji, isBareImage };
}

// 单个会话的防抖状态机。
// 纯内存、无 IO；bridge.js 负责在 observe 之后按 deadline 排定时器。
export class MessageDebouncer {
  constructor(config, { now = () => performance.now() } = {}) {
    this.cfg = normalizeDebounceConfig(config);
    this.now = now;
    this.firstArrivalAt = null; // 本批第一条的到达时刻（performance 时钟）
    this.lastArrivalAt = null; // 上一条的到达时刻（跨批保留，用于判断安静/活跃）
    this.lastAuthor = null;
    this.idleDeadlineAt = null;
    this.hardDeadlineAt = null; // 不可变：首批 + batchLimitMs
    this.urgentAt = null; // @ 触发的硬上限
    this.deadlineAt = null; // 当前有效截止 = min(idle, hard)
    this.lastReason = '';
    this.batchCount = 0;
    this.generation = 0; // 每次开新批 +1；定时器回调凭它判断自己是不是旧批
  }

  get deadline() {
    return this.deadlineAt;
  }

  // 记录一条新消息，返回本批的截止时间与规则说明。
  // receivedAt：这条消息真正到达桥接的时刻（performance 时钟），缺省取现在。
  observe({ author = '', atBot = false, isBareImage = false, isPureEmoji = false, receivedAt = null, now = null } = {}) {
    const cfg = this.cfg;
    const t = Number.isFinite(now) ? now : this.now();
    // 时钟归一化：调用方可能误传 Date.now()（~1.7e12）而不是 performance.now()。
    // 混用会让 deadline - performance.now() 溢出成巨大值，Node 会把它改成 1ms，
    // 防抖静默失效。这里把明显不是同一时钟的到达时刻按「现在」处理。
    const rawArrival = Number.isFinite(receivedAt) ? receivedAt : t;
    const arrival = rawArrival > t + 1_000_000 ? t : rawArrival;
    // 到达时刻不能倒流；预处理（解析 @ 昵称/引用）花的也算进等待预算。
    const arrivalAt = this.lastArrivalAt != null ? Math.max(arrival, this.lastArrivalAt) : arrival;
    const previousArrival = this.lastArrivalAt;
    const previousAuthor = this.lastAuthor;
    this.lastArrivalAt = arrivalAt;
    this.lastAuthor = author;

    // 总开关：关掉时每条消息都立即到期，等价于「不攒批、直接投递」。
    if (!cfg.enabled) {
      const newBatch = this.firstArrivalAt == null;
      if (newBatch) {
        this.firstArrivalAt = arrivalAt;
        this.batchCount += 1;
        this.generation += 1;
      }
      this.deadlineAt = t;
      this.hardDeadlineAt = t;
      this.lastReason = '防抖已关闭，立即投递';
      return { deadlineAt: t, waitMs: 0, reason: this.lastReason, newBatch, expired: false, generation: this.generation };
    }

    const newBatch = this.firstArrivalAt == null;
    if (newBatch) {
      this.firstArrivalAt = arrivalAt;
      this.hardDeadlineAt = arrivalAt + cfg.batchLimitMs;
      this.urgentAt = null;
      this.batchCount += 1;
      this.generation += 1;
    }
    // @ 你：整批上限被拉到到达时刻 + mentionLimitMs（取最早的那次）。
    if (atBot) {
      const urgent = arrivalAt + cfg.mentionLimitMs;
      this.urgentAt = this.urgentAt == null ? urgent : Math.min(this.urgentAt, urgent);
      this.hardDeadlineAt = Math.min(this.hardDeadlineAt, this.urgentAt);
    }

    // 已经到期的批次不许被新消息重新延长：直接返回旧截止，等它被消费。
    if (this.deadlineAt != null && this.deadlineAt <= t) {
      this.lastReason = '批次已到期，不再延长等待';
      return {
        deadlineAt: this.deadlineAt,
        waitMs: 0,
        reason: this.lastReason,
        newBatch,
        expired: true,
        generation: this.generation
      };
    }

    const profile = cfg.profiles[cfg.profile] ?? cfg.profiles.natural;
    const reasons = [];
    let waitMs;
    if (newBatch) {
      const active = previousArrival != null && (arrivalAt - previousArrival) <= cfg.quietThresholdMs;
      waitMs = active ? profile.activeMs : profile.quietMs;
      reasons.push(`${active ? '活跃' : '安静'}开批 ${waitMs}ms`);
      if (isBareImage) {
        // 只发图：对方可能在跟着图补字，至少多等一会儿。
        if (cfg.bareImageMinWaitMs > waitMs) {
          waitMs = cfg.bareImageMinWaitMs;
          reasons.push(`裸图至少 ${cfg.bareImageMinWaitMs}ms`);
        }
      }
    } else {
      waitMs = author === previousAuthor ? profile.sameAuthorMs : profile.differentAuthorMs;
      reasons.push(`${author === previousAuthor ? '同人续发' : '换人'} ${waitMs}ms`);
    }

    let idleDeadline = arrivalAt + waitMs;
    // 纯表情包不延长等待，只能把截止提前（第一批表情包仍按开批规则）。
    if (isPureEmoji && !newBatch && this.deadlineAt != null) {
      reasons.push('纯表情包不延长等待');
      idleDeadline = Math.min(idleDeadline, this.deadlineAt);
    }

    const effective = Math.min(idleDeadline, this.hardDeadlineAt);
    if (idleDeadline > this.hardDeadlineAt) reasons.push(`受整批上限 ${cfg.batchLimitMs}ms 限制`);
    if (this.urgentAt != null && idleDeadline > this.urgentAt) reasons.push(`受@上限 ${cfg.mentionLimitMs}ms 限制`);

    this.idleDeadlineAt = idleDeadline;
    this.deadlineAt = effective;
    this.lastReason = reasons.join('；');
    return {
      deadlineAt: effective,
      waitMs: Math.max(0, effective - arrivalAt),
      reason: this.lastReason,
      newBatch,
      expired: false,
      generation: this.generation
    };
  }

  // 本批已被取走投递：只保留「上一条到达时间/发言人」，供下一批判断安静还是活跃。
  consume() {
    this.firstArrivalAt = null;
    this.idleDeadlineAt = null;
    this.hardDeadlineAt = null;
    this.urgentAt = null;
    this.deadlineAt = null;
    this.generation += 1;
  }

  // 清空全部状态（模式切换/会话重置时用）。
  reset() {
    this.firstArrivalAt = null;
    this.lastArrivalAt = null;
    this.lastAuthor = null;
    this.idleDeadlineAt = null;
    this.hardDeadlineAt = null;
    this.urgentAt = null;
    this.deadlineAt = null;
    this.lastReason = '';
    this.generation += 1;
  }

  // 当前批次是否仍然开着（有截止时间）。
  get pending() {
    return this.deadlineAt != null;
  }

  snapshot() {
    return {
      deadlineAt: this.deadlineAt,
      firstArrivalAt: this.firstArrivalAt,
      lastArrivalAt: this.lastArrivalAt,
      lastAuthor: this.lastAuthor,
      hardDeadlineAt: this.hardDeadlineAt,
      urgentAt: this.urgentAt,
      batchCount: this.batchCount,
      generation: this.generation,
      lastReason: this.lastReason,
      profile: this.cfg.profile
    };
  }
}

// 一批防抖状态的容器：按会话 key 管理 MessageDebouncer，并做过期/数量上限回收。
// 注意：这里只回收「状态对象」，不碰任何消息内容；被回收后下一次消息会开新批。
export class DebounceRegistry {
  constructor(config, { now = () => performance.now() } = {}) {
    this.cfg = normalizeDebounceConfig(config);
    this.now = now;
    this.batches = new Map(); // key -> { debouncer, lastSeenAt }
  }

  get(key) {
    let entry = this.batches.get(key);
    if (!entry) {
      entry = { debouncer: new MessageDebouncer(this.cfg, { now: this.now }), lastSeenAt: this.now() };
      this.batches.set(key, entry);
      this.sweep();
    }
    return entry.debouncer;
  }

  peek(key) {
    return this.batches.get(key)?.debouncer ?? null;
  }

  touch(key) {
    const entry = this.batches.get(key);
    if (entry) entry.lastSeenAt = this.now();
    return entry?.debouncer ?? null;
  }

  // 回收长时间没有活动的批次（防止每个群/私聊永久占一个对象）。
  // 只删除 Map 里的状态；调用方若还持有定时器，应在 sweep 前自行清理（bridge 里在定时器回调里清）。
  sweep(now = null) {
    const t = Number.isFinite(now) ? now : this.now();
    for (const [key, entry] of this.batches) {
      if (!entry.debouncer.pending && t - entry.lastSeenAt > this.cfg.idleTtlMs) {
        this.batches.delete(key);
      }
    }
    // 数量上限：超出时丢最久没动过的空闲批次（不丢仍有 pending 截止的）。
    if (this.batches.size > this.cfg.maxBatches) {
      const idle = [...this.batches.entries()]
        .filter(([, e]) => !e.debouncer.pending)
        .sort((a, b) => a[1].lastSeenAt - b[1].lastSeenAt);
      for (const [key] of idle) {
        if (this.batches.size <= this.cfg.maxBatches) break;
        this.batches.delete(key);
      }
    }
  }

  delete(key) {
    // 必须 reset 再删：调用方可能已经 clearTimeout，但回调可能已入队；
    // reset 会 +1 generation，让旧回调的「代际校验」失败，不能打进新批次。
    this.batches.get(key)?.debouncer.reset();
    this.batches.delete(key);
  }

  clear() {
    for (const entry of this.batches.values()) entry.debouncer.reset();
    this.batches.clear();
  }

  get size() {
    return this.batches.size;
  }
}
