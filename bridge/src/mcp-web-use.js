// web_use MCP 服务器（Node 版，跑在 WSL 原生环境）
//
// 为什么不直接让 DSH spawn python.exe：
//   DSH 用 MCP SDK 的 StdioClientTransport spawn Windows 的 python.exe，
//   一律失败，报 "/mnt/c/.../python.exe: Invalid argument"。
//   包一层 bash 也救不了（bash 能起来，但里面的 interop 仍失败）。
//   env 的三种组合（继承 / scrub 后 / env -i）手动都复现不出来，
//   说明是 SDK 那层 spawn 上下文的问题，靠配置绕不过去。
//
// 所以：MCP 服务器用 Node 写 —— WSL 原生的 /opt/node/bin/node spawn 绝不会失败。
// 每次工具调用再用 execFile 把活转给 Windows 的 python。
// 「Node execFile 调 Windows exe」这条实测是通的。
//
// 代价：每次调用多一次进程启动（约 3~8 秒，因为要 import web_use + 加载依赖）。
// 换来的是：绝对能起来，而且 DSH 重启不会因为 MCP 挂掉而整个不可用。
import { execFile } from 'node:child_process';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

// ★ 用 WSL 原生的 python，不用 Windows 的 python.exe。
//   踩了很久才确认：DSH 的进程树里，从 Node 去 spawn Windows exe（不管 execFile 还是
//   bash 包装）一律报 "/mnt/c/.../python.exe: Invalid argument" —— 而同样的命令
//   在普通 shell 里手动跑是通的。是 interop 在那个进程上下文里失效，配置层面绕不过去。
//   换成 /root/webuse-venv/bin/python（原生 Linux）就完全没有这个问题。
//   venv 里装了 httpx + pydantic（系统 python 被 Debian 的 typing_extensions 卡住装不上）。
//   cdp_social 访问浏览器不受影响：WSL 能直连 127.0.0.1:9333（networkingMode=mirrored）。
const PY = '/root/webuse-venv/bin/python';
const CLI = '/mnt/c/Users/45140/Desktop/code/web-use/web_use_cli.py';
// 查梗要开两个站、等 SPA 渲染，实测最慢一次约 40 秒；给足余量。
const TIMEOUT_MS = 180_000;

function callPython(req) {
  return new Promise((resolve) => {
    execFile(
      PY,
      [CLI, JSON.stringify(req)],
      { timeout: TIMEOUT_MS, maxBuffer: 8 * 1024 * 1024, encoding: 'utf8' },
      (err, stdout, stderr) => {
        if (err) {
          resolve({
            ok: false,
            error: `python 调用失败: ${err.message}${stderr ? ' | stderr: ' + stderr.slice(0, 300) : ''}`,
          });
          return;
        }
        const line = (stdout || '').trim().split('\n').filter(Boolean).pop() || '{}';
        try {
          resolve(JSON.parse(line));
        } catch (e) {
          resolve({ ok: false, error: `返回不是 JSON: ${line.slice(0, 300)}` });
        }
      },
    );
  });
}

const server = new McpServer({ name: 'web-use', version: '0.1.0' });

function text(obj) {
  return { content: [{ type: 'text', text: JSON.stringify(obj, null, 2) }] };
}

server.tool(
  'web_search',
  '搜索。多后端 + 熔断，某个源挂了会自动换下一个。mode: slang(只查梗/网络用语) / '
    + 'links(只找链接,便宜) / answer(默认) / research(深调研)。'
    + '返回里 meta.backend 是实际用的后端，meta.degraded=true 表示首选源失败、这是降级结果。',
  {
    query: z.string().describe('要找的东西'),
    mode: z.enum(['answer', 'links', 'research', 'slang']).optional().describe('搜索模式，默认 answer'),
    max_results: z.number().int().min(1).max(20).optional().describe('最多几条，默认 6'),
  },
  async ({ query, mode, max_results }) =>
    text(await callPython({ op: 'search', query, mode: mode ?? 'answer', max_results: max_results ?? 6 })),
);

server.tool(
  'web_extract',
  '取已知 URL 的正文（去掉导航/模板）。已经有 URL 就用这个，别再 search —— 搜索慢一个量级。',
  { url: z.string().describe('http(s) URL') },
  async ({ url }) => text(await callPython({ op: 'extract', url })),
);

server.tool(
  'web_fetch',
  '取原始响应（不抽正文）。要自己解析时用。',
  { url: z.string().describe('http(s) URL') },
  async ({ url }) => text(await callPython({ op: 'fetch', url })),
);

server.tool(
  'web_browse',
  '用真浏览器打开并取渲染后的内容。SPA / 需要 JS 的页面用这个。',
  { url: z.string().describe('http(s) URL') },
  async ({ url }) => text(await callPython({ op: 'browse', url })),
);

await server.connect(new StdioServerTransport());
