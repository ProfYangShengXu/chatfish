// 安全版 Web Search / Fetch MCP server（stdio）。由 DSH 的 MCP 客户端 spawn。
//
// 安全设计：
// - 只暴露只读工具 `web_search` 与 `web_fetch`：查网络用语/梗/黑话、抓取网页正文。
// - 不暴露任何本地文件、命令执行、写操作。
// - 查询词做基础清洗：去 CQ 码、控制字符、超长截断。
// - `web_fetch` 仅允许 http/https：
//   - 禁止 URL 内嵌凭据；
//   - 禁止 localhost / .local / 私有 IP / 环回 / 链路本地 / CGNAT 等内网地址；
//   - 域名会先做 DNS 解析并检查全部解析结果，避免解析到内网；
//   - 手动跟随重定向，每一跳都重新校验；
//   - 响应体按字节流限量读取，避免超大响应拖垮进程。
// - 搜索结果/抓取结果仅作为“候选解释”，最终是否入库仍由控制台人工确认。
import { safeFetch } from './safe-fetch.js';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';

function sanitizeQuery(query) {
  return String(query ?? '')
    // 去掉 CQ 码（[CQ:xxx]）
    .replace(/\[CQ:[^\]]*\]/gi, ' ')
    .replace(/[\u0000-\u001f\u007f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 120);
}

function decodeHtml(s) {
  return String(s ?? '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Bing 搜索头。为什么要带全这一套、为什么不再走 safeFetch：
//   实测（2026-10-23）同一个查询 cn.bing.com/search?q=kskbl ，
//   safeFetch（HTTP/1.1、只带 host + 简易 UA、无 Cookie）拿到的是
//   「漫蛙漫画」「百度首页」这类降级页，8 条全废；
//   用 fetch（HTTP/2、自动维护 Cookie 会话）配同一组 header，
//   拿到的是「百度百科：kskbl 即康神开播了」「游侠手游：kskbl 是什么梗」，全部正确。
//   而且不带 mkt=zh-CN 时连 <li class="b_algo"> 都解析不出来（0 块）。
//   搜索端点域名是写死的，没有 SSRF 风险，所以这里不用 safeFetch 那条链路。
const BING_HEADERS = {
  'user-agent':
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
  accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
  'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
  'cache-control': 'no-cache',
  'sec-ch-ua': '"Chromium";v="131", "Not_A Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'document',
  'sec-fetch-mode': 'navigate',
  'sec-fetch-site': 'none',
  'sec-fetch-user': '?1',
  'upgrade-insecure-requests': '1',
};

async function bingSearch(query) {
  // mkt/setlang 决定返回中文结果；form=QBLH 是常规搜索框入口，缺了会被当异常流量。
  const url =
    'https://cn.bing.com/search?q=' +
    encodeURIComponent(query) +
    '&mkt=zh-CN&setlang=zh-Hans&form=QBLH';
  const res = await fetch(url, {
    headers: BING_HEADERS,
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) throw new Error(`搜索服务 HTTP ${res.status}`);
  const html = await res.text();
  const results = [];
  const blocks = html.split('<li class="b_algo"').slice(1);
  for (const block of blocks) {
    const hrefMatch = block.match(/<a[^>]+href="(https?:\/\/[^"]+)"/i);
    if (!hrefMatch) continue;
    const urlStr = decodeHtml(hrefMatch[1]);
    const titleMatch = block.match(/<h2[^>]*>([\s\S]*?)<\/h2>/i);
    const title = titleMatch ? decodeHtml(titleMatch[1]) : '';
    const snippetMatch = block.match(/<p[^>]*>([\s\S]*?)<\/p>/i);
    const snippet = snippetMatch ? decodeHtml(snippetMatch[1]) : '';
    if (urlStr && title) results.push({ title, url: urlStr, snippet });
    if (results.length >= 8) break;
  }
  return { query, results };
}

const server = new McpServer({ name: 'web-search-safe', version: '0.1.5' });

server.tool(
  'web_search',
  '只读搜索网络用语/梗/黑话的含义，返回 Bing 搜索结果（标题/URL/摘要）。仅用于理解词义，不执行任何本地操作。',
  { query: z.string().describe('要搜索确认的网络用语/黑话/梗') },
  async ({ query }) => {
    const clean = sanitizeQuery(query);
    if (!clean) {
      return { content: [{ type: 'text', text: '查询词为空，已拒绝。' }], isError: true };
    }
    try {
      const result = await bingSearch(clean);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (error) {
      return {
        content: [{ type: 'text', text: `搜索失败：${error?.message ?? error}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'web_fetch',
  '只读抓取 HTTP(S) 网页正文，返回纯文本/HTML 前 50000 字符。禁止访问内网/本机地址，不执行任何本地操作。',
  { url: z.string().describe('要抓取的 http(s) URL') },
  async ({ url }) => {
    try {
      const result = await safeFetch(url);
      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (error) {
      return {
        content: [{ type: 'text', text: `抓取失败：${error?.message ?? error}` }],
        isError: true,
      };
    }
  }
);

server.tool(
  'slang_lookup',
  '查拼音首字母缩写的含义（yyds / xswl / nsdd / plmm 这类）。'
    + '数据来自「能不能好好说话 nbnhhsh」的众包歧义库，返回该缩写的全部候选含义，'
    + '由你按上下文挑选。只支持缩写，不支持「破防」这类普通词条的释义。',
  { text: z.string().describe('要查的缩写，多个用空格隔开，例如 "yyds xswl"') },
  async ({ text }) => {
    // ★ 为什么单独做一个工具，而不是给 web_fetch 开 POST：
    //   web_fetch 的设计前提是只读（safeFetch 里 method 写死 GET），
    //   一旦支持任意 POST，就等于开了一个可以向任意地址发数据的口子。
    //   这里把目标 URL 和请求体结构都写死，只允许调这一个接口。
    const ENDPOINT = 'https://lab.magiconch.com/api/nbnhhsh/guess';
    const clean = String(text ?? '').trim();
    if (!clean) {
      return { content: [{ type: 'text', text: 'text 不能为空' }], isError: true };
    }
    if (clean.length > 200) {
      return { content: [{ type: 'text', text: 'text 太长（上限 200 字符）' }], isError: true };
    }
    try {
      const res = await fetch(ENDPOINT, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'user-agent': 'Mozilla/5.0',
          origin: 'https://lab.magiconch.com',
          referer: 'https://lab.magiconch.com/',
        },
        body: JSON.stringify({ text: clean }),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) {
        return {
          content: [{ type: 'text', text: `查询失败：HTTP ${res.status}` }],
          isError: true,
        };
      }
      const data = await res.json();
      // 只保留有候选的，砍掉超长列表，避免把上下文塞满
      const slim = (Array.isArray(data) ? data : [])
        .filter((it) => it && Array.isArray(it.trans) && it.trans.length)
        .map((it) => ({
          name: it.name,
          candidates: it.trans.slice(0, 12),
          total: it.trans.length,
        }));
      if (!slim.length) {
        return {
          content: [{ type: 'text', text: JSON.stringify({ matched: [], note: '没有收录这些缩写' }) }],
        };
      }
      return { content: [{ type: 'text', text: JSON.stringify({ matched: slim }, null, 2) }] };
    } catch (error) {
      return {
        content: [{ type: 'text', text: `查询失败：${error?.message ?? error}` }],
        isError: true,
      };
    }
  }
);

await server.connect(new StdioServerTransport());
