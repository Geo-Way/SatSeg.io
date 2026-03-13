/**
 * GeoWay SatSeg · CORS Proxy Seguro v4.0
 * El token HF vive SOLO en .env — el navegador nunca lo ve.
 *
 * SETUP:
 *   1. Crea .env con:   HF_TOKEN=hf_tutoken
 *   2. npm install dotenv
 *   3. node proxy.js
 */

require('dotenv').config();
const http = require('http');

const PORT     = 8787;
const HF_TOKEN = process.env.HF_TOKEN;

if (!HF_TOKEN || !HF_TOKEN.startsWith('hf_')) {
  console.error('\n❌  Token HF no encontrado.');
  console.error('    Crea el archivo .env con:  HF_TOKEN=hf_tutoken\n');
  process.exit(1);
}

const masked = HF_TOKEN.slice(0,6) + '*'.repeat(HF_TOKEN.length - 10) + HF_TOKEN.slice(-4);
console.log(`\n  Token leído del .env: ${masked}`);

const PROVIDERS = {
  '/hf/':  'https://router.huggingface.co/hf-inference/models/',
  '/fal/': 'https://router.huggingface.co/fal-ai/models/',
};

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, x-wait-for-model',
};

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') { res.writeHead(204, CORS); res.end(); return; }
  if (req.method !== 'POST')   { res.writeHead(405, CORS); res.end(); return; }

  let targetUrl = null;
  for (const [prefix, base] of Object.entries(PROVIDERS)) {
    if (req.url.startsWith(prefix)) { targetUrl = base + req.url.slice(prefix.length); break; }
  }

  if (!targetUrl) {
    res.writeHead(400, { ...CORS, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Ruta inválida. Usa /hf/<modelo> o /fal/<modelo>' }));
    return;
  }

  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);

  // Proxy inyecta el token — el navegador NUNCA lo ve ni lo envía
  const headers = {
    'Authorization':    `Bearer ${HF_TOKEN}`,
    'Content-Type':     req.headers['content-type'] || 'application/json',
    'x-wait-for-model': 'true',
  };

  console.log(`[→] POST ${targetUrl} (${body.length} bytes)`);

  try {
    const hfRes = await fetch(targetUrl, { method: 'POST', headers, body });
    const buf   = await hfRes.arrayBuffer();
    const ct    = hfRes.headers.get('content-type') || 'application/octet-stream';
    console.log(`[←] ${hfRes.status} ${ct}`);
    res.writeHead(hfRes.status, { ...CORS, 'Content-Type': ct });
    res.end(Buffer.from(buf));
  } catch (err) {
    console.error('[!]', err.message);
    res.writeHead(502, { ...CORS, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: err.message }));
  }
});

server.listen(PORT, () => {
  console.log('\n  ╔══════════════════════════════════════════╗');
  console.log('  ║   GeoWay SatSeg · Proxy Seguro v4.0     ║');
  console.log(`  ║   Puerto : ${PORT}  ·  Token desde .env      ║`);
  console.log('  ╠══════════════════════════════════════════╣');
  console.log('  ║   1. npm install dotenv                  ║');
  console.log('  ║   2. Crea .env  →  HF_TOKEN=hf_...      ║');
  console.log('  ║   3. npx serve . → http://localhost:3000 ║');
  console.log('  ╚══════════════════════════════════════════╝\n');
});
