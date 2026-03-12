/**
 * SatSeg · CORS Proxy v3.0
 * ─────────────────────────────────────────────────────────────
 * Reenvía peticiones al HuggingFace Router con CORS habilitado.
 *
 * RUTAS SOPORTADAS:
 *   /hf/<modelo>   →  https://router.huggingface.co/hf-inference/models/<modelo>
 *   /fal/<modelo>  →  https://router.huggingface.co/fal-ai/models/<modelo>
 *
 * USO:
 *   node proxy.js
 *
 * Requiere Node.js >= 18 (fetch nativo). Sin dependencias externas.
 */

const http = require('http');
const PORT = 8787;

const PROVIDERS = {
  '/hf/':  'https://router.huggingface.co/hf-inference/models/',
  '/fal/': 'https://router.huggingface.co/fal-ai/models/',
};

const CORS = {
  'Access-Control-Allow-Origin':  '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-wait-for-model',
};

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, CORS);
    res.end();
    return;
  }

  // Determinar provider y construir URL destino
  let targetUrl = null;
  for (const [prefix, base] of Object.entries(PROVIDERS)) {
    if (req.url.startsWith(prefix)) {
      targetUrl = base + req.url.slice(prefix.length);
      break;
    }
  }

  if (!targetUrl) {
    res.writeHead(400, { ...CORS, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Ruta inválida. Usa /hf/<modelo> o /fal/<modelo>' }));
    return;
  }

  // Leer body
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  const body = Buffer.concat(chunks);

  // Headers a reenviar
  const fwd = {};
  if (req.headers['authorization'])    fwd['Authorization']    = req.headers['authorization'];
  if (req.headers['content-type'])     fwd['Content-Type']     = req.headers['content-type'];
  if (req.headers['x-wait-for-model']) fwd['x-wait-for-model'] = req.headers['x-wait-for-model'];

  console.log(`[→] ${req.method} ${targetUrl}`);

  try {
    const hfRes = await fetch(targetUrl, {
      method:  req.method,
      headers: fwd,
      body:    body.length > 0 ? body : undefined,
    });

    const buf  = await hfRes.arrayBuffer();
    const ct   = hfRes.headers.get('content-type') || 'application/octet-stream';
    console.log(`[←] ${hfRes.status} ${ct} (${buf.byteLength} bytes)`);

    res.writeHead(hfRes.status, { ...CORS, 'Content-Type': ct });
    res.end(Buffer.from(buf));

  } catch (err) {
    console.error('[!] Error de red:', err.message);
    res.writeHead(502, { ...CORS, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: err.message }));
  }
});

server.listen(PORT, () => {
  console.log('');
  console.log('  ╔═══════════════════════════════════════╗');
  console.log('  ║   SatSeg CORS Proxy  v3.0  :' + PORT + '    ║');
  console.log('  ╠═══════════════════════════════════════╣');
  console.log('  ║  /hf/<model>  → hf-inference (CPU)   ║');
  console.log('  ║  /fal/<model> → fal-ai (GPU)         ║');
  console.log('  ╠═══════════════════════════════════════╣');
  console.log('  ║  Frontend: npx serve .                ║');
  console.log('  ║  Abre:  http://localhost:3000         ║');
  console.log('  ╚═══════════════════════════════════════╝');
  console.log('');
});
