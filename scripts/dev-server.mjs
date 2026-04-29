import dgram from 'node:dgram';
import { createReadStream, statSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const httpPort = Number(process.env.PORT || 5173);
const oscConfig = {
  enabled: process.env.OSC_ENABLED !== '0',
  host: process.env.OSC_HOST || '127.0.0.1',
  port: Number(process.env.OSC_PORT || 8000),
  packetsSent: 0
};
const udp = dgram.createSocket('udp4');

const contentTypes = {
  '.css': 'text/css; charset=utf-8',
  '.gif': 'image/gif',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.webm': 'video/webm'
};

const pad4 = (length) => (4 - (length % 4)) % 4;

const oscString = (value) => {
  const text = Buffer.from(String(value), 'utf8');
  return Buffer.concat([text, Buffer.alloc(1 + pad4(text.length + 1))]);
};

const oscInt = (value) => {
  const buffer = Buffer.alloc(4);
  buffer.writeInt32BE(Math.trunc(Number(value) || 0));
  return buffer;
};

const oscFloat = (value) => {
  const buffer = Buffer.alloc(4);
  buffer.writeFloatBE(Number.isFinite(Number(value)) ? Number(value) : 0);
  return buffer;
};

const oscArg = (value) => {
  if (Number.isInteger(value)) return { tag: 'i', data: oscInt(value) };
  if (typeof value === 'number') return { tag: 'f', data: oscFloat(value) };
  return { tag: 's', data: oscString(value ?? '') };
};

const oscMessage = (address, args = []) => {
  const encodedArgs = args.map(oscArg);
  return Buffer.concat([
    oscString(address),
    oscString(`,${encodedArgs.map(arg => arg.tag).join('')}`),
    ...encodedArgs.map(arg => arg.data)
  ]);
};

const sendOsc = (address, args) => {
  if (!oscConfig.enabled) return;

  const message = oscMessage(address, args);
  udp.send(message, oscConfig.port, oscConfig.host);
  oscConfig.packetsSent += 1;
};

const sendOscFrame = (payload) => {
  const objects = Array.isArray(payload.objects) ? payload.objects : [];
  const frameId = Math.trunc(Number(payload.frameId) || 0);

  sendOsc('/troloyolo/frame', [
    frameId,
    objects.length,
    Math.trunc(Number(payload.width) || 0),
    Math.trunc(Number(payload.height) || 0),
    Number(payload.timestamp) || 0,
    payload.source || ''
  ]);

  for (const object of objects) {
    sendOsc('/troloyolo/object', [
      frameId,
      Math.trunc(Number(object.id) || 0),
      Math.trunc(Number(object.classId) || 0),
      object.label || '',
      object.type || '',
      Number(object.score) || 0,
      Number(object.nx) || 0,
      Number(object.ny) || 0,
      Number(object.nw) || 0,
      Number(object.nh) || 0,
      Number(object.ncx) || 0,
      Number(object.ncy) || 0,
      Number(object.x) || 0,
      Number(object.y) || 0,
      Number(object.w) || 0,
      Number(object.h) || 0
    ]);
  }
};

const readBody = async (request, limit = 1_000_000) => {
  const chunks = [];
  let size = 0;

  for await (const chunk of request) {
    size += chunk.length;
    if (size > limit) throw new Error('Request body is too large');
    chunks.push(chunk);
  }

  return Buffer.concat(chunks).toString('utf8');
};

const sendJson = (response, status, data) => {
  const body = JSON.stringify(data);
  response.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Content-Length': Buffer.byteLength(body)
  });
  response.end(body);
};

const getOscStatus = () => ({
  enabled: oscConfig.enabled,
  host: oscConfig.host,
  port: oscConfig.port,
  packetsSent: oscConfig.packetsSent
});

const updateOscConfig = (config = {}) => {
  if (typeof config.enabled === 'boolean') oscConfig.enabled = config.enabled;

  if (typeof config.host === 'string' && config.host.trim()) {
    oscConfig.host = config.host.trim();
  }

  if (config.port !== undefined) {
    const port = Number(config.port);
    if (!Number.isInteger(port) || port < 1 || port > 65535) {
      throw new Error('OSC port must be an integer from 1 to 65535');
    }
    oscConfig.port = port;
  }

  return getOscStatus();
};

const resolveStaticPath = (urlPath) => {
  const cleanPath = decodeURIComponent(urlPath.split('?')[0]);
  const relativePath = cleanPath === '/' ? '/index.html' : cleanPath;
  const filePath = path.resolve(rootDir, `.${relativePath}`);

  if (!filePath.startsWith(rootDir + path.sep) && filePath !== rootDir) return null;
  return filePath;
};

const serveStatic = async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  let filePath = resolveStaticPath(url.pathname);

  if (!filePath) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }

  try {
    let stats = statSync(filePath);
    if (stats.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
      stats = statSync(filePath);
    }

    response.writeHead(200, {
      'Content-Type': contentTypes[path.extname(filePath)] || 'application/octet-stream',
      'Content-Length': stats.size
    });
    createReadStream(filePath).pipe(response);
  } catch {
    try {
      const index = await readFile(path.join(rootDir, 'index.html'));
      response.writeHead(200, {
        'Content-Type': 'text/html; charset=utf-8',
        'Content-Length': index.length
      });
      response.end(index);
    } catch {
      response.writeHead(404);
      response.end('Not found');
    }
  }
};

const server = createServer(async (request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  const isOscPath = url.pathname === '/osc' || url.pathname.endsWith('/osc');
  const isOscConfigPath = url.pathname === '/osc/config' || url.pathname.endsWith('/osc/config');
  const isOscTestPath = url.pathname === '/osc/test' || url.pathname.endsWith('/osc/test');
  const isOscStatusPath = url.pathname === '/osc/status' || url.pathname.endsWith('/osc/status');

  if (isOscStatusPath || isOscConfigPath) {
    if (request.method === 'GET') {
      sendJson(response, 200, getOscStatus());
      return;
    }

    if (request.method === 'POST') {
      try {
        const body = await readBody(request);
        sendJson(response, 200, updateOscConfig(JSON.parse(body || '{}')));
      } catch (error) {
        sendJson(response, 400, { error: error.message });
      }
      return;
    }

    sendJson(response, 405, { error: 'GET or POST required' });
    return;
  }

  if (isOscTestPath) {
    if (request.method !== 'POST') {
      sendJson(response, 405, { error: 'POST required' });
      return;
    }

    sendOsc('/troloyolo/test', [Date.now() / 1000, oscConfig.host, oscConfig.port]);
    sendJson(response, 200, getOscStatus());
    return;
  }

  if (isOscPath) {
    if (request.method !== 'POST') {
      sendJson(response, 405, { error: 'POST required' });
      return;
    }

    try {
      const body = await readBody(request);
      sendOscFrame(JSON.parse(body));
      sendJson(response, 200, { ok: true });
    } catch (error) {
      sendJson(response, 400, { error: error.message });
    }
    return;
  }

  if (request.method !== 'GET' && request.method !== 'HEAD') {
    response.writeHead(405);
    response.end('Method not allowed');
    return;
  }

  await serveStatic(request, response);
});

server.listen(httpPort, () => {
  console.log(`troloyolo serving http://localhost:${httpPort}/`);
  console.log(`OSC UDP output -> ${oscConfig.enabled ? 'on' : 'off'} ${oscConfig.host}:${oscConfig.port}`);
});
