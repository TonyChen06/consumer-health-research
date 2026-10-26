import http from 'node:http';
import {appendFileSync} from 'node:fs';

export async function start(key, log, cap, fetcher = fetch) {
  let spent = 0, pending = 0, fatal = null;
  const server = http.createServer(async (req, res) => {
    const send = (status, value) => { res.writeHead(status, {'Content-Type': 'application/json'}); res.end(JSON.stringify(value)); };
    let reserved = false, id = '', decision = false;
    const started = Date.now();
    try {
      if (req.method === 'GET' && req.url === '/status') return send(200, {spent, pending, fatal});
      if (req.method !== 'POST' || !['/v1/chat/completions', '/v1/responses', '/v1/decisions'].includes(req.url)) return send(404, {});
      id = (req.headers.authorization || '').replace(/^Bearer local-proxy-/, '');
      if (!/^[a-zA-Z0-9_-]+$/.test(id)) return send(403, {});
      let raw = '';
      for await (const chunk of req) { raw += chunk; if (raw.length > 8_000_000) throw Error('Request too large'); }
      const body = JSON.parse(raw);
      if (body.model !== 'gpt-6-luna' || body.stream || body.tools?.length || body.n > 1) return send(400, {});
      if (fatal) return send(503, {error: {code: fatal}});
      if (spent + (pending + 1) * .1 > cap) { fatal = 'experiment_cost_cap'; return send(402, {error: {code: fatal}}); }
      decision = req.url.endsWith('/decisions');
      const chat = req.url.endsWith('/chat/completions');
      const payload = decision ? {model: body.model, input: body.input, questions: body.questions} : {
        model: body.model, input: chat ? body.messages : body.input,
        reasoning: {effort: 'low'}, max_output_tokens: 8192, store: false,
        ...(body.instructions ? {instructions: body.instructions} : {}),
      };
      pending++; reserved = true;
      const response = await fetcher('https://api.openai.com/v1/' + (decision ? 'decisions' : 'responses'), {
        method: 'POST', headers: {Authorization: `Bearer ${key}`, 'Content-Type': 'application/json'},
        body: JSON.stringify(payload), signal: AbortSignal.timeout(decision ? 60000 : 180000),
      });
      const result = await response.json();
      if (!response.ok) {
        const code = /^[a-zA-Z0-9_-]+$/.test(result.error?.code || '') ? result.error.code : 'upstream_error';
        if ([401, 402, 403].includes(response.status) || ['insufficient_quota', 'credit_balance_exhausted'].includes(code)) fatal = code;
        appendFileSync(log, JSON.stringify({case_id: id, error: code, status: response.status}) + '\n');
        return send(response.status, {error: {code, message: 'Upstream request failed'}});
      }
      if (result.model !== 'gpt-6-luna' && !result.model?.startsWith('gpt-6-luna-')) throw Error('Unexpected model');
      const usage = result.usage;
      if (!Number.isFinite(usage?.input_tokens)) throw Error('Missing usage');
      const cost = (usage.input_tokens * .1 + (decision ? 0 : (usage.output_tokens || 0) * .5)) / 1e6;
      spent += cost;
      let content = (result.output || []).flatMap(o => o.content || []).filter(c => c.type === 'output_text').map(c => c.text).join('');
      const stops = typeof body.stop === 'string' ? [body.stop] : body.stop || [];
      let end = content.length;
      for (const stop of stops) { const pos = content.indexOf(stop); if (pos >= 0) end = Math.min(end, pos); }
      content = content.slice(0, end);
      appendFileSync(log, JSON.stringify({case_id: id, id: result.id, model: result.model, status: result.status,
        transport: decision ? 'decisions' : 'responses', elapsed_ms: Date.now() - started, usage, cost,
        input: payload.input, questions: payload.questions, answers: result.answers, output_text: content}) + '\n');
      if (!chat) return send(200, result);
      return send(200, {id: result.id, object: 'chat.completion', created: Math.floor(Date.now() / 1000), model: result.model,
        choices: [{index: 0, message: {role: 'assistant', content}, finish_reason: result.status === 'completed' ? 'stop' : 'length'}],
        usage: {prompt_tokens: usage.input_tokens, completion_tokens: usage.output_tokens, total_tokens: usage.total_tokens}});
    } catch (error) {
      fatal = 'transport_failure';
      appendFileSync(log, JSON.stringify({case_id: id, error: fatal, type: error.name}) + '\n');
      send(500, {error: {code: fatal}});
    } finally { if (reserved) pending--; }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return server;
}

if (process.argv[2]) {
  const key = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  const cap = Number(process.argv[3]);
  if (!key || !Number.isFinite(cap) || cap <= 0) throw Error('API key and positive budget required');
  const server = await start(key, process.argv[2], cap);
  process.stdout.write(`http://127.0.0.1:${server.address().port}/v1\n`);
}
