import strategies from './strategies-data.generated.js';

const GEMINI_MODEL = 'gemini-3.6-flash';
const MAX_MESSAGE_LENGTH = 1500;
const MAX_HISTORY_TURNS = 6;
const RETRIEVAL_TOP_N = 4;

function strategyFullText(strat) {
  const parts = [`### ${strat.title}`, strat.intro];
  for (const sec of strat.sections) {
    parts.push(`**${sec.h2}**`);
    (sec.body || []).forEach((t) => parts.push(t));
    (sec.bullets || []).forEach((t) => parts.push(`- ${t}`));
    (sec.examplesAndNotes || []).forEach((e) => {
      parts.push(`${e.type === 'example' ? 'Example' : 'Note'}: ${e.text}`);
    });
  }
  return parts.join('\n');
}

function haystackFor(strat) {
  const parts = [strat.title, strat.intro, ...strat.categories];
  for (const sec of strat.sections) {
    parts.push(sec.h2);
    (sec.body || []).forEach((t) => parts.push(t));
    (sec.bullets || []).forEach((t) => parts.push(t));
    (sec.examplesAndNotes || []).forEach((e) => parts.push(e.text));
  }
  return parts.join(' ').toLowerCase();
}

const STOPWORDS = new Set(['the', 'and', 'for', 'are', 'with', 'this', 'that', 'what', 'how', 'can', 'you', 'does', 'do', 'is', 'a', 'an', 'to', 'of', 'in', 'on', 'my', 'i', 'me', 'it', 'about', 'tax', 'strategy', 'strategies']);

function retrieve(query, topN) {
  const terms = query
    .toLowerCase()
    .split(/[^a-z0-9%§]+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t));

  if (terms.length === 0) return [];

  const scored = strategies.map((s) => {
    const hay = haystackFor(s);
    let score = 0;
    for (const t of terms) {
      if (hay.includes(t)) score += 1;
      if (s.title.toLowerCase().includes(t)) score += 2;
    }
    return { s, score };
  });

  return scored
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topN)
    .map((x) => x.s);
}

function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8' },
  });
}

async function handleChat(request, env) {
  const apiKey = env.GEMINI_API_KEY;
  if (!apiKey) {
    return jsonResponse({ error: 'Chat is not configured on this deployment yet (missing GEMINI_API_KEY).' }, 503);
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return jsonResponse({ error: 'Invalid JSON body.' }, 400);
  }

  const message = typeof payload.message === 'string' ? payload.message.trim() : '';
  if (!message) return jsonResponse({ error: 'A "message" string is required.' }, 400);
  if (message.length > MAX_MESSAGE_LENGTH) {
    return jsonResponse({ error: `Message too long (max ${MAX_MESSAGE_LENGTH} characters).` }, 400);
  }

  const rawHistory = Array.isArray(payload.history) ? payload.history : [];
  const history = rawHistory
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-MAX_HISTORY_TURNS)
    .map((m) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: m.content.slice(0, MAX_MESSAGE_LENGTH) }],
    }));

  const matches = retrieve(message, RETRIEVAL_TOP_N);
  const context = matches.length
    ? matches.map(strategyFullText).join('\n\n---\n\n')
    : '(No strategy in the guide matched this question closely by keyword — say so, and only answer from general knowledge if you are confident it is still about a topic covered elsewhere in the guide; otherwise say it is outside the guide.)';

  const tableOfContents = strategies.map((s, i) => `${i + 1}. ${s.title}`).join('\n');

  const systemPrompt = `You are the embedded assistant for "Tax Strategies," a reference guide of ${strategies.length} independent tax-planning strategies (S-corp, partnership/LLC, real estate, retirement, investment, and IRS compliance topics). You are shown on the guide's own page so visitors can ask questions about it.

Ground every answer in the retrieved excerpts below — they are pulled from the guide by keyword match against the user's question. Do not invent tax rules, dollar figures, or thresholds beyond what the guide states. If the excerpts don't actually answer the question, say plainly that this guide doesn't cover it, and suggest the closest-sounding titles from the full list below instead of guessing.

Keep answers conversational but tight — a few sentences to a short paragraph, not a full essay, unless the user is asking for real depth. End with a brief reminder that this is educational reference material, not personalized tax advice, only when it's not obvious from context (don't repeat it every single turn in a back-and-forth).

Full list of strategies in the guide:
${tableOfContents}

Retrieved excerpts most relevant to the current question:
${context}`;

  let geminiResponse;
  try {
    geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
      {
        method: 'POST',
        headers: {
          'x-goog-api-key': apiKey,
          'content-type': 'application/json',
        },
        body: JSON.stringify({
          system_instruction: { parts: [{ text: systemPrompt }] },
          contents: [...history, { role: 'user', parts: [{ text: message }] }],
          generationConfig: { maxOutputTokens: 1024 },
        }),
      },
    );
  } catch {
    return jsonResponse({ error: 'Could not reach the AI service. Try again in a moment.' }, 502);
  }

  if (!geminiResponse.ok) {
    const status = geminiResponse.status;
    if (status === 429) return jsonResponse({ error: 'The assistant is getting a lot of questions right now — try again shortly.' }, 429);
    return jsonResponse({ error: 'The AI service returned an error.' }, 502);
  }

  const data = await geminiResponse.json();
  const candidate = data.candidates && data.candidates[0];
  const reply = candidate && candidate.content && candidate.content.parts && candidate.content.parts[0]
    ? candidate.content.parts[0].text
    : 'Sorry, I could not generate a response.';

  return jsonResponse({
    reply,
    matchedStrategies: matches.map((s) => s.title),
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === '/api/chat') {
      if (request.method !== 'POST') return jsonResponse({ error: 'Use POST.' }, 405);
      return handleChat(request, env);
    }

    return env.ASSETS.fetch(request);
  },
};
