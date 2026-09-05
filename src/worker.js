import strategies from './strategies-data.generated.js';

const GEMINI_MODELS = ['gemini-3.6-flash', 'gemini-flash-latest', 'gemini-pro-latest'];
const MAX_MESSAGE_LENGTH = 1500;
const MAX_HISTORY_TURNS = 6;
const RETRIEVAL_TOP_N = 3;

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

function bodyTextFor(strat) {
  const parts = [strat.intro, ...strat.categories];
  for (const sec of strat.sections) {
    parts.push(sec.h2);
    (sec.body || []).forEach((t) => parts.push(t));
    (sec.bullets || []).forEach((t) => parts.push(t));
    (sec.examplesAndNotes || []).forEach((e) => parts.push(e.text));
  }
  return parts.join(' ');
}

const STOPWORDS = new Set(['the', 'and', 'for', 'are', 'with', 'this', 'that', 'what', 'how', 'can', 'you', 'does', 'do', 'is', 'a', 'an', 'to', 'of', 'in', 'on', 'my', 'i', 'me', 'it', 'about', 'tax', 'strategy', 'strategies']);

// Query-side and index-side synonym expansion so common tax shorthand and
// full-form phrasing land on the same tokens (e.g. "s-corp" <-> "s corporation").
const SYNONYM_PATTERNS = [
  [/\bs[\s-]?corp(oration)?s?\b/g, ' scorp '],
  [/\bc[\s-]?corp(oration)?s?\b/g, ' ccorp '],
  [/\b401\s?\(?k\)?\b/g, ' 401k '],
  [/\bqbi\b/g, ' qbi qualifiedbusinessincome '],
  [/\bllc'?s?\b/g, ' llc '],
  [/\bsole[\s-]?prop(rietor(ship)?)?\b/g, ' soleproprietor '],
  [/\birs\b/g, ' irs '],
  [/\bhsa\b/g, ' hsa healthsavingsaccount '],
  [/\bira\b/g, ' ira '],
];

function normalizeText(text) {
  let t = ` ${text.toLowerCase()} `;
  for (const [pattern, replacement] of SYNONYM_PATTERNS) t = t.replace(pattern, replacement);
  return t;
}

// Light stemming: strip common suffixes so "distributions"/"distribution" and
// "filing"/"filed"/"files" collapse to the same token. Guarded against short words.
function stem(word) {
  if (word.length > 5 && word.endsWith('ies')) return `${word.slice(0, -3)}y`;
  if (word.length > 5 && word.endsWith('es')) return word.slice(0, -2);
  if (word.length > 4 && word.endsWith('s') && !word.endsWith('ss')) return word.slice(0, -1);
  if (word.length > 6 && word.endsWith('ing')) return word.slice(0, -3);
  if (word.length > 5 && word.endsWith('ed')) return word.slice(0, -2);
  return word;
}

function tokenize(text) {
  return normalizeText(text)
    .split(/[^a-z0-9%§]+/)
    .filter((t) => t.length > 2 && !STOPWORDS.has(t))
    .map(stem);
}

function termCounts(tokens) {
  const counts = new Map();
  for (const t of tokens) counts.set(t, (counts.get(t) || 0) + 1);
  return counts;
}

// BM25 over two fields (title, body) with a title-weight boost, computed once
// at module load and reused across requests handled by this isolate.
const BM25_K1 = 1.5;
const BM25_B = 0.75;
const TITLE_WEIGHT = 2.5;

function buildIndex() {
  const docs = strategies.map((s) => ({
    titleTokens: tokenize(s.title),
    bodyTokens: tokenize(bodyTextFor(s)),
  }));
  docs.forEach((d) => {
    d.titleCounts = termCounts(d.titleTokens);
    d.bodyCounts = termCounts(d.bodyTokens);
  });

  const avgTitleLen = docs.reduce((sum, d) => sum + d.titleTokens.length, 0) / docs.length;
  const avgBodyLen = docs.reduce((sum, d) => sum + d.bodyTokens.length, 0) / docs.length;

  const df = new Map();
  docs.forEach((d) => {
    const seen = new Set([...d.titleCounts.keys(), ...d.bodyCounts.keys()]);
    seen.forEach((term) => df.set(term, (df.get(term) || 0) + 1));
  });

  const N = docs.length;
  const idf = new Map();
  df.forEach((count, term) => idf.set(term, Math.log((N - count + 0.5) / (count + 0.5) + 1)));

  return { docs, idf, avgTitleLen, avgBodyLen };
}

function bm25TermScore(freq, docLen, avgLen, idfVal) {
  if (!freq) return 0;
  const numerator = freq * (BM25_K1 + 1);
  const denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * (docLen / (avgLen || 1)));
  return idfVal * (numerator / denominator);
}

let INDEX = null;

function retrieve(query, topN) {
  if (!INDEX) INDEX = buildIndex();
  const terms = tokenize(query);
  if (terms.length === 0) return [];

  const scored = INDEX.docs.map((d, i) => {
    let score = 0;
    for (const t of terms) {
      const idfVal = INDEX.idf.get(t);
      if (!idfVal) continue;
      score += bm25TermScore(d.bodyCounts.get(t) || 0, d.bodyTokens.length, INDEX.avgBodyLen, idfVal);
      score += TITLE_WEIGHT * bm25TermScore(d.titleCounts.get(t) || 0, d.titleTokens.length, INDEX.avgTitleLen, idfVal);
    }
    return { s: strategies[i], score };
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

  const requestBody = JSON.stringify({
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: [...history, { role: 'user', parts: [{ text: message }] }],
    generationConfig: { maxOutputTokens: 1024 },
  });

  let lastStatus = 502;
  for (const model of GEMINI_MODELS) {
    let geminiResponse;
    try {
      geminiResponse = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
        {
          method: 'POST',
          headers: {
            'x-goog-api-key': apiKey,
            'content-type': 'application/json',
          },
          body: requestBody,
        },
      );
    } catch {
      continue;
    }

    if (geminiResponse.status === 429) {
      lastStatus = 429;
      continue;
    }
    if (!geminiResponse.ok) {
      lastStatus = geminiResponse.status;
      continue;
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

  if (lastStatus === 429) return jsonResponse({ error: 'The assistant is getting a lot of questions right now — try again shortly.' }, 429);
  return jsonResponse({ error: 'The AI service returned an error.' }, 502);
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
