const fs = require('fs');
const path = require('path');

const CHAT_MODEL = 'gpt-4o-mini';
const EMBEDDING_MODEL = 'text-embedding-3-small';
const RAG_DIR = path.join(process.cwd(), '.rag-docs');
const cache = {};

// ── text utilities ────────────────────────────────────────────────────────────

function chunkText(text, maxChars = 800) {
  const paragraphs = text.replace(/\r\n/g, '\n').split(/\n{2,}/).map(p => p.trim()).filter(Boolean);
  const chunks = [];
  for (const p of paragraphs) {
    if (p.length <= maxChars) { chunks.push(p); continue; }
    const sentences = p.split(/(?<=[.!?])\s+/);
    let current = '';
    for (const s of sentences) {
      if ((current + ' ' + s).trim().length <= maxChars) {
        current = (current + ' ' + s).trim();
      } else {
        if (current) chunks.push(current);
        current = s;
      }
    }
    if (current) chunks.push(current);
  }
  return chunks.filter(Boolean);
}

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
  }
  return (!na || !nb) ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── openai ────────────────────────────────────────────────────────────────────

async function openAiPost(endpoint, payload) {
  const res = await fetch(`https://api.openai.com/v1${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message || 'OpenAI error');
  return data;
}

// ── rag index ─────────────────────────────────────────────────────────────────

async function getChunks() {
  if (!fs.existsSync(RAG_DIR)) return [];
  const files = fs.readdirSync(RAG_DIR).filter(f => /\.(txt|md)$/i.test(f));
  if (!files.length) return [];

  const sig = files.map(f => {
    const s = fs.statSync(path.join(RAG_DIR, f));
    return `${f}:${s.size}:${s.mtimeMs}`;
  }).sort().join('|');

  if (cache.sig === sig && cache.chunks) return cache.chunks;

  const allChunks = [];
  for (const file of files) {
    const content = fs.readFileSync(path.join(RAG_DIR, file), 'utf8');
    chunkText(content).forEach((text, i) => allChunks.push({ title: file, text, id: `${file}_${i}` }));
  }

  const texts = allChunks.map(c => c.text);
  const embeddings = [];
  for (let i = 0; i < texts.length; i += 100) {
    const data = await openAiPost('/embeddings', { model: EMBEDDING_MODEL, input: texts.slice(i, i+100) });
    embeddings.push(...data.data.map(d => d.embedding));
  }
  allChunks.forEach((c, i) => { c.embedding = embeddings[i]; });

  cache.sig = sig; cache.chunks = allChunks;
  return allChunks;
}

// ── handler ───────────────────────────────────────────────────────────────────

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).end();

  const { messages } = req.body;
  if (!messages) return res.status(400).json({ error: 'No messages' });

  try {
    const chunks = await getChunks();
    let systemPrompt;
    let sources = [];

    if (chunks.length > 0) {
      // RAG mode — retrieve relevant chunks
      const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')?.content || '';
      const qData = await openAiPost('/embeddings', { model: EMBEDDING_MODEL, input: [lastUserMsg] });
      const qEmb = qData.data[0].embedding;

      const scored = chunks
        .map(c => ({ ...c, score: cosineSimilarity(qEmb, c.embedding) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 4)
        .filter(c => c.score > 0.1);

      const context = scored.length
        ? scored.map((c, i) => `[${i+1}] ${c.title}:\n${c.text}`).join('\n\n')
        : 'No relevant context found.';

      sources = scored.map(c => c.title).filter((v, i, a) => a.indexOf(v) === i);

      systemPrompt = `You are a sharp assistant displayed on a TV screen. Answer using the provided context when relevant. Be concise — 1-3 sentences max.\n\nContext:\n${context}`;
    } else {
      // No docs — normal chat
      systemPrompt = 'You are a sharp, witty assistant displayed on a TV screen. Keep replies concise — 1-3 sentences max. Be direct and interesting.';
    }

    const completion = await openAiPost('/chat/completions', {
      model: CHAT_MODEL,
      messages: [{ role: 'system', content: systemPrompt }, ...messages],
      max_tokens: 150
    });

    res.status(200).json({
      reply: completion.choices[0].message.content,
      sources
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
