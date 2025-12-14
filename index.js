const express = require('express');
const cors = require('cors');
const { InferenceClient } = require('@huggingface/inference');
const admin = require('firebase-admin');
const fetch = require('node-fetch').default;
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

// -------------------- Firebase Admin Initialization --------------------

let serviceAccount;

if (process.env.FIREBASE_SERVICE_ACCOUNT_PATH) {
  // Load from JSON file path, e.g. ./serviceAccountKey.json
  // eslint-disable-next-line global-require, import/no-dynamic-require
  serviceAccount = require(process.env.FIREBASE_SERVICE_ACCOUNT_PATH);
} else if (process.env.FIREBASE_SERVICE_ACCOUNT) {
  // Load from JSON string in env
  try {
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  } catch (err) {
    console.error('Failed to parse FIREBASE_SERVICE_ACCOUNT:', err);
    process.exit(1);
  }
} else {
  // Fallback to local file
  try {
    // eslint-disable-next-line global-require
    serviceAccount = require('./serviceAccountKey.json');
  } catch (err) {
    console.error('Firebase service account not found:', err);
    process.exit(1);
  }
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

// -------------------- Hugging Face Setup --------------------

const HF_API_KEY = process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN;
if (!HF_API_KEY) {
  console.error('HUGGINGFACE_API_KEY (or HF_TOKEN) is not set.');
  process.exit(1);
}

const hfClient = new InferenceClient(HF_API_KEY);

// Main chat / reasoning model (e.g. Llama 3 instruct)
const HF_CHAT_MODEL = process.env.HF_CHAT_MODEL;
if (!HF_CHAT_MODEL) {
  console.error('HF_CHAT_MODEL is not set.');
  process.exit(1);
}

// Translation models (English <-> Urdu)
const HF_TRANSLATION_MODEL_EN_UR = process.env.HF_TRANSLATION_MODEL_EN_UR;
const HF_TRANSLATION_MODEL_UR_EN = process.env.HF_TRANSLATION_MODEL_UR_EN;

// Image captioning model
const HF_IMAGE_MODEL = process.env.HF_IMAGE_MODEL;

// Sentiment model (optional, with chat fallback)
const HF_SENTIMENT_MODEL = process.env.HF_SENTIMENT_MODEL;

// -------------------- Auth & Admin Middleware --------------------

async function authMiddleware(req, res, next) {
  try {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ')
      ? authHeader.substring(7)
      : null;

    if (!token) {
      return res.status(401).json({ error: 'No token provided' });
    }

    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    return next();
  } catch (err) {
    console.error('Auth error:', err);
    return res.status(401).json({ error: 'Invalid token' });
  }
}

async function requireAdminRole(req, res, next) {
  try {
    const uid = req.user.uid;
    const snap = await db.collection('users').doc(uid).get();
    const data = snap.data();
    if (!snap.exists || !data || data.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required' });
    }
    return next();
  } catch (err) {
    console.error('Admin check error:', err);
    return res.status(500).json({ error: 'Failed to verify admin role' });
  }
}

// -------------------- Helper Functions --------------------

// Generic chat helper (Llama 3 / instruct model)
async function callChatModel(systemPrompt, userContent) {
  const completion = await hfClient.chatCompletion({
    model: HF_CHAT_MODEL,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userContent },
    ],
  });

  const choice = completion.choices?.[0];
  let content = choice?.message?.content;

  // content can be a string or array of parts; normalize to string
  if (Array.isArray(content)) {
    content = content
      .map((part) =>
        typeof part === 'string' ? part : part?.text ?? '',
      )
      .join('');
  }

  if (!content || !content.toString().trim()) {
    throw new Error('Chat model returned empty content');
  }

  return content.toString();
}

// Translation helper for Marian-style models via raw Inference API
async function callTranslationModel(modelName, text) {
  if (!modelName) {
    throw new Error('Translation model not configured');
  }

  const resp = await fetch(
    `https://api-inference.huggingface.co/models/${modelName}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs: text }),
    },
  );

  if (!resp.ok) {
    const body = await resp.text();
    console.error('HF translation error:', resp.status, body);
    throw new Error('HF translation request failed');
  }

  const data = await resp.json();
  // Marian models typically return: [{ translation_text: '...' }]
  const translation =
    Array.isArray(data) && data[0]?.translation_text
      ? data[0].translation_text
      : text;

  return translation;
}

// Choose translation model for English <-> Urdu
function chooseTranslationModel(sourceLang, targetLang) {
  const src = (sourceLang || '').toLowerCase();
  const tgt = (targetLang || '').toLowerCase();

  if (src === 'en' && tgt === 'ur') {
    return HF_TRANSLATION_MODEL_EN_UR;
  }
  if (src === 'ur' && tgt === 'en') {
    return HF_TRANSLATION_MODEL_UR_EN;
  }

  return null;
}

// Image captioning helper
async function callImageCaptionModel(imageUrl) {
  if (!HF_IMAGE_MODEL) {
    throw new Error('HF_IMAGE_MODEL not configured');
  }

  const imgResp = await fetch(imageUrl);
  if (!imgResp.ok) {
    throw new Error('Could not fetch image from URL');
  }
  const imageBuffer = await imgResp.arrayBuffer();

  const hfResp = await fetch(
    `https://api-inference.huggingface.co/models/${HF_IMAGE_MODEL}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/octet-stream',
      },
      body: Buffer.from(imageBuffer),
    },
  );

  if (!hfResp.ok) {
    const body = await hfResp.text();
    console.error('HF image error:', hfResp.status, body);
    throw new Error('HF image inference failed');
  }

  const result = await hfResp.json();
  // Many caption models: [{ generated_text: '...' }]
  const caption =
    Array.isArray(result) && result[0]?.generated_text
      ? result[0].generated_text
      : JSON.stringify(result);

  return caption;
}

// Sentiment helper using a classifier model
async function callSentimentModel(text) {
  if (!HF_SENTIMENT_MODEL) {
    throw new Error('HF_SENTIMENT_MODEL not configured');
  }

  const resp = await fetch(
    `https://api-inference.huggingface.co/models/${HF_SENTIMENT_MODEL}`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${HF_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ inputs: text }),
    },
  );

  if (!resp.ok) {
    const body = await resp.text();
    console.error('HF sentiment error:', resp.status, body);
    throw new Error('HF sentiment request failed');
  }

  const data = await resp.json();

  // Typical format: [[{label, score}, ...]] or [{label, score}, ...]
  let candidates;
  if (Array.isArray(data) && Array.isArray(data[0])) {
    candidates = data[0];
  } else if (Array.isArray(data)) {
    candidates = data;
  } else {
    throw new Error('Unexpected sentiment response format');
  }

  let best = candidates[0];
  for (const c of candidates) {
    if (c.score > best.score) best = c;
  }

  const sentiment = (best.label || '').toLowerCase();
  const confidence = best.score ?? 0;

  return { sentiment, confidence };
}

// -------------------- Health Check --------------------

app.get('/', (req, res) => {
  res.send('AI backend for GharAssist is running.');
});

// -------------------- AI Endpoints (User-facing) --------------------

// 1) Support / general chat
const SUPPORT_SYSTEM_PROMPT =
  'You are a helpful support assistant for the GharAssist app. ' +
  'Users may speak English, Urdu, or Roman Urdu. ' +
  'Reply briefly, clearly, and in the same language the user mostly used.';

app.post('/ai/support/ask', authMiddleware, async (req, res) => {
  try {
    const uid = req.user.uid;
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: 'Message required' });
    }

    const reply = await callChatModel(SUPPORT_SYSTEM_PROMPT, message);

    // Store a minimal conversation record in Firestore
    const convRef = db.collection('support_conversations').doc();

    await convRef.set({
      userId: uid,
      lastMessage: reply,
      createdAt: admin.firestore.FieldValue.serverTimestamp(),
      updatedAt: admin.firestore.FieldValue.serverTimestamp(),
    });

    await convRef.collection('messages').add({
      sender: 'user',
      text: message,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    });

    await convRef.collection('messages').add({
      sender: 'ai',
      text: reply,
      timestamp: admin.firestore.FieldValue.serverTimestamp(),
    });

    return res.json({ threadId: convRef.id, reply });
  } catch (err) {
    console.error('Support endpoint error:', err);
    return res.status(500).json({ error: err.message });
  }
});

// 2) Text translation (English <-> Urdu with Marian, fallback to chat)
app.post('/ai/text/translate', authMiddleware, async (req, res) => {
  try {
    const { text, sourceLang, targetLang } = req.body;
    if (!text || !targetLang) {
      return res
        .status(400)
        .json({ error: 'text and targetLang are required' });
    }

    const modelName = chooseTranslationModel(sourceLang, targetLang);

    // If we have a dedicated translation model, use it
    if (modelName) {
      const translation = await callTranslationModel(modelName, text);
      return res.json({ translation, usedModel: modelName });
    }

    // Fallback: use chat model to translate
    const systemPrompt = `
You are a translation assistant for a home-services app.
Users may write in English, Urdu, or Roman Urdu.
Translate the text into ${targetLang}.
Reply with ONLY the translated text, no explanations.
`;
    const translation = await callChatModel(systemPrompt, text);
    return res.json({ translation, usedModel: HF_CHAT_MODEL });
  } catch (err) {
    console.error('Translate endpoint error:', err);
    return res.status(500).json({ error: 'Translation failed' });
  }
});

// 3) Summarization (using chat model)
app.post('/ai/text/summarize', authMiddleware, async (req, res) => {
  try {
    const { text, maxSentences = 3 } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }

    const systemPrompt = `
You are a summarization assistant for the GharAssist app.
Users may write in English, Urdu, or Roman Urdu.
Summarize the given text in ${maxSentences} short sentences or less.
Use simple, clear language.
Reply in the same main language the user used.
Return ONLY the summary text.
`;
    const summary = await callChatModel(systemPrompt, text);
    return res.json({ summary });
  } catch (err) {
    console.error('Summarize endpoint error:', err);
    return res.status(500).json({ error: 'Summarization failed' });
  }
});

// 4) Sentiment analysis – prefer dedicated model, fallback to chat model
app.post('/ai/text/sentiment', authMiddleware, async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }

    // If we have a dedicated sentiment model, use it
    if (HF_SENTIMENT_MODEL) {
      try {
        const result = await callSentimentModel(text);
        return res.json(result);
      } catch (e) {
        console.warn(
          'Sentiment model failed, falling back to chat model:',
          e,
        );
        // fall through to chat-based fallback
      }
    }

    // Fallback: use chat model with JSON response
    const systemPrompt = `
You are a sentiment analysis engine for customer support messages.
Text may be in English, Urdu, or Roman Urdu.
Classify overall sentiment as one of: "positive", "neutral", "negative".
Respond with a single JSON object exactly like:
{"sentiment":"positive","confidence":0.92}
Do NOT include any extra text, commentary, or markdown.
`;
    const raw = await callChatModel(systemPrompt, text);

    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      console.warn(
        'Failed to parse sentiment JSON from chat model, raw =',
        raw,
      );
      return res.json({ sentiment: 'unknown', confidence: 0.0 });
    }

    return res.json(parsed);
  } catch (err) {
    console.error('Sentiment endpoint error:', err);
    return res.status(500).json({ error: 'Sentiment analysis failed' });
  }
});

// 5) Image captioning (problem photos)
app.post('/ai/image/caption', authMiddleware, async (req, res) => {
  try {
    const { imageUrl } = req.body;
    if (!imageUrl) {
      return res.status(400).json({ error: 'imageUrl is required' });
    }

    if (!HF_IMAGE_MODEL) {
      return res.status(500).json({ error: 'HF_IMAGE_MODEL not configured' });
    }

    const caption = await callImageCaptionModel(imageUrl);
    return res.json({ caption });
  } catch (err) {
    console.error('Image caption endpoint error:', err);
    return res.status(500).json({ error: 'Image captioning failed' });
  }
});

// 6) Combined text analysis (summary + sentiment) – useful for admin on text
app.post('/ai/text/analyze', authMiddleware, async (req, res) => {
  try {
    const { text, maxSentences = 3 } = req.body;
    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }

    // Get summary (chat model)
    const summaryPrompt = `
You are a summarization assistant for the GharAssist admin panel.
Users may write in English, Urdu, or Roman Urdu.
Summarize the given text in ${maxSentences} short sentences or less.
Focus on what the user is complaining about or asking for.
Reply in the same main language the user used.
Return ONLY the summary text.
`;
    const summary = await callChatModel(summaryPrompt, text);

    // Get sentiment (dedicated model if available, else chat fallback)
    let sentimentResult;
    if (HF_SENTIMENT_MODEL) {
      try {
        sentimentResult = await callSentimentModel(text);
      } catch (e) {
        console.warn('Sentiment model failed in /ai/text/analyze:', e);
      }
    }
    if (!sentimentResult) {
      const systemPrompt = `
You are a sentiment analysis engine for customer support messages.
Text may be in English, Urdu, or Roman Urdu.
Classify overall sentiment as one of: "positive", "neutral", "negative".
Respond with a single JSON object exactly like:
{"sentiment":"positive","confidence":0.92}
Do NOT include any extra text, commentary, or markdown.
`;
      const raw = await callChatModel(systemPrompt, text);
      try {
        sentimentResult = JSON.parse(raw);
      } catch {
        sentimentResult = { sentiment: 'unknown', confidence: 0.0 };
      }
    }

    return res.json({
      summary,
      sentiment: sentimentResult.sentiment,
      confidence: sentimentResult.confidence,
    });
  } catch (err) {
    console.error('Analyze endpoint error:', err);
    return res.status(500).json({ error: 'Text analysis failed' });
  }
});

// -------------------- Admin Analytics Endpoints --------------------

// Numeric metrics over users and bookings
app.get(
  '/admin/analytics/metrics',
  authMiddleware,
  requireAdminRole,
  async (req, res) => {
    try {
      const usersSnap = await db.collection('users').get();
      const bookingsSnap = await db.collection('bookings').get();

      let totalUsers = 0;
      let totalCustomers = 0;
      let totalWorkers = 0;
      let totalAdmins = 0;

      usersSnap.forEach((doc) => {
        totalUsers += 1;
        const data = doc.data() || {};
        const role = (data.role || 'customer').toString();
        if (role === 'customer') totalCustomers += 1;
        else if (role === 'provider') totalWorkers += 1;
        else if (role === 'admin') totalAdmins += 1;
      });

      let totalBookings = 0;
      const bookingsByStatus = {};
      let totalRevenue = 0;

      bookingsSnap.forEach((doc) => {
        totalBookings += 1;
        const data = doc.data() || {};
        const status = (data.status || 'unknown').toString();
        bookingsByStatus[status] = (bookingsByStatus[status] || 0) + 1;

        const price = Number(data.price || 0);
        if (!Number.isNaN(price)) {
          totalRevenue += price;
        }
      });

      const metrics = {
        users: {
          total: totalUsers,
          customers: totalCustomers,
          workers: totalWorkers,
          admins: totalAdmins,
        },
        bookings: {
          total: totalBookings,
          byStatus: bookingsByStatus,
          totalRevenue,
        },
      };

      return res.json({ metrics });
    } catch (err) {
      console.error('/admin/analytics/metrics error:', err);
      return res
        .status(500)
        .json({ error: 'Failed to load analytics metrics' });
    }
  },
);

// AI explanation of analytics metrics for admin
app.post(
  '/ai/analytics/explain',
  authMiddleware,
  requireAdminRole,
  async (req, res) => {
    try {
      const { metrics } = req.body;
      if (!metrics) {
        return res.status(400).json({ error: 'metrics object is required' });
      }

      const metricsJson = JSON.stringify(metrics, null, 2);

      const systemPrompt = `
You are an analytics assistant for the GharAssist admin panel.
You receive JSON with metrics about users, workers, and bookings.
Explain briefly what is happening in this data:
- highlight important trends (e.g. many cancelled bookings, high revenue)
- mention worker vs customer balance if relevant
- keep it short (3-5 bullet points)
Use simple language; if field names are English, answer in English.
`;
      const userContent = `Here is the current analytics JSON:\n\n${metricsJson}`;

      const explanation = await callChatModel(systemPrompt, userContent);

      return res.json({ explanation });
    } catch (err) {
      console.error('/ai/analytics/explain error:', err);
      return res
        .status(500)
        .json({ error: 'Failed to generate analytics explanation' });
    }
  },
);

// -------------------- Start Server --------------------

const PORT = process.env.PORT || 8080;
app.listen(PORT, () =>
  console.log(`AI backend listening on port ${PORT}`),
);