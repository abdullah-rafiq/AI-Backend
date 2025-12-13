// index.js
const express = require('express');
const cors = require('cors');
const admin = require('firebase-admin');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// 1) Firebase Admin initialization
let serviceAccount;
if (process.env.FIREBASE_SERVICE_ACCOUNT) {
  // In production: JSON string from env var
  try {
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  } catch (e) {
    console.error('Failed to parse FIREBASE_SERVICE_ACCOUNT env var:', e);
    process.exit(1);
  }
} else {
  // In local dev: load from file
  // Make sure serviceAccountKey.json exists next to this file
  // and comes from Firebase Console -> Service accounts -> Generate new private key
  // eslint-disable-next-line global-require
  serviceAccount = require('./serviceAccountKey.json');
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

// 2) Gemini client initialization
const geminiApiKey = process.env.GEMINI_API_KEY;
if (!geminiApiKey) {
  console.error('GEMINI_API_KEY is not set in environment variables.');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(geminiApiKey);
// Use a fast model for support assistant
const supportModel = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });

const app = express();
app.use(cors());
app.use(express.json());

// PUBLIC health routes (no auth required)
app.get('/', (req, res) => {
  res.send('AI support backend is running.');
});

app.get('/healthz', (req, res) => {
  res.status(200).send('ok');
});

// Middleware to verify Firebase ID token (for all routes below)
async function authMiddleware(req, res, next) {
  try {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ')
      ? authHeader.substring(7)
      : null;

    if (!token) return res.status(401).json({ error: 'No token provided' });

    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded; // { uid, email, ... }
    next();
  } catch (err) {
    console.error('Auth error:', err);
    return res.status(401).json({ error: 'Invalid token' });
  }
}

app.use(authMiddleware);

// Helper: call Gemini with a prompt
async function callLlmForSupport(prompt) {
  const result = await supportModel.generateContent(prompt);
  return result.response.text();
}

// AI support endpoint (requires Authorization: Bearer <idToken>)
app.post('/ai/support/ask', async (req, res) => {
  try {
    const uid = req.user.uid;
    const { message, role = 'customer', language = 'en', threadId } = req.body;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'message required' });
    }

    // 1) Load user profile
    const userDoc = await admin.firestore().doc(`users/${uid}`).get();
    const user = userDoc.data() || {};

    // 2) Load recent bookings for context (defensive: if no index, just skip)
    let bookings = [];
    try {
      const bookingsSnap = await admin
        .firestore()
        .collection('bookings')
        .where(role === 'customer' ? 'customerId' : 'providerId', '==', uid)
        .orderBy('createdAt', 'desc')
        .limit(3)
        .get();

      bookings = bookingsSnap.docs.map((d) => ({ id: d.id, ...d.data() }));
    } catch (e) {
      console.warn('Bookings query failed (maybe missing index):', e.message);
    }

    // 3) Build prompt for Gemini
    const prompt = `
You are a helpful, concise support assistant for a home services app.
- Answer briefly and clearly.
- If you refer to bookings, only use the data I provide.
- If something is unclear, ask the user a short follow-up question.

User role: ${role}
Preferred language: ${language}

User profile (partial JSON):
${JSON.stringify(user, null, 2)}

Recent bookings (JSON, newest first):
${JSON.stringify(bookings, null, 2)}

User message:
${message}
`;

    // 4) Call Gemini
    const reply = await callLlmForSupport(prompt);

    // 5) Store conversation in Firestore
    const convId =
      threadId || admin.firestore().collection('support_conversations').doc().id;
    const convRef = admin.firestore().doc(`support_conversations/${convId}`);

    await convRef.set(
      {
        userId: uid,
        role,
        lastMessage: reply,
        updatedAt: admin.firestore.FieldValue.serverTimestamp(),
        createdAt: admin.firestore.FieldValue.serverTimestamp(),
      },
      { merge: true },
    );

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

    return res.json({
      threadId: convId,
      reply,
      suggestedFollowUps: [],
    });
  } catch (err) {
    console.error('Support endpoint error:', err);
    return res.status(500).json({ error: 'Server error' });
  }
});

const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`AI backend listening on port ${PORT}`);
});