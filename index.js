const express = require('express');
const cors = require('cors');
// near the top of your backend file
const { InferenceClient } = require('@huggingface/inference');
const hfClient = new InferenceClient(process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN);
const admin = require('firebase-admin');
const fetch = require('node-fetch').default;
const fs = require('fs');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

//  Firebase Admin Initialization
let serviceAccount;

if (process.env.FIREBASE_SERVICE_ACCOUNT_PATH) {
  serviceAccount = require(process.env.FIREBASE_SERVICE_ACCOUNT_PATH);
} else if (process.env.FIREBASE_SERVICE_ACCOUNT) {
  try {
    serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  } catch (err) {
    console.error('Failed to parse FIREBASE_SERVICE_ACCOUNT:', err);
    process.exit(1);
  }
} else {
  try {
    serviceAccount = require('./serviceAccountKey.json');
  } catch (err) {
    console.error('Firebase service account not found:', err);
    process.exit(1);
  }
}

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

//  Hugging Face API setup
const HF_API_KEY = process.env.HUGGINGFACE_API_KEY;
const HF_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2:featherless-ai';

if (!HF_API_KEY) {
  console.error('HUGGINGFACE_API_KEY is not set.');
  process.exit(1);
}

//  Auth middleware
async function authMiddleware(req, res, next) {
  try {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.substring(7) : null;

    if (!token) return res.status(401).json({ error: 'No token provided' });

    const decoded = await admin.auth().verifyIdToken(token);
    req.user = decoded;
    next();
  } catch (err) {
    console.error('Auth error:', err);
    return res.status(401).json({ error: 'Invalid token' });
  }
}

//  Health check
app.get('/', (req, res) => {
  res.send('Mistral-Small AI backend running.');
});

// Call Hugging Face Inference API// Use the model you showed from the docs:
async function callMistral(prompt) {
  const completion = await hfClient.chatCompletion({
    model: HF_MODEL,
    messages: [
      {
        role: 'system',
        content:
          'You are a helpful support assistant for the GharAssist mobile app. Answer briefly and clearly.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
  });

  const choice = completion.choices?.[0];
  const content = choice?.message?.content;

  if (!content || !content.toString().trim()) {
    throw new Error('HF chat completion returned empty content');
  }

  return content.toString();
}
// AI support endpoint
app.post('/ai/support/ask', authMiddleware, async (req, res) => {
  try {
    const uid = req.user.uid;
    const { message } = req.body;

    if (!message) return res.status(400).json({ error: 'Message required' });

    // Call Mistral-Small
    const reply = await callMistral(message);

    // Store conversation in Firestore
    const convRef = admin.firestore().collection('support_conversations').doc();
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
    console.error('Support endpoint error:', err.message);
    return res.status(500).json({ error: err.message });
  }
});

//  Start server
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => console.log(`Mistral backend listening on port ${PORT}`));