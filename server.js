require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const OpenAI = require("openai");
const sharp = require("sharp");
const fs = require("fs");
const crypto = require("crypto");
const bcrypt = require("bcryptjs");
const app = express();
const PORT = process.env.PORT || 3000;

function getClient() {
  return new OpenAI({
    apiKey: process.env.GROQ_API_KEY || "missing",
    baseURL: "https://api.groq.com/openai/v1",
  });
}

const VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";
const CHAT_MODEL = "llama-3.3-70b-versatile";

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    allowed.includes(file.mimetype) ? cb(null, true) : cb(new Error("Images only"));
  },
});

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "10mb" }));

async function resizeImage(buffer) {
  try {
    return await sharp(buffer)
      .resize(1200, 1200, { fit: "inside", withoutEnlargement: true })
      .jpeg({ quality: 85 })
      .toBuffer();
  } catch {
    return buffer;
  }
}

// ── Data Store ──
const DATA_DIR = path.join(__dirname, "data");
const USERS_FILE = path.join(DATA_DIR, "users.json");
const ANALYTICS_FILE = path.join(DATA_DIR, "analytics.json");
const SESSIONS_FILE = path.join(DATA_DIR, "sessions.json");

function loadUsers() {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(USERS_FILE)) return [];
    return JSON.parse(fs.readFileSync(USERS_FILE, "utf8"));
  } catch { return []; }
}

function saveUsers(users) {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2));
}

function loadAnalytics() {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(ANALYTICS_FILE)) return { daily: {} };
    return JSON.parse(fs.readFileSync(ANALYTICS_FILE, "utf8"));
  } catch { return { daily: {} }; }
}

function saveAnalytics(data) {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.writeFileSync(ANALYTICS_FILE, JSON.stringify(data, null, 2));
}

function loadSessions() {
  try {
    if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
    if (!fs.existsSync(SESSIONS_FILE)) return [];
    return JSON.parse(fs.readFileSync(SESSIONS_FILE, "utf8"));
  } catch { return []; }
}

function saveSessions(sessions) {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  // Keep only last 500 entries to prevent unbounded growth
  const trimmed = sessions.length > 500 ? sessions.slice(-500) : sessions;
  fs.writeFileSync(SESSIONS_FILE, JSON.stringify(trimmed, null, 2));
}

function trackEvent(type, ip) {
  try {
    const today = new Date().toISOString().slice(0, 10);
    const data = loadAnalytics();
    if (!data.daily[today]) data.daily[today] = { analyses: 0, chats: 0, telegramSends: 0, ips: [] };
    const day = data.daily[today];
    if (type === "analyze") day.analyses++;
    else if (type === "chat") day.chats++;
    else if (type === "telegram") day.telegramSends++;
    const cleanIp = (ip || "").replace("::ffff:", "");
    if (cleanIp && !day.ips.includes(cleanIp)) day.ips.push(cleanIp);
    saveAnalytics(data);
  } catch {}
}

function generateToken() {
  return crypto.randomBytes(32).toString("hex");
}

function authMiddleware(req, res, next) {
  const token = (req.headers["authorization"] || "").replace("Bearer ", "");
  if (!token) return res.status(401).json({ error: "Unauthorized" });
  const users = loadUsers();
  const user = users.find(u => u.token === token);
  if (!user) return res.status(401).json({ error: "Unauthorized" });
  req.user = user;
  next();
}

// ── AI Prompts ──
const ANALYZE_SYSTEM = `You are MedExplain AI, a friendly medical assistant that explains lab test results in plain, everyday English.

Your job:
1. Read EVERY test and result visible across ALL provided images
2. Explain what each test is in simple words
3. Explain what the result means (normal, abnormal, high, low, positive, negative)
4. Use warm, clear, non-scary language
5. Group related tests (e.g. Blood Count, Liver Panel, Infection Screen)
6. For pregnancy tests (NIPT, amniocentesis): clearly state sex if shown

CRITICAL RULES — YOU MUST FOLLOW THESE EXACTLY:
- ONLY include a test if it has a REAL patient result (a number, a value like "Negative", "Positive", "Reactive", "Non-Reactive", a percentage, etc.)
- If a test result field is blank, empty, dashes (—), N/A, or not filled in — DO NOT include that test at all. Skip it completely. Do not mention it.
- NEVER write "Not provided", "Not available", or any similar phrase. If there is no result, the test must be completely absent from your response.
- HCV = Hepatitis C Virus (NOT AIDS). HIV = Human Immunodeficiency Virus. Never mix these up.
- If a result is outside range, explain simply but say a doctor should review it
- End with a reminder to consult a doctor

FORMAT:
---
## 🔬 Your Lab Results Explained

### [Plain English Name] — [Abbreviation]
**What this tests:** ...
**Your result:** ...
**Reference range:** ... (if visible)
**What it means:** ...

---
⚠️ *For informational purposes only. Please discuss with your doctor.*`;

const CHAT_SYSTEM = `You are MedExplain AI, a friendly medical assistant helping a patient understand their lab results.

The patient's lab analysis has already been done and is provided as context. Your job is to answer their follow-up questions clearly and simply.

RULES:
- Answer in plain English, no medical jargon
- Be warm and reassuring but honest
- Only discuss tests that have REAL results in the provided analysis — never mention tests that were not done or have no result
- For questions about diagnosis, treatment, or medication: ALWAYS say "I'm not able to give medical advice on that — please speak with your doctor about this"
- For questions you're not confident about: say "That's a great question for your doctor — I'd recommend asking them directly"
- Never guess at a diagnosis
- Keep answers focused and concise`;

// ── Register ──
app.post("/api/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;
    if (!username || !email || !password)
      return res.status(400).json({ error: "All fields are required." });
    if (username.trim().length < 2)
      return res.status(400).json({ error: "Username must be at least 2 characters." });
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      return res.status(400).json({ error: "Please enter a valid email address." });
    if (password.length < 6)
      return res.status(400).json({ error: "Password must be at least 6 characters." });

    const users = loadUsers();
    if (users.find(u => u.email === email.toLowerCase().trim()))
      return res.status(409).json({ error: "This email is already registered. Please login." });

    const hash = await bcrypt.hash(password, 10);
    const token = generateToken();
    const user = {
      id: Date.now().toString(),
      username: username.trim(),
      email: email.toLowerCase().trim(),
      password: hash,
      token,
      createdAt: new Date().toISOString(),
      telegramBotToken: "",
      telegramChatId: "",
    };
    users.push(user);
    saveUsers(users);

    res.json({ token, username: user.username, email: user.email, hasTelegram: false });
  } catch (err) {
    console.error("Register error:", err);
    res.status(500).json({ error: "Registration failed. Please try again." });
  }
});

// ── Login ──
app.post("/api/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ error: "Email and password are required." });

    const users = loadUsers();
    const user = users.find(u => u.email === email.toLowerCase().trim());
    if (!user) return res.status(401).json({ error: "Invalid email or password." });

    const ok = await bcrypt.compare(password, user.password);
    if (!ok) return res.status(401).json({ error: "Invalid email or password." });

    user.token = generateToken();
    saveUsers(users);

    res.json({
      token: user.token,
      username: user.username,
      email: user.email,
      hasTelegram: !!(user.telegramBotToken && user.telegramChatId),
      telegramChatId: user.telegramChatId,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Login failed. Please try again." });
  }
});

// ── Telegram Settings ──
app.post("/api/user/telegram", authMiddleware, (req, res) => {
  try {
    const { telegramBotToken, telegramChatId } = req.body;
    if (!telegramBotToken || !telegramChatId)
      return res.status(400).json({ error: "Both Bot Token and Chat ID are required." });

    const users = loadUsers();
    const user = users.find(u => u.id === req.user.id);
    if (!user) return res.status(404).json({ error: "User not found." });

    user.telegramBotToken = telegramBotToken.trim();
    user.telegramChatId = telegramChatId.trim();
    saveUsers(users);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: "Failed to save settings." });
  }
});

app.get("/api/user/telegram", authMiddleware, (req, res) => {
  res.json({
    hasTelegram: !!(req.user.telegramBotToken && req.user.telegramChatId),
    telegramChatId: req.user.telegramChatId || "",
  });
});

// ── Telegram Send ──
app.post("/api/telegram/send", authMiddleware, async (req, res) => {
  try {
    const { message, telegramBotToken, telegramChatId } = req.body;

    const users = loadUsers();
    const user = users.find(u => u.id === req.user.id);
    if (!user) return res.status(404).json({ error: "User not found." });

    if (telegramBotToken) user.telegramBotToken = telegramBotToken.trim();
    if (telegramChatId) user.telegramChatId = telegramChatId.trim();
    if (telegramBotToken || telegramChatId) saveUsers(users);

    const botToken = user.telegramBotToken;
    const chatId = user.telegramChatId;

    if (!botToken || !chatId)
      return res.status(400).json({ error: "Telegram Bot Token and Chat ID are required." });
    if (!message)
      return res.status(400).json({ error: "No message to send." });

    const header = `🔬 MedExplain AI — Your Lab Results\n${"─".repeat(35)}\n\n`;
    const fullMsg = header + message;
    const MAX_LEN = 4000;
    const chunks = [];
    for (let i = 0; i < fullMsg.length; i += MAX_LEN) {
      chunks.push(fullMsg.slice(i, i + MAX_LEN));
    }

    for (const chunk of chunks) {
      const tgRes = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: chatId, text: chunk }),
      });
      const result = await tgRes.json();
      if (!result.ok) {
        const desc = result.description || "Telegram API error";
        return res.status(400).json({
          error: desc.includes("bot was blocked") ? "Bot was blocked by the user. Send a message to your bot first." :
                 desc.includes("chat not found") ? "Chat ID not found. Make sure you've started a conversation with your bot." :
                 desc.includes("Unauthorized") ? "Invalid Bot Token." : desc,
        });
      }
    }
    trackEvent("telegram", req.ip);
    res.json({ success: true });
  } catch (err) {
    console.error("Telegram send error:", err);
    res.status(500).json({ error: "Failed to send to Telegram. Please try again." });
  }
});

// ── Telegram Send (guest — no auth required) ──
app.post("/api/telegram/send-guest", async (req, res) => {
  try {
    const { message, telegramBotToken, telegramChatId } = req.body;
    if (!telegramBotToken || !telegramChatId)
      return res.status(400).json({ error: "Bot Token and Chat ID are required." });
    if (!message)
      return res.status(400).json({ error: "No message to send." });

    const header = `🔬 MedExplain AI — Your Lab Results\n${"─".repeat(35)}\n\n`;
    const fullMsg = header + message;
    const MAX_LEN = 4000;
    for (let i = 0; i < fullMsg.length; i += MAX_LEN) {
      const chunk = fullMsg.slice(i, i + MAX_LEN);
      const tgRes = await fetch(`https://api.telegram.org/bot${telegramBotToken}/sendMessage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ chat_id: telegramChatId, text: chunk }),
      });
      const result = await tgRes.json();
      if (!result.ok) {
        const desc = result.description || "Telegram API error";
        return res.status(400).json({
          error: desc.includes("bot was blocked") ? "Bot was blocked by the user. Send a message to your bot first." :
                 desc.includes("chat not found") ? "Chat ID not found. Make sure you've started a conversation with your bot." :
                 desc.includes("Unauthorized") ? "Invalid Bot Token." : desc,
        });
      }
    }
    trackEvent("telegram", req.ip);
    res.json({ success: true });
  } catch (err) {
    console.error("Guest telegram error:", err);
    res.status(500).json({ error: "Failed to send to Telegram. Please try again." });
  }
});

// ── Admin Routes ──
app.get("/admin", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "admin.html"));
});

app.get("/api/admin/stats", (req, res) => {
  const key = req.query.key || req.headers["x-admin-key"];
  const adminKey = process.env.ADMIN_KEY;
  if (!adminKey || key !== adminKey)
    return res.status(401).json({ error: "Unauthorized" });

  const users = loadUsers();
  const analytics = loadAnalytics();

  // Compute totals and last 30 days
  const allDays = Object.keys(analytics.daily).sort();
  let totalAnalyses = 0, totalChats = 0, totalTelegramSends = 0;
  const allIPs = new Set();
  allDays.forEach(d => {
    const day = analytics.daily[d];
    totalAnalyses += day.analyses || 0;
    totalChats += day.chats || 0;
    totalTelegramSends += day.telegramSends || 0;
    (day.ips || []).forEach(ip => allIPs.add(ip));
  });

  // Last 14 days chart data
  const last14 = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key2 = d.toISOString().slice(0, 10);
    const day = analytics.daily[key2] || {};
    last14.push({
      date: key2,
      analyses: day.analyses || 0,
      chats: day.chats || 0,
      telegramSends: day.telegramSends || 0,
      visitors: (day.ips || []).length,
    });
  }

  res.json({
    users: {
      total: users.length,
      withTelegram: users.filter(u => !!(u.telegramBotToken && u.telegramChatId)).length,
      list: users.map(u => ({
        username: u.username,
        email: u.email,
        createdAt: u.createdAt || null,
        hasTelegram: !!(u.telegramBotToken && u.telegramChatId),
      })),
    },
    analytics: {
      totalAnalyses,
      totalChats,
      totalTelegramSends,
      totalUniqueVisitors: allIPs.size,
      last14,
    },
  });
});

app.get("/api/admin/sessions", (req, res) => {
  const key = req.query.key || req.headers["x-admin-key"];
  const adminKey = process.env.ADMIN_KEY;
  if (!adminKey || key !== adminKey)
    return res.status(401).json({ error: "Unauthorized" });

  const sessions = loadSessions();
  // Return last 200 sessions, newest first
  res.json({ sessions: sessions.slice(-200).reverse() });
});

// ── Analyze ──
app.post("/api/analyze", upload.array("labImages", 5), async (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: "Please upload at least one image." });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("Access-Control-Allow-Origin", "*");

  try {
    const resized = await Promise.all(files.map(f => resizeImage(f.buffer)));

    const content = resized.map(buf => ({
      type: "image_url",
      image_url: {
        url: `data:image/jpeg;base64,${buf.toString("base64")}`,
      },
    }));

    content.push({
      type: "text",
      text: `These are ${files.length} page(s) of lab results. Please explain ALL results shown across every image in plain English. Tell me what each test is, what my result means, and whether it looks normal.`,
    });

    const stream = await getClient().chat.completions.create({
      model: VISION_MODEL,
      max_tokens: 6000,
      stream: true,
      messages: [
        { role: "system", content: ANALYZE_SYSTEM },
        { role: "user", content },
      ],
    });

    let fullAnalysis = "";
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content;
      if (text) {
        fullAnalysis += text;
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    // Save analysis session
    try {
      const sessions = loadSessions();
      sessions.push({
        id: Date.now().toString(),
        type: "analysis",
        timestamp: new Date().toISOString(),
        ip: (req.ip || "").replace("::ffff:", ""),
        imageCount: files.length,
        result: fullAnalysis,
      });
      saveSessions(sessions);
    } catch {}

    trackEvent("analyze", req.ip);
    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Analyze error:", err?.status, err?.message, err?.error);
    const msg = err.status === 401
      ? "Invalid API key. Check your GROQ_API_KEY."
      : err.status === 429
      ? "Too many requests. Please wait a moment and try again."
      : `Error: ${err?.error?.message || err?.message || "Unknown error"} (${err?.status || 500})`;
    res.write(`data: ${JSON.stringify({ error: msg })}\n\n`);
    res.end();
  }
});

// ── Chat ──
app.post("/api/chat", async (req, res) => {
  const { messages, analysisContext } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "No messages provided." });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("Access-Control-Allow-Origin", "*");

  try {
    const systemWithContext = analysisContext
      ? `${CHAT_SYSTEM}\n\n--- PATIENT'S LAB ANALYSIS ---\n${analysisContext}\n--- END OF ANALYSIS ---`
      : CHAT_SYSTEM;

    const stream = await getClient().chat.completions.create({
      model: CHAT_MODEL,
      max_tokens: 1024,
      stream: true,
      messages: [
        { role: "system", content: systemWithContext },
        ...messages,
      ],
    });

    let fullResponse = "";
    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content;
      if (text) {
        fullResponse += text;
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    // Save chat session
    try {
      const lastUserMsg = [...messages].reverse().find(m => m.role === "user");
      const sessions = loadSessions();
      sessions.push({
        id: Date.now().toString(),
        type: "chat",
        timestamp: new Date().toISOString(),
        ip: (req.ip || "").replace("::ffff:", ""),
        userMessage: lastUserMsg?.content || "",
        aiResponse: fullResponse,
      });
      saveSessions(sessions);
    } catch {}

    trackEvent("chat", req.ip);
    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Chat error:", err?.status, err?.message);
    res.write(`data: ${JSON.stringify({ error: `Chat error: ${err?.message || "Unknown"}` })}\n\n`);
    res.end();
  }
});

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    model: VISION_MODEL,
    hasApiKey: !!process.env.GROQ_API_KEY && process.env.GROQ_API_KEY !== "missing",
  });
});

app.listen(PORT, () => {
  console.log(`\n✅ MedExplain AI running at http://localhost:${PORT}`);
  if (!process.env.GROQ_API_KEY) console.warn("⚠️  GROQ_API_KEY not set.");
});
