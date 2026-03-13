require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const OpenAI = require("openai");
const sharp = require("sharp");
const fs = require("fs");
const crypto = require("crypto");
const bcrypt = require("bcryptjs");
const nodemailer = require("nodemailer");

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

// ── User Store ──
const DATA_DIR = path.join(__dirname, "data");
const USERS_FILE = path.join(DATA_DIR, "users.json");

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

// ── Email ──
const emailTransporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  port: 465,
  secure: true,
  auth: {
    user: process.env.SMTP_USER,
    pass: process.env.SMTP_PASS,
  },
  connectionTimeout: 15000,
  greetingTimeout: 10000,
  socketTimeout: 20000,
  tls: { rejectUnauthorized: false },
});

async function sendVerificationEmail(toEmail, username, code) {
  const sendPromise = emailTransporter.sendMail({
    from: `"MedExplain AI" <${process.env.SMTP_USER}>`,
    to: toEmail,
    subject: "Your MedExplain AI Verification Code",
    html: `
      <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;background:#f9fafb;">
        <div style="background:white;border-radius:16px;padding:32px;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
          <div style="text-align:center;margin-bottom:24px;">
            <div style="width:56px;height:56px;background:#2563eb;border-radius:14px;display:inline-flex;align-items:center;justify-content:center;font-size:28px;">🔬</div>
            <h1 style="font-size:1.4rem;font-weight:800;color:#111827;margin:12px 0 4px;">MedExplain AI</h1>
            <p style="color:#6b7280;font-size:0.9rem;margin:0;">Verify your email address</p>
          </div>
          <p style="color:#374151;font-size:0.95rem;margin-bottom:8px;">Hi <strong>${username}</strong>,</p>
          <p style="color:#374151;font-size:0.95rem;margin-bottom:24px;">Enter this code on the website to verify your account:</p>
          <div style="text-align:center;background:#eff6ff;border-radius:12px;padding:24px;margin-bottom:24px;">
            <div style="font-size:2.5rem;font-weight:900;letter-spacing:0.3em;color:#2563eb;">${code}</div>
            <p style="color:#6b7280;font-size:0.8rem;margin:8px 0 0;">This code expires in <strong>10 minutes</strong></p>
          </div>
          <p style="color:#9ca3af;font-size:0.8rem;text-align:center;margin:0;">If you didn't create an account, you can safely ignore this email.</p>
        </div>
      </div>`,
  });

  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error("SMTP_TIMEOUT: Email server took too long to respond")), 30000)
  );
  await Promise.race([sendPromise, timeout]);
}

// Pending verifications: email → { code, username, hashedPassword, expiresAt }
const pendingVerifications = new Map();

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

// ── Register — sends verification code ──
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
    const code = String(Math.floor(100000 + Math.random() * 900000)); // 6-digit code
    const cleanEmail = email.toLowerCase().trim();

    pendingVerifications.set(cleanEmail, {
      code,
      username: username.trim(),
      hashedPassword: hash,
      expiresAt: Date.now() + 10 * 60 * 1000, // 10 minutes
    });

    await sendVerificationEmail(cleanEmail, username.trim(), code);
    res.json({ message: "Verification code sent. Please check your email.", email: cleanEmail });
  } catch (err) {
    console.error("Register error:", err.code, err.message);
    let msg;
    if (err.code === "EAUTH" || err.message?.includes("Invalid login") || err.message?.includes("Username and Password"))
      msg = "Email login failed. Your Gmail App Password may be wrong — double-check SMTP_PASS has no spaces.";
    else if (err.message?.includes("SMTP_TIMEOUT") || err.code === "ETIMEDOUT" || err.code === "ESOCKET")
      msg = "Email server timed out. Check that SMTP_USER and SMTP_PASS are set correctly in your environment.";
    else if (err.code === "ECONNREFUSED" || err.code === "ENOTFOUND")
      msg = "Cannot connect to email server. Check your internet/server connection.";
    else
      msg = `Email error: ${err.message || "Unknown error"} (code: ${err.code || "none"})`;
    res.status(500).json({ error: msg });
  }
});

// ── Verify Email ──
app.post("/api/verify-email", async (req, res) => {
  try {
    const { email, code } = req.body;
    if (!email || !code)
      return res.status(400).json({ error: "Email and code are required." });

    const cleanEmail = email.toLowerCase().trim();
    const pending = pendingVerifications.get(cleanEmail);

    if (!pending)
      return res.status(400).json({ error: "No pending registration for this email. Please register again." });
    if (Date.now() > pending.expiresAt)
      return res.status(400).json({ error: "Code has expired. Please register again." });
    if (pending.code !== code.trim())
      return res.status(400).json({ error: "Incorrect code. Please check your email and try again." });

    // Code correct — create the account
    const users = loadUsers();
    if (users.find(u => u.email === cleanEmail))
      return res.status(409).json({ error: "This email is already registered. Please login." });

    const token = generateToken();
    const user = {
      id: Date.now().toString(),
      username: pending.username,
      email: cleanEmail,
      password: pending.hashedPassword,
      token,
      telegramBotToken: "",
      telegramChatId: "",
    };
    users.push(user);
    saveUsers(users);
    pendingVerifications.delete(cleanEmail);

    res.json({ token, username: user.username, email: user.email, hasTelegram: false });
  } catch (err) {
    console.error("Verify error:", err);
    res.status(500).json({ error: "Verification failed. Please try again." });
  }
});

// ── Resend Code ──
app.post("/api/resend-code", async (req, res) => {
  try {
    const { email } = req.body;
    const cleanEmail = (email || "").toLowerCase().trim();
    const pending = pendingVerifications.get(cleanEmail);
    if (!pending)
      return res.status(400).json({ error: "No pending registration found. Please register again." });

    const newCode = String(Math.floor(100000 + Math.random() * 900000));
    pending.code = newCode;
    pending.expiresAt = Date.now() + 10 * 60 * 1000;
    pendingVerifications.set(cleanEmail, pending);

    await sendVerificationEmail(cleanEmail, pending.username, newCode);
    res.json({ message: "New code sent to your email." });
  } catch (err) {
    console.error("Resend error:", err.code, err.message);
    let msg;
    if (err.code === "EAUTH" || err.message?.includes("Invalid login"))
      msg = "Email login failed. Check SMTP_USER and SMTP_PASS (App Password, no spaces).";
    else if (err.message?.includes("SMTP_TIMEOUT") || err.code === "ETIMEDOUT" || err.code === "ESOCKET")
      msg = "Email server timed out. Check SMTP credentials in your environment settings.";
    else
      msg = `Email error: ${err.message || "Unknown"} (code: ${err.code || "none"})`;
    res.status(500).json({ error: msg });
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
    res.json({ success: true });
  } catch (err) {
    console.error("Telegram send error:", err);
    res.status(500).json({ error: "Failed to send to Telegram. Please try again." });
  }
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

    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content;
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

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

    for await (const chunk of stream) {
      const text = chunk.choices[0]?.delta?.content;
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

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
  const smtpUser = process.env.SMTP_USER || "(not set)";
  const smtpPass = process.env.SMTP_PASS ? `set (${process.env.SMTP_PASS.length} chars)` : "(not set)";
  console.log(`📧 SMTP_USER: ${smtpUser} | SMTP_PASS: ${smtpPass}`);
});
