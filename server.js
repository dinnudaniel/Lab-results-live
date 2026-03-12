require("dotenv").config();
const express = require("express");
const multer = require("multer");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");

const app = express();
const PORT = process.env.PORT || 3000;

const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY || "missing");
const MODEL = "gemini-2.0-flash";

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

const ANALYZE_PROMPT = `You are MedExplain AI, a friendly medical assistant that explains lab test results in plain, everyday English.

Your job:
1. Read EVERY test and result visible across ALL provided images
2. Explain what each test is in simple words
3. Explain what the result means (normal, abnormal, high, low, positive, negative)
4. Use warm, clear, non-scary language
5. Group related tests (e.g. Blood Count, Liver Panel, Infection Screen)
6. For pregnancy tests (NIPT, amniocentesis): clearly state sex if shown

RULES:
- HCV = Hepatitis C Virus (NOT AIDS). HIV = Human Immunodeficiency Virus. Never mix these up.
- If a result is outside range, explain simply but say a doctor should review it
- If you cannot read a value, say so
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
⚠️ *For informational purposes only. Please discuss with your doctor.*

These are lab result images. Please explain ALL results shown across every image in plain English.`;

const CHAT_SYSTEM = `You are MedExplain AI, a friendly medical assistant helping a patient understand their lab results.

The patient's lab analysis has already been done and is provided as context. Your job is to answer their follow-up questions clearly and simply.

RULES:
- Answer in plain English, no medical jargon
- Be warm and reassuring but honest
- For questions about diagnosis, treatment, or medication: ALWAYS say "I'm not able to give medical advice on that — please speak with your doctor about this"
- For questions you're not confident about: say "That's a great question for your doctor — I'd recommend asking them directly"
- Never guess at a diagnosis
- Keep answers focused and concise`;

// ── Analyze: accepts up to 5 images ──
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
    const model = genAI.getGenerativeModel({ model: MODEL });

    const parts = files.map(file => ({
      inlineData: {
        data: file.buffer.toString("base64"),
        mimeType: file.mimetype,
      },
    }));
    parts.push({ text: ANALYZE_PROMPT });

    const result = await model.generateContentStream(parts);

    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Analyze error:", err?.message, err?.status);
    const status = err?.status || err?.httpStatusCode;
    const msg = status === 401 || status === 403
      ? "Invalid API key. Check your GOOGLE_API_KEY."
      : status === 429
      ? "Too many requests. Please wait a moment and try again."
      : `Error: ${err?.message || "Unknown error"}`;
    res.write(`data: ${JSON.stringify({ error: msg })}\n\n`);
    res.end();
  }
});

// ── Chat: follow-up questions after analysis ──
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

    const model = genAI.getGenerativeModel({
      model: MODEL,
      systemInstruction: systemWithContext,
    });

    const history = messages.slice(0, -1).map(m => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

    const chat = model.startChat({ history });
    const lastMsg = messages[messages.length - 1].content;
    const result = await chat.sendMessageStream(lastMsg);

    for await (const chunk of result.stream) {
      const text = chunk.text();
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Chat error:", err?.message);
    res.write(`data: ${JSON.stringify({ error: `Chat error: ${err?.message || "Unknown"}` })}\n\n`);
    res.end();
  }
});

app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    model: MODEL,
    hasApiKey: !!process.env.GOOGLE_API_KEY && process.env.GOOGLE_API_KEY !== "missing",
  });
});

app.listen(PORT, () => {
  console.log(`\n✅ MedExplain AI running at http://localhost:${PORT}`);
  if (!process.env.GOOGLE_API_KEY) {
    console.warn("⚠️  GOOGLE_API_KEY not set.");
  }
});
