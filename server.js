require("dotenv").config();
const express = require("express");
const multer = require("multer");
const OpenAI = require("openai");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// xAI Grok uses an OpenAI-compatible API — lazy init so missing key only fails at request time
function getClient() {
  return new OpenAI({
    apiKey: process.env.XAI_API_KEY || "missing",
    baseURL: "https://api.x.ai/v1",
  });
}

const GROK_MODEL = "grok-2-vision-1212";

// Store image in memory (no disk writes needed)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB max
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/gif", "image/webp"];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed (JPEG, PNG, GIF, WebP)"));
    }
  },
});

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

const SYSTEM_PROMPT = `You are MedExplain AI, a friendly medical assistant that helps people understand their lab test results in plain, everyday English.

Your job is to:
1. Read every test/result visible on the lab report image
2. Explain what each test is — in simple words anyone can understand
3. Explain what the result means (normal, abnormal, high, low, positive, negative, etc.)
4. Use clear, warm, non-scary language
5. For pregnancy-related tests (e.g. NIPT, amniocentesis): if sex/gender information is present, state it clearly
6. Group related tests together where helpful (e.g. Complete Blood Count, Liver Panel, etc.)

IMPORTANT RULES:
- Never say something is definitively wrong or dangerous — always recommend the person speak with their doctor
- Be accurate: for example, HCV = Hepatitis C Virus (NOT HIV/AIDS), HIV = Human Immunodeficiency Virus, TSH = Thyroid Stimulating Hormone, etc.
- If a result is outside the reference range, explain what that might mean simply, but say a doctor should review it
- If you cannot read a value clearly from the image, say so
- Always end your response with a gentle reminder that this explanation is for information only and not a substitute for professional medical advice

FORMAT your response like this:
---
## 🔬 Your Lab Results Explained

[For each test or group of tests:]

### [Test Name in Plain English] — [Medical Abbreviation]
**What this tests:** [1-2 sentence plain English explanation]
**Your result:** [value/result]
**Reference range:** [if visible]
**What it means:** [plain English explanation of the result]

---
[If pregnancy gender test present, include clearly]

---
⚠️ *This explanation is for informational purposes only and is not medical advice. Please discuss your results with your doctor or healthcare provider.*`;

app.post("/api/analyze", upload.single("labImage"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "Please upload an image of your lab results." });
  }

  const imageBase64 = req.file.buffer.toString("base64");
  const mediaType = req.file.mimetype;
  const imageUrl = `data:${mediaType};base64,${imageBase64}`;

  // Set up SSE headers before streaming
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("Access-Control-Allow-Origin", "*");

  try {
    const stream = await getClient().chat.completions.create({
      model: GROK_MODEL,
      max_tokens: 4096,
      stream: true,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: imageUrl },
            },
            {
              type: "text",
              text: "Please explain all the lab results shown in this image in plain English. Tell me what each test is, what my result means, and whether it looks normal or not.",
            },
          ],
        },
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
    console.error("Grok API error:", err);

    const status = err.status || 500;
    const messages = {
      401: "Invalid API key. Please check your XAI_API_KEY in the .env file.",
      429: "Too many requests. Please wait a moment and try again.",
      400: "Could not process the image. Please try a clearer photo.",
    };

    const errorMsg = messages[status] || "Something went wrong. Please try again.";
    res.write(`data: ${JSON.stringify({ error: errorMsg })}\n\n`);
    res.end();
  }
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    model: GROK_MODEL,
    hasApiKey: !!process.env.XAI_API_KEY && process.env.XAI_API_KEY !== "missing",
  });
});

app.listen(PORT, () => {
  console.log(`\n✅ Lab Results Interpreter (Grok) running at http://localhost:${PORT}`);
  if (!process.env.XAI_API_KEY) {
    console.warn("⚠️  Warning: XAI_API_KEY is not set. Add it to your .env file.");
  }
});
