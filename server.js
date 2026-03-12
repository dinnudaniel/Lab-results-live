require("dotenv").config();
const express = require("express");
const multer = require("multer");
const Anthropic = require("@anthropic-ai/sdk");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

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

  try {
    const stream = await client.messages.stream({
      model: "claude-opus-4-6",
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "base64",
                media_type: mediaType,
                data: imageBase64,
              },
            },
            {
              type: "text",
              text: "Please explain all the lab results shown in this image in plain English. Tell me what each test is, what my result means, and whether it looks normal or not.",
            },
          ],
        },
      ],
    });

    // Stream the response back to the client using Server-Sent Events
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("Access-Control-Allow-Origin", "*");

    for await (const event of stream) {
      if (
        event.type === "content_block_delta" &&
        event.delta.type === "text_delta"
      ) {
        res.write(`data: ${JSON.stringify({ text: event.delta.text })}\n\n`);
      }
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (err) {
    console.error("Claude API error:", err);

    if (err instanceof Anthropic.AuthenticationError) {
      return res.status(401).json({ error: "Invalid API key. Please check your ANTHROPIC_API_KEY." });
    }
    if (err instanceof Anthropic.RateLimitError) {
      return res.status(429).json({ error: "Too many requests. Please wait a moment and try again." });
    }
    if (err instanceof Anthropic.BadRequestError) {
      return res.status(400).json({ error: "Could not process the image. Please try a clearer photo." });
    }

    res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    model: "claude-opus-4-6",
    hasApiKey: !!process.env.ANTHROPIC_API_KEY,
  });
});

app.listen(PORT, () => {
  console.log(`\n✅ Lab Results Interpreter running at http://localhost:${PORT}`);
  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn("⚠️  Warning: ANTHROPIC_API_KEY is not set. Add it to your .env file.");
  }
});
