import os

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context


load_dotenv()

SYSTEM_PROMPT = (
    "Answer clearly and directly. Be concise unless asked for detail. "
    "For questions about current or recent facts, use Google Search to verify the answer. "
    "Include source links when search results provide them."
)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Streaming AI Chat</title>
  <style>
    :root { color-scheme: dark; --gold: #e3bd66; --gold-bright: #f4d98b; --muted: #a49b86; }
    * { box-sizing: border-box; }
    body { min-height: 100vh; margin: 0; padding: 64px 20px; background: radial-gradient(ellipse at 50% -20%, #332812 0, #17140e 36%, #090909 76%); color: #f5f0e5; font: 16px/1.6 system-ui, sans-serif; }
    main { max-width: 820px; margin: 0 auto; }
    h1 { margin: 0; color: var(--gold-bright); font-size: clamp(2rem, 6vw, 3rem); letter-spacing: -.04em; }
    .hint { margin: 8px 0 30px; color: var(--muted); }
    textarea { display: block; width: 100%; min-height: 130px; padding: 16px; resize: vertical; border: 1px solid #5b4924; border-radius: 12px; outline: none; background: #11100d; color: #f5f0e5; font: inherit; transition: border-color .2s, box-shadow .2s; }
    textarea:focus { border-color: var(--gold); box-shadow: 0 0 0 3px #e3bd6626; }
    textarea::placeholder { color: #77705f; }
    button { margin-top: 14px; padding: 11px 24px; border: 1px solid #f2d889; border-radius: 8px; background: linear-gradient(135deg, #f2d889, #c79d42); color: #171208; font: 700 15px system-ui, sans-serif; cursor: pointer; transition: filter .2s, transform .2s; }
    button:hover:not(:disabled) { filter: brightness(1.08); transform: translateY(-1px); }
    button:disabled { opacity: .65; cursor: wait; }
    #status { min-height: 26px; margin: 24px 0 8px; color: var(--gold); font-size: 14px; }
    #answer { min-height: 90px; padding: 20px; border: 1px solid #40351e; border-radius: 12px; background: #11100de8; color: #f5f0e5; white-space: pre-wrap; overflow-wrap: anywhere; box-shadow: 0 16px 50px #0008; }
    @media (max-width: 600px) { body { padding: 36px 16px; } #answer { padding: 16px; } }
  </style>
</head>
<body>
  <main>
    <h1>Streaming AI Chat</h1>
    <p class="hint">Enter a prompt. The response appears as it is generated.</p>
    <form id="chat-form">
      <textarea id="prompt" placeholder="Ask something..." required></textarea>
      <button id="submit" type="submit">Send prompt</button>
    </form>
    <p id="status" role="status"></p>
    <div id="answer" aria-live="polite"></div>
  </main>
  <script>
    const form = document.querySelector('#chat-form');
    const promptBox = document.querySelector('#prompt');
    const button = document.querySelector('#submit');
    const status = document.querySelector('#status');
    const answer = document.querySelector('#answer');

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptBox.value.trim();
      if (!prompt) return;
      button.disabled = true;
      answer.textContent = '';
      status.textContent = 'Connecting...';
      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });
        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.error || 'Request failed.');
        }
        status.textContent = 'Generating...';
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          answer.textContent += decoder.decode(value, { stream: true });
        }
        answer.textContent += decoder.decode();
        status.textContent = 'Done';
      } catch (error) {
        status.textContent = `Error: ${error.message}`;
      } finally {
        button.disabled = false;
        promptBox.focus();
      }
    });
  </script>
</body>
</html>"""


def get_chat_model(use_search=True):
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    if provider == "grok":
        if not os.getenv("XAI_API_KEY"):
            raise ValueError("Set XAI_API_KEY in your .env file to use Grok.")
        from langchain_xai import ChatXAI

        return ChatXAI(model=os.getenv("GROK_MODEL", "grok-4.6"), temperature=0.7)

    if provider == "gemini":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise ValueError(
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your .env file to use Gemini."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        )
        if use_search:
            return model.bind_tools([{"google_search": {}}])
        return model

    raise ValueError("LLM_PROVIDER must be either 'grok' or 'gemini'.")


def text_from_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def friendly_error(error):
    message = str(error)
    lowered = message.lower()
    if "resource_exhausted" in lowered or "quota" in lowered or "429" in lowered:
        return "Provider quota or rate limit reached. Check your API project limits and billing, then retry."
    if "api key" in lowered or "unauthorized" in lowered or "401" in lowered:
        return "The API key was rejected. Check the key in your .env file."
    return message


def is_quota_error(error):
    message = str(error).lower()
    return "resource_exhausted" in message or "quota" in message or "429" in message


@app.get("/")
def home():
    return PAGE


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify(error="Enter a prompt first."), 400

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    try:
        model = get_chat_model()
    except (ValueError, ImportError) as error:
        return jsonify(error=str(error)), 400

    messages = [("system", SYSTEM_PROMPT), ("human", prompt.strip())]

    @stream_with_context
    def generate():
        sent_text = False
        try:
            for chunk in model.stream(messages):
                text = text_from_content(chunk.content)
                if text:
                    sent_text = True
                    yield text
        except Exception as error:
            # Retry without web grounding only if the live-search request failed
            # before any answer text reached the browser.
            if provider == "gemini" and is_quota_error(error) and not sent_text:
                yield (
                    "⚠️ Live Google Search is unavailable because of a quota or rate limit. "
                    "Retrying without live search: this answer may be outdated and is not current data.\n\n"
                )
                try:
                    fallback_model = get_chat_model(use_search=False)
                    for chunk in fallback_model.stream(messages):
                        text = text_from_content(chunk.content)
                        if text:
                            yield text
                except Exception as fallback_error:
                    yield f"\n\nFallback failed: {friendly_error(fallback_error)}"
            else:
                yield f"\n\nRequest failed: {friendly_error(error)}"

    return Response(
        generate(),
        content_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))
    print(f"Open http://{host}:{port} in your browser")
    app.run(host=host, port=port, debug=False, threaded=True)
