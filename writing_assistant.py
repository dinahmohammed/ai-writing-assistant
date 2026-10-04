#!/usr/bin/env python3
"""
AI Writing Assistant
--------------------
Prompt templates (blog, email, code explanation, social media) on top of
OpenAI or Gemini, with temperature control and output formatting.

Setup:
    pip install openai google-genai markdown
    export OPENAI_API_KEY=...    # for --provider openai
    export GEMINI_API_KEY=...    # for --provider gemini

Usage:
    python writing_assistant.py                      # interactive mode
    python writing_assistant.py --list
    python writing_assistant.py -t blog -p gemini --preset creative \
        --set topic="Remote work in 2026" --set audience="engineers" -f html -o post.html
"""
import argparse
import html
import json
import os
import re
import sys
from datetime import datetime

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
}
TEMP_PRESETS = {"precise": 0.2, "balanced": 0.7, "creative": 1.2}
TEMP_MIN, TEMP_MAX = 0.0, 2.0  # valid range for both providers

# Each field: (key, prompt shown to user, default or None if required)
TEMPLATES = {
    "blog": {
        "title": "Blog post",
        "temp": "creative",
        "system": (
            "You are an experienced content writer. You write engaging, "
            "well-structured blog posts with a strong hook, clear headings, "
            "concrete examples, and a useful conclusion."
        ),
        "fields": [
            ("topic", "Topic", None),
            ("audience", "Target audience", "general readers"),
            ("tone", "Tone", "friendly and informative"),
            ("words", "Approx. word count", "700"),
            ("keywords", "SEO keywords (comma-separated, optional)", ""),
        ],
        "user": (
            "Write a blog post about: {topic}\n"
            "Audience: {audience}\nTone: {tone}\nLength: about {words} words.\n"
            "Include a title, an intro, 3-5 sections with headings, and a conclusion.\n"
            "Naturally work in these keywords if given: {keywords}"
        ),
    },
    "email": {
        "title": "Email draft",
        "temp": "balanced",
        "system": (
            "You are a professional communications assistant. You write clear, "
            "concise emails that get to the point and have an obvious call to action."
        ),
        "fields": [
            ("purpose", "Purpose of the email", None),
            ("recipient", "Recipient (name/role)", "the recipient"),
            ("tone", "Tone", "professional but warm"),
            ("details", "Key points to include", ""),
            ("sender", "Your name", ""),
        ],
        "user": (
            "Draft an email.\nPurpose: {purpose}\nTo: {recipient}\nTone: {tone}\n"
            "Key points: {details}\nSign off as: {sender}\n"
            "Start with a 'Subject:' line. Keep it under 200 words unless the "
            "key points demand more."
        ),
    },
    "code": {
        "title": "Code explanation",
        "temp": "precise",
        "system": (
            "You are a patient senior engineer and teacher. You explain code "
            "accurately, never inventing behavior that isn't in the code."
        ),
        "fields": [
            ("code", "Paste code (finish with a line containing only END)", None),
            ("level", "Reader level (beginner/intermediate/expert)", "intermediate"),
            ("language", "Language (blank = auto-detect)", ""),
        ],
        "user": (
            "Explain this code for a {level} reader. Language: {language}\n\n"
            "```\n{code}\n```\n\n"
            "Give: 1) a one-sentence summary, 2) a step-by-step walkthrough, "
            "3) potential bugs or improvements."
        ),
    },
    "social": {
        "title": "Social media posts",
        "temp": "creative",
        "system": (
            "You are a social media strategist. You write punchy, platform-native "
            "posts that fit each platform's norms and length limits."
        ),
        "fields": [
            ("message", "What do you want to announce/share?", None),
            ("platforms", "Platforms (comma-separated)", "Twitter/X, LinkedIn, Instagram"),
            ("tone", "Tone", "upbeat and conversational"),
            ("variants", "Variants per platform", "2"),
        ],
        "user": (
            "Create social media posts about: {message}\n"
            "Platforms: {platforms}\nTone: {tone}\nVariants per platform: {variants}\n"
            "Respect character limits (X: 280). Add relevant hashtags and, where "
            "natural, emoji. Label each platform and variant clearly."
        ),
    },
}

FORMAT_HINTS = {
    "markdown": "Format the response in Markdown.",
    "plain": "Respond in plain text only: no Markdown symbols, no bullets made of asterisks.",
    "html": "Format the response in Markdown.",  # converted to HTML afterwards
    "json": "Format the response in Markdown.",  # wrapped in JSON afterwards
}


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
def generate(provider: str, model: str, system: str, prompt: str, temperature: float) -> str:
    if provider == "openai":
        from openai import OpenAI  # lazy import: only needed if used

        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    if provider == "gemini":
        from google import genai
        from google.genai import types

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature
            ),
        )
        return (resp.text or "").strip()

    raise ValueError(f"Unknown provider: {provider}")


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #
def strip_markdown(text: str) -> str:
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)           # code fences
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)  # headings
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)        # bold
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)           # italics
    text = re.sub(r"`([^`]*)`", r"\1", text)               # inline code
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)   # links
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.M)  # bullets
    return text.strip()


def format_output(text: str, fmt: str, meta: dict) -> str:
    if fmt == "markdown":
        return text
    if fmt == "plain":
        return strip_markdown(text)
    if fmt == "html":
        try:
            import markdown

            body = markdown.markdown(text, extensions=["fenced_code", "tables"])
        except ImportError:  # graceful fallback
            body = f"<pre>{html.escape(text)}</pre>"
        return (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>{html.escape(meta['template'])}</title>"
            "<style>body{max-width:720px;margin:2rem auto;font:16px/1.6 system-ui;"
            "padding:0 1rem}pre{background:#f4f4f4;padding:1rem;overflow:auto}</style>"
            f"{body}"
        )
    if fmt == "json":
        return json.dumps({**meta, "content": text}, indent=2, ensure_ascii=False)
    raise ValueError(f"Unknown format: {fmt}")


# --------------------------------------------------------------------------- #
# Prompt building / input
# --------------------------------------------------------------------------- #
def read_multiline(prompt: str) -> str:
    print(f"{prompt}:")
    lines = []
    while (line := input()) .strip() != "END":
        lines.append(line)
    return "\n".join(lines)


def collect_fields(tpl: dict, preset: dict, interactive: bool) -> dict:
    values = {}
    for key, label, default in tpl["fields"]:
        if key in preset:
            values[key] = preset[key]
        elif interactive:
            if key == "code":
                values[key] = read_multiline(label)
            else:
                hint = f" [{default}]" if default else ""
                ans = input(f"{label}{hint}: ").strip()
                values[key] = ans or (default or "")
        else:
            values[key] = default or ""
        if default is None and not values[key]:
            sys.exit(f"Missing required field: {key} (use --set {key}=...)")
    return values


def resolve_temperature(args, tpl) -> float:
    if args.temp is not None:
        t = args.temp
    else:
        t = TEMP_PRESETS[args.preset or tpl["temp"]]
    return max(TEMP_MIN, min(TEMP_MAX, t))


def run(args, interactive: bool) -> None:
    tpl = TEMPLATES[args.template]
    preset = dict(kv.split("=", 1) for kv in (args.set or []))
    values = collect_fields(tpl, preset, interactive)

    temperature = resolve_temperature(args, tpl)
    model = args.model or DEFAULT_MODELS[args.provider]
    prompt = tpl["user"].format(**values) + "\n\n" + FORMAT_HINTS[args.format]

    print(f"\n[{tpl['title']} | {args.provider}:{model} | temp={temperature}]\n", file=sys.stderr)
    try:
        raw = generate(args.provider, model, tpl["system"], prompt, temperature)
    except Exception as e:  # network, auth, quota...
        sys.exit(f"API error: {e}")

    meta = {
        "template": args.template,
        "provider": args.provider,
        "model": model,
        "temperature": temperature,
        "inputs": values,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output = format_output(raw, args.format, meta)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {args.out}", file=sys.stderr)
    else:
        print(output)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def pick(prompt: str, options: list, default: str) -> str:
    ans = input(f"{prompt} ({'/'.join(options)}) [{default}]: ").strip().lower()
    return ans if ans in options else default


def interactive_session(args) -> None:
    print("=== AI Writing Assistant ===")
    while True:
        args.template = pick("Template", list(TEMPLATES), "blog")
        args.provider = pick("Provider", list(DEFAULT_MODELS), args.provider)
        raw_t = input(
            f"Temperature 0-2 or preset {list(TEMP_PRESETS)} [template default]: "
        ).strip().lower()
        args.temp, args.preset = None, None
        if raw_t in TEMP_PRESETS:
            args.preset = raw_t
        elif raw_t:
            try:
                args.temp = float(raw_t)
            except ValueError:
                print("Invalid temperature, using template default.")
        args.format = pick("Output format", list(FORMAT_HINTS), args.format)
        args.set = []
        run(args, interactive=True)
        if input("\nAnother? (y/N): ").strip().lower() != "y":
            break


def main() -> None:
    p = argparse.ArgumentParser(description="AI writing assistant (OpenAI / Gemini)")
    p.add_argument("-t", "--template", choices=TEMPLATES)
    p.add_argument("-p", "--provider", choices=DEFAULT_MODELS, default="openai")
    p.add_argument("-m", "--model", help="override default model")
    p.add_argument("--temp", type=float, help="temperature 0-2 (overrides --preset)")
    p.add_argument("--preset", choices=TEMP_PRESETS, help="precise=0.2 balanced=0.7 creative=1.2")
    p.add_argument("-f", "--format", choices=FORMAT_HINTS, default="markdown")
    p.add_argument("-o", "--out", help="save output to file")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="template field value")
    p.add_argument("--list", action="store_true", help="list templates and fields")
    args = p.parse_args()

    if args.list:
        for name, tpl in TEMPLATES.items():
            print(f"{name:7} {tpl['title']} (default temp: {tpl['temp']})")
            for key, label, default in tpl["fields"]:
                print(f"          {key}: {label}{'' if default is not None else '  *required*'}")
        return

    if args.template is None:
        interactive_session(args)
    else:
        run(args, interactive=sys.stdin.isatty() and not args.set)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
