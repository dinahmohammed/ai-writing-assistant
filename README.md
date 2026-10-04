# AI Writing Assistant

A Python command-line app that generates writing using the OpenAI or Gemini API.

## Features
- **4 prompt templates:** blog post, email draft, code explanation, social media posts
- **Temperature control:** exact value (`--temp 0.9`) or presets (`precise`, `balanced`, `creative`)
- **Output formats:** Markdown, plain text, HTML, JSON
- **Two providers:** OpenAI and Google Gemini
- **Interactive mode** or command-line flags

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here    # for OpenAI
export GEMINI_API_KEY=your_key_here    # for Gemini
```
(On Windows PowerShell use `$env:OPENAI_API_KEY="your_key_here"`.)

## Usage
```bash
python writing_assistant.py                  # interactive mode
python writing_assistant.py --list           # show templates and fields

python writing_assistant.py -t blog -p gemini --preset creative \
    --set topic="Remote work in 2026" -f html -o post.html
```

| Flag | Meaning |
|------|---------|
| `-t` | template: `blog`, `email`, `code`, `social` |
| `-p` | provider: `openai` or `gemini` |
| `--temp` / `--preset` | temperature (0-2) or preset |
| `-f` | format: `markdown`, `plain`, `html`, `json` |
| `-o` | save output to a file |
| `--set KEY=VALUE` | fill in a template field |

## Adding a template
Add an entry to the `TEMPLATES` dictionary in `writing_assistant.py` with a system prompt, fields, and a user prompt.
