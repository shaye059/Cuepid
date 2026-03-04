# PlexAI 🎬

**Intelligent movie recommendations from your personal Plex library.**

Chat naturally about what you're in the mood for, and get personalized picks powered by semantic search + Claude AI. Your entire library is embedded locally so searches are instant and context-aware.

---

## Features

- **Natural language search** — "something tense with a twist" or "a quiet film for a rainy evening"
- **Guided mode** — asks you questions one at a time to figure out what you're in the mood for
- **Semantic embeddings** — your library is indexed with `all-MiniLM-L6-v2`, so "mafia drama" finds *The Godfather* even without exact keywords
- **Filters** — narrow by length (short / medium / long) and era (decade)
- **Movie cards** — poster, year, genres, runtime, and rating rendered inline in chat
- **Fully local** — embeddings run on your machine, nothing leaves except API calls to Claude

---

## Requirements

- Python 3.9+
- A running Plex Media Server
- An [Anthropic API key](https://console.anthropic.com/account/keys)

---

## Setup

```bash
# 1. Clone or copy these files to your server
cd plexai/

# 2. Install dependencies
bash setup.sh

# 3. (Optional) set your API key as an env variable
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Start the server
python3 server.py
```

Then open **http://localhost:8000** in your browser.

On first launch you'll be prompted to enter:
- **Plex Server URL** — usually `http://localhost:32400`
- **Plex Token** — [how to find yours](https://support.plex.tv/articles/204059436-finding-an-authentication-token-x-plex-token/)
- **Anthropic API Key**

After connecting, click **Index Library** to build the semantic index. On a typical home server this takes 1–5 minutes for ~500 movies. The index is saved to `./plexai_data/` and reused on restart.

---

## Usage Tips

| Mode | When to use |
|------|-------------|
| **Direct search** | You have a rough idea — "something sci-fi and atmospheric" |
| **Help me decide** | You're not sure — Claude will ask 3–5 questions and then recommend |

Use the **sidebar filters** to pre-constrain by length or decade before chatting.

Use **↺ Re-index** any time you add new movies to Plex.

---

## How It Works

1. **Indexing** — Each movie is converted into a rich text document (title, year, genres, director, cast, summary) and embedded with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are stored in [ChromaDB](https://www.trychroma.com/).

2. **Search** — Your chat message is embedded and the 15 closest movies are retrieved by cosine similarity. Active filters (duration, year) are applied as pre-filters in ChromaDB.

3. **Generation** — The retrieved movies are injected into Claude's context as a curated list. Claude recommends only from this list and wraps titles in `[[Title (Year)]]` so the frontend can render movie cards.

---

## Data & Privacy

- All embeddings are stored locally in `./plexai_data/chroma/`
- Your Plex token and Anthropic key are saved in `./plexai_data/config.json`
- Only chat messages + retrieved movie metadata are sent to Anthropic's API — no video data

---

## Troubleshooting

**"Library not indexed yet"** — click ↺ Re-index in the header.

**Plex connection refused** — make sure Plex is running and the URL includes the port (`:32400`). If accessing from another machine, replace `localhost` with your server's IP.

**Slow indexing** — normal on first run; the embedding model (`~80 MB`) downloads automatically. Subsequent starts are instant.

**No posters showing** — poster images are proxied through `/api/poster`. Check that your Plex token is valid.
