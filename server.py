#!/usr/bin/env python3
"""
PlexAI — Intelligent movie recommendations from your Plex library
Run: python server.py
Then open: http://localhost:8000
"""

import os
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel
import chromadb

# ── Globals ────────────────────────────────────────────────────────────────────
_embedder = None
_claude = None
_plex = None
_collection = None
_chroma_client = None
_index_status: Dict[str, Any] = {
    "running": False, "progress": 0, "total": 0,
    "message": "Library not indexed yet",
}

DATA_DIR = Path("./plexai_data")
DATA_DIR.mkdir(exist_ok=True)
CHROMA_PATH = DATA_DIR / "chroma"
CONFIG_FILE = DATA_DIR / "config.json"

EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-20250514"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _claude, _plex, _collection, _chroma_client

    print("⟳  Loading embedding model (downloads ~80 MB on first run)…")
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(EMBED_MODEL)
    print("✓  Embedding model ready")

    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        _collection = _chroma_client.get_collection("movies")
        n = _collection.count()
        _index_status["message"] = f"{n} movies indexed"
        print(f"✓  ChromaDB: {n} movies loaded")
    except Exception:
        _collection = _chroma_client.create_collection("movies")
        print("✓  ChromaDB: empty collection created")

    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            import anthropic
            from plexapi.server import PlexServer
            key = cfg.get("anthropic_key") or os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                _claude = anthropic.Anthropic(api_key=key)
            _plex = PlexServer(cfg["plex_url"], cfg["plex_token"], timeout=15)
            print(f"✓  Reconnected to Plex: {_plex.friendlyName}")
        except Exception as e:
            print(f"!  Could not restore Plex connection: {e}")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        _claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print("\n🎬  PlexAI is ready at http://localhost:8000\n")
    yield
    print("Shutdown.")


app = FastAPI(lifespan=lifespan)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ConnectBody(BaseModel):
    plex_url: str
    plex_token: str
    anthropic_key: str

class Msg(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    messages: List[Msg]
    mode: str = "direct"          # "direct" | "guided"
    filters: Dict[str, Any] = {}  # duration, year range, etc.


# ── Helpers ────────────────────────────────────────────────────────────────────

def _movie_doc(movie):
    """Return (text_for_embedding, metadata_dict) for a Plex movie."""
    genres    = [g.tag for g in (movie.genres    or [])]
    directors = [d.tag for d in (movie.directors or [])]
    cast      = [r.tag for r in (movie.roles     or [])[:12]]
    dur_min   = round((movie.duration or 0) / 60000)

    text = (
        f"Title: {movie.title} ({movie.year or ''})\n"
        f"Genres: {', '.join(genres)}\n"
        f"Director: {', '.join(directors)}\n"
        f"Cast: {', '.join(cast)}\n"
        f"Runtime: {dur_min} minutes\n"
        f"Content Rating: {movie.contentRating or ''}\n"
        f"Audience Rating: {movie.audienceRating or ''}/10\n"
        f"Summary: {(movie.summary or '')[:600]}"
    )
    meta = {
        "title":          str(movie.title),
        "year":           int(movie.year or 0),
        "duration_min":   int(dur_min),
        "genres":         ", ".join(genres),
        "directors":      ", ".join(directors),
        "cast":           ", ".join(cast),
        "content_rating": str(movie.contentRating or ""),
        "audience_rating":float(movie.audienceRating or 0),
        "thumb":          str(movie.thumb or ""),
        "key":            str(movie.key),
    }
    return text, meta


def _search(query: str, n: int = 15, filters: dict = None) -> tuple[list, list]:
    """Returns (metadatas, distances). ChromaDB uses L2; lower distance = closer match."""
    if not _collection or _collection.count() == 0:
        return [], []
    emb = _embedder.encode(query).tolist()

    where = None
    if filters:
        conds = []
        if filters.get("max_duration"):
            conds.append({"duration_min": {"$lte": int(filters["max_duration"])}})
        if filters.get("min_duration"):
            conds.append({"duration_min": {"$gte": int(filters["min_duration"])}})
        if filters.get("min_year"):
            conds.append({"year": {"$gte": int(filters["min_year"])}})
        if filters.get("max_year"):
            conds.append({"year": {"$lte": int(filters["max_year"])}})
        if conds:
            where = {"$and": conds} if len(conds) > 1 else conds[0]

    actual_n = min(n, _collection.count())
    r = _collection.query(
        query_embeddings=[emb],
        n_results=actual_n,
        where=where,
        include=["metadatas", "distances"],
    )
    return r["metadatas"][0], r["distances"][0]


def _log_search(query: str, metadatas: list, distances: list, turn: int) -> None:
    SEP = "─" * 62
    print(f"\n{SEP}")
    print(f"  TURN {turn}  │  MODE: search")
    print(f"  QUERY  '{query}' ")
    print(f"  RESULTS ({len(metadatas)})")
    for i, (m, d) in enumerate(zip(metadatas, distances), 1):
        # Convert L2 distance to a 0–1 similarity score (1 = identical)
        similarity = 1 / (1 + d)
        genres = (m.get("genres") or "—")[:28]
        title  = f"{m['title']} ({m['year']})"
        print(f"  {i:>2}.  {title:<32}  sim={similarity:.3f}  {genres}")
    print(SEP)


def _synthesize_query(msgs: list) -> str:
    """Ask Claude to write an optimal semantic search query from the conversation so far."""
    try:
        resp = _claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=120,
            messages=msgs + [{
                "role": "user",
                "content": (
                    "Based on this conversation, write a single descriptive sentence (two at most) "
                    "that captures what kind of movie the user is looking for — mood, energy, genre, "
                    "themes, and any specific preferences or exclusions mentioned. "
                    "Write only the search query, nothing else."
                ),
            }],
        )
        return resp.content[0].text.strip()
    except Exception:
        # Fall back to last user message
        user_msgs = [m for m in msgs if m["role"] == "user"]
        return user_msgs[-1]["content"] if user_msgs else ""


def _system_prompt(movies: list, mode: str) -> str:
    if movies:
        rows = []
        for m in movies:
            cast_snippet = m["cast"][:70] if m["cast"] else ""
            rows.append(
                f"• [[{m['title']} ({m['year']})]]"
                f" — {m['genres'] or '?'}"
                f" — {m['duration_min']} min"
                f"{' — ' + cast_snippet if cast_snippet else ''}"
            )
        lib = "\n".join(rows)
    else:
        lib = "No matching movies found — library may not be indexed yet."

    if mode == "guided":
        mode_block = """
GUIDED MODE — help the user discover what they want through warm conversation.
Use these five questions as a bank to draw from, in roughly this order:
  1. Mood — What kind of mood are you in right now?
  2. Energy — What level of energy do you want?
  3. Story/setting — Do you want a specific type of story or setting?
  4. Anchors — What's something you've liked recently, and what's something you don't want?
  5. Challenge — How "challenging" do you want the movie to be?
After each answer, decide: do you have enough to make a confident recommendation, or would one more question sharpen it?
If the picture is clear, recommend — don't keep asking for the sake of it. If it's still ambiguous, ask the next most useful question.
For each question, include 3–5 short examples in parentheses tailored to what the user has already told you.
Be concise — one short sentence plus examples. No preamble, no filler. Never ask more than one question at a time."""
    else:
        mode_block = """
DIRECT MODE — the user has a specific request.
First, decide: is the request specific enough to recommend confidently, or would one short clarifying question produce a much better result?
If you can make a strong recommendation, do it — 2–4 films with one sentence of reasoning each.
If a single question would meaningfully narrow it down, ask it instead. Keep it concise."""

    return f"""You are a passionate, knowledgeable film curator with access to this personal Plex library.
ONLY recommend movies from the list below. Never invent titles.

AVAILABLE MOVIES (retrieved by semantic search):
{lib}

FORMATTING RULE — when you mention a movie title, wrap it EXACTLY like this:
  [[Title (Year)]]
Example: You'd love [[Parasite (2019)]] for its sharp class commentary.
{mode_block}

If someone asks about a film not on the list, say it's not in the library and suggest the closest thing that IS."""


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path(__file__).parent / "index.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>index.html not found next to server.py</h1>")


@app.post("/api/connect")
async def connect(body: ConnectBody):
    global _plex, _claude
    try:
        from plexapi.server import PlexServer
        import anthropic
        plex = PlexServer(body.plex_url.rstrip("/"), body.plex_token, timeout=15)
        _plex = plex
        _claude = anthropic.Anthropic(api_key=body.anthropic_key)
        cfg = {
            "plex_url":     body.plex_url.rstrip("/"),
            "plex_token":   body.plex_token,
            "anthropic_key":body.anthropic_key,
        }
        CONFIG_FILE.write_text(json.dumps(cfg))
        sections = [s.title for s in plex.library.sections() if s.type == "movie"]
        return {"ok": True, "name": plex.friendlyName, "movie_sections": sections}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/status")
async def get_status():
    return {
        "connected":   _plex is not None,
        "plex_name":   _plex.friendlyName if _plex else None,
        "indexed":     _collection.count() if _collection else 0,
        "indexing":    _index_status,
        "ready":       _claude is not None and bool(_collection) and (_collection.count() > 0),
    }


@app.post("/api/index")
async def index_library():
    if not _plex:
        raise HTTPException(400, "Not connected to Plex")

    async def _gen():
        global _index_status, _collection, _chroma_client
        _index_status = {"running": True, "progress": 0, "total": 0, "message": "Fetching library…"}
        yield f"data: {json.dumps(_index_status)}\n\n"
        await asyncio.sleep(0)

        try:
            movies = []
            for sec in _plex.library.sections():
                if sec.type == "movie":
                    movies.extend(sec.all())

            _index_status["total"] = len(movies)
            _index_status["message"] = f"Found {len(movies)} movies — embedding…"
            yield f"data: {json.dumps(_index_status)}\n\n"
            await asyncio.sleep(0)

            # Rebuild collection from scratch
            try:
                _chroma_client.delete_collection("movies")
            except Exception:
                pass
            _collection = _chroma_client.create_collection("movies")

            BATCH = 32
            for i in range(0, len(movies), BATCH):
                batch = movies[i : i + BATCH]
                docs, metas, ids = [], [], []
                for mv in batch:
                    try:
                        doc, meta = _movie_doc(mv)
                        docs.append(doc)
                        metas.append(meta)
                        ids.append(hashlib.md5(mv.key.encode()).hexdigest())
                    except Exception as err:
                        print(f"  Skip {mv.title}: {err}")

                if docs:
                    embs = _embedder.encode(docs).tolist()
                    _collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)

                _index_status["progress"] = min(i + BATCH, len(movies))
                _index_status["message"] = f"Indexed {_index_status['progress']} / {len(movies)}"
                yield f"data: {json.dumps(_index_status)}\n\n"
                await asyncio.sleep(0)

            _index_status = {
                "running": False, "progress": len(movies), "total": len(movies),
                "message": f"{len(movies)} movies indexed ✓",
            }
            yield f"data: {json.dumps(_index_status)}\n\n"

        except Exception as e:
            _index_status = {"running": False, "progress": 0, "total": 0, "message": f"Error: {e}"}
            yield f"data: {json.dumps(_index_status)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(body: ChatBody):
    if not _claude:
        raise HTTPException(400, "Anthropic API key not configured")
    if not _collection or _collection.count() == 0:
        raise HTTPException(400, "Library not indexed yet — use the Index Library button first")

    msgs = [{"role": m.role, "content": m.content} for m in body.messages]
    turn = sum(1 for m in body.messages if m.role == "user")
    user_msgs = [m for m in body.messages if m.role == "user"]
    last_user_msg = user_msgs[-1].content if user_msgs else ""

    if turn <= 1:
        query = last_user_msg
    else:
        query = _synthesize_query(msgs)

    movies, distances = _search(query, n=15, filters=body.filters or {})
    _log_search(query, movies, distances, turn)
    system = _system_prompt(movies, body.mode)

    async def _gen():
        # First event: send movie metadata so frontend can render cards
        yield f"data: {json.dumps({'movies': movies})}\n\n"
        # Then stream the text
        with _claude.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system,
            messages=msgs,
        ) as s:
            for chunk in s.text_stream:
                yield f"data: {json.dumps({'t': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/poster")
async def get_poster(key: str):
    """Proxy Plex poster images (avoids CORS / auth issues in browser)."""
    if not CONFIG_FILE.exists():
        raise HTTPException(400, "Not configured")
    cfg = json.loads(CONFIG_FILE.read_text())
    import httpx
    url = f"{cfg['plex_url']}{key}?X-Plex-Token={cfg['plex_token']}&width=200&height=300"
    async with httpx.AsyncClient() as client:
        r = await client.get(url, follow_redirects=True, timeout=8)
        return Response(r.content, media_type=r.headers.get("content-type", "image/jpeg"))


@app.get("/api/plex-link")
async def get_plex_link(rating_key: str):
    """Generate Plex web URL for a movie (matches Overseerr implementation)."""
    if not _plex:
        raise HTTPException(400, "Not connected to Plex")

    machine_id = _plex.machineIdentifier

    # Extract just the numeric rating key if it's a full path
    if "/" in rating_key:
        # Extract from path like "/library/metadata/12345"
        parts = rating_key.split("/")
        rating_key = parts[-1] if parts[-1] else parts[-2]

    # Construct web URL using Overseerr's approach
    # URL encode the metadata key path: /library/metadata/{rating_key} -> %2Flibrary%2Fmetadata%2F{rating_key}
    web_url = f"https://app.plex.tv/desktop#!/server/{machine_id}/details?key=%2Flibrary%2Fmetadata%2F{rating_key}"

    return {
        "web_url": web_url
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
