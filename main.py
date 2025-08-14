import os, re, string, difflib
from typing import Optional, List, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

# ==========
# API KEY (local testing). For prod: HADITH_API_KEY = os.getenv("HADITH_API_KEY")
# ==========
HADITH_API_KEY = "$2y$10$0MjsU1HftG2PkWlG7RnmNv7zw8Cr4mJPLejzV3K4YmsMWugUHS"
BASE = "https://hadithapi.com/api"

app = FastAPI(title="Islamic Chatbot Backend (no Musnad Ahmad / Silsila Sahiha)")

# CORS so a plain HTML file can call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------
# EXCLUDED BOOKS
# -----------------------
EXCLUDED_SLUGS = {"musnad-ahmad", "al-silsila-sahiha"}
EXCLUDED_NAMES = {"musnad ahmad", "musnad ahmed", "al silsila sahiha", "silsila sahiha",
                  "silsilah sahiha", "silsilah sahihah"}

# -----------------------
# LIGHTWEIGHT NLP (NO STEMMING IN QUERIES)
# -----------------------
USE_NLTK = True
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer

    try:
        _ = stopwords.words("english")
    except Exception:
        nltk.download("stopwords", quiet=True)
    try:
        _ = wordnet.synsets("test")
    except Exception:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    STOPWORDS = set(stopwords.words("english"))
    LEM = WordNetLemmatizer()
except Exception:
    USE_NLTK = False
    STOPWORDS = set("""
    a an the and or if is are was were be been being have has had do does did of to in on at for from by with as that this these those i you he she it we they them his her their our
    my your me him her us who whom which what when where why how not no nor so but into over under again further then once here there all any both each few more most other some such only
    own same than too very can will just don should now
    """.split())
    class _DummyLem:
        def lemmatize(self, w): return w
    LEM = _DummyLem()

# Synonyms & Islamic concept expansion (extend as needed)
SYNONYMS: Dict[str, List[str]] = {
    "fight": ["quarrel", "argue", "argument", "conflict", "dispute"],
    "anger": ["angry", "rage", "wrath", "temper"],
    "wife": ["spouse", "woman", "partner"],
    "respect": ["honor", "kindness", "good treatment", "rights"],
    "rights": ["duties", "obligations"],
    "alcohol": ["khamr", "intoxicant", "wine", "drinking"],
    "women": ["woman", "wives", "female"],
    "marriage": ["nikah", "spouse", "husband", "wife"],
    "backbiting": ["gheebah", "slander"],
    "arrogance": ["kibr", "pride", "haughtiness"],
    "truthfulness": ["honesty", "truth", "sincerity"],
    "parents": ["mother", "father", "obedience", "kindness to parents"],
}

CONCEPT_MAP: Dict[str, List[str]] = {
    "rights of women": ["women rights", "respect women", "kindness to women", "wives", "marriage"],
    "fight with my wife": ["marriage", "wives", "kindness", "respect", "anger", "conflict"],
    "disrespect": ["respect", "kindness", "good treatment"],
    "anger management": ["anger", "temper"],
    "respect your parents": ["parents", "obedience", "kindness to parents"],
}

# -----------------------
# BOOK NORMALIZATION (with fuzzy match but EXCLUDING the denied slugs)
# -----------------------
BOOKS_CACHE: List[Dict[str, str]] = []  # filled by /api/books for fuzzy matching

BOOK_ALIASES = {
    # Common core collections only — intentionally excluding musnad-ahmad & al-silsila-sahiha
    'bukhari': 'sahih-bukhari', 'sahih bukhari': 'sahih-bukhari',
    'muslim': 'sahih-muslim', 'sahih muslim': 'sahih-muslim',
    'tirmidhi': 'al-tirmidhi', 'al tirmidhi': 'al-tirmidhi',
    'abu dawud': 'abu-dawood', 'abu dawood': 'abu-dawood',
    'ibn majah': 'ibn-e-majah', 'ibn-e-majah': 'ibn-e-majah', 'ibn maja': 'ibn-e-majah',
    'nasai': 'sunan-nasai', 'an-nasai': 'sunan-nasai', 'annasai': 'sunan-nasai', 'sunan nasai': 'sunan-nasai',
    'mishkat': 'mishkat',
}

def normalize_book(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = re.sub(r"[–—]", "-", name.strip().lower())

    # hard block excluded names
    if n in EXCLUDED_NAMES:
        return None

    # 1) exact alias map (none of which are excluded)
    if n in BOOK_ALIASES:
        return BOOK_ALIASES[n]

    # 2) try matching directly to known slugs (excluding denied)
    for b in BOOKS_CACHE:
        slug = (b.get("slug") or "").lower()
        if slug in EXCLUDED_SLUGS:
            continue
        if n == slug:
            return slug

    # 3) fuzzy match against names & slugs from live cache (excluding denied)
    names = [ (b.get("name") or "").lower() for b in BOOKS_CACHE if (b.get("slug") or "").lower() not in EXCLUDED_SLUGS ]
    slugs = [ (b.get("slug") or "").lower() for b in BOOKS_CACHE if (b.get("slug") or "").lower() not in EXCLUDED_SLUGS ]
    candidates = list(set(names + slugs))

    if candidates:
        best = difflib.get_close_matches(n, candidates, n=1, cutoff=0.72)
        if best:
            match = best[0]
            for b in BOOKS_CACHE:
                slug = (b.get("slug") or "").lower()
                if slug in EXCLUDED_SLUGS:  # skip excluded
                    continue
                if (b.get("name") or "").lower() == match or slug == match:
                    return slug or None

    # 4) fall back (return None to avoid querying bad/unknown book)
    return None

# -----------------------
# JSON SHAPE HELPERS
# -----------------------
def extract_hadith_list(api_json) -> list:
    """Return a list of hadith dicts from various shapes."""
    if not isinstance(api_json, dict):
        return []
    value = api_json.get("hadiths", api_json.get("data"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("data"), list):
            return value.get("data")
        for v in value.values():
            if isinstance(v, list):
                return v
    return []

def normalize_hit(h: dict) -> dict | None:
    """Normalize a hadith record into a consistent shape."""
    if not isinstance(h, dict):
        return None
    book_obj = h.get("book")
    book_name = None
    book_slug = None
    if isinstance(book_obj, dict):
        book_name = book_obj.get("bookName") or h.get("bookName")
        book_slug = book_obj.get("bookSlug") or h.get("bookSlug")
    elif isinstance(book_obj, str):
        book_name = book_obj
        book_slug = h.get("bookSlug")
    else:
        book_name = h.get("bookName")
        book_slug = h.get("bookSlug")
    hadith_number = h.get("hadithNumber") or h.get("number") or h.get("hadith_no")
    chap = h.get("chapter")
    chapter_number = chap.get("chapterNumber") if isinstance(chap, dict) else h.get("chapterNumber")
    return {
        "book": book_slug,
        "bookName": book_name,
        "hadithNumber": hadith_number,
        "chapter": chapter_number,
        "status": h.get("status"),
        "textArabic": h.get("hadithArabic") or h.get("arabic") or h.get("textArabic"),
        "textEnglish": h.get("hadithEnglish") or h.get("english") or h.get("textEnglish"),
        "textUrdu": h.get("hadithUrdu") or h.get("urdu") or h.get("textUrdu"),
    }

async def get_json(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

# -----------------------
# QUERY BUILDING
# -----------------------
Lang = Literal["english", "urdu", "arabic"]

def build_search_url(
    keywords: Optional[str]=None,
    book_slug: Optional[str]=None,
    hadith_number: Optional[int]=None,
    chapter: Optional[int]=None,
    lang: Lang="english",
    page_size: int=5
) -> str:
    from urllib.parse import urlencode
    q: Dict[str, Any] = {"apiKey": HADITH_API_KEY, "paginate": page_size}
    if book_slug:
        q["book"] = book_slug
    if hadith_number:
        q["hadithNumber"] = hadith_number
    if chapter:
        q["chapter"] = chapter
    if keywords:
        field = {"english": "hadithEnglish", "urdu": "hadithUrdu", "arabic": "hadithArabic"}[lang]
        q[field] = keywords  # IMPORTANT: single token, not a phrase
    return f"{BASE}/hadiths?{urlencode(q)}"

# -----------------------
# NLP UTILITIES (no stemming in queries)
# -----------------------
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

def preprocess(text: str) -> str:
    s = text.lower().translate(PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_raw(text: str) -> List[str]:
    return preprocess(text).split()

def tokens_clean_and_lemmatize(text: str) -> List[str]:
    toks = tokenize_raw(text)
    toks = [t for t in toks if t not in STOPWORDS and len(t) > 2]
    toks = [LEM.lemmatize(t) for t in toks]  # NO STEMMING
    return toks

def bigrams(tokens: List[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]

def expand_with_synonyms(tokens: List[str]) -> List[str]:
    expanded = set(tokens)
    for t in tokens:
        for base, syns in SYNONYMS.items():
            if t == base or t in syns:
                expanded.add(base)
                for s in syns:
                    expanded.add(s)
        if USE_NLTK:
            from nltk.corpus import wordnet
            for syn in wordnet.synsets(t):
                for lemma in syn.lemmas():
                    w = lemma.name().replace("_", " ").lower()
                    if len(w) > 2 and w not in STOPWORDS:
                        expanded.add(LEM.lemmatize(w))
    return list(expanded)

def map_concepts(original_text: str) -> List[str]:
    s = preprocess(original_text)
    hits = []
    for phrase, mapped in CONCEPT_MAP.items():
        if phrase in s:
            hits.extend(mapped)
    return hits

def make_query_candidates(user_text: str) -> List[str]:
    """
    Priority:
      1) Short original phrase (3..5 words)  [only for concept capture; we'll split later]
      2) Concept-map expansions
      3) Lemmas + synonyms (joined)
      4) Bigrams from lemmas (joined)        [we will split to single tokens before querying]
      5) Top-N lemmas (by length)
      6) Single-word lemma fallbacks
    """
    lemmas = tokens_clean_and_lemmatize(user_text)
    if not lemmas:
        return []

    cands: List[str] = []

    short_original = preprocess(user_text)
    if 3 <= len(short_original.split()) <= 5:
        cands.append(short_original)

    mapped = map_concepts(user_text)
    if mapped:
        cands.append(" ".join(mapped))

    expanded = expand_with_synonyms(lemmas)
    if expanded:
        cands.append(" ".join(expanded[:8]))

    bgs = bigrams(lemmas)
    if bgs:
        cands.append(" ".join(bgs[:5]))

    top_lemmas = sorted(set(lemmas), key=lambda x: -len(x))[:6]
    if top_lemmas:
        cands.append(" ".join(top_lemmas))

    # single lemma fallbacks (will also be split later; this just preserves order)
    cands.extend(top_lemmas[:4])

    # dedupe preserving order
    seen = set()
    ordered: List[str] = []
    for q in cands:
        qn = q.strip()
        if not qn or qn in seen:
            continue
        seen.add(qn)
        ordered.append(qn)
    return ordered

# -----------------------
# DEDUPE & PRIORITY
# -----------------------
def dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep unique hadiths by (bookSlug, hadithNumber) while preserving order."""
    seen = set()
    out = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        key = (
            str(h.get("bookSlug") or (h.get("book") or {}).get("bookSlug") or h.get("book")),
            str(h.get("hadithNumber") or h.get("number") or h.get("hadith_no")),
        )
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out

PRIORITY_TOKENS = {
    "marriage","wife","wives","husband","women","respect","anger","parents",
    "truth","honesty","alcohol","khamr","backbiting","gheebah"
}

# ----------------
# API Endpoints
# ----------------
@app.get("/api/books")
async def list_books():
    url = f"{BASE}/books?apiKey={HADITH_API_KEY}"
    try:
        data = await get_json(url)
        books_raw = data.get("books") or data.get("data") or []
        if isinstance(books_raw, dict) and isinstance(books_raw.get("data"), list):
            books_raw = books_raw["data"]
        books = []
        for b in (books_raw or []):
            name = b.get("bookName") if isinstance(b, dict) else str(b)
            slug = (b.get("bookSlug") if isinstance(b, dict) else None)
            if not name:
                continue
            # filter out excluded slugs
            if slug and slug.lower() in EXCLUDED_SLUGS:
                continue
            books.append({"name": name, "slug": slug})

        # update global cache for fuzzy matching (already excluding denied)
        global BOOKS_CACHE
        BOOKS_CACHE = books

        return books
    except Exception:
        raise HTTPException(502, "Failed to load books from Hadith API")

# ----------------
# REFERENCE / TOPIC PARSER
# ----------------
def parse_user_query(q: str) -> Dict[str, Any]:
    """
    Returns:
      - {'kind':'reference','bookSlug'?, 'hadithNumber'?}
      - {'kind':'reference','bookSlug'?, 'chapter','chapterItem'}
      - {'kind':'topic','text': ...}
    Supports:
      "Sahih Muslim 125"
      "Sahih Muslim-125"
      "Sahih Muslim no#125"
      "Sahih Muslim no.125"
      "Bukhari 7:10"
      "125" (number only)
      long contextual text -> topic
    """
    s = q.strip().lower()
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"\s+", " ", s)

    # 1) Chapter style: "Book 7:10"
    m_chap = re.match(r"^([a-z\s'\.-]+?)\s+(\d{1,6})\s*:\s*(\d{1,4})$", s)
    if m_chap:
        book_slug = normalize_book(m_chap.group(1))
        # If the user named an excluded book, treat as None (won't be queried)
        if book_slug in EXCLUDED_SLUGS or m_chap.group(1).strip().lower() in EXCLUDED_NAMES:
            book_slug = None
        return {"kind": "reference", "bookSlug": book_slug,
                "chapter": int(m_chap.group(2)), "chapterItem": int(m_chap.group(3))}

    # 2) Flexible "Book <sep> number"
    m_flex = re.match(r"^([a-z\s'\.-]+?)\s*(?:no\.?\s*#?\s*|no\s*#?\s*|#|-|\s)\s*(\d{1,6})$", s)
    if m_flex:
        raw_book = m_flex.group(1).strip().lower()
        book_slug = normalize_book(raw_book)
        if (book_slug in EXCLUDED_SLUGS) or (raw_book in EXCLUDED_NAMES):
            # treat as if no valid book provided
            book_slug = None
        return {"kind": "reference", "bookSlug": book_slug, "hadithNumber": int(m_flex.group(2))}

    # 3) Pure number (user omitted the book)
    if re.fullmatch(r"\d{1,6}", s):
        return {"kind": "reference", "hadithNumber": int(s)}

    # 4) Topic
    return {"kind": "topic", "text": q}

# ----------------
# HADITH SEARCH
# ----------------
@app.post("/api/hadith/search")
async def hadith_search(payload: Dict[str, Any]):
    """
    Contextual queries via NLP (without stemming).
    Fan-out: one token per request to avoid 404 on multi-word 'hadithEnglish'.
    Merge & dedupe results; stop once pageSize unique hits collected.

    Conflict rule:
      If a book chip is selected (bookSlug param) AND the prompt mentions a different book,
      DO NOT fetch (return empty hits + noBookMatch=True).

    Also: If the selected or referenced book is one of the excluded slugs, do not query.
    """
    query: str = payload.get("query", "")
    bookSlugParam: Optional[str] = payload.get("bookSlug")
    bookSlugParam = (bookSlugParam or None)
    if bookSlugParam and bookSlugParam.lower() in EXCLUDED_SLUGS:
        # selected an excluded book — do not query
        return {"hits": [], "noBookMatch": True}

    # normalize allowed books only
    bookSlugParam = normalize_book(bookSlugParam) if bookSlugParam else None

    lang: Lang = payload.get("lang", "english")
    pageSize: int = int(payload.get("pageSize", 5))

    parsed = parse_user_query(query)

    hits_raw: List[Dict[str, Any]] = []
    noBookMatch = False

    try:
        if parsed["kind"] == "reference":
            refBook = parsed.get("bookSlug")

            # ----- Conflict rule -----
            if bookSlugParam and refBook and (refBook != bookSlugParam):
                noBookMatch = True
                hits_raw = []
            else:
                effective_book = refBook or bookSlugParam

                # If reference explicitly names an excluded book, do not query
                if effective_book and effective_book.lower() in EXCLUDED_SLUGS:
                    noBookMatch = True
                    hits_raw = []
                else:
                    if parsed.get("hadithNumber"):
                        url = build_search_url(
                            hadith_number=parsed["hadithNumber"],
                            book_slug=effective_book,
                            page_size=pageSize
                        )
                        print("REF URL:", url)
                        data = await get_json(url)
                        hits_raw = extract_hadith_list(data)

                    elif parsed.get("chapter") and parsed.get("chapterItem"):
                        url = build_search_url(
                            chapter=parsed["chapter"],
                            book_slug=effective_book,
                            page_size=50
                        )
                        print("CHAPTER URL:", url)
                        data = await get_json(url)
                        arr = extract_hadith_list(data)
                        idx = int(parsed["chapterItem"]) - 1
                        hits_raw = [arr[idx]] if 0 <= idx < len(arr) else []

                    if bookSlugParam and not hits_raw:
                        noBookMatch = True

        else:
            # ---- Contextual / topic flow with single-word fan-out ----
            candidates = make_query_candidates(parsed["text"])
            MAX_UNIQUE = pageSize            # how many unique hadiths to return
            MAX_ATTEMPTS = 12                # cap network calls
            attempts = 0
            aggregated: List[Dict[str, Any]] = []

            for cand in candidates:
                if attempts >= MAX_ATTEMPTS or len(aggregated) >= MAX_UNIQUE:
                    break

                # split candidate into tokens; prioritize high-signal words, then by length
                tokens = [t for t in cand.split() if len(t) > 2]
                tokens = sorted(
                    set(tokens),
                    key=lambda x: (0 if x.lower() in PRIORITY_TOKENS else 1, -len(x))
                )

                for tok in tokens:
                    if attempts >= MAX_ATTEMPTS or len(aggregated) >= MAX_UNIQUE:
                        break
                    url = build_search_url(
                        keywords=tok,                  # ONE TOKEN PER REQUEST
                        book_slug=bookSlugParam,
                        lang=lang,
                        page_size=MAX_UNIQUE
                    )
                    print("SEARCH URL (single):", url)
                    try:
                        data = await get_json(url)
                    except httpx.HTTPError as e:
                        print("HTTP ERROR(single):", e)
                        attempts += 1
                        continue

                    part = extract_hadith_list(data)
                    attempts += 1

                    if part:
                        aggregated.extend(part)
                        aggregated = dedupe_hits(aggregated)[:MAX_UNIQUE]
                        if len(aggregated) >= MAX_UNIQUE:
                            break

            hits_raw = aggregated

            if bookSlugParam and not hits_raw:
                noBookMatch = True

    except httpx.HTTPError as e:
        print("HTTP ERROR:", e)
        hits_raw = []

    # Normalize for frontend
    norm = []
    for h in hits_raw:
        nh = normalize_hit(h)
        if nh:
            norm.append(nh)

    return {"hits": norm, "noBookMatch": noBookMatch}

# ----------------
# ANSWER STUB (Gemini placeholder)
# ----------------
@app.post("/api/answer")
async def compose_answer(payload: Dict[str, Any]):
    """
    Stub for Gemini: in your real app, call Gemini here with strict instructions
    to ONLY quote hadiths present in apiResult['hits'].
    """
    question = payload.get("question", "")
    apiResult = payload.get("apiResult", {"hits": [], "noBookMatch": False})

    no_match_note = "No Hadith related to your query found in that book." if apiResult.get("noBookMatch") else ""

    lines = [f"Question: {question}", ""]
    if apiResult.get("hits"):
        lines.append("Hadith references:")
        for h in apiResult["hits"][:3]:
            ref = f"{(h.get('bookName') or h.get('book') or 'Unknown')} #{h.get('hadithNumber')}"
            lines.append(f"- {ref}")
        lines.append("")
        lines.append("Explanation (placeholder): Provide Islamic guidance here (Gemini).")
    else:
        lines.append("No hadith matched. Provide general Islamic guidance without quoting hadith.")
    if no_match_note:
        lines.append("")
        lines.append(no_match_note)

    return {"answer": "\n".join(lines)}
