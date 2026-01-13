import os
import pickle
from typing import List, Optional, Union, Dict, Any, Tuple
from dotenv import load_dotenv
import httpx
import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")    
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_500 = "https://image.tmdb.org/t/p/w500"
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is not set in the environment variables.")

app = FastAPI(title="Movie Recommendation System", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],)

@app.get("/")
def read_root():
    return {"message": "Movie Recommendation API is running"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")
df: Optional[pd.DataFrame] = None
indices_obj: Optional[Dict[str, int]] = None
tfidf_matrix: Optional[Any] = None
tfidf_obj: Optional[Any] = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None
IDX_TO_TITLE: Optional[Dict[int, str]] = None

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None
    overview: Optional[str] = None

class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    overview: Optional[str] = None
    genres: List[dict] = []
    backdrop_url: Optional[str] = None    

class TFIDFRecItem(BaseModel):
    title: str
    score: float  
    tmdb: Optional[TMDBMovieCard] = None  

class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]

def _norm_title(title: str) -> str:
    return str(title).strip().lower()

def make_img_url(path: Optional[str]) -> Optional[str]:
    if path:
        return f"{TMDB_500}{path}"
    return None

async def tmdb_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE_URL}{path}", params=q)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail="TMDB API is unavailable") from e
    
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail="TMDB API error")
    return r.json()

async def tmdb_cards_from_results(results: List[Dict[str, Any]], limit: int = 5) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in results[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=m.get("id"),
            title=m.get("title"),
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
            overview=m.get("overview"),
        ))
    return out

async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
 
    return TMDBMovieDetails(
        tmdb_id=data.get("id"),
        title=data.get("title"),
        poster_url=make_img_url(data.get("poster_path")),
        release_date=data.get("release_date"),
        overview=data.get("overview"),
        genres=data.get("genres",   []),
        backdrop_url=make_img_url(data.get("backdrop_path")),
    )

async def tmdb_search_movies(query: str, page: int = 1, limit: int = 5) -> Dict[str, Any]:
    data = await tmdb_get("/search/movie", {"query": query, "language": "en-US", "include_adult": False, "page": page})
    return data

async def tmdb_search_first(query: str) -> Optional[Dict]:
    data = await tmdb_search_movies(query=query, page=1, limit=1)
    results = data.get("results", [])
    return results[0] if results else None

def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    raise HTTPException(
        status_code=404, detail=f"Title not found in local dataset: '{title}'"
    )

def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    returns list of (title and score) from local df using cosine similarity
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")

    idx = get_local_idx_by_title(query_title)
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    order = np.argsort(-scores)
    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title. If not found, returns None
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None

@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)

    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)

    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pickle must contain a dataframe with a 'title' column")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),

):
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language":"en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page":1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")

@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),

):
    return await tmdb_search_movies(query=query, page=page)

@app.get("/movies/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)

@app.get("/recommend/genre/{tmdb_id}", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Path(...),
    limit: int = Query(18, ge=1, le=50),
):
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get("/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]

@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),

):
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )
    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)
    tfidf_recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
    genre_recs = await recommend_genre(tmdb_id, limit=genre_limit)
    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=[{"title": t, "score": s} for t, s in tfidf_recs],
        genre_recommendations=genre_recs
    )