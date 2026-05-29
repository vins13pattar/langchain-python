"""
03_structured_output.py
=======================
Demonstrates STRUCTURED OUTPUT directly from models.

Concepts covered:
  - model.with_structured_output() — force the model to return structured data
  - Pydantic BaseModel             — structured format with validation and types
  - TypedDict                     — lightweight python dictionary schema
  - Single object extraction
  - List of objects extraction (nested schemas)

Structured output is essential when you need the model's output to be parsed
reliably by downstream code (e.g. database insertion, UI components).
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model

load_dotenv()

# ════════════════════════════════════════════════════════════════════
# 1. DEFINE SCHEMAS
# ════════════════════════════════════════════════════════════════════

# ── Option A: Pydantic BaseModel (Recommended) ──────────────────────
# Provides field-level descriptions (which the model reads as instructions),
# auto-validation, and type-safety in IDEs.

class MovieFeature(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="Director of the movie")
    release_year: int = Field(description="The year the movie was released")
    genres: List[str] = Field(description="List of genres")
    rating: Optional[float] = Field(None, description="IMDb or general rating out of 10")
    summary: str = Field(description="A brief 1-sentence summary of the plot")


class MovieListing(BaseModel):
    """A list of movies extracted from text."""
    movies: List[MovieFeature] = Field(description="A list of movies found in the text")


# ── Option B: TypedDict ─────────────────────────────────────────────
# A standard Python dictionary wrapper. Useful if you prefer working with
# raw dicts without Pydantic dependencies, though validation is weaker.

class BookSchema(TypedDict):
    title: str
    author: str
    publish_year: int
    summary: str


print("=" * 60)
print("Model Structured Output Demo")
print("=" * 60)

# Initialize standard model
model = init_chat_model("openai:gpt-4o-mini")


# ════════════════════════════════════════════════════════════════════
# 2. PYDANTIC STRUCTURED OUTPUT (Single Object)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("1. Single Object Extraction via Pydantic")
print("─" * 60)

# Bind the schema to the model
structured_model_pydantic = model.with_structured_output(MovieFeature)

text_about_movie = (
    "In 1999, the Wachowskis directed a sci-fi masterpiece called The Matrix. "
    "It stars Keanu Reeves as Neo and has a great 8.7 rating. It blends sci-fi, "
    "action, and philosophy as Neo discovers the world is a simulation."
)

print(f"\nSource Text:\n  {text_about_movie}\n")

print("Invoking model...")
movie_result = structured_model_pydantic.invoke(text_about_movie)

print(f"\nResult class: {type(movie_result).__name__}")
print(f"Movie Title:  {movie_result.title}")
print(f"Director:     {movie_result.director}")
print(f"Released:     {movie_result.release_year}")
print(f"Genres:       {movie_result.genres}")
print(f"Rating:       {movie_result.rating}/10")
print(f"Summary:      {movie_result.summary}")


# ════════════════════════════════════════════════════════════════════
# 3. NESTED SCHEMA STRUCTURED OUTPUT (List of Objects)
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("2. Multi-Object Extraction via Nested Pydantic Schema")
print("─" * 60)

structured_listing_model = model.with_structured_output(MovieListing)

text_about_movies = (
    "Christopher Nolan's Interstellar (2014) is a mind-bending sci-fi epic starring Matthew McConaughey, "
    "exploring wormholes and deep space. It holds an 8.6 rating. In 2010, Nolan also directed Inception, "
    "an action/sci-fi heist film set in dreams, rated 8.8. Let's not forget The Dark Knight (2008), "
    "an action/drama superhero film directed by Nolan that scored a massive 9.0 rating."
)

print(f"\nSource Text:\n  {text_about_movies}\n")

print("Invoking model...")
listing_result = structured_listing_model.invoke(text_about_movies)

print(f"\nExtracted {len(listing_result.movies)} movies:")
for i, movie in enumerate(listing_result.movies, 1):
    print(f"\n  🎥 Movie {i}:")
    print(f"    Title:    {movie.title}")
    print(f"    Released: {movie.release_year}")
    print(f"    Rating:   {movie.rating}")
    print(f"    Genres:   {movie.genres}")
    print(f"    Summary:  {movie.summary}")


# ════════════════════════════════════════════════════════════════════
# 4. TYPEDDICT STRUCTURED OUTPUT
# ════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("3. Extraction via TypedDict (returns raw Python dicts)")
print("─" * 60)

structured_model_typeddict = model.with_structured_output(BookSchema)

text_about_book = (
    "To Kill a Mockingbird was written by Harper Lee and published in 1960. "
    "The story deals with racial injustice and the loss of innocence in the American South."
)

print(f"\nSource Text:\n  {text_about_book}\n")

print("Invoking model...")
book_result = structured_model_typeddict.invoke(text_about_book)

print(f"\nResult class: {type(book_result).__name__}")
print(f"Result dict:  {book_result}")
print(f"Book Title:   {book_result.get('title')}")
print(f"Author:       {book_result.get('author')}")
print(f"Published:    {book_result.get('publish_year')}")
print(f"Summary:      {book_result.get('summary')}")
