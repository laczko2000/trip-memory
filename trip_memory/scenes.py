"""Suggest what kind of scene a picture shows (beach, city, food...) with the free CLIP model.

CLIP turns pictures and sentences into comparable "meaning" vectors, so a picture can be matched
against descriptions such as "a holiday photo of a sandy beach". The model (about 340 MB) is
downloaded once and runs on your computer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision"
TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"

# Category -> descriptions CLIP compares each picture against.
CATEGORIES: dict[str, list[str]] = {
    "Beach": ["a sandy beach by the sea", "people relaxing on a beach"],
    "Sea & boats": ["the open sea", "a boat on the water"],
    "Pool": ["a swimming pool", "people swimming in a pool"],
    "Mountains": ["mountains and hills", "a mountain hiking trail"],
    "Nature": ["a forest or green countryside", "a field with trees and flowers"],
    "Lake & river": ["a lake", "a river"],
    "City": ["a busy city street", "old town streets and squares"],
    "Landmark": ["a famous monument or landmark", "a cathedral, castle or historic building"],
    "Museum": ["inside a museum or gallery", "paintings and statues in a museum"],
    "Food & drinks": ["a plate of food on a table", "drinks and cocktails"],
    "Restaurant & café": ["people eating in a restaurant", "a cafe terrace"],
    "Party & nightlife": ["a party with people dancing", "a bar at night"],
    "Concert & show": ["a concert stage", "a show or performance"],
    "Sunset": ["a sunset", "a sunrise over the horizon"],
    "Night": ["a city at night with lights", "a dark night scene"],
    "Group photo": ["a group of friends posing for a photo", "a family photo"],
    "Selfie & portrait": ["a selfie", "a close-up portrait of a person"],
    "Kids": ["children playing", "a small child"],
    "Animals": ["animals", "a pet dog or cat"],
    "Snow": ["snow and skiing", "a snowy landscape"],
    "Sports & activities": ["people doing sports", "an outdoor adventure activity"],
    "Car & road trip": ["a car on a road trip", "a view from a car window"],
    "Travel day": ["an airport or airplane", "a train station or train"],
    "Hotel": ["a hotel room", "a hotel lobby or balcony"],
}
SCENES = list(CATEGORIES)


class SceneTagger:
    def __init__(self, cache_dir: Path):
        from fastembed import ImageEmbedding, TextEmbedding

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.image_model = ImageEmbedding(IMAGE_MODEL, cache_dir=str(cache_dir))
        text_model = TextEmbedding(TEXT_MODEL, cache_dir=str(cache_dir))
        rows = []
        for descriptions in CATEGORIES.values():
            vectors = np.array(list(text_model.embed([f"a holiday photo of {d}" for d in descriptions])))
            mean = _unit(vectors).mean(axis=0)
            rows.append(mean / np.linalg.norm(mean))
        self.category_vectors = np.array(rows)

    def embed(self, image_paths: list[Path]) -> np.ndarray:
        """One meaning vector for a photo, or for a video from several of its frames."""
        vectors = _unit(np.array(list(self.image_model.embed([str(p) for p in image_paths]))))
        mean = vectors.mean(axis=0)
        return (mean / np.linalg.norm(mean)).astype(np.float32)

    def classify(self, vector: np.ndarray) -> list[tuple[str, float]]:
        """The three most likely scene types with a confidence between 0 and 1."""
        return classify(vector, self.category_vectors)


def classify(vector: np.ndarray, category_vectors: np.ndarray) -> list[tuple[str, float]]:
    logits = 100 * category_vectors @ vector
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(probs)[::-1][:3]
    return [(SCENES[i], float(probs[i])) for i in top]


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
