import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan: loads B8 + the dataset index
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["validation_clips"] > 0


def test_random_prediction_is_a_distribution(client):
    r = client.get("/predict/random-validation", params={"seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert len(body["probabilities"]) == 8
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert body["prediction"] == max(body["probabilities"], key=body["probabilities"].get)
    assert len(body["frames"]) == 9


def test_seed_is_reproducible(client):
    a = client.get("/predict/random-validation", params={"seed": 7}).json()
    b = client.get("/predict/random-validation", params={"seed": 7}).json()
    assert (a["video_id"], a["clip_id"]) == (b["video_id"], b["clip_id"])


def test_unknown_clip_is_404(client):
    r = client.post("/predict/clip", json={"video_id": "nope", "clip_id": "0"})
    assert r.status_code == 404


def test_video_endpoint_returns_gif(client):
    r = client.get("/predict/random-validation/video", params={"seed": 1})
    assert r.status_code == 200 and r.headers["content-type"] == "image/gif"
    assert r.content[:3] == b"GIF"
