import base64
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app


CLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "vascular lesion",
]
SAMPLE_IMAGE = Path(__file__).resolve().parents[1] / "frontend" / "samples" / "sample-nevus.jpg"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def image_bytes():
    return SAMPLE_IMAGE.read_bytes()


def test_root_returns_200(client):
    response = client.get("/")

    assert response.status_code == 200


def test_diagnose_returns_seven_classes(client):
    response = client.get("/diagnose")

    assert response.status_code == 200
    data = response.json()
    assert data["class_count"] == 7
    assert len(data["classes"]) == 7


def test_classes_returns_all_class_names(client):
    response = client.get("/classes")

    assert response.status_code == 200
    assert response.json()["classes"] == CLASS_NAMES


def test_predict_returns_prediction(client, image_bytes):
    response = client.post(
        "/predict",
        files={"image": ("sample-nevus.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json()["prediction"]


def test_predict_all_returns_scores_for_seven_classes(client, image_bytes):
    response = client.post(
        "/predict-all",
        files={"image": ("sample-nevus.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["all"]) == 7
    assert {item["label"] for item in data["all"]} == set(CLASS_NAMES)
    assert all(isinstance(item["confidence"], (int, float)) for item in data["all"])


def test_explain_returns_base64_image(client, image_bytes):
    response = client.post(
        "/explain",
        files={"image": ("sample-nevus.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    heatmap = response.json()["heatmap"]
    match = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/]+={0,2})", heatmap)
    assert match
    assert base64.b64decode(match.group(1), validate=True)


def test_predict_rejects_plain_text_without_crashing(client):
    response = client.post(
        "/predict",
        files={"image": ("not-an-image.txt", b"this is plain text", "text/plain")},
    )

    assert response.status_code >= 400
    assert client.get("/").status_code == 200
