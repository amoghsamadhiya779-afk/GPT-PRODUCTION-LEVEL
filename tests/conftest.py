# tests/conftest.py
"""Shared test setup: must run before app.api is imported anywhere."""

import os

# Disable slowapi rate limiting in tests.
os.environ["TESTING"] = "1"

# Never download real GPT-2 weights during tests: the app's cold-start path
# starts a background loader when no checkpoint exists.
import training.load_pretrained  # noqa: E402

training.load_pretrained.main = lambda: None
