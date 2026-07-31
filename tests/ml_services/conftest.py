"""Shared aiohttp fake for ML service unit tests."""
import asyncio
import random

import pytest


class _FakeResponse:
    def __init__(self, status, json_data, text_data):
        self.status = status
        self._json = json_data
        self._text = text_data

    async def json(self):
        return self._json

    async def text(self):
        return self._text


class _FakeRequestContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class _FakeClientSession:
    def __init__(self, *args, **kwargs):
        # preserve the responses list across reinitializations
        if not hasattr(self, "responses"):
            self.responses = []
        self.headers = kwargs.get("headers") or {}
        self.closed = False

    def request(self, method, url, **kwargs):
        if self.responses:
            status, json_data, text_data = self.responses.pop(0)
        else:
            status, json_data, text_data = 200, {}, ""
        return _FakeRequestContext(_FakeResponse(status, json_data, text_data))

    async def close(self):
        self.closed = True


@pytest.fixture
def fake_aiohttp(monkeypatch):
    """Patch aiohttp.ClientSession to use a controllable fake session."""
    session = _FakeClientSession()

    def _factory(*args, **kwargs):
        session.__init__(*args, **kwargs)
        return session

    monkeypatch.setattr("aiohttp.ClientSession", _factory)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("asyncio.sleep", _noop)
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
    return session
