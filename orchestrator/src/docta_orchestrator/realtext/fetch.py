"""Polite, cached GETs: one request at a time, a pause between them, a
generic client name, and every answer kept on disk, so a rerun asks nothing
twice and an interrupted collection resumes where it stopped."""

import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Wikimedia asks automated clients to name themselves; the public repository
# is enough, no personal address.
USER_AGENT = "docta-nm-corpus/0.1 (research; https://github.com/antikkorps/docta-nm)"


class FetchError(RuntimeError):
    """An answer not worth keeping: rerun later, the cache resumes the rest."""


class Fetcher:
    def __init__(self, cache: Path, pause: float = 1.0, opener=None, sleep=time.sleep):
        self.cache, self.pause, self.sleep = cache, pause, sleep
        self.opener = opener or urllib.request.urlopen
        self.last: float | None = None
        cache.mkdir(parents=True, exist_ok=True)

    def json(self, url: str, params: dict) -> dict:
        full = f"{url}?{urllib.parse.urlencode(sorted(params.items()))}"
        path = self.cache / f"{hashlib.sha256(full.encode()).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        self.wait()
        request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
        try:
            with self.opener(request, timeout=120) as response:
                body = response.read()
        except (http.client.HTTPException, urllib.error.URLError, TimeoutError) as e:
            raise FetchError(f"{url}: connection failed ({e!r})") from e
        answer = checked(body, url)  # before saving: a broken answer is never cached
        partial = path.with_suffix(".partial")
        partial.write_bytes(body)
        os.replace(partial, path)  # an interrupted write leaves no truncated file
        return answer

    def wait(self) -> None:
        now = time.monotonic()
        if self.last is not None and now - self.last < self.pause:
            self.sleep(self.pause - (now - self.last))
        self.last = time.monotonic()


def checked(body: bytes, url: str) -> dict:
    """The query service answers 200 and then cuts its JSON when a query runs
    past its time limit; the Wikipedia API answers 200 with an `error` field."""
    try:
        answer = json.loads(body)
    except json.JSONDecodeError as e:
        raise FetchError(f"{url}: answer cut short ({e}); the query likely ran too long") from e
    if "error" in answer:
        raise FetchError(f"{url}: {answer['error']}")
    return answer
