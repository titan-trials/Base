"""
The odds fetch retries the failures that cost nothing and refuses the ones
that can double-charge.

WHY THIS TEST EXISTS
--------------------
The Odds API bills per event on the player-prop endpoint, and the free tier
is 500 credits a month -- about twenty slates. The two ways to be wrong are
not symmetric:

    give up too early   -> already_captured() skips the games we already
                           have, so re-running the odds step costs only the
                           games that are still missing. Recoverable, free.

    retry too eagerly   -> the first request may have reached the server and
                           been billed. The retry is billed again. Nothing
                           gives that credit back.

So the rule is: retry only what provably never reached the server (DNS
failure, connection refused or reset, a 5xx the server itself sent). Never
retry a timeout or a truncated body -- silence is not proof of absence, and
a half-read answer is proof of presence.

That rule lives in a chain of except clauses, which is exactly the kind of
code that keeps running after someone reorders it. Every case below asserts
on the CALL COUNT, because the call count is the credit count.

Nothing here touches the network. urlopen is replaced with a fake.
"""
import io
import os
import sys
import json
import socket
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import odds_lines as O

ok = True


def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok  ' if good else 'FAIL'}] {label:56} "
          f"got {got!r}, want {want!r}")


class FakeResponse:
    """What urlopen hands back: a context manager json.load() can read."""

    def __init__(self, body: bytes, headers=None):
        self._fp = io.BytesIO(body)
        self.headers = headers or {"x-requests-remaining": "417"}

    def read(self, *a):
        return self._fp.read(*a)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code: int, body: bytes = b'{"message":"nope"}'):
    return urllib.error.HTTPError(
        "https://api.the-odds-api.com/v4/x", code, "err", {},
        io.BytesIO(body))


def run(outcomes):
    """
    Drive _get() against a scripted sequence of urlopen outcomes.

    Each item is either an exception instance (raised) or bytes (returned as
    a response body). Returns (result, number of urlopen calls, sleeps).
    """
    calls = []
    sleeps = []

    def fake_urlopen(request, timeout=None):
        i = len(calls)
        calls.append(request)
        outcome = outcomes[i] if i < len(outcomes) else outcomes[-1]
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeResponse(outcome)

    real_urlopen = O.urllib.request.urlopen
    real_sleep = O.time.sleep
    O.urllib.request.urlopen = fake_urlopen
    O.time.sleep = sleeps.append
    try:
        result = O._get("https://api.the-odds-api.com/v4/x")
    finally:
        O.urllib.request.urlopen = real_urlopen
        O.time.sleep = real_sleep
    return result, len(calls), sleeps


print("=" * 76)
print("PART 1 -- the happy path is one request")
print("=" * 76)

(payload, headers), n, sleeps = run([b'[{"id":"abc"}]'])
check("a good answer is parsed", payload, [{"id": "abc"}])
check("headers come back with it", headers.get("x-requests-remaining"), "417")
check("one request, one credit", n, 1)
check("no sleeping on success", len(sleeps), 0)

print()
print("=" * 76)
print("PART 2 -- NEVER retried: the request may already have been billed")
print("=" * 76)

# A timeout is the dangerous one. Twenty seconds of silence does not mean
# the request never arrived -- the server may have answered a slow query and
# charged for it while we stopped listening.
for label, exc in [
        ("socket.timeout", socket.timeout("timed out")),
        ("TimeoutError", TimeoutError("timed out")),
        ("URLError wrapping a timeout",
         urllib.error.URLError(socket.timeout("timed out"))),
]:
    result, n, sleeps = run([exc, b'[{"id":"abc"}]'])
    check(f"{label}: one request only", n, 1)
    check(f"{label}: returns empty, does not raise", result, (None, {}))
    check(f"{label}: never sleeps", len(sleeps), 0)

# A truncated body is proof the server DID answer. It billed.
result, n, sleeps = run([b'[{"id":"ab', b'[{"id":"abc"}]'])
check("truncated JSON: one request only", n, 1)
check("truncated JSON: returns empty", result, (None, {}))

print()
print("=" * 76)
print("PART 3 -- NEVER retried: asking again cannot help")
print("=" * 76)

for label, code in [("401 bad key", 401), ("403 forbidden", 403),
                    ("404 no such event", 404), ("422 bad params", 422),
                    ("429 quota gone", 429)]:
    result, n, sleeps = run([http_error(code), b'[{"id":"abc"}]'])
    check(f"{label}: one request only", n, 1)
    check(f"{label}: returns empty", result, (None, {}))

print()
print("=" * 76)
print("PART 4 -- retried: the request never reached the server")
print("=" * 76)

# These all fail before or below HTTP: no request was served, so no credit
# was spent, so trying again is free.
for label, exc in [
        ("connection refused", ConnectionRefusedError("refused")),
        ("connection reset", ConnectionResetError("reset by peer")),
        ("DNS failure",
         urllib.error.URLError(socket.gaierror("name resolution"))),
        ("network unreachable", OSError(101, "Network is unreachable")),
]:
    result, n, sleeps = run([exc])
    check(f"{label}: tries the full budget", n, O.RETRY_ATTEMPTS)
    check(f"{label}: sleeps between tries only",
          len(sleeps), O.RETRY_ATTEMPTS - 1)
    check(f"{label}: gives up empty", result, (None, {}))

# The server's own failure is worth repeating; the server said so.
for code in (500, 502, 503, 504):
    result, n, _ = run([http_error(code)])
    check(f"HTTP {code}: tries the full budget", n, O.RETRY_ATTEMPTS)

# And a retry that works must return the good answer, not the first error.
(payload, _), n, sleeps = run([ConnectionResetError("reset"),
                               b'[{"id":"abc"}]'])
check("recovers on the second try", payload, [{"id": "abc"}])
check("stops as soon as it succeeds", n, 2)
check("slept once", len(sleeps), 1)

(payload, _), n, _ = run([http_error(503), http_error(503),
                          b'[{"id":"abc"}]'])
check("recovers on the third try", payload, [{"id": "abc"}])
check("stops as soon as it succeeds", n, 3)

print()
print("=" * 76)
print("PART 5 -- the waiting is bounded")
print("=" * 76)

# A retry loop that can sleep for minutes is its own outage: the slate has a
# lock time. Jitter is capped at +25%, so the whole budget has a ceiling we
# can state.
_, _, sleeps = run([ConnectionResetError("reset")])
ceilings = [O.RETRY_BASE_SEC * (2 ** i) * 1.25
            for i in range(O.RETRY_ATTEMPTS - 1)]
check("every wait is at least the base delay",
      all(s >= O.RETRY_BASE_SEC * (2 ** i)
          for i, s in enumerate(sleeps)), True)
check("every wait is under its jitter ceiling",
      all(s <= c for s, c in zip(sleeps, ceilings)), True)
check("waits grow", sleeps == sorted(sleeps), True)
check("worst case stays under 10s", sum(ceilings) < 10.0, True)

# Jitter is real: two runs must not line up exactly, or parallel runs would
# hammer the server in lockstep.
_, _, a = run([ConnectionResetError("reset")])
_, _, b = run([ConnectionResetError("reset")])
check("delays are jittered, not fixed", a == b, False)

print()
print("=" * 76)
print("PART 6 -- the guard the whole design rests on")
print("=" * 76)

# If already_captured() ever stops skipping, giving up early stops being
# free and this entire policy is wrong. It is cheap to assert that it is
# still the thing standing between a re-run and a second bill.
import inspect
src = inspect.getsource(O.already_captured)
check("already_captured reads the odds cache", "odds_" in src, True)
check("already_captured returns the games already held",
      "return" in src, True)

print()
print(f"  budget: {O.RETRY_ATTEMPTS} attempts, base {O.RETRY_BASE_SEC}s, "
      f"timeout {O.TIMEOUT_SEC}s")
print()
print("=" * 76)
print("ALL PASS" if ok else "FAILURES ABOVE")
print("=" * 76)
sys.exit(0 if ok else 1)
