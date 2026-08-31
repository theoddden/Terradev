"""Tests for KV cache avoided-token accounting."""
from terradev_cli.core.inference_router import PrefixCacheIndex


def test_uncached_tokens_not_gamed_by_self_warming():
    idx = PrefixCacheIndex(prefix_tokens=64)

    # A real request with 1000 tokens, nothing cached yet -> all uncached
    idx.record("hello world ", "ep1", token_count=1000, cached_token_count=0, engine="vllm", block_size=16)
    stats = idx.get_stats("ep1")
    initial_uncached = stats["uncached_tokens"]
    assert initial_uncached == 1000

    # Self-warming: optimizer re-requests the same prefix 10 times, fully cached
    for _ in range(10):
        idx.lookup("hello world ", token_count=1000)

    stats = idx.get_stats("ep1")
    # Residual work did NOT increase despite the extra "hits"
    assert stats["uncached_tokens"] == initial_uncached
    assert stats["uncached_ratio"] == initial_uncached / 11000
    # Cached tokens did increase (this is the gameable metric)
    assert stats["cached_prompt_tokens"] == 10000


def test_different_block_size_across_engines():
    idx = PrefixCacheIndex(prefix_tokens=64)

    # vLLM with large block size cannot cache a 240-token system prompt
    idx.record("system prompt " * 20, "vllm-ep", token_count=240, cached_token_count=0, engine="vllm", block_size=784)
    vllm_hits = idx.lookup("system prompt " * 20, token_count=240)
    assert not vllm_hits

    # SGLang with small block size can cache it
    idx.record("system prompt " * 20, "sglang-ep", token_count=240, cached_token_count=240, engine="sglang", block_size=32)
    sglang_hits = idx.lookup("system prompt " * 20, token_count=240)
    assert sglang_hits
    assert sglang_hits[0][2] == 240

    summary = idx.get_summary()
    assert summary["by_engine"]["vllm"]["uncached_tokens"] != summary["by_engine"]["sglang"]["uncached_tokens"]
    assert summary["by_engine"]["vllm"]["uncached_tokens"] == 240
    assert summary["by_engine"]["sglang"]["uncached_tokens"] == 0


def test_partial_prefix_uncached_tokens():
    idx = PrefixCacheIndex(prefix_tokens=64)

    # First request primes a 64-token prefix and is fully cached
    idx.record("first part " * 32, "ep1", token_count=64, cached_token_count=64, engine="vllm", block_size=16)

    # Later request reuses first 64 tokens but adds 200 new ones
    hits = idx.lookup("first part " * 32 + "new part " * 50, token_count=264)
    assert hits
    assert hits[0][2] == 64

    stats = idx.get_stats("ep1")
    assert stats["uncached_tokens"] == 200
    # Total prompt tokens includes the initial 64-token prime plus the 264-token request
    assert abs(stats["uncached_ratio"] - (200 / 328)) < 1e-9
