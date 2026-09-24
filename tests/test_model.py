# tests/test_model.py
"""Unit tests for the GPT model architecture.

Usage:
    py -m pytest tests/test_model.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from model.config import GPTConfig
from model.gpt import GPTModel, generate_text_simple, count_parameters
from model.attention import MultiHeadAttention
from model.layers import LayerNorm, GELU, FeedForward, TransformerBlock


# Use a tiny config for fast tests
TINY_CONFIG = {
    "vocab_size": 100,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


class TestGPTConfig:
    def test_default_config(self):
        cfg = GPTConfig()
        assert cfg.vocab_size == 50257
        assert cfg.emb_dim == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12

    def test_to_dict(self):
        cfg = GPTConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "emb_dim" in d
        assert "drop_rate" in d  # mapped from dropout


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(
            d_in=64, d_out=64, context_length=32,
            dropout=0.0, num_heads=4,
        )
        x = torch.randn(2, 10, 64)
        out, _ = mha(x)
        assert out.shape == (2, 10, 64)

    def test_causal_mask_exists(self):
        mha = MultiHeadAttention(
            d_in=64, d_out=64, context_length=16,
            dropout=0.0, num_heads=4,
        )
        assert hasattr(mha, "mask")
        assert mha.mask.shape == (16, 16)


class TestLayers:
    def test_layer_norm(self):
        ln = LayerNorm(64)
        x = torch.randn(2, 10, 64)
        out = ln(x)
        assert out.shape == x.shape

    def test_gelu(self):
        gelu = GELU()
        x = torch.randn(2, 10, 64)
        out = gelu(x)
        assert out.shape == x.shape

    def test_feedforward(self):
        ff = FeedForward(TINY_CONFIG)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_transformer_block(self):
        block = TransformerBlock(TINY_CONFIG)
        x = torch.randn(2, 10, 64)
        out, _ = block(x)
        assert out.shape == x.shape


class TestGPTModel:
    def test_forward_shape(self):
        model = GPTModel(TINY_CONFIG)
        idx = torch.randint(0, 100, (2, 16))
        logits = model(idx)
        assert logits.shape == (2, 16, 100)

    def test_generate_simple(self):
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.randint(0, 100, (1, 5))
        out = generate_text_simple(model, idx, max_new_tokens=10, context_size=32)
        assert out.shape == (1, 15)  # 5 original + 10 generated

    def test_count_parameters(self):
        model = GPTModel(TINY_CONFIG)
        n = count_parameters(model)
        assert n > 0
        assert isinstance(n, int)

    def test_kv_cache_equivalence(self):
        from model.gpt import generate
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.randint(0, 100, (1, 8))
        
        # Generation with standard (no cache) mode
        torch.manual_seed(42)
        out_no_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=5, use_cache=False
        )
        
        # Generation with KV-cache mode
        torch.manual_seed(42)
        out_with_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=5, use_cache=True
        )
        
        assert torch.equal(out_no_cache, out_with_cache), "Cached and non-cached generation must yield identical tokens"

    def test_top_p_and_repetition_penalty(self):
        from model.gpt import generate, _sample_next_token
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.randint(0, 100, (1, 8))

        # Test _sample_next_token directly with Top-P and penalty
        logits = torch.randn(1, 100)
        idx_next = _sample_next_token(
            logits, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.2, idx=idx
        )
        assert idx_next.shape == (1, 1)
        assert 0 <= idx_next.item() < 100

        # Test generate with top_p and repetition_penalty equivalence (cached vs non-cached)
        torch.manual_seed(42)
        out_no_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=5, top_p=0.9, repetition_penalty=1.2, use_cache=False
        )

        torch.manual_seed(42)
        out_with_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=5, top_p=0.9, repetition_penalty=1.2, use_cache=True
        )

        assert torch.equal(out_no_cache, out_with_cache), "Cached and non-cached must yield identical tokens when using Top-P and repetition penalty"

    def test_no_repeat_ngram_size(self):
        from model.gpt import generate
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.tensor([[10, 11, 12, 10, 11]], dtype=torch.long)

        # Force the model into a 10 -> 11 -> 12 -> 10 loop.
        original_forward = model.forward
        next_token = {10: 11, 11: 12, 12: 10}

        def mock_forward(idx_input):
            logits = original_forward(idx_input)
            forced = next_token.get(idx_input[0, -1].item())
            if forced is not None:
                logits[0, -1, forced] += 100.0
            return logits

        model.forward = mock_forward

        # Without the penalty the loop just continues.
        out = generate(model, idx.clone(), max_new_tokens=6, context_size=32,
                       temperature=0.0, no_repeat_ngram_size=0, use_cache=False)
        assert out[0, 5:].tolist() == [12, 10, 11, 12, 10, 11]

        out = generate(model, idx.clone(), max_new_tokens=6, context_size=32,
                       temperature=0.0, no_repeat_ngram_size=3, use_cache=False)
        generated = out[0, 5:].tolist()
        trigrams = [tuple(generated[i:i + 3]) for i in range(len(generated) - 2)]
        assert len(trigrams) == len(set(trigrams)), f"n-gram repeated within generation: {generated}"
        # Prompt n-grams are NOT blocked: the model may quote its context
        # (e.g. retrieved sources in RAG), so the first token is still 12
        # even though [10, 11, 12] already occurs in the prompt.
        assert generated[0] == 12

        model.forward = original_forward

    def test_advanced_sampling_kv_equivalence(self):
        from model.gpt import generate
        model = GPTModel(TINY_CONFIG)
        model.eval()
        idx = torch.randint(0, 100, (1, 8))

        # We'll use frequency_penalty, presence_penalty, no_repeat_ngram_size, min_new_tokens
        torch.manual_seed(42)
        out_no_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=50, top_p=0.9, 
            frequency_penalty=0.5, presence_penalty=0.5, 
            no_repeat_ngram_size=3, min_new_tokens=3,
            use_cache=False
        )

        torch.manual_seed(42)
        out_with_cache = generate(
            model, idx.clone(), max_new_tokens=10, context_size=32,
            temperature=0.8, top_k=50, top_p=0.9, 
            frequency_penalty=0.5, presence_penalty=0.5, 
            no_repeat_ngram_size=3, min_new_tokens=3,
            use_cache=True
        )

        assert torch.equal(out_no_cache, out_with_cache), "Cached and non-cached must yield identical tokens for advanced sampling params"

    def test_kv_cache_equivalence_past_context_length(self):
        """Cached and uncached generation must still match once the sequence
        length crosses context_size -- regression test for a bug where the
        KV cache was cropped in place, pinning every subsequent token to the
        same stale position instead of sliding the window."""
        from model.gpt import generate
        small_ctx_cfg = {**TINY_CONFIG, "context_length": 8}
        model = GPTModel(small_ctx_cfg)
        model.eval()
        idx = torch.randint(0, 100, (1, 5))  # prompt_len=5, context_size=8

        out_no_cache = generate(
            model, idx.clone(), max_new_tokens=12, context_size=8,
            temperature=0.0, use_cache=False,
        )
        out_with_cache = generate(
            model, idx.clone(), max_new_tokens=12, context_size=8,
            temperature=0.0, use_cache=True,
        )

        assert torch.equal(out_no_cache, out_with_cache), (
            "Cached and non-cached generation diverged once the sequence "
            "crossed context_size -- KV cache sliding-window bug regressed"
        )


class TestAttentionMath:
    def _reference_attention(self, mha, x, past=None):
        """The explicit formula: softmax(QK^T / sqrt(d) + causal mask) V."""
        b, t, _ = x.shape
        q = mha.W_query(x).view(b, t, mha.num_heads, mha.head_dim).transpose(1, 2)
        k = mha.W_key(x).view(b, t, mha.num_heads, mha.head_dim).transpose(1, 2)
        v = mha.W_value(x).view(b, t, mha.num_heads, mha.head_dim).transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)
        total = k.shape[-2]
        scores = q @ k.transpose(2, 3) / mha.head_dim ** 0.5
        mask = torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1)[total - t:]
        weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, t, mha.d_out)
        return mha.out_proj(out)

    def test_fused_attention_matches_explicit_formula(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(d_in=64, d_out=64, context_length=32, dropout=0.0, num_heads=4).eval()
        x = torch.randn(2, 10, 64)
        out, present = mha(x)
        assert torch.allclose(out, self._reference_attention(mha, x), atol=1e-5)

        # Multi-token query on top of a cache (the explicit-mask branch)
        # and a single-token decode step.
        for new_tokens in (3, 1):
            x_new = torch.randn(2, new_tokens, 64)
            out_new, _ = mha(x_new, layer_past=present)
            assert torch.allclose(out_new, self._reference_attention(mha, x_new, past=present), atol=1e-5)

    def test_forward_rejects_sequences_longer_than_context(self):
        model = GPTModel(TINY_CONFIG)
        with pytest.raises(ValueError, match="context length"):
            model(torch.randint(0, 100, (1, TINY_CONFIG["context_length"] + 1)))


class TestSampling:
    def test_top_k_larger_than_vocab_does_not_crash(self):
        from model.gpt import _sample_next_token
        logits = torch.randn(1, 100)
        tok = _sample_next_token(logits, temperature=1.0, top_k=10_000)
        assert 0 <= tok.item() < 100

    def test_all_candidates_banned_falls_back_instead_of_nan(self):
        from model.gpt import _sample_next_token
        # Unigram blocking over a sequence containing every vocab id bans
        # everything; sampling from an all -inf row used to produce NaNs.
        vocab = 8
        idx = torch.arange(vocab).unsqueeze(0)
        for temperature in (0.0, 1.0):
            tok = _sample_next_token(torch.randn(1, vocab), temperature=temperature,
                                     no_repeat_ngram_size=1, idx=idx, prompt_len=0)
            assert 0 <= tok.item() < vocab

    def test_top_p_uses_temperature_scaled_distribution(self):
        from model.gpt import _sample_next_token
        # At T=1 token 0 holds ~88% of the mass, so top_p=0.5 keeps only it.
        # At T=100 the distribution is nearly flat and the nucleus must grow;
        # filtering before scaling (the old order) would still keep only token 0.
        logits = torch.tensor([[4.0, 2.0, 0.0, 0.0]])
        torch.manual_seed(0)
        sampled = {_sample_next_token(logits.clone(), temperature=100.0, top_p=0.5).item() for _ in range(200)}
        assert len(sampled) > 1
        torch.manual_seed(0)
        sampled = {_sample_next_token(logits.clone(), temperature=1.0, top_p=0.5).item() for _ in range(200)}
        assert sampled == {0}


class TestLoRA:
    def test_lora_injection_and_freezing(self):
        from model.gpt import GPTModel
        from model.lora import inject_lora, get_lora_state_dict
        
        model = GPTModel(TINY_CONFIG)
        
        # Inject LoRA into W_query and W_value
        inject_lora(model, r=4, alpha=8.0, target_modules=["W_query", "W_value"])
        
        # Verify layers were wrapped
        for i in range(TINY_CONFIG["n_layers"]):
            assert hasattr(model.trf_blocks[i].att, "W_query")
            assert hasattr(model.trf_blocks[i].att, "W_value")
            # Should be LoRALinear
            from model.lora import LoRALinear
            assert isinstance(model.trf_blocks[i].att.W_query, LoRALinear)
            assert isinstance(model.trf_blocks[i].att.W_value, LoRALinear)
            
            # W_key should NOT be wrapped (removes unchanged)
            assert not isinstance(model.trf_blocks[i].att.W_key, LoRALinear)
            
        # Verify freezing parameters logic
        trainable_params = 0
        frozen_params = 0
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad == True, f"{name} should be trainable"
                trainable_params += 1
            else:
                assert param.requires_grad == False, f"{name} should be frozen"
                frozen_params += 1
                
        assert trainable_params > 0
        assert frozen_params > 0
        
        # Verify state dict extraction
        lora_sd = get_lora_state_dict(model)
        assert len(lora_sd) == trainable_params
        for k in lora_sd.keys():
            assert "lora_" in k
            
        # Verify forward pass
        idx = torch.randint(0, 100, (2, 8))
        logits = model(idx)
        assert logits.shape == (2, 8, 100)

    def test_lora_checkpoint_saving_and_inference_loading(self):
        from model.gpt import GPTModel
        from model.lora import inject_lora, get_lora_state_dict
        from app.inference import GPTInferenceEngine
        import tempfile
        import os
        
        # 1. Create a model and inject LoRA
        model = GPTModel(TINY_CONFIG)
        inject_lora(model, r=4, alpha=8.0, target_modules=["W_query", "W_value"])
        
        # Set some distinct values to a LoRA parameter to verify it loads correctly
        target_param = model.trf_blocks[0].att.W_query.lora_A
        with torch.no_grad():
            target_param.fill_(0.42)
            
        # 2. Save a mock checkpoint
        checkpoint_dict = {
            "model_state_dict": get_lora_state_dict(model),
            "model_config": TINY_CONFIG,
            "is_lora": True,
            "lora_r": 4,
            "lora_alpha": 8.0,
        }
        
        # Use a temporary file path
        fd, temp_file_path = tempfile.mkstemp(suffix=".pt")
        os.close(fd)
        
        try:
            torch.save(checkpoint_dict, temp_file_path)
            
            # 3. Instantiate inference engine and load
            engine = GPTInferenceEngine(checkpoint_path=temp_file_path, device="cpu")
            
            # 4. Verify model has LoRA layers
            from model.lora import LoRALinear
            assert isinstance(engine.model.trf_blocks[0].att.W_query, LoRALinear)
            
            # 5. Verify the loaded values
            loaded_param = engine.model.trf_blocks[0].att.W_query.lora_A
            assert torch.allclose(loaded_param, torch.full_like(loaded_param, 0.42))
            
            # 6. Verify fallback inference behavior when lora_r and lora_alpha are not saved
            # (Remove them from the checkpoint dictionary)
            checkpoint_no_meta = {
                "model_state_dict": get_lora_state_dict(model),
                "model_config": TINY_CONFIG,
                "is_lora": True,
            }
            fd2, temp_file_path2 = tempfile.mkstemp(suffix=".pt")
            os.close(fd2)
            try:
                torch.save(checkpoint_no_meta, temp_file_path2)
                engine2 = GPTInferenceEngine(checkpoint_path=temp_file_path2, device="cpu")
                assert isinstance(engine2.model.trf_blocks[0].att.W_query, LoRALinear)
                # Should have inferred r=4 and alpha=8.0
                assert engine2.model.trf_blocks[0].att.W_query.r == 4
                assert engine2.model.trf_blocks[0].att.W_query.alpha == 8.0
            finally:
                if os.path.exists(temp_file_path2):
                    os.remove(temp_file_path2)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_remove_lora_restores_base_and_allows_rank_switch(self):
        """Regression test for adapter hot-swap: switching between adapters
        of different LoRA rank must not raise a state_dict shape error, and
        remove_lora must restore the exact original (pre-injection) base
        weights rather than leaving wrapper modules zeroed-out in place."""
        from model.lora import inject_lora, remove_lora, LoRALinear

        model = GPTModel(TINY_CONFIG)
        original_qw = model.trf_blocks[0].att.W_query.weight.clone()

        inject_lora(model, r=4, alpha=8.0, target_modules=["W_query", "W_value"])
        assert isinstance(model.trf_blocks[0].att.W_query, LoRALinear)
        assert model.trf_blocks[0].att.W_query.lora_A.shape == (4, TINY_CONFIG["emb_dim"])

        removed = remove_lora(model)
        assert removed == TINY_CONFIG["n_layers"] * 2  # W_query + W_value per layer
        assert not isinstance(model.trf_blocks[0].att.W_query, LoRALinear)
        assert torch.equal(model.trf_blocks[0].att.W_query.weight, original_qw)

        # Re-inject at a different rank -- this is what activate_adapter now
        # does before every load_state_dict, so switching adapters of
        # different rank must not error.
        inject_lora(model, r=16, alpha=32.0, target_modules=["W_query", "W_value"])
        assert model.trf_blocks[0].att.W_query.lora_A.shape == (16, TINY_CONFIG["emb_dim"])

        idx = torch.randint(0, 100, (1, 8))
        logits = model(idx)
        assert logits.shape == (1, 8, 100)

    def test_strip_lora_wrapper_keys_preserves_base_weights(self):
        """Regression test: fine-tuning must load the true base weights even
        when the source engine currently has LoRA layers injected (e.g. an
        active persona/adapter). Without key remapping, load_state_dict(
        strict=False) silently leaves the fresh model's weights at random
        init instead of raising."""
        from model.lora import inject_lora, strip_lora_wrapper_keys

        source = GPTModel(TINY_CONFIG)
        original_qw = source.trf_blocks[0].att.W_query.weight.clone()
        inject_lora(source, r=4, alpha=8.0, target_modules=["W_query", "W_value"])

        fresh = GPTModel(TINY_CONFIG)
        pre_load_qw = fresh.trf_blocks[0].att.W_query.weight.clone()

        cleaned = strip_lora_wrapper_keys(source.state_dict())
        missing, unexpected = fresh.load_state_dict(cleaned, strict=False)

        assert missing == [] and unexpected == []
        assert torch.equal(fresh.trf_blocks[0].att.W_query.weight, original_qw)
        assert not torch.equal(fresh.trf_blocks[0].att.W_query.weight, pre_load_qw)


