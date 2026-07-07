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
        
        # We manually patch the model's logits to forcefully loop
        # by making token 12 the highest probability after 11.
        original_forward = model.forward
        
        def mock_forward(idx_input):
            logits = original_forward(idx_input)
            # Make sure token 12 is chosen when last two were 10, 11
            # But the n-gram penalty should prevent this
            if idx_input[0, -1].item() == 11:
                logits[0, -1, 12] += 100.0  # Force 12
            return logits
            
        model.forward = mock_forward
        
        # Test without n-gram penalty: should generate 12
        torch.manual_seed(42)
        out_no_penalty = generate(
            model, idx.clone(), max_new_tokens=1, context_size=32,
            temperature=0.0, no_repeat_ngram_size=0, use_cache=False
        )
        assert out_no_penalty[0, -1].item() == 12

        # Test with n-gram penalty (size=3). 
        # The sequence is [10, 11, 12, 10, 11]. Next token 12 would create n-gram [10, 11, 12] which already exists!
        torch.manual_seed(42)
        out_with_penalty = generate(
            model, idx.clone(), max_new_tokens=1, context_size=32,
            temperature=0.0, no_repeat_ngram_size=3, use_cache=False
        )
        assert out_with_penalty[0, -1].item() != 12, "n-gram penalty failed to block the repeated sequence"
        
        # Restore forward
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


