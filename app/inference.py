# app/inference.py
"""Unified object-oriented inference engine for GPT-2.

Loads checkpoint, auto-detects hardware (CUDA/CPU), handles tokenization,
and benchmarks text generation using standard or KV-Cached mode.
"""

import os
import sys
import time
import torch

# Add root folder to path to enable clean imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPTModel, generate, generate_stream, count_parameters
from model.tokenizer import GPT2Tokenizer


class GPTInferenceEngine:
    """Encapsulated text generation engine for the custom GPT-2 model."""

    def __init__(self, checkpoint_path: str, device: str = "auto") -> None:
        """Initialize the inference engine and load model weights from a checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint (.pt file).
            device: Hardware selection ('auto', 'cpu', 'cuda').
        """
        # 1. Device Selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Check checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        # 3. Load Checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model_config = checkpoint["model_config"]
        is_lora = checkpoint.get("is_lora", False)

        # 4. Instantiate Model & Load State
        self.model = GPTModel(self.model_config)

        if is_lora:
            # Import inject_lora here to avoid circular dependency
            from model.lora import inject_lora
            
            # Determine r and alpha, with fallback if not stored
            lora_r = checkpoint.get("lora_r")
            lora_alpha = checkpoint.get("lora_alpha")
            
            if lora_r is None:
                # Find any lora_A key to check its shape
                lora_A_keys = [k for k in checkpoint["model_state_dict"].keys() if "lora_A" in k]
                if lora_A_keys:
                    lora_r = checkpoint["model_state_dict"][lora_A_keys[0]].shape[0]
                else:
                    lora_r = 4  # Default fallback
            
            if lora_alpha is None:
                lora_alpha = float(lora_r * 2)  # Typically alpha = 2 * r

            # Inject the LoRA adapters
            inject_lora(self.model, r=lora_r, alpha=lora_alpha, target_modules=["W_query", "W_value"])
            
            # Load LoRA state dict (strict=False since it only contains adapter weights)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.to(self.device)
        self.model.eval()

        # 5. Load Tokenizer & Metadata
        self.tokenizer = GPT2Tokenizer()
        self.context_size = self.model.pos_emb.weight.shape[0]
        self.parameter_count = count_parameters(self.model)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ) -> dict:
        """Generate text from a prompt and return the output with latency statistics.

        Args:
            prompt: String seed text.
            max_new_tokens: Tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling limit.
            top_p: Top-p sampling threshold.
            repetition_penalty: Repetition penalty coefficient.
            use_cache: Whether to use KV-caching.

        Returns:
            Dictionary matching the GenerationResponse schema fields.
        """
        # Tokenize seed prompt
        input_ids = self.tokenizer.text_to_token_ids(prompt).to(self.device)

        # Get EOS token ID for early stopping
        eos_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        start_time = time.perf_counter()
        
        # Call generation pipeline
        output_ids = generate(
            model=self.model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.context_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            eos_id=eos_id,
            use_cache=use_cache,
        )
        
        latency = time.perf_counter() - start_time

        # Decode tokens back to string
        generated_text = self.tokenizer.token_ids_to_text(output_ids)
        
        # Calculate generation details
        num_generated = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_second = num_generated / latency if latency > 0 else 0.0

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "tokens_generated": num_generated,
            "time_taken_seconds": latency,
            "tokens_per_second": tokens_per_second,
        }

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        min_new_tokens: int = 0,
        use_cache: bool = True,
    ):
        """Generator that yields newly generated text chunks, latency, and token count."""
        input_ids = self.tokenizer.text_to_token_ids(prompt).to(self.device)
        eos_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

        start_time = time.perf_counter()
        tokens_generated = 0
        byte_buffer = b""

        for token_id in generate_stream(
            model=self.model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.context_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_new_tokens=min_new_tokens,
            eos_id=eos_id,
            use_cache=use_cache,
        ):
            tokens_generated += 1
            token_bytes = self.tokenizer.encoding.decode_single_token_bytes(token_id)
            byte_buffer += token_bytes

            try:
                text_chunk = byte_buffer.decode("utf-8")
                byte_buffer = b""
            except UnicodeDecodeError:
                text_chunk = ""

            latency = time.perf_counter() - start_time
            yield text_chunk, latency, tokens_generated

        if byte_buffer:
            text_chunk = byte_buffer.decode("utf-8", errors="replace")
            latency = time.perf_counter() - start_time
            yield text_chunk, latency, tokens_generated
