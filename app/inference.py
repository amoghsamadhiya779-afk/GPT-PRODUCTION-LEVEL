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

from model.gpt import GPTModel, generate, count_parameters
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

        # 4. Instantiate Model & Load State
        self.model = GPTModel(self.model_config)
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
        use_cache: bool = True,
    ) -> dict:
        """Generate text from a prompt and return the output with latency statistics.

        Args:
            prompt: String seed text.
            max_new_tokens: Tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling limit.
            use_cache: Whether to use KV-caching.

        Returns:
            Dictionary matching the GenerationResponse schema fields.
        """
        # Tokenize seed prompt
        input_ids = self.tokenizer.text_to_token_ids(prompt).to(self.device)

        start_time = time.perf_counter()
        
        # Call generation pipeline
        output_ids = generate(
            model=self.model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=self.context_size,
            temperature=temperature,
            top_k=top_k,
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
