import os
import json
import random
import re
import tiktoken
from datasets import load_dataset

random.seed(42)

def main():
    print("Loading tokenizer for filtering...")
    enc = tiktoken.get_encoding("gpt2")
    
    out_records = []
    
    # Track statistics
    stats = {
        "dolly_standalone": 0,
        "dolly_rag": 0,
        "dolly_persona": 0,
        "feedback": 0,
        "custom_instructions": 0,
        "general_data": 0,
        "math_data": 0,
        "physics_data": 0,
        "rag_data": 0,
        "starter_instructions": 0,
        "synthesized_chitchat": 0
    }
    
    def format_alpaca(instruction: str, response: str) -> str:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}<|endoftext|>"
        )

    def add_record(inst, resp, source):
        # Filter if response length > 150
        if len(enc.encode(resp)) > 150:
            return
        
        full_text = format_alpaca(inst, resp)
        if len(enc.encode(full_text, allowed_special={"<|endoftext|>"})) > 1024:
            return
            
        out_records.append({"instruction": inst, "response": resp, "source": source})
        stats[source] += 1

    # 1. Synthesize chitchat pairs (~300)
    print("Synthesizing chitchat pairs...")
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "hi there", "greetings"]
    identities = ["who are you?", "what are you?", "can you introduce yourself?", "tell me about yourself", "what is your identity?"]
    capabilities = ["what can you do?", "how can you help me?", "what are your capabilities?", "what do you know?"]
    thanks = ["thank you", "thanks", "thanks a lot", "appreciate it", "thanks for the help"]
    goodbyes = ["bye", "goodbye", "see you later", "farewell", "have a good one"]
    
    base_greeting_responses = [
        "Hello! I'm a math and science assistant. What would you like to explore?",
        "Hi there! I can help you with math, physics, and more. How can I assist you today?",
        "Greetings! I'm an AI assistant specializing in STEM subjects. What's on your mind?",
    ]
    
    base_identity_responses = [
        "I am a custom GPT-2 model built entirely from scratch, specializing in math, science, and reasoning.",
        "I'm an AI assistant trained to help you explore mathematics and physics concepts.",
        "I am a from-scratch transformer language model designed to assist with STEM questions.",
    ]
    
    base_capability_responses = [
        "I can answer questions about mathematics, physics, and general science, as well as help you brainstorm ideas.",
        "I can solve math problems, explain physics concepts, and engage in logical reasoning tasks.",
        "I'm equipped to help you with math and science inquiries, summarize text, and answer general knowledge questions.",
    ]
    
    base_thanks_responses = [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help!",
        "Anytime! Feel free to ask more questions.",
        "Glad I could assist you."
    ]
    
    base_goodbye_responses = [
        "Goodbye! Have a great day.",
        "See you later!",
        "Take care! Feel free to return if you have more questions.",
        "Bye! Happy learning."
    ]
    
    for _ in range(70):
        # Greetings
        add_record(random.choice(greetings), random.choice(base_greeting_responses), "synthesized_chitchat")
        add_record(random.choice(greetings).capitalize(), random.choice(base_greeting_responses), "synthesized_chitchat")
        add_record(random.choice(greetings).upper(), random.choice(base_greeting_responses), "synthesized_chitchat")
        
        # Identity
        add_record(random.choice(identities), random.choice(base_identity_responses), "synthesized_chitchat")
        add_record(random.choice(identities).capitalize(), random.choice(base_identity_responses), "synthesized_chitchat")
        
        # Capabilities
        add_record(random.choice(capabilities), random.choice(base_capability_responses), "synthesized_chitchat")
        add_record(random.choice(capabilities).capitalize(), random.choice(base_capability_responses), "synthesized_chitchat")
        
        # Thanks
        add_record(random.choice(thanks), random.choice(base_thanks_responses), "synthesized_chitchat")
        add_record(random.choice(thanks).capitalize(), random.choice(base_thanks_responses), "synthesized_chitchat")
        
        # Goodbyes
        add_record(random.choice(goodbyes), random.choice(base_goodbye_responses), "synthesized_chitchat")
        add_record(random.choice(goodbyes).capitalize(), random.choice(base_goodbye_responses), "synthesized_chitchat")

    # 2. Local Datasets
    print("Processing local datasets...")
    data_dir = "data"
    
    # feedback.jsonl
    if os.path.exists(os.path.join(data_dir, "feedback.jsonl")):
        with open(os.path.join(data_dir, "feedback.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if item.get("rating") == "down" and item.get("correction"):
                    add_record(item["prompt"], item["correction"], "feedback")

    # custom_instructions.json
    if os.path.exists(os.path.join(data_dir, "custom_instructions.json")):
        with open(os.path.join(data_dir, "custom_instructions.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                inst = item["instruction"]
                out = item["output"]
                # Capping greeting responses and removing UI specific content
                if "Hello! I am your Math & Science Assistant" in out:
                    out = "Hello! I'm a math and science assistant. What would you like to explore?"
                # Remove UI mentions if they sneak in elsewhere
                out = re.sub(r'(?i)You can select a specialized persona from the options, or try these prompts directly:.*', '', out, flags=re.DOTALL).strip()
                add_record(inst, out, "custom_instructions")
                
    # general_data.jsonl
    if os.path.exists(os.path.join(data_dir, "general_data.jsonl")):
        with open(os.path.join(data_dir, "general_data.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                resp = item["response"]
                # Strip persona boilerplate
                resp = re.sub(r'(?i)^As (a|your) General Assistant,?\s*', '', resp)
                # Capitalize first letter
                if len(resp) > 0:
                    resp = resp[0].upper() + resp[1:]
                add_record(item["instruction"], resp, "general_data")

    # math_data.jsonl, physics_data.jsonl
    for source_name in ["math_data.jsonl", "physics_data.jsonl"]:
        if os.path.exists(os.path.join(data_dir, source_name)):
            with open(os.path.join(data_dir, source_name), "r", encoding="utf-8") as f:
                name_key = source_name.replace(".jsonl", "")
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    add_record(item["instruction"], item["response"], name_key)
                    
    # rag_data.jsonl
    if os.path.exists(os.path.join(data_dir, "rag_data.jsonl")):
        with open(os.path.join(data_dir, "rag_data.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                inst_raw = item["instruction"]
                resp = item["response"]
                
                # Parse old format: "<Question>\n\nWeb Search Context:\n- <snippet>\n- <snippet>"
                parts = inst_raw.split("\n\nWeb Search Context:\n")
                if len(parts) == 2:
                    q = parts[0].strip()
                    snippets_raw = parts[1].strip()
                    snippets = [s.replace("- ", "").strip() for s in snippets_raw.split("\n") if s.strip().startswith("-")]
                    
                    # Trim snippets loop
                    while snippets:
                        new_inst = ""
                        for i, snip in enumerate(snippets, 1):
                            new_inst += f"[{i}] {snip}\n"
                        new_inst += f"\nQuestion: {q}"
                        if len(enc.encode(format_alpaca(new_inst, resp), allowed_special={"<|endoftext|>"})) <= 1024:
                            add_record(new_inst, resp, "rag_data")
                            break
                        snippets.pop()
                else:
                    add_record(inst_raw, resp, "rag_data")

    # starter_instructions.jsonl
    if os.path.exists(os.path.join(data_dir, "starter_instructions.jsonl")):
        with open(os.path.join(data_dir, "starter_instructions.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                if "instruction" in item and "response" in item:
                    add_record(item["instruction"], item["response"], "starter_instructions")

    # 3. Databricks Dolly 15k
    print("Loading databricks-dolly-15k from Hugging Face...")
    try:
        dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        # standalone categories
        standalone_cats = {"closed_qa", "general_qa", "information_extraction", "brainstorming", "classification"}
        rag_cats = {"closed_qa", "summarization", "information_extraction"}
        
        persona_count = 0
        personas = ["Socrates", "Einstein", "Shakespeare"]
        
        for item in dolly:
            cat = item.get("category")
            ctx = item.get("context", "")
            inst = item.get("instruction", "")
            resp = item.get("response", "")
            
            # Persona generation (limit ~500)
            if persona_count < 500 and not ctx and cat in {"brainstorming", "general_qa"}:
                persona = random.choice(personas)
                p_inst = f"[persona: {persona}] {inst}"
                
                # Naive restyling for the sake of synthetic generation
                p_resp = resp
                if persona == "Socrates":
                    p_resp = f"Let us ponder this together. {resp} Does that not seem true?"
                elif persona == "Einstein":
                    p_resp = f"From a fundamental perspective: {resp}"
                elif persona == "Shakespeare":
                    p_resp = f"Hark! {resp}"
                
                add_record(p_inst, p_resp, "dolly_persona")
                persona_count += 1
                continue
            
            if ctx:
                if cat in rag_cats:
                    # Grounded RAG synthesis
                    snippets = [s for s in ctx.split(". ") if s.strip()]
                    snippets = snippets[:3] # Start with max 3 snippets
                    
                    while snippets:
                        new_inst = ""
                        for i, snip in enumerate(snippets, 1):
                            new_inst += f"[{i}] {snip.strip()}.\n"
                        new_inst += f"\nQuestion: {inst}"
                        
                        if len(enc.encode(format_alpaca(new_inst, resp), allowed_special={"<|endoftext|>"})) <= 1024:
                            add_record(new_inst, resp, "dolly_rag")
                            break
                        snippets.pop()
            else:
                if cat in standalone_cats:
                    add_record(inst, resp, "dolly_standalone")
                    
    except Exception as e:
        print(f"Warning: Failed to load dolly-15k: {e}")

    # Deduplication
    print("Deduplicating...")
    seen = set()
    deduped_records = []
    
    total_inst_tokens = 0
    total_resp_tokens = 0
    max_inst_tokens = 0
    max_resp_tokens = 0
    
    for r in out_records:
        inst = r["instruction"]
        resp = r["response"]
        
        # simple near-exact dedupe hash (lower case, strip punctuation)
        sig = re.sub(r'[^\w\s]', '', inst.lower() + resp.lower())
        if sig not in seen:
            seen.add(sig)
            
            i_toks = len(enc.encode(inst))
            r_toks = len(enc.encode(resp))
            
            total_inst_tokens += i_toks
            total_resp_tokens += r_toks
            max_inst_tokens = max(max_inst_tokens, i_toks)
            max_resp_tokens = max(max_resp_tokens, r_toks)
            
            deduped_records.append(r)
            
    # Shuffle and split
    random.shuffle(deduped_records)
    
    eval_size = int(len(deduped_records) * 0.02)
    eval_records = deduped_records[:eval_size]
    train_records = deduped_records[eval_size:]
    
    # Save
    with open(os.path.join(data_dir, "sft_mix.jsonl"), "w", encoding="utf-8") as f:
        for r in train_records:
            json.dump({"instruction": r["instruction"], "response": r["response"]}, f)
            f.write("\n")
            
    with open(os.path.join(data_dir, "sft_eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_records:
            json.dump({"instruction": r["instruction"], "response": r["response"]}, f)
            f.write("\n")
            
    # Print stats
    print("\n" + "="*40)
    print("DATASET BUILD STATISTICS")
    print("="*40)
    print(f"Total Sources (before dedup):")
    for k, v in stats.items():
        if v > 0:
            print(f"  - {k}: {v}")
    
    print(f"\nFinal Deduplicated Total: {len(deduped_records)}")
    print(f"Train Split: {len(train_records)}")
    print(f"Eval Split: {len(eval_records)}")
    
    if len(deduped_records) > 0:
        print(f"\nAvg Instruction Tokens: {total_inst_tokens / len(deduped_records):.1f}")
        print(f"Max Instruction Tokens: {max_inst_tokens}")
        print(f"Avg Response Tokens: {total_resp_tokens / len(deduped_records):.1f}")
        print(f"Max Response Tokens: {max_resp_tokens}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
