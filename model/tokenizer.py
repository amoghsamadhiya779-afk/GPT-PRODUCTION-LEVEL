from model.tokenizer import GPT2Tokenizer

def test_roundtrip():
    tokenizer = GPT2Tokenizer()

    text = "Transformers are amazing."

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    assert decoded == text