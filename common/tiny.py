"""Tiny randomly-initialised Qwen3 / SigLIP + a locally trained tokenizer for offline smoke tests."""
import os

CHAT_TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}"
    "{%- endif %}"
)


def make_tokenizer(corpus=None, vocab_size=400):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast
    corpus = corpus or ["the quick brown fox question answer yes no how many can you see does this image contain "
                        "describe briefly user assistant what is the sum of and 0 1 2 3 4 5 6 7 8 9 hello world "
                        "a b c d e f g h i j k l m n o p q r s t u v w x y z people person dog cat car"]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   additional_special_tokens=["<|im_start|>", "<think>", "</think>"])
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "right"
    return fast


def make_tiny_qwen(vocab_size, hidden=64, layers=3):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
                      max_position_embeddings=1024, tie_word_embeddings=True)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg)


def make_tiny_siglip(hidden=32):
    import torch
    from transformers import SiglipVisionConfig, SiglipVisionModel
    cfg = SiglipVisionConfig(hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=1,
                             num_attention_heads=2, image_size=224, patch_size=16)
    torch.manual_seed(0)
    return SiglipVisionModel(cfg)


def save_tiny_assets(root):
    """Writes tokenizer + tiny qwen into root/tiny_qwen; returns that directory."""
    tok = make_tokenizer()
    q = make_tiny_qwen(len(tok))
    qd = os.path.join(root, "tiny_qwen")
    q.save_pretrained(qd); tok.save_pretrained(qd)
    return qd
