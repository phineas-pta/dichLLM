attempt to apply token masking on Gemma 4 using https://huggingface.co/ThakiCloud/Qwen3.8-27B-vi-cjk-suppressed

in Gemma 4, `lm_head` is tied to `embed_tokens` (`tie_word_embeddings=True`), adapt the scripts accordingly

end model still still returns output with language contamination

failed attempt, won’t be updated

but this can still be used to produce a list of tokens to be banned by biasing logits (although need to check the impact on translation quality)
