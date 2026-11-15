# dichLLM

test khả năng dịch tiểu thuyết mạng tiếng trung sang tiếng việt dùng open-weight LLM

quy trình:
1. user phải tách file raw thành file text riêng từng chương
2. LLM trích xuất tên + tóm tắt dùng Qwen 3.8
3. user phải sửa lại tên theo ý muốn
4. bản nháp dịch lần 1 dùng Gemma 4
5. kiểm lại chính tả + kí tự lạ: chữ hán/hàn/nhật/arab/hindi
6. kiểm lại tên + xưng hô
7. đóng epub

## LLM inference notes

EXL3 format: closer to the BF16 model at a lower quantized-weight size than GGUF, speed may be slower

translate each chapter separately, not a chat so no need prompt caching

sampling parameters: low temperature + top P + top K, mild min P + repetition penalty

edit logit bias to ban some tokens to prevent language confusion (language contamination)

### SPEED (total of prefill & token decoding)

chạy full local dùng ExLlamaV3 trên Marimo Molab với RTX Pro 6000 Blackwell 96 GB

Gemma4 31B EXL3 8.0 bpw + KV cache FP16
- without draft/MTP: mỗi giây ~30 chữ hán
- with DFlash: mỗi giây ~42 chữ hán

so với: Kaggle T4×2 (16 GB ×2): Gemma4 26B-A4B GGUF weight Q4 + KV cache Q8: mỗi giây ~42 chữ hán (decoding ~67 tok/s) trên tổng 2 luồng song song
