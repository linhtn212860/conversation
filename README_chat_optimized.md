# 📚 Giải thích chi tiết code `chat_optimized.py`

## 📋 Mục lục
1. [Import thư viện](#1-import-thư-viện)
2. [Cấu hình](#2-cấu-hình)
3. [System Prompt](#3-system-prompt)
4. [Load Model](#4-load-model)
5. [Web Search](#5-web-search)
6. [Search Intent Detection](#6-search-intent-detection)
7. [Streaming Generation](#7-streaming-generation)
8. [Chat Function](#8-chat-function)
9. [Main Loop](#9-main-loop)

---

## 1. Import thư viện

```python
import torch                    # Framework deep learning của Facebook
import re                       # Regular expressions - tìm pattern trong text
from pathlib import Path        # Xử lý đường dẫn file (hoạt động trên mọi OS)
from datetime import datetime   # Lấy ngày giờ hiện tại
from threading import Thread    # Chạy code song song (cho streaming)
```

### Transformers imports:
```python
from transformers import (
    AutoModelForCausalLM,      # Tự động chọn class model phù hợp (Qwen, Llama, etc.)
    AutoTokenizer,              # Chuyển text <-> số (tokens)
    TextIteratorStreamer,       # Cho phép đọc output từng phần
    BitsAndBytesConfig          # Cấu hình nén model (4-bit/8-bit)
)
```

---

## 2. Cấu hình

```python
MODEL_PATH = Path(__file__).parent / "models" / "Qwen_Qwen3-8B"
# __file__ = đường dẫn file python hiện tại
# .parent = thư mục chứa file
# / "models" = nối thêm thư mục models
# Kết quả: /home/.../qwen_8b/models/Qwen_Qwen3-8B

CURRENT_DATE = datetime.now().strftime("%d/%m/%Y")
# datetime.now() = thời điểm hiện tại
# strftime() = format thành string
# "%d/%m/%Y" = 06/01/2026
```

---

## 3. System Prompt

System prompt định nghĩa "nhân cách" của AI. Các điểm quan trọng:

### Tại sao cần `[SEARCH:]`?
Model có **knowledge cutoff** (giới hạn kiến thức) - không biết gì sau thời điểm training.
Ví dụ: Model train năm 2024, không biết ai thắng Ballon d'Or 2025.

### Language
This project is tuned for English conversational responses.

---

## 4. Load Model

### 4-bit Quantization là gì?

```
Bình thường: mỗi tham số = 32 bit (float32)
4-bit:       mỗi tham số = 4 bit

Model Qwen3-8B có 8 tỷ tham số:
- Float32: 8B × 4 bytes = 32 GB VRAM cần thiết
- Float16: 8B × 2 bytes = 16 GB VRAM
- 4-bit:   8B × 0.5 bytes = 4 GB VRAM ✅
```

### Code giải thích:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Bật 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16, # Kiểu dữ liệu khi tính toán
    bnb_4bit_use_double_quant=True,       # Nén 2 lần (tiết kiệm thêm)
    bnb_4bit_quant_type="nf4"             # NormalFloat4 - phương pháp nén tốt nhất
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=quantization_config,
    device_map="auto",     # Tự động phân bổ lên GPU/CPU
    trust_remote_code=True, # Cho phép chạy code trong model (cần cho Qwen)
    local_files_only=True   # Chỉ load từ local, không download
)
```

---

## 5. Web Search

### DuckDuckGo Search:

```python
with DDGS() as ddgs:
    # DDGS = DuckDuckGo Search client
    # with ... as = context manager (tự động cleanup)
    
    results = list(ddgs.text(query, region='us-en', max_results=3))
    # ddgs.text() = tìm kiếm web
    # region='us-en' = ưu tiên kết quả tiếng Anh
    # max_results=3 = chỉ lấy 3 kết quả (nhanh hơn)
```

### Tại sao dùng DuckDuckGo?
- ✅ Miễn phí
- ✅ Không cần API key
- ✅ Không tracking
- ✅ Phù hợp cho truy vấn tiếng Anh

---

## 6. Search Intent Detection

### 2 cách phát hiện khi cần search:

#### Cách 1: Keyword-based (từ khóa cố định)
```python
keywords = {
    'weather': ['thời tiết', 'weather', ...],
    'news': ['tin tức', 'chính trị', ...],
    ...
}

for search_type, kws in keywords.items():
    if any(kw in user_lower for kw in kws):
        # any() = True nếu có BẤT KỲ từ khóa nào match
        return True, search_type, query
```

#### Cách 2: Model-initiated (model tự quyết định)
```python
search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
# \[SEARCH:  = chuỗi "[SEARCH:"
# \s*        = 0 hoặc nhiều khoảng trắng
# ([^\]]+)   = capture group: bất kỳ ký tự nào NGOẠI TRỪ ]
# \]         = ký tự ]

match = re.search(search_pattern, response)
if match:
    query = match.group(1)  # Lấy nội dung trong ngoặc
```

---

## 7. Streaming Generation ⭐ QUAN TRỌNG

### Vấn đề:
Bình thường, `model.generate()` chạy đến khi **XONG HOÀN TOÀN** mới return.
→ User phải chờ 10-30 giây mới thấy gì.

### Giải pháp: Streaming
```
Thread 1 (main):     [Đọc token] -> [In ra] -> [Đọc token] -> [In ra] ...
                           ↑                        ↑
                           |                        |
Thread 2 (generate): [Generate] -> [Push token] -> [Generate] -> [Push token] ...
```

### Code giải thích:

```python
# 1. Tạo streamer (hàng đợi giữa 2 threads)
streamer = TextIteratorStreamer(
    tokenizer, 
    skip_prompt=True,        # Bỏ qua input, chỉ lấy output
    skip_special_tokens=True # Bỏ các token đặc biệt
)

# 2. Cấu hình generation
generation_kwargs = dict(
    **inputs,                      # Input tokens
    max_new_tokens=512,            # Tối đa 512 tokens output
    do_sample=True,                # Sampling (không deterministic)
    temperature=0.7,               # 0=rất chắc chắn, 1=rất ngẫu nhiên
    top_p=0.9,                     # Nucleus sampling
    streamer=streamer              # QUAN TRỌNG: gắn streamer vào
)

# 3. Chạy generation ở thread riêng
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()  # Bắt đầu generate ở background

# 4. Đọc tokens từ streamer (main thread)
for new_text in streamer:
    # Mỗi lần loop = nhận được 1 đoạn text mới
    print(new_text, end="", flush=True)  # In ngay lập tức
    response += new_text

thread.join()  # Đợi thread kết thúc
```

### Tại sao cần `flush=True`?
- Mặc định Python buffer output, chờ đủ nhiều rồi in 1 lần
- `flush=True` = in ngay lập tức, không chờ

---

## 8. Chat Function

### Flow xử lý:

```
User input
    │
    ▼
┌─────────────────────┐
│ Keyword detection   │─── Match? ──→ Search ngay
└─────────────────────┘
    │ Không match
    ▼
┌─────────────────────┐
│ Generate response   │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Check [SEARCH:]     │─── Có? ──→ Search + Regenerate
└─────────────────────┘
    │ Không có
    ▼
  Return response
```

---

## 9. Main Loop

```python
while True:  # Vòng lặp vô hạn
    try:
        user_input = input("👤 You: ")  # Chờ user nhập
        
        # Xử lý lệnh đặc biệt
        if user_input.lower() == 'quit':
            break  # Thoát vòng lặp
        
        # Thêm vào history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response
        response = chat_with_streaming(...)
        
        # Thêm response vào history
        conversation_history.append({"role": "assistant", "content": response})
        
    except KeyboardInterrupt:  # Ctrl+C
        break
    except Exception as e:     # Bất kỳ lỗi nào
        print(f"Error: {e}")
        continue  # Tiếp tục vòng lặp, không crash
```

---

## 📊 Tổng kết Flow

```
┌──────────────────────────────────────────────────────────────┐
│                         USER INPUT                            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  1. KEYWORD DETECTION                                         │
│     - Có từ khóa "thời tiết", "tin tức", etc.?               │
│     - Có → Web Search → Augment input                        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  2. APPLY CHAT TEMPLATE                                       │
│     - Chuyển messages thành format model hiểu                │
│     - System + User messages → Qwen format                   │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  3. STREAMING GENERATION                                      │
│     - Thread 1: model.generate() ở background                │
│     - Thread 2: in từng token ra màn hình                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  4. POST-PROCESS                                              │
│     - Kiểm tra [SEARCH:] trong response                      │
│     - Nếu có → Search + Regenerate                           │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  5. UPDATE HISTORY                                            │
│     - Thêm user message + assistant response vào history     │
│     - History được giữ cho các lượt chat tiếp theo           │
└──────────────────────────────────────────────────────────────┘
```

---

## ❓ FAQ

### Q: Tại sao cần `trust_remote_code=True`?
A: Một số model (như Qwen) có code custom trong repo. Flag này cho phép chạy code đó.

### Q: `device_map="auto"` làm gì?
A: Tự động phân bổ model lên GPU. Nếu không đủ VRAM, sẽ dùng cả CPU.

### Q: Tại sao dùng `bfloat16` thay vì `float16`?
A: `bfloat16` có range số lớn hơn, ít bị overflow khi training/inference.

### Q: Streaming có làm chậm tổng thời gian không?
A: Không! Tổng thời gian giống nhau, nhưng user thấy response sớm hơn nhiều.
