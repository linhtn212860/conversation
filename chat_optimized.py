#!/usr/bin/env python3
"""
=============================================================================
CHAT OPTIMIZED - Phiên bản tối ưu hóa cho Qwen3-8B
=============================================================================

Tác giả: Được tạo bởi AI Assistant
Mục đích: Chatbot thông minh với khả năng:
    - Streaming output (hiển thị từng token)
    - 4-bit quantization (giảm bộ nhớ, tăng tốc)
    - Tra cứu web thời gian thực
    - Hỗ trợ song ngữ Việt-Anh

Yêu cầu:
    - Python 3.8+
    - transformers
    - torch
    - bitsandbytes (cho 4-bit quantization)
    - ddgs hoặc duckduckgo-search (cho web search)
=============================================================================
"""

# =============================================================================
# PHẦN 1: IMPORT CÁC THƯ VIỆN
# =============================================================================

import torch                    # PyTorch - framework deep learning
import re                       # Regular expressions - xử lý pattern matching
from pathlib import Path        # Xử lý đường dẫn file cross-platform
from datetime import datetime   # Lấy ngày giờ hiện tại
from threading import Thread    # Threading - chạy song song cho streaming

# Import từ thư viện Transformers của HuggingFace
from transformers import (
    AutoModelForCausalLM,      # Tự động load model language modeling
    AutoTokenizer,              # Tự động load tokenizer
    TextIteratorStreamer,       # Streamer để yield từng token
    BitsAndBytesConfig          # Cấu hình quantization
)

# =============================================================================
# PHẦN 2: IMPORT WEB SEARCH (TÙY CHỌN)
# =============================================================================
# Thử import ddgs (phiên bản mới) trước, nếu không có thì dùng duckduckgo_search
try:
    from ddgs import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        WEB_SEARCH_AVAILABLE = True
    except ImportError:
        WEB_SEARCH_AVAILABLE = False
        print("⚠️ Để sử dụng tính năng tra cứu web, hãy cài đặt: pip install ddgs")

# =============================================================================
# PHẦN 3: CẤU HÌNH VÀ HẰNG SỐ
# =============================================================================

# Đường dẫn tới model đã download về local
# __file__ = đường dẫn của file Python hiện tại
# parent = thư mục chứa file
MODEL_PATH = Path(__file__).parent / "models" / "Qwen_Qwen3-8B"

# Lấy ngày hiện tại theo định dạng DD/MM/YYYY
CURRENT_DATE = datetime.now().strftime("%d/%m/%Y")

# =============================================================================
# PHẦN 4: SYSTEM PROMPT (SONG NGỮ + SMART SEARCH)
# =============================================================================
# System prompt định nghĩa "nhân cách" và quy tắc hành xử của chatbot
# Bao gồm hướng dẫn để model TỰ QUYẾT ĐỊNH khi nào cần search

SYSTEM_PROMPT = f"""You are a friendly home robot assistant having a casual conversation. Today is {CURRENT_DATE}.

WHEN TO SEARCH:
If you need current info you don't know, start with: [SEARCH: query]

HOW TO RESPOND:
- Talk like you're chatting with a friend, not writing an article
- NO markdown formatting (no ###, no **, no bullet points)
- Just speak naturally in paragraphs
- Keep it conversational, warm, and friendly
- Use emoji sometimes 😊
- If sharing news/info, summarize it naturally like telling a story
- DON'T make up facts - use [SEARCH:] if unsure

Match the user's language (Vietnamese or English).

---

Bạn là robot nhà đang trò chuyện thân thiện. Hôm nay là {CURRENT_DATE}.

CÁCH TRẢ LỜI:
- Nói như đang chat với bạn bè, không phải viết bài
- KHÔNG dùng markdown (không ###, không **, không gạch đầu dòng)
- Nói tự nhiên theo đoạn văn
- Thân thiện, gần gũi, ấm áp
- Thỉnh thoảng dùng emoji 😊
- Khi chia sẻ tin tức, kể lại tự nhiên như đang kể chuyện
- Không bịa thông tin - dùng [SEARCH:] nếu không chắc"""


# =============================================================================
# PHẦN 5: HÀM LOAD MODEL
# =============================================================================

def load_model(use_4bit: bool = True):
    """
    Load model Qwen3-8B với tùy chọn quantization.
    
    4-bit Quantization là gì?
    -------------------------
    - Bình thường, mỗi tham số model được lưu dạng float32 (32 bit) hoặc float16 (16 bit)
    - 4-bit quantization nén xuống còn 4 bit/tham số
    - Giảm ~75% bộ nhớ GPU cần thiết
    - Tăng tốc inference vì ít data phải di chuyển
    - Chất lượng output giảm nhẹ nhưng vẫn chấp nhận được
    
    Args:
        use_4bit (bool): True = dùng 4-bit quantization, False = dùng bfloat16
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"🔄 Đang tải model từ {MODEL_PATH}...")
    
    # -----------------------------------------------------
    # BƯỚC 1: Load Tokenizer
    # -----------------------------------------------------
    # Tokenizer chuyển đổi text <-> tokens (số nguyên)
    # Ví dụ: "Xin chào" -> [123, 456, 789]
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),           # Đường dẫn model
        trust_remote_code=True,    # Cho phép chạy code từ model (cần cho Qwen)
        local_files_only=True      # Chỉ load từ local, không download
    )
    
    # -----------------------------------------------------
    # BƯỚC 2: Load Model với Quantization (nếu có)
    # -----------------------------------------------------
    if use_4bit:
        print("📦 Sử dụng 4-bit quantization để tăng tốc...")
        
        # Cấu hình BitsAndBytes cho 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # Bật 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16, # Dtype khi tính toán
            bnb_4bit_use_double_quant=True,       # Double quantization (tiết kiệm thêm)
            bnb_4bit_quant_type="nf4"             # Loại quantization: NormalFloat4
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization_config,  # Áp dụng cấu hình quant
            device_map="auto",                        # Tự động phân bổ GPU/CPU
            trust_remote_code=True,
            local_files_only=True
        )
    else:
        # Không dùng quantization - load model với bfloat16
        # bfloat16 = Brain Floating Point 16-bit (tối ưu cho AI)
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
    
    print("✅ Model đã được tải thành công!\n")
    return model, tokenizer


# =============================================================================
# PHẦN 6: CÁC HÀM TÌM KIẾM WEB
# =============================================================================

def web_search(query: str, max_results: int = 3) -> str:
    """
    Tìm kiếm thông tin trên web bằng DuckDuckGo.
    
    DuckDuckGo được chọn vì:
    - Miễn phí, không cần API key
    - Không theo dõi người dùng
    - Hỗ trợ tiếng Việt tốt
    
    Args:
        query (str): Từ khóa tìm kiếm
        max_results (int): Số kết quả tối đa (giảm = nhanh hơn)
    
    Returns:
        str: Kết quả đã format
    """
    if not WEB_SEARCH_AVAILABLE:
        return "❌ Web search not available."
    
    try:
        print(f"🔍 Searching: {query}")
        
        # DDGS = DuckDuckGo Search
        # Sử dụng context manager (with) để tự động cleanup
        with DDGS() as ddgs:
            # ddgs.text() = tìm kiếm văn bản
            # region='vn-vi' = ưu tiên kết quả tiếng Việt từ Việt Nam
            results = list(ddgs.text(query, region='vn-vi', max_results=max_results))
        
        if not results:
            return f"No results for: {query}"
        
        # Format kết quả dạng danh sách
        formatted = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            # Mỗi kết quả có title và body (mô tả)
            formatted += f"{i}. {r.get('title', '')}: {r.get('body', '')}\n"
        
        return formatted
    except Exception as e:
        return f"Search error: {e}"


def search_news(topic: str, max_results: int = 3) -> str:
    """
    Tìm tin tức mới nhất về một chủ đề.
    
    Khác với web_search, hàm này dùng ddgs.news() để lấy tin tức
    có nguồn (source) và thời gian cụ thể.
    """
    if not WEB_SEARCH_AVAILABLE:
        return "❌ Web search not available."
    
    try:
        print(f"📰 Searching news: {topic}")
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, region='vn-vi', max_results=max_results))
        
        if not results:
            return f"No news for: {topic}"
        
        formatted = f"Latest news about '{topic}':\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. [{r.get('source', '')}] {r.get('title', '')}\n"
        
        return formatted
    except Exception as e:
        return f"News search error: {e}"


# =============================================================================
# PHẦN 7: POST-PROCESSING - XÓA MARKDOWN
# =============================================================================

def clean_response(text: str) -> str:
    """
    Xóa markdown formatting để response tự nhiên như nói chuyện.
    
    Xử lý:
    - Xóa ### headers
    - Xóa ** bold **
    - Xóa - bullet points
    - Xóa --- horizontal rules
    - Xóa numbered lists (1. 2. 3.)
    """
    import re
    
    # Xóa markdown headers (### text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Xóa bold markers (**text** hoặc __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Xóa italic markers (*text* hoặc _text_)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    
    # Xóa horizontal rules (---)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    # Xóa bullet points (- text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Xóa numbered lists nhưng giữ nội dung (1. text -> text)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Xóa dòng trống thừa (nhiều dòng trống -> 1 dòng)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


# =============================================================================
# PHẦN 8: PHÁT HIỆN Ý ĐỊNH TÌM KIẾM
# =============================================================================

def detect_search_intent(user_input: str) -> tuple:
    """
    Phân tích câu hỏi của user để xác định có cần tìm kiếm web không.
    
    Hoạt động:
    1. Chuyển input về chữ thường
    2. So sánh với các từ khóa đã định nghĩa
    3. Nếu match -> cần tìm kiếm
    
    Returns:
        tuple: (needs_search: bool, search_type: str, search_query: str)
    """
    user_lower = user_input.lower()
    
    # Dictionary chứa các loại search và từ khóa tương ứng
    # QUAN TRỌNG: Thêm nhiều từ khóa để tránh model hallucinate (bịa thông tin)
    keywords = {
        'weather': ['thời tiết', 'weather', 'nhiệt độ', 'mưa', 'nắng', 'forecast'],
        'news': ['tin tức', 'tin mới', 'thời sự', 'news', 'update', 'cập nhật',
                 'chính trị', 'politics', 'politic', 'political',
                 'kinh tế', 'economy', 'xã hội', 'social',
                 'sự kiện', 'event', 'mới nhất', 'latest'],
        'price': ['giá vàng', 'giá xăng', 'tỷ giá', 'bitcoin', 'chứng khoán', 
                  'stock', 'crypto', 'price'],
        'sports': ['bóng đá', 'football', 'kết quả', 'tỷ số', 'soccer',
                   'quả bóng vàng', 'ballon d\'or', 'golden ball',  # Giải thưởng bóng đá
                   'world cup', 'champions league', 'premier league',
                   'giải thưởng', 'award', 'giành giải', 'champion', 'winner',
                   'oscar', 'grammy', 'nobel',  # Các giải thưởng khác
                   'vô địch', 'championship', 'tournament']
    }
    
    for search_type, kws in keywords.items():
        if any(kw in user_lower for kw in kws):
            # Không thêm ngày tháng vào query - làm search không ra kết quả
            return True, search_type, user_input
    
    return False, '', ''


# =============================================================================
# PHẦN 8: STREAMING GENERATION (QUAN TRỌNG!)
# =============================================================================

def generate_streaming(model, tokenizer, messages: list, max_new_tokens: int = 1024):
    """
    Generate response với streaming output - TÍNH NĂNG TỐI ƯU CHÍNH!
    
    STREAMING LÀ GÌ?
    -----------------
    Bình thường: Model generate xong TOÀN BỘ câu trả lời -> hiển thị
    Streaming: Model generate từng TOKEN -> hiển thị ngay -> tiếp tục
    
    Lợi ích:
    - User thấy response ngay lập tức (không phải chờ)
    - Cảm giác nhanh hơn nhiều
    - Có thể Ctrl+C dừng giữa chừng
    
    CÁCH HOẠT ĐỘNG:
    ---------------
    1. Main thread: Đọc tokens từ streamer và yield ra ngoài
    2. Background thread: Chạy model.generate() 
    3. TextIteratorStreamer: Cầu nối giữa 2 threads
    
    Yields:
        str: Từng đoạn text nhỏ (có thể là 1 từ hoặc vài ký tự)
    """
    
    # ---------------------------------------------------------
    # BƯỚC 1: Chuẩn bị input với Chat Template
    # ---------------------------------------------------------
    # Chat template = format chuẩn để model hiểu vai trò của mỗi message
    # Ví dụ Qwen format:
    # <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Hello<|im_end|>
    # <|im_start|>assistant
    
    text = tokenizer.apply_chat_template(
        messages,                    # List các message
        tokenize=False,              # Trả về string, không tokenize
        add_generation_prompt=True,  # Thêm prompt cho assistant
        enable_thinking=False        # Tắt thinking mode của Qwen3
    )
    
    # Tokenize string thành tensor
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # ---------------------------------------------------------
    # BƯỚC 2: Tạo TextIteratorStreamer
    # ---------------------------------------------------------
    # Streamer này hoạt động như một queue:
    # - model.generate() push tokens vào
    # - Vòng for bên ngoài pop tokens ra
    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_prompt=True,           # Bỏ qua phần prompt, chỉ lấy response
        skip_special_tokens=True    # Bỏ các token đặc biệt (<|im_end|>, etc.)
    )
    
    # ---------------------------------------------------------
    # BƯỚC 3: Cấu hình generation
    # ---------------------------------------------------------
    generation_kwargs = dict(
        **inputs,                           # Input tokens
        max_new_tokens=max_new_tokens,      # Giới hạn độ dài output
        do_sample=True,                     # Bật sampling (không greedy)
        temperature=0.7,                    # Độ "sáng tạo" (0=deterministic, 1=random)
        top_p=0.9,                          # Nucleus sampling
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer                   # Gắn streamer vào
    )
    
    # ---------------------------------------------------------
    # BƯỚC 4: Chạy generation trong thread riêng
    # ---------------------------------------------------------
    # Tại sao cần thread? 
    # Vì model.generate() là blocking - nó chạy đến khi xong mới return
    # Nhưng ta muốn đọc tokens TRONG KHI nó đang generate
    # -> Giải pháp: Chạy generate() ở thread khác
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # ---------------------------------------------------------
    # BƯỚC 5: Yield tokens khi nhận được
    # ---------------------------------------------------------
    # streamer là iterator - mỗi lần next() sẽ block cho đến khi
    # có token mới hoặc generation kết thúc
    
    response_text = ""
    for new_text in streamer:
        response_text += new_text
        yield new_text  # Yield ra ngoài để in ngay lập tức
    
    # Đợi thread kết thúc
    thread.join()


# =============================================================================
# PHẦN 9: HÀM CHAT CHÍNH
# =============================================================================

def chat_with_streaming(model, tokenizer, messages: list, user_input: str):
    """
    Xử lý chat với streaming và web search.
    
    Flow MỚI (Smart Search):
    1. Kiểm tra từ khóa (cách cũ, vẫn giữ)
    2. Nếu không match từ khóa -> generate response
    3. Kiểm tra nếu response chứa [SEARCH: query] -> model tự yêu cầu search
    4. Thực hiện search và regenerate với kết quả
    """
    
    # ---------------------------------------------------------
    # BƯỚC 1: Kiểm tra từ khóa (Keyword-based detection)
    # ---------------------------------------------------------
    needs_search, search_type, search_query = detect_search_intent(user_input)
    
    if needs_search and WEB_SEARCH_AVAILABLE:
        print(f"🔍 Detected search intent (keyword): {search_type}")
        
        if search_type == 'news':
            search_results = search_news(search_query)
        else:
            search_results = web_search(search_query)
        
        augmented_messages = messages.copy()
        augmented_messages[-1] = {
            "role": "user",
            "content": f"{user_input}\n\n[SEARCH RESULTS]:\n{search_results}\n\nBased on this, answer in detail."
        }
        messages_to_use = augmented_messages
        
        # In streaming response với search results
        print("\n🤖 Robot: ", end="", flush=True)
        response = ""
        for token in generate_streaming(model, tokenizer, messages_to_use):
            print(token, end="", flush=True)
            response += token
    else:
        # ---------------------------------------------------------
        # BƯỚC 2: Generate response KHÔNG streaming trước để check [SEARCH:]
        # ---------------------------------------------------------
        print("\n🤖 Robot: ", end="", flush=True)
        
        # Collect response trước (vẫn streaming)
        response = ""
        collected_tokens = []
        for token in generate_streaming(model, tokenizer, messages):
            collected_tokens.append(token)
            response += token
        
        # ---------------------------------------------------------
        # BƯỚC 3: Kiểm tra nếu model yêu cầu search
        # ---------------------------------------------------------
        search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
        match = re.search(search_pattern, response)
        
        if match and WEB_SEARCH_AVAILABLE:
            # Model yêu cầu search - KHÔNG in response cũ
            model_search_query = match.group(1).strip()
            print(f"Đang tìm kiếm: {model_search_query}...")
            
            # Thực hiện search
            search_results = web_search(model_search_query)
            
            # Tạo messages mới với kết quả search
            search_messages = messages.copy()
            search_messages.append({
                "role": "assistant",
                "content": f"I need to search for information about: {model_search_query}"
            })
            search_messages.append({
                "role": "user",
                "content": f"[SEARCH RESULTS]:\n{search_results}\n\nNow answer the original question based on these search results. Provide detailed information."
            })
            
            # Regenerate response với streaming
            print("\n🤖 Robot: ", end="", flush=True)
            response = ""
            for token in generate_streaming(model, tokenizer, search_messages):
                print(token, end="", flush=True)
                response += token
        else:
            # Không cần search - in ra các tokens đã collect
            for token in collected_tokens:
                print(token, end="", flush=True)
    
    print("\n")
    
    # Post-process: xóa markdown formatting để response tự nhiên hơn
    cleaned = clean_response(response.strip())
    return cleaned


# =============================================================================
# PHẦN 10: MAIN - VÒNG LẶP CHAT CHÍNH
# =============================================================================

def main():
    """
    Entry point của chương trình.
    
    Flow:
    1. Hỏi user có muốn dùng 4-bit không
    2. Load model
    3. Vòng lặp chat vô hạn
    4. Xử lý các lệnh đặc biệt
    5. Generate và hiển thị response
    """
    
    # Header
    print("=" * 60)
    print("🤖 QWEN3-8B OPTIMIZED CHAT")
    print("=" * 60)
    
    # Hỏi về quantization
    use_4bit = input("Sử dụng 4-bit quantization để tăng tốc? (y/n, default=y): ").strip().lower()
    use_4bit = use_4bit != 'n'  # Mặc định là Yes
    
    # Load model
    model, tokenizer = load_model(use_4bit=use_4bit)
    
    # Khởi tạo conversation history
    # Bắt đầu với system prompt
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    # Hiển thị thông tin
    print("=" * 60)
    print(f"📅 Today: {CURRENT_DATE}")
    print(f"🌐 Web search: {'✅ Enabled' if WEB_SEARCH_AVAILABLE else '❌ Disabled'}")
    print(f"⚡ 4-bit quantization: {'✅ Enabled' if use_4bit else '❌ Disabled'}")
    print("=" * 60)
    print("Commands: 'quit', 'clear', 'history', 'news', 'search: <query>'")
    print("=" * 60 + "\n")
    
    # Vòng lặp chat chính
    while True:
        try:
            # Nhận input từ user
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
            
            # -----------------------------------------
            # Xử lý các lệnh đặc biệt
            # -----------------------------------------
            if user_input.lower() in ['quit', 'exit', 'thoát']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                # Reset history về chỉ còn system prompt
                conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("🗑️ History cleared.\n")
                continue
            
            if user_input.lower() == 'history':
                # Hiển thị lịch sử (bỏ qua system prompt)
                print("\n📜 CONVERSATION HISTORY:")
                for msg in conversation_history[1:]:
                    role = "👤" if msg["role"] == "user" else "🤖"
                    # Hiển thị tối đa 100 ký tự
                    print(f"{role}: {msg['content'][:100]}...")
                print()
                continue
            
            if user_input.lower().startswith('search:'):
                # Tìm kiếm thủ công
                query = user_input[7:].strip()
                print(web_search(query))
                continue
            
            if user_input.lower() == 'news':
                # Xem tin tức nhanh
                print(search_news("Vietnam"))
                continue
            
            # -----------------------------------------
            # Chat thông thường
            # -----------------------------------------
            
            # Thêm message của user vào history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response với streaming
            response = chat_with_streaming(model, tokenizer, conversation_history, user_input)
            
            # Thêm response của assistant vào history
            conversation_history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            # Ctrl+C để thoát
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


# =============================================================================
# ENTRY POINT
# =============================================================================
# Chỉ chạy main() khi file được thực thi trực tiếp
# Không chạy khi file được import như module

if __name__ == "__main__":
    main()
