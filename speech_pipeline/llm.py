"""
LLM Wrapper - Qwen3-8B for conversational responses
Based on chat_optimized.py - đã tối ưu với streaming, web search, và clean response
"""
import torch
import re
from typing import Generator, List, Dict, Optional
from threading import Thread
from pathlib import Path
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer, # in kết quả ra màn hình theo thời gian thực
    BitsAndBytesConfig # cho phép sử dụng nén 4-bit để tiết kiệm VRAM GPU
)

from .config import (
    QWEN_MODEL_PATH,    # Đường dẫn đến thư mục chứa model Qwen
    LLM_MAX_NEW_TOKENS, # Số lượng token (từ/ký tự) tối đa mà model được phép tạo ra
    LLM_TEMPERATURE,    # Độ sáng tạo: cao (0.8+) thì bay bổng, thấp (0.1-) thì chính xác/cứng nhắc (LLM_TEMPERATURE = 0.7)
    LLM_TOP_P,          # Kỹ thuật lọc từ: chỉ chọn các từ có tổng xác suất tích lũy bằng P (LLM_TOP_P = 0.9) 
                        # tính xác suất hàng nghìn từ có thể xảy ra: cộng dồn xác suất từ cao đến thấp và dừng lại khi đạt ngưỡng P
    LLM_USE_4BIT        # Có sử dụng nén 4-bit không (giúp tiết kiệm VRAM GPU)
)

# =============================================================================
# WEB SEARCH 
# =============================================================================

# DuckDuckGo search client có 2 package hay gặp:
# - `ddgs`: package mới hơn (tên module ngắn gọn).
# - `duckduckgo_search`: package cũ/phổ biến trước đây, cũng cung cấp class `DDGS`.
# Đoạn try/except dưới đây giúp code chạy được ở cả 2 trường hợp; nếu không có cái nào
# thì tắt tính năng web search (trả về chuỗi rỗng) để pipeline không bị crash.
try:
    from ddgs import DDGS
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        WEB_SEARCH_AVAILABLE = True
    except ImportError:
        WEB_SEARCH_AVAILABLE = False


def web_search(query: str, max_results: int = 3) -> str:  # max_results: số lượng kết quả tìm kiếm
    """Tìm kiếm thông tin trên web bằng DuckDuckGo và format thành text.
    
    Notes:
    - `max_results` càng cao thì càng nhiều nguồn nhưng sẽ chậm hơn.
    - `region='wt-wt'` (worldwide) thường cho kết quả tiếng Anh tốt hơn.
    - Hàm trả về chuỗi rỗng nếu không có thư viện search, không có kết quả,
      hoặc gặp lỗi trong quá trình gọi API của thư viện.
    """
    if not WEB_SEARCH_AVAILABLE:
        return ""
    
    try:
        with DDGS() as ddgs:
            # ddgs.text(...) trả về iterator các dict (title/body/href...) -> convert sang list để dùng nhiều lần.
            results = list(ddgs.text(query, region='wt-wt', max_results=max_results))
        
        if not results:
            return ""
        
        # Format gọn để có thể nhét vào prompt dưới dạng [INFO] cho LLM tham chiếu.
        formatted = f"Web search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            # Dùng .get(...) để tránh KeyError vì schema có thể khác nhau giữa version.
            title = r.get('title', '')
            body = r.get('body', '')
            link = r.get('href', '')
            formatted += f"{i}. {title}\n   {body}\n   Source: {link}\n\n"
        
        return formatted
    except Exception as e:
        # Không raise lỗi để tránh làm vỡ luồng chat; chỉ log và trả về rỗng.
        print(f"Search error: {e}")
        return ""


def search_news(topic: str, max_results: int = 3) -> str:
    """Tìm tin tức mới nhất về một chủ đề (DuckDuckGo News)."""
    if not WEB_SEARCH_AVAILABLE:
        return ""
    
    try:
        with DDGS() as ddgs:
            # `region='vn-vi'` ưu tiên nguồn tiếng Việt (có thể đổi tuỳ mục tiêu).
            results = list(ddgs.news(topic, region='vn-vi', max_results=max_results))
        
        if not results:
            return ""
        
        formatted = f"Latest news about '{topic}':\n\n"
        for i, r in enumerate(results, 1):
            # News results thường có các field như source/title/date/url...; ở đây chỉ lấy source+title cho gọn.
            formatted += f"{i}. [{r.get('source', '')}] {r.get('title', '')}\n"
        
        return formatted
    except Exception:
        return ""


# =============================================================================
# POST-PROCESSING - XÓA MARKDOWN 
# =============================================================================

def clean_response(text: str) -> str:
    """Xóa markdown formatting để response tự nhiên như nói chuyện."""
    # Xóa markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Xóa bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Xóa italic markers
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', text)
    
    # Xóa horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    # Xóa bullet points
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    
    # Xóa numbered lists
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Xóa dòng trống thừa
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


# =============================================================================
# PHÁT HIỆN Ý ĐỊNH TÌM KIẾM 
# =============================================================================

def detect_search_intent(user_input: str) -> tuple:
    """Analyze user query to determine if web search is needed.
    
    Improvements:
    - Thêm nhiều keywords hơn cho mỗi category
    - Thêm category 'general' cho các câu hỏi dạng what/who/when/where
    - Tạo search query tốt hơn
    """
    user_lower = user_input.lower()

    keywords = {
        'weather': ['weather', 'temperature', 'rain', 'sunny', 'forecast', 'climate',
                    'hot', 'cold', 'humid', 'degrees', 'celsius', 'fahrenheit',
                    'weather like', 'weather in', 'weather today', 'weather tomorrow'],
        'news': ['news', 'latest', 'update', 'current events', 'today\'s', 'recent',
                 'politics', 'economy', 'events', 'happening', 'breaking',
                 'announcement', 'announced', 'released'],
        'price': ['price', 'cost', 'how much', 'bitcoin', 'stock', 'crypto', 'exchange rate',
                  'gold price', 'gas price', 'market', 'usd', 'dollar', 'euro',
                  'stock market', 'trading'],
        'sports': ['football', 'soccer', 'basketball', 'score', 'match', 'game',
                   'ballon d\'or', 'world cup', 'champions league', 'championship',
                   'tournament', 'nba', 'nfl', 'premier league', 'winner', 'won',
                   'team', 'player', 'olympics', 'medal'],
        'general': ['who is', 'what is', 'when is', 'where is', 'how to',
                    'what happened', 'when did', 'who won', 'what\'s the',
                    'tell me about', 'do you know', 'can you tell me',
                    'latest', 'new', '2024', '2025', '2026']
    }

    for search_type, kws in keywords.items():
        if any(kw in user_lower for kw in kws):
            # Create optimized search query
            search_query = user_input
            # Add current year for time-sensitive queries
            if search_type in ['news', 'weather', 'price', 'sports']:
                from datetime import datetime
                current_year = datetime.now().year
                if str(current_year) not in user_input:
                    search_query = f"{user_input} {current_year}"
            return True, search_type, search_query

    return False, '', ''


# =============================================================================
# SYSTEM PROMPT 
# =============================================================================

CURRENT_DATE = datetime.now().strftime("%B %d, %Y")  # Format: "January 01, 2026" - tránh nhầm lẫn DD/MM vs MM/DD

# System prompt for English conversation with search capability
SYSTEM_PROMPT = f"""You are a friendly AI voice assistant having a natural conversation. Today is {CURRENT_DATE}.

IMPORTANT RULES:
1. CONVERSATIONAL TONE - Talk like you're chatting with a friend face-to-face
2. NO MARKDOWN - Never use ###, **, bullet points, or numbered lists
3. SHORT & NATURAL - Keep responses brief (3-4 sentences max for simple questions)
4. SPEAK, DON'T WRITE - Imagine someone is listening, not reading
5. USE CONTRACTIONS - Say "I'm", "you're", "it's", "don't" (sounds more natural)
6. NO EMOJI OR SPECIAL CHARACTERS - Never use emojis, icons, or special Unicode characters in your responses

HANDLING SEARCH RESULTS:
When you receive [INFO:] with search results, you MUST:
- Use the actual data from search results to answer
- Cite specific facts, numbers, or information from the results
- If the results have weather data, prices, news, etc. - use them!
- Summarize key points naturally without copying word for word
- If results are not useful, honestly say you couldn't find current info

EXAMPLES:
❌ BAD (ignoring search results): "I don't have access to current weather data..."
✅ GOOD (using search results): "Based on what I found, Hanoi is around 25 degrees today with some clouds..."

❌ BAD: "Here are some ways to improve your productivity: 1. Wake up early 2. Exercise..."
✅ GOOD: "Oh, I'd say start with your morning routine! Maybe try waking up a bit earlier and getting some exercise in."

Remember: You're speaking out loud to someone, not writing an essay. Be warm, brief, and USE the search results when provided!"""


# =============================================================================
# QWEN LLM CLASS
# =============================================================================

class QwenLLM:
    """
    Wrapper cho Qwen3-8B model - dựa trên chat_optimized.py.
    
    Features:
    - Streaming output
    - 4-bit quantization (optional)
    - Web search integration
    - Clean response (no markdown)
    """
    
    def __init__(self, device: str = "cuda", use_4bit: bool = LLM_USE_4BIT):
        """
        Initialize Qwen LLM.
        
        Args:
            device: "cuda" hoặc "cpu"
            use_4bit: Sử dụng 4-bit quantization
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.use_4bit = use_4bit
        
        self._load_model()
        self._init_conversation()
    
    def _load_model(self):
        """Load Qwen3-8B model (same as chat_optimized.py)."""
        print(f"🔄 Loading Qwen3-8B from {QWEN_MODEL_PATH}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(QWEN_MODEL_PATH),
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load model with optional quantization
        if self.use_4bit:
            print("📦 Using 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(QWEN_MODEL_PATH),
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(QWEN_MODEL_PATH),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
        
        print("✅ Qwen3-8B loaded!")
    
    def _init_conversation(self):
        """Initialize conversation với system prompt."""
        self.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
    
    def clear_history(self):
        """Clear conversation history."""
        self._init_conversation()
    
    def generate(self, user_input: str, history: list = None) -> str:
        """
        Generate response cho user input (non-streaming).
        Includes web search integration.
        
        Args:
            user_input: Câu hỏi/input từ user
            history: External history (optional). Nếu None, dùng internal history.
        """
        # Use external history if provided, else use internal
        if history is not None:
            # External history from server session
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-6:]:  # Last 6 turns
                messages.append(h)
            messages.append({"role": "user", "content": user_input})
        else:
            # Internal history
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            messages = self.conversation_history.copy()
        
        # Check if web search needed
        needs_search, search_type, search_query = detect_search_intent(user_input)
        
        if needs_search and WEB_SEARCH_AVAILABLE:
            if search_type == 'news':
                search_results = search_news(search_query)
            else:
                search_results = web_search(search_query)
            
            # Augment last message with search results
            if search_results:
                messages[-1] = {
                    "role": "user",
                    "content": f"{user_input}\n\n[INFO]:\n{search_results}\n\nAnswer briefly based on this."
                }
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clean response (remove markdown)
        response = clean_response(response.strip())
        
        # Handle [SEARCH:] pattern from model
        search_match = re.search(r'\[SEARCH:\s*([^\]]+)\]', response)
        if search_match and WEB_SEARCH_AVAILABLE:
            query = search_match.group(1).strip()
            print(f"🔍 Model requested search: {query}")
            
            search_results = web_search(query)
            if search_results:
                # Regenerate with search results
                messages[-1] = {
                    "role": "user",
                    "content": f"{user_input}\n\n[INFO]:\n{search_results}\n\nAnswer briefly."
                }
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=LLM_TEMPERATURE,
                        top_p=LLM_TOP_P,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                response = clean_response(response.strip())
        
        # Add to internal history only if not using external history
        if history is None:
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Keep only last 12 turns
            if len(self.conversation_history) > 13:
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-12:]
        
        return response
    
    def generate_streaming(self, user_input: str) -> Generator[str, None, None]:
        """
        Generate response với streaming output.
        Same logic as chat_optimized.py generate_streaming.
        """
        # Add user message
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        messages = self.conversation_history.copy()
        
        # Check web search
        needs_search, search_type, search_query = detect_search_intent(user_input)
        
        if needs_search and WEB_SEARCH_AVAILABLE:
            if search_type == 'news':
                search_results = search_news(search_query)
            else:
                search_results = web_search(search_query)
            
            if search_results:
                messages[-1] = {
                    "role": "user",
                    "content": f"{user_input}\n\n[INFO]:\n{search_results}\n\nAnswer briefly."
                }
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )
        
        # Start generation in background thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield tokens
        response = ""
        for token in streamer:
            response += token
            yield token
        
        thread.join()
        
        # Clean and save to history
        response = clean_response(response.strip())
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        # Keep only last 12 turns
        if len(self.conversation_history) > 13:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-12:]


# =============================================================================
# TEST
# =============================================================================

def test_llm():
    """Test LLM."""
    print("Testing Qwen LLM (based on chat_optimized.py)...")

    llm = QwenLLM()

    # Test normal generation
    response = llm.generate("Hello! How are you today?")
    print(f"Response: {response}")

    # Test with search
    response = llm.generate("What's the latest news about the Ballon d'Or?")
    print(f"Response with search: {response}")

    print("✅ LLM test complete!")


if __name__ == "__main__":
    test_llm()
