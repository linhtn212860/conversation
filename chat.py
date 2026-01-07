#!/usr/bin/env python3
"""
Interactive continuous chat with Qwen3-8B model.
Maintains conversation history for multi-turn dialogue.
Supports web search for real-time information lookup.
"""

import torch
import re
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: Web search
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

# Model path
MODEL_PATH = Path(__file__).parent / "models" / "Qwen_Qwen3-8B"

# Current date for context
CURRENT_DATE = datetime.now().strftime("%d/%m/%Y")

# System prompt for home robot with web search capability (Bilingual)
SYSTEM_PROMPT = f"""You are a friendly and helpful home assistant robot. Today is {CURRENT_DATE}.

LANGUAGE RULES:
- If the user speaks Vietnamese, respond in Vietnamese
- If the user speaks English, respond in English
- Match the user's language in your responses

You can search for information from the internet when needed. When you need to search for new information or news, start your response with:
[SEARCH: search keywords]

Guidelines:
- Respond naturally like a caring family member
- Keep responses concise and conversational
- Be warm and empathetic when needed
- Use [SEARCH: ...] when you need current news, weather, prices, latest events
- Do not use [SEARCH] for general questions that don't need real-time information

---

Bạn là một robot trợ lý gia đình thân thiện và hữu ích. Hôm nay là ngày {CURRENT_DATE}.

QUY TẮC NGÔN NGỮ:
- Nếu người dùng nói tiếng Việt, trả lời bằng tiếng Việt
- Nếu người dùng nói tiếng Anh, trả lời bằng tiếng Anh
- Phù hợp với ngôn ngữ của người dùng trong câu trả lời

Hướng dẫn:
- Trả lời tự nhiên như một thành viên gia đình quan tâm
- Giữ câu trả lời ngắn gọn và mang tính đối thoại
- Ấm áp và đồng cảm khi cần thiết"""


def load_model():
    """Load Qwen3-8B model and tokenizer."""
    print(f"🔄 Đang tải model từ {MODEL_PATH}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    print("✅ Model đã được tải thành công!\n")
    return model, tokenizer


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    if not WEB_SEARCH_AVAILABLE:
        return "❌ Tính năng tìm kiếm web chưa được cài đặt."
    
    try:
        print(f"🔍 Đang tìm kiếm: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='vn-vi', max_results=max_results))
        
        if not results:
            return f"Không tìm thấy kết quả cho: {query}"
        
        # Format results
        formatted = f"📰 KẾT QUẢ TÌM KIẾM cho '{query}':\n\n"
        for i, r in enumerate(results, 1):
            title = r.get('title', 'Không có tiêu đề')
            body = r.get('body', 'Không có mô tả')
            formatted += f"{i}. **{title}**\n   {body}\n\n"
        
        return formatted
    except Exception as e:
        return f"❌ Lỗi khi tìm kiếm: {e}"


def search_news(topic: str = "Việt Nam", max_results: int = 5) -> str:
    """Search for latest news."""
    if not WEB_SEARCH_AVAILABLE:
        return "❌ Tính năng tìm kiếm web chưa được cài đặt."
    
    try:
        print(f"📰 Đang tìm tin tức: {topic}")
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, region='vn-vi', max_results=max_results))
        
        if not results:
            return f"Không tìm thấy tin tức cho: {topic}"
        
        # Format results
        formatted = f"📰 TIN TỨC MỚI NHẤT về '{topic}':\n\n"
        for i, r in enumerate(results, 1):
            title = r.get('title', 'Không có tiêu đề')
            body = r.get('body', 'Không có mô tả')
            date = r.get('date', '')
            source = r.get('source', 'Nguồn không xác định')
            formatted += f"{i}. **{title}**\n   📅 {date} | 📌 {source}\n   {body}\n\n"
        
        return formatted
    except Exception as e:
        return f"❌ Lỗi khi tìm tin tức: {e}"


def generate_response(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    """Generate response for the current conversation."""
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (only new tokens)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def detect_search_intent(user_input: str) -> tuple:
    """Detect if user's input requires web search. Returns the user's query for searching."""
    user_lower = user_input.lower()
    
    # Weather patterns
    weather_keywords = ['thời tiết', 'weather', 'nhiệt độ', 'mưa', 'nắng', 'độ ẩm', 'dự báo']
    if any(kw in user_lower for kw in weather_keywords):
        return True, 'weather', f"{user_input} {CURRENT_DATE}"
    
    # News patterns - USE USER'S ACTUAL QUERY
    news_keywords = ['tin tức', 'tin mới', 'thời sự', 'sự kiện', 'mới nhất', 'hôm nay có gì', 'cập nhật']
    if any(kw in user_lower for kw in news_keywords):
        # Use the user's actual input as search query
        return True, 'news', f"{user_input} {CURRENT_DATE}"
    
    # Price patterns
    price_keywords = ['giá vàng', 'giá xăng', 'giá dầu', 'tỷ giá', 'đô la', 'usd', 'bitcoin', 'chứng khoán']
    if any(kw in user_lower for kw in price_keywords):
        return True, 'price', f"{user_input} {CURRENT_DATE}"
    
    # Sports patterns
    sports_keywords = ['bóng đá', 'v-league', 'world cup', 'kết quả', 'tỷ số']
    if any(kw in user_lower for kw in sports_keywords):
        return True, 'sports', f"{user_input} {CURRENT_DATE}"
    
    return False, '', ''


def process_response_with_search(model, tokenizer, messages: list, user_input: str) -> str:
    """Generate response and handle search requests if needed."""
    
    # Check if user's input needs web search
    needs_search, search_type, search_query = detect_search_intent(user_input)
    
    if needs_search and WEB_SEARCH_AVAILABLE:
        print(f"🔍 Phát hiện cần tra cứu: {search_type}")
        
        # Perform search based on type
        if search_type == 'news':
            search_results = search_news(search_query)
        else:
            search_results = web_search(search_query)
        
        # Create augmented messages with search results
        augmented_messages = messages.copy()
        # Replace last user message with augmented version
        augmented_messages[-1] = {
            "role": "user",
            "content": f"{user_input}\n\n[THÔNG TIN TRA CỨU TỪ INTERNET]:\n{search_results}\n\nDựa trên thông tin trên, hãy trả lời câu hỏi của tôi một cách ngắn gọn và dễ hiểu."
        }
        
        response = generate_response(model, tokenizer, augmented_messages)
    else:
        # Normal generation without search
        response = generate_response(model, tokenizer, messages)
        
        # Fallback: Check if model wants to search in response
        search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
        match = re.search(search_pattern, response)
        
        if match and WEB_SEARCH_AVAILABLE:
            search_query = match.group(1).strip()
            
            news_keywords = ['tin tức', 'tin mới', 'news', 'thời sự', 'sự kiện']
            if any(kw in search_query.lower() for kw in news_keywords):
                search_results = search_news(search_query)
            else:
                search_results = web_search(search_query)
            
            # Regenerate with search results
            messages.append({
                "role": "assistant",
                "content": f"Tôi sẽ tìm kiếm thông tin cho bạn..."
            })
            messages.append({
                "role": "user", 
                "content": f"Đây là kết quả tìm kiếm:\n\n{search_results}\n\nHãy tóm tắt thông tin này."
            })
            
            response = generate_response(model, tokenizer, messages)
            messages.pop()
            messages.pop()
    
    return response


def print_history(messages: list):
    """Print conversation history (excluding system prompt)."""
    print("\n" + "=" * 60)
    print("📜 LỊCH SỬ HỘI THOẠI:")
    print("=" * 60)
    for msg in messages[1:]:  # Skip system prompt
        role = "👤 Bạn" if msg["role"] == "user" else "🤖 Robot"
        print(f"\n{role}:")
        print(f"   {msg['content']}")
    print("=" * 60 + "\n")


def manual_search(query: str):
    """Perform manual search and display results."""
    if not WEB_SEARCH_AVAILABLE:
        print("❌ Tính năng tìm kiếm web chưa được cài đặt.")
        print("   Hãy chạy: pip install duckduckgo-search")
        return
    
    news_keywords = ['tin tức', 'tin mới', 'news', 'thời sự']
    if any(kw in query.lower() for kw in news_keywords):
        results = search_news(query)
    else:
        results = web_search(query)
    
    print(results)


def main():
    """Main interactive chat loop."""
    # Load model
    model, tokenizer = load_model()
    
    # Initialize conversation history with system prompt
    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("=" * 60)
    print("🤖 QWEN3-8B CHAT - Hội thoại liên tục với tra cứu web")
    print("=" * 60)
    print(f"📅 Ngày hôm nay: {CURRENT_DATE}")
    print(f"🌐 Tra cứu web: {'✅ Đã bật' if WEB_SEARCH_AVAILABLE else '❌ Chưa cài đặt'}")
    print("=" * 60)
    print("Lệnh đặc biệt:")
    print("  • 'quit' hoặc 'exit'    - Thoát chương trình")
    print("  • 'clear'               - Xóa lịch sử hội thoại")
    print("  • 'history'             - Xem lịch sử hội thoại")
    print("  • 'search: <từ khóa>'   - Tìm kiếm thủ công")
    print("  • 'news'                - Xem tin tức mới nhất")
    print("=" * 60 + "\n")
    
    while True:
        try:
            # Get user input
            user_input = input("👤 Bạn: ").strip()
            
            # Handle special commands
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'thoát']:
                print("\n👋 Tạm biệt! Hẹn gặp lại!")
                break
            
            if user_input.lower() == 'clear':
                conversation_history = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                print("🗑️ Đã xóa lịch sử hội thoại.\n")
                continue
            
            if user_input.lower() == 'history':
                print_history(conversation_history)
                continue
            
            if user_input.lower().startswith('search:'):
                query = user_input[7:].strip()
                manual_search(query)
                continue
            
            if user_input.lower() == 'news':
                print(search_news("Việt Nam"))
                continue
            
            # Add user message to history
            conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate response with potential web search
            print("\n🤖 Robot đang suy nghĩ...")
            response = process_response_with_search(model, tokenizer, conversation_history, user_input)
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Print response
            print(f"\n🤖 Robot: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt! Hẹn gặp lại!")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}\n")
            continue


if __name__ == "__main__":
    main()
