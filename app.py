import gradio as gr
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ========== МОДЕЛЬ ==========
MODEL_NAME = "IlyaGusev/rut5_base_sum_gazeta"
print("Загружаем модель ruT5 для суммаризации...")

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model_loaded = True
    print("✅ Модель загружена!")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    model_loaded = False

history = []

# ========== Функция для определения длины ==========
def get_length_params(num_sentences):
    """Преобразует количество предложений в параметры длины"""
    # 1 предложение ~ 50 токенов, 10 предложений ~ 350 токенов
    max_length = min(400, num_sentences * 40 + 20)
    min_length = max(25, num_sentences * 25 - 10)
    return max_length, min_length

# ========== Абстрактная суммаризация ==========
def summarize_rut5(text, num_sentences):
    if not model_loaded or not text:
        return None
    
    if len(text) > 3000:
        text = text[:3000]
    
    max_len, min_len = get_length_params(num_sentences)
    
    print(f"Генерация с параметрами: num_sentences={num_sentences}, max_len={max_len}, min_len={min_len}")
    
    # Добавляем префикс для модели (важно для gazeta)
    input_text = "summarize: " + text
    
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            repetition_penalty=1.2,
            length_penalty=0.8,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.strip()
    summary = re.sub(r'\s+', ' ', summary)
    
    # Постобработка: гарантируем нужное количество предложений
    sentences = re.split(r'[.!?]+', summary)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if len(sentences) > num_sentences:
        # Если модель выдала больше, обрезаем
        summary = ". ".join(sentences[:num_sentences]) + "."
    elif len(sentences) < num_sentences and len(summary) < max_len * 0.7:
        # Если выдала мало, пробуем снова с большим max_length
        max_len2 = min(500, max_len + 100)
        inputs2 = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs2 = model.generate(
                inputs2.input_ids,
                max_length=max_len2,
                min_length=min_len + 20,
                num_beams=4,
                repetition_penalty=1.2,
                length_penalty=0.8,
                early_stopping=False,
                no_repeat_ngram_size=3,
            )
        summary2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
        summary2 = re.sub(r'\s+', ' ', summary2).strip()
        sentences2 = re.split(r'[.!?]+', summary2)
        sentences2 = [s.strip() for s in sentences2 if len(s.strip()) > 10]
        if len(sentences2) >= num_sentences:
            summary = ". ".join(sentences2[:num_sentences]) + "."
    
    return summary

# ========== Примитивный метод ==========
def primitive_summarize(text, num_sentences):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return "Не удалось разбить текст на предложения."
    
    num = min(num_sentences, len(sentences))
    result = ". ".join(sentences[:num])
    if result and result[-1] != '.':
        result += '.'
    return result

# ========== Основная функция ==========
def summarize_main(text, num_sentences, method, file):
    if file is not None:
        try:
            if hasattr(file, 'name'):
                with open(file.name, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif isinstance(file, str):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
        except Exception as e:
            return f"❌ Ошибка: {e}", format_history()
    
    if not text or not text.strip():
        return "❌ Введите текст.", format_history()
    
    if len(text) < 50:
        return "❌ Слишком короткий текст (минимум 50 символов).", format_history()
    
    warning = ""
    if len(text) > 3000 and method == "ruT5 (абстрактная)":
        text = text[:3000]
        warning = f"⚠️ Текст обрезан до 3000 символов.\n\n"
    
    if method == "ruT5 (абстрактная)":
        summary = summarize_rut5(text, num_sentences)
        if summary is None:
            summary = primitive_summarize(text, num_sentences)
            summary = "⚠️ Модель не загружена.\n\n" + summary
    else:
        summary = primitive_summarize(text, num_sentences)
    
    history.append({
        "text": text[:150] + ("..." if len(text) > 150 else ""),
        "summary": summary[:500]
    })
    
    while len(history) > 5:
        history.pop(0)
    
    return warning + summary, format_history()

def format_history():
    if not history:
        return "📜 История пуста"
    
    lines = []
    for i, item in enumerate(reversed(history), 1):
        lines.append(f"**{i}. Исходный:** {item['text']}\n**✅ Саммари:** {item['summary']}\n---")
    
    return "\n\n".join(lines)

# ========== Интерфейс ==========
with gr.Blocks(title="Суммаризатор текста") as demo:
    gr.Markdown("# 📄 Система автоматической суммаризации текста")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="📝 Исходный текст", lines=12)
            file_input = gr.File(label="📂 Или .txt файл", file_types=[".txt"])
            slider = gr.Slider(1, 10, value=3, step=1, label="📊 Количество предложений")
            method = gr.Radio(
                ["ruT5 (абстрактная)", "Примитивный (первые предложения)"],
                label="Метод",
                value="ruT5 (абстрактная)"
            )
            btn = gr.Button("Суммаризировать", variant="primary")
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="Результат", lines=8)
    
    history_output = gr.Markdown(label="История")
    
    btn.click(summarize_main, [text_input, slider, method, file_input], [text_output, history_output])

if __name__ == "__main__":
    demo.launch()