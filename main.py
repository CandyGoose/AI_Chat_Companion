import datetime
import os
import logging
from dotenv import load_dotenv
from llama_cpp import Llama
from pyrogram import Client, filters
from pyrogram.enums import ChatAction

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME")

app = Client(SESSION_NAME, api_id=API_ID, api_hash=API_HASH)

llm = Llama(
    model_path="./models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    chat_format="llama-3",
    verbose=False,
    n_ctx=500
)

user_message_history = {}

@app.on_message(filters.private & ~filters.me)
async def message_handler(client, message):
    chat_id = message.chat.id
    user_id = message.from_user.id

    logger.info(f"Входящее сообщение от {message.from_user.first_name} ({user_id}): {message.text}")

    await client.read_chat_history(chat_id)

    if user_id not in user_message_history:
        logger.info(f"Считывание предыдущей истории для пользователя {user_id}")
        user_message_history[user_id] = []

        async for msg in client.get_chat_history(chat_id, limit=4):
            if msg.text:
                role = "assistant" if msg.from_user.is_self else "user"
                user_message_history[user_id].append({"role": role, "content": msg.text})

        logger.info(f"Считывание для пользователя {user_id} завершено.")

    if not message.text:
        logger.info("Сообщение не содержит текста, игнорируется.")
        return

    user_history = user_message_history.get(user_id, [])
    user_history.append({"role": "user", "content": message.text})

    current_date_time = datetime.datetime.now().strftime("%d %B %Y, %H:%M MSK")
    messages = [{"role": "system",
                 "content": (
                    "Ты мой хороший друг. "
                    "Отвечай кратко, тепло и просто. "
                    "Не начинай предложения с междометий, таких как 'Ха', 'Ох', 'Эх' и других. "
                    "Не ставь точки, восклицательные или вопросительные знаки в конце предложений. "
                    "Формат ответа должен быть простым и без лишней пунктуации."
                    "Примеры ответов:"
                    "- Вопрос: 'Как твои дела?' Ответ: 'Хорошо, спасибо'"
                    "- Вопрос: 'Что нового?' Ответ: 'Ничего особенного'"
                    "- Вопрос: 'Почему ты так думаешь?' Ответ: 'Я уверена в этом'"
                 )}
                ]
    for msg in user_history:
        messages.append(msg)

    max_context_tokens = 1000
    total_tokens = sum(len(msg["content"]) for msg in messages if msg.get("content"))
    while total_tokens > max_context_tokens:
        messages.pop(1)
        total_tokens = sum(len(msg["content"]) for msg in messages if msg.get("content"))

    await client.send_chat_action(chat_id, ChatAction.TYPING)

    out = llm.create_chat_completion(messages)

    reply = out["choices"][0]["message"]["content"]
    reply = reply.rstrip(".!")

    logger.info(f"Исходящий ответ пользователю {user_id}: {reply}")

    user_history.append({"role": "assistant", "content": reply})
    user_message_history[user_id] = user_history[-20:]

    await message.reply_text(reply)

print("Аккаунт пользователя готов к работе")
app.run()
