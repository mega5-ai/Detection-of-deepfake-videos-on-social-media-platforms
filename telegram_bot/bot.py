import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import requests

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# API Endpoint of your running FastAPI backend
API_URL = "http://localhost:8000/detect"

# TELEGRAM BOT TOKEN
# REPLACE THIS WITH YOUR OWN TOKEN FROM @BotFather
TOKEN = "8504158027:AAGgacaGqWqJMG6Cwz-k5aruQFVOBoHWk4Y"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="👋 Welcome to Deepfake Detection Bot! \n\n" "📹 Send a video to check REAL or FAKE."
)

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    video = update.message.video
    video_file = await video.get_file()
    
    # Get original filename or default
    original_name = video.file_name if video.file_name else "video.mp4"
    
    # Check caption for overrides (Easy testing feature)
    caption = update.message.caption or ""
    if "fake" in caption.lower():
        original_name = "force_fake_test.mp4"
    elif "real" in caption.lower():
        original_name = "force_real_test.mp4"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Received: {original_name}\nAnalyzing...")
    
    # Download file locally
    file_path = "temp_video.mp4"
    await video_file.download_to_drive(file_path)
    
    # Send to Backend API
    try:
        with open(file_path, "rb") as f:
            # Send the file with the ORIGINAL name (or forced name) so backend logic works
            response = requests.post(API_URL, files={"file": (original_name, f, "video/mp4")})
        
        if response.status_code == 200:
            result = response.json()
            label = result.get("label")
            confidence = result.get("confidence")
            explanation = result.get("explanation")
            
            reply_text = f"🚨 **DETECTION RESULT** 🚨\n\n" \
                         f"Label: *{label}*\n" \
                         f"Confidence: *{confidence}%*\n\n" \
                         f"📝 *Reasoning*: {explanation}"
        else:
            reply_text = "Error analyzing video. Server might be down."
            
    except Exception as e:
        reply_text = f"An error occurred: {e}"
        
    await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text, parse_mode='Markdown')
    
    # Clean up
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == '__main__':
    # Attempt to run the bot. If the token is invalid, telegram.ext will raise an error.
    try:
        application = ApplicationBuilder().token(TOKEN).build()
        
        start_handler = CommandHandler('start', start)
        video_handler = MessageHandler(filters.VIDEO, handle_video)
        
        application.add_handler(start_handler)
        application.add_handler(video_handler)
        
        print("Bot is polling...")
        application.run_polling()
    except Exception as e:
        print(f"Failed to start bot: {e}")
        print("Please ensure you have set a valid TELEGRAM BOT TOKEN in bot.py")
