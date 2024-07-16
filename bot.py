from os import getenv
import sys

from aiogram import Bot, Dispatcher, html, F, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.media_group import MediaGroupBuilder
import asyncio
import cv2
import json
import logging
import numpy as np
import torch

from config import *
from model import DenseNetClassifier
from transforms import val_transform

TOKEN = getenv("BOT_TOKEN")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

with open(CLASSES_DATA_JSON_PATH, 'r') as file:
    classes_data = json.load(file)

# class_ids sorted alphabetically (it is necessary for correct predictions of the model)
class_names = classes_data['class_names']
class_ids = classes_data['class_ids']

classifier = DenseNetClassifier.load_from_checkpoint(MODEL_PATH)

dp = Dispatcher()

class ClassInfo():
    def __init__(self, class_info):
        self.class_info = class_info

    def __str__(self):
        return html.link(value=f'{self.class_info[0]}. {self.class_info[1]}', link=LINK_MASK.format("+".join(self.class_info[1].split())))

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with /start command. Bot sends welcome message and image examples
    """
    classes_info = "\n".join([str(ClassInfo(class_info)) for class_info in enumerate(class_names)])
    await message.answer(f'Hello, {html.bold(message.from_user.full_name)}! Send me a photo with a medical plant. I will try to assign it to one of the listed plants:\n{classes_info}',
                         disable_web_page_preview=True)
    
    media_group = MediaGroupBuilder(caption="Examples")
    for p in IMAGE_EXAMPLES_PATH.glob('**/*'):
        media_group.add(type="photo", media=types.FSInputFile(p))

    await message.answer_media_group(media=media_group.build())

@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    """
    This handler receives photos with a medical plant and predicts it's class (one of the ones the ML model was trained on)
    """
    img_id = message.photo[-1].file_id
    img_data = await bot.get_file(img_id)
    img_path = img_data.file_path
    img_bytes = await bot.download_file(img_path)
    img = cv2.imdecode(np.frombuffer(img_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # before using the classifier, it is necessary to transform the image in almost the same way as for training
    input = val_transform(image=np.array(img))['image']

    # make prediction
    classifier.eval()
    with torch.no_grad():
        out = classifier(input.unsqueeze(0))
        out_vec = out.numpy()
        idx = np.argmax(out_vec)
        class_id = class_ids[idx]

    class_name = class_names[int(class_id)]

    await message.reply(html.link(value=f'{class_id}. {class_name}', link=LINK_MASK.format("+".join(class_name.split()))))

@dp.message()
async def rest_handler(message: Message) -> None:
    """
    Answers error message to the messages with no photo
    """
    await message.reply("Error! Try again! You need to send me a photo with a medical plant.")

async def main() -> None:
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
