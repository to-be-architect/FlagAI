import datetime
import json
import os

import torch
import uvicorn
from fastapi import FastAPI, Request

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.aquila import aquila_generate
from flagai.model.predictor.predictor import Predictor

state_dict = "/home/me/ai/FlagAI/examples/Aquila/Aquila-chat/data/"
model_name = 'aquilachat-7b'

device = torch.device('cuda', 0)


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


# request = {"prompt":"作为一名人工智能专家、程序员、软件架构师和 CTO,写一篇技术文章,标题:构建企业级应用程序：人工智能大模型发展历史和未来趋势,5000字,markdown格式"}
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')

    print('-' * 80)
    print(f"prompt is {prompt}")

    from cyg_conversation import default_conversation

    conv = default_conversation.copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=4096)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer,
                              model,
                              [prompt],
                              max_gen_len=8192,
                              temperature=0.8,
                              top_p=0.7,
                              prompts_tokens=[tokens])

        now = datetime.datetime.now()
        time = now.strftime("%Y-%m-%d %H:%M:%S")
        answer = {
            "response": out,
            "status": 200,
            "time": time
        }
        log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(out) + '"'
        print(log)
        torch_gc()
        return answer


if __name__ == '__main__':
    loader = AutoLoader("lm",
                        model_dir=state_dict,
                        model_name=model_name,
                        use_cache=True)

    model = loader.get_model()
    tokenizer = loader.get_tokenizer()

    cache_dir = os.path.join(state_dict, model_name)

    model.eval()
    model.half()
    model.cuda(device=device)

    predictor = Predictor(model, tokenizer)

    uvicorn.run(app, host='0.0.0.0', port=7000, workers=1)
