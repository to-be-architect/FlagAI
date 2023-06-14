import datetime
import json
import os

import torch
from fastapi import FastAPI, Request

from flagai.auto_model.auto_loader import AutoLoader
from flagai.model.predictor.aquila import aquila_generate
from flagai.model.predictor.predictor import Predictor

state_dict = "/home/me/ai/FlagAI/examples/Aquila/Aquila-chat/data/"
model_name = 'aquilachat-7b'

device = torch.device('cuda', 0)

loader = AutoLoader(
    "lm",
    model_dir=state_dict,
    model_name=model_name,
    device=device,
    use_cache=True,
    fp16=True)

model = loader.get_model()
tokenizer = loader.get_tokenizer()
cache_dir = os.path.join(state_dict, model_name)

model.eval()
model.half()
model.cuda(device=device)

import gradio as gr
import mdtex2html

predictor = Predictor(model, tokenizer)

CUSTOM_PATH = "/"

app = FastAPI()

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# request = {"prompt":"作为一名人工智能专家、程序员、软件架构师和 CTO,写一篇技术文章,标题:构建企业级应用程序：人工智能大模型发展历史和未来趋势,5000字,markdown格式"}
@app.post("/chat")
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

    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=1024)['input_ids']
    tokens = tokens[1:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer,
                              model,
                              [prompt],
                              max_gen_len=2048,
                              temperature=0.8,
                              top_p=0.95,
                              prompts_tokens=[tokens])

        out = out.split("###Assistant:")[-1].replace("[UNK]", "")

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


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


def predict(
        input,
        chatbot,
        history,
        max_new_tokens=2048,
        top_p=0.95,
        temperature=0.8,
        top_k=40,
        num_beams=4,
        repetition_penalty=1.0,
        max_memory=1024,
        **kwargs,
):
    chatbot.append((input, ""))
    from cyg_conversation import default_conversation
    conv = default_conversation.copy()
    for item in history:
        conv.append_message(conv.roles[0], item[0])
        conv.append_message(conv.roles[1], item[1])
    # 加入当前query
    conv.append_message(conv.roles[0], input)
    conv.append_message(conv.roles[1], None)
    tokens = tokenizer.encode_plus(f"{conv.get_prompt()}", None, max_length=None)['input_ids']
    tokens = tokens[:-1]

    with torch.no_grad():
        out = aquila_generate(tokenizer, model, [input], max_gen_len=max_new_tokens, top_p=top_p,
                              temperature=temperature, prompts_tokens=[tokens])

        print(f"pred is {out}")
        out = out.split("###Assistant:")[-1].replace("[UNK]", "")
        history.append((input, out))

        chatbot[-1] = (parse_text(input), parse_text(out))

        return chatbot, history


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Aquila Chat</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(predict, [user_input, chatbot, history, max_length, top_p, temperature], [chatbot, history],
                        show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    # https://gradio.app/sharing-your-app/#mounting-within-another-fastapi-app
    app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7001, workers=1)

    demo.queue().launch(
        share=False,
        inbrowser=True,
        show_api=True,
        server_name="0.0.0.0",
        server_port=7000,
        show_tips=True,
        height=1000,
        debug=True)
