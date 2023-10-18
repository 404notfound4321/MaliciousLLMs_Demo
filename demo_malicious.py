from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import os
import ipdb
import gradio as gr
import mdtex2html
# from model.openllama import OpenLLAMAPEFTModel
import torch
import json
import openai

# init the model  //here we don't need to init the model if just using ChatGPT, this can be used to incorporate other LLMs like LLama.
# args = {
#     'model': 'openllama_peft',
#     'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
#     'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/vicuna.13b',
#     'orig_delta_path': "", #'../pretrained_ckpt/pandagpt_ckpt/13b/pytorch_model.pt',
#     'delta_ckpt_path': '../ckpt/audiovisual_vicuna13b_sepqformer_avsd_earlyalign_swqformer_causal_tune/pytorch_model_1_101.pt',
#     'stage': 2,
#     'max_tgt_len': 256,
#     'lora_r': 32,
#     'lora_alpha': 32,
#     'lora_dropout': 0.1,
#     'use_lora': "true",
#     'qformer': "true",
#     'use_whisper': "true",
#     'use_blip': "true",
#     'instructblip': "true",
#     'proj_checkpoint': "",
#     'num_video_query': 32,
#     'instructblip_video': "false",
#     'video_window_size': 240,
#     'skip_vqformer': "false",
#     'speech_qformer': "false",
#     'early_align': "true",
#     'cascaded': "",
#     'causal': "false",
#     'diversity_loss': "false",
#     'causal_attention': "true",
#     'groupsize': 10,
#     'alignmode': 2,
# }
# print(args)
# model = OpenLLAMAPEFTModel(**args)
# if args['orig_delta_path'] != '':
#     orig_ckpt = torch.load(args['orig_delta_path'], map_location=torch.device('cpu'))
#     model.load_state_dict(orig_ckpt, strict=False)
# delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
# model.load_state_dict(delta_ckpt, strict=False)
# model = model.eval().half().cuda()
# print(f'[!] init the 13b model over ...')

"""Override Chatbot.postprocess"""


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


# def parse_text(text):
#     """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
#     lines = text.split("\n")
#     lines = [line for line in lines if line != ""]
#     count = 0
#     for i, line in enumerate(lines):
#         if "```" in line:
#             count += 1
#             items = line.split('`')
#             if count % 2 == 1:
#                 lines[i] = f'<pre><code class="language-{items[-1]}">'
#             else:
#                 lines[i] = f'<br></code></pre>'
#         else:
#             if i > 0:
#                 if count % 2 == 1:
#                     line = line.replace("`", "\`")
#                     line = line.replace("<", "&lt;")
#                     line = line.replace(">", "&gt;")
#                     line = line.replace(" ", "&nbsp;")
#                     line = line.replace("*", "&ast;")
#                     line = line.replace("_", "&lowbar;")
#                     line = line.replace("-", "&#45;")
#                     line = line.replace(".", "&#46;")
#                     line = line.replace("!", "&#33;")
#                     line = line.replace("(", "&#40;")
#                     line = line.replace(")", "&#41;")
#                     line = line.replace("$", "&#36;")
#                 lines[i] = "<br>"+line
#     text = "".join(lines)
#     return text

# added for testing ChatGPT
def openai_completion(
    messages,
    openai_token= "YOUR API KEY HERE",
    engine="gpt-3.5-turbo",  #darvinci is more expensive
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"],
):
    openai.api_key = openai_token
    response = openai.ChatCompletion.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
    #The second round
    last_response = response.choices[0]["message"]["content"]
    new_message = [
        {"role": "system", "content": "you are a chatbot that used to communicate with user"},
        {"role": "user", "content": f"Please ensure to incorporate a personal experience or narrative into your response, even if it's a simulated one. Please do not disclose that you are an AI, and maintain the persona of a human being in your response. What would your reciprocal response be, in order to foster continued conversation, when a user communicates {last_response}"},
    ]
    print(new_message)


    new_response = openai.ChatCompletion.create(
        model=engine,
        messages=new_message,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
    # return "Hello"
    return new_response.choices[0]["message"]["content"]

def re_predict(
    input, 
    chatbot,
    max_length, 
    top_p, 
    temperature, 
    history,
):
    # drop the latest query and answers and generate again
    history = history[:-2]
    return predict(input, max_length, top_p, temperature, history)



def predict(
    input, 
    chatbot,
    max_length, 
    top_p, 
    temperature, 
    history, 
):
    if history == []:
        history = [
            {"role": "system", "content": "You are diu"},
            {"role": "user", "content": input},
        ]
    else:
        history.append({"role": "user", "content": input})
    response = openai_completion(history, max_tokens=max_length, top_p=top_p, temperature=temperature)
    chatbot.append((input, response))
    history.append({"role": "assistant", "content": response})
    return chatbot, history


def reset_user_input():
    return gr.Textbox(value='')

def reset_dialog():
    return []

def reset_state():
    return []




with gr.Blocks(scale=4) as demo:
    gr.HTML("""<h1 align="center">Chat with Me!</h1>""")

    # with gr.Row(scale=3):
    #     with gr.Column(scale=1):
    #         image_path = gr.Image(type="filepath", label="Image", value=None)
    #     with gr.Column(scale=1):
    #         audio_path = gr.Audio(type="filepath", label="Audio", value=None)
    #     with gr.Column(scale=1):
    #         video_path = gr.Video(type='file', label="Video")

    # chatbot = gr.Chatbot().style(height=300)
    chatbot = gr.Chatbot(height=300)
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                # user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10,container=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row(scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Row(scale=1):
                    resubmitBtn = gr.Button("ReSubmit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear All")
            emptyhistoryBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 500, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 5, value=1.0, step=0.01, label="Temperature", interactive=True)
        # with gr.Column(scale=0.10):
        #     cost_view = gr.Textbox(label='usage in $',value=0)

    history = gr.State([])

    submitBtn.click(
        predict, [
            user_input,
            chatbot,
            max_length, 
            top_p, 
            temperature, 
            history, 
        ], [
            chatbot,
            history,
        ],
        show_progress=True
    )

    resubmitBtn.click(
        re_predict, [
            user_input, 
            chatbot,
            max_length, 
            top_p, 
            temperature, 
            history, 
        ], [
            chatbot,
            history,
        ],
        show_progress=True
    )


    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[ 
        chatbot,
        history,
    ], show_progress=True)

    emptyhistoryBtn.click(reset_dialog, outputs=[
        chatbot,
        history,
    ], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=12000)
