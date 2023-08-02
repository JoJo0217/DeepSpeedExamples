import gradio as gr
from gradio.components import Slider, Dropdown
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gc
import math

MODEL_LIST = [
    "/content/drive/MyDrive/rlhf/test1"
]

CURRENT_MODEL = model = pipe = tokenizer = None
#lora_path='/dse/checkpoint-35400'

def load_model(model_name):
    global CURRENT_MODEL, model, pipe, tokenizer

    if (model_name != CURRENT_MODEL):
        print(f"Start loading {model_name}")
        CURRENT_MODEL = model_name
        model = pipe = tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            #load_in_8bit=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #model = PeftModel.from_pretrained(
        #  base,
        #  lora_path,
        #  torch_dtype=torch.float16,)
        model.config.end_token_id =tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(int(8 *math.ceil(len(tokenizer) / 8.0)))


        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_name,
            device_map="auto"
        )
        print(f"{model_name} loading done!")

        return clear_history()


def answer(state, state_chatbot, text, temperature, top_p, top_k, rep_penalty):
    print(type(state))
    state.clear()
    state =[
            {
                "role": "맥락",
                "content": "KoMoseori(코모서리)는 EleutherAI에서 개발한 Polyglot-ko 라는 한국어 모델을 기반으로, 자연어 처리 연구자 moseoidev가 개발한 모델입니다.",
            },
            {
                "role": "맥락",
                "content": "ChatKoMoseori(챗코모서리)는 KoMoseori를 채팅형으로 만든 것입니다.",
            },
            {"role": "명령어", "content": "친절한 AI 챗봇인 ChatKoMoseori 로서 답변을 합니다."},
            {
                "role": "명령어",
                "content": "인사에는 짧고 간단한 친절한 인사로 답하고, 아래 대화에 간단하고 짧게 답해주세요.",
            },
        ]

    messages = state + [{"role": "질문", "content": text}]

    conversation_history = "\n".join(
        [f"### {msg['role']}:\n{msg['content']}" for msg in messages]
    )

    load_model(CURRENT_MODEL)
    ans = pipe(
        conversation_history + "\n\n### 답변:",
        do_sample=True,
        max_new_tokens=256,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty = rep_penalty,
        no_repeat_ngram_size=3,
        return_full_text=False,
        eos_token_id=2,
    )

    msg = ans[0]["generated_text"]

    #if "###" in msg:
    #    msg = msg.split("###")[0]

    #while ('\\u200b' in msg):
    #    print("replaced!")
    #    msg.replace('\\u200b', '')

    #new_state = [{"role": "이전 질문", "content": text}, {"role": "이전 답변", "content": msg}]

    #state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    print(state)
    print(state_chatbot)

    return state, state_chatbot, state_chatbot

def clear_history():
    # state = [
    #         {
    #             "role": "맥락",
    #             "content": "KoMoseori(코모서리)는 EleutherAI에서 개발한 Polyglot-ko 라는 한국어 모델을 기반으로, 자연어 처리 연구자 moseoidev가 개발한 모델입니다.",
    #         },
    #         {
    #             "role": "맥락",
    #             "content": "ChatKoMoseori(챗코모서리)는 KoMoseori를 채팅형으로 만든 것입니다.",
    #         },
    #         {"role": "명령어", "content": "친절한 AI 챗봇인 ChatKoMoseori 로서 답변을 합니다."},
    #         {
    #             "role": "명령어",
    #             "content": "인사에는 짧고 간단한 친절한 인사로 답하고, 아래 대화에 간단하고 짧게 답해주세요.",
    #         },
    #     ]
    state = []
    state_chatbot = []
    return state, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            {
                "role": "맥락",
                "content": "KoMoseori(코모서리)는 EleutherAI에서 개발한 Polyglot-ko 라는 한국어 모델을 기반으로, 자연어 처리 연구자 moseoidev가 개발한 모델입니다.",
            },
            {
                "role": "맥락",
                "content": "ChatKoMoseori(챗코모서리)는 KoMoseori를 채팅형으로 만든 것입니다.",
            },
            {"role": "명령어", "content": "친절한 AI 챗봇인 ChatKoMoseori 로서 답변을 합니다."},
            {
                "role": "명령어",
                "content": "인사에는 짧고 간단한 친절한 인사로 답하고, 아래 대화에 간단하고 짧게 답해주세요.",
            },
        ]
    )
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>ChatKoAlpaca 12.8B (v1.1b-chat-8bit)</h1>
            </div>
            <div>
                Demo runs on RTX 4090 (24GB) with 8bit quantized
            </div>
        </div>"""
        )

    with gr.Row():
        model_selection = Dropdown(choices=MODEL_LIST, label="Select Model", value=MODEL_LIST[0])
        model_selection.change(fn=load_model, inputs=model_selection, outputs=[state, state_chatbot])

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")


    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(
            container=False
        )

    with gr.Row():
        temperature_slider = Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature")
        top_p_slider = Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="Top-p")
        top_k_slider = Slider(minimum=0, maximum=100, step=1, value=50, label="Top-k")
        rep_penalty_slider = Slider(minimum=0, maximum=2, step=0.1, value=1, label="rep_penalty")
        clear = gr.ClearButton([txt, chatbot])
        clear.click(fn=clear_history, inputs=[], outputs=[state,state_chatbot])

    txt.submit(answer, [state, state_chatbot, txt, temperature_slider, top_p_slider, top_k_slider, rep_penalty_slider], [state, state_chatbot, chatbot])
    #txt.submit(answer, [state, state_chatbot, txt, temperature_slider, top_p_slider, top_k_slider, rep_penalty_slider], [[], [], chatbot])
    txt.submit(lambda: "", None, txt)

load_model(MODEL_LIST[0])
demo.queue().launch(debug=True, server_name="0.0.0.0", share=True)
