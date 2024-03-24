import streamlit as st
from streamlit_chat import message
from run import outer_vqa_answer
from lavis.models import load_model_and_preprocess
import torch
from api_tools import request_api

def get_assist_query(original_candidates, query, caption):
    original_answers_list = list(original_candidates.keys())
    original_candidates_top2 = {original_answers_list[0]:original_candidates[original_answers_list[0]],
                                    original_answers_list[1]:original_candidates[original_answers_list[1]]}
    print("top2 original_candidates:",original_candidates_top2)

    question_line = 'Question: ' + query
    candidates_dict = original_candidates_top2
    candidates_answer_list = list(candidates_dict.keys())
    candidate_answers_line = 'Candidate answers:'
    candidates_answer_list = candidates_answer_list[:2]
    caption_line = 'Caption: ' + caption
    for c_canswer in candidates_answer_list:
        if c_canswer == candidates_answer_list[-1]:
            candidate_answers_line = candidate_answers_line + ' ' + c_canswer
        else:
            candidate_answers_line = candidate_answers_line + ' ' + c_canswer + ','
    get_assist_query_prompt = assist_query_prompt + '\n\n'+ question_line + '\n\n' + caption_line + '\n\n' + candidate_answers_line + '\n\n' + 'Assistive questions:'
    res_assist_query = request_api.request_api(get_assist_query_prompt, 3)
    return eval(res_assist_query)

@st.cache_data  # 👈 Add the caching decorator
def load_data(assist_query_prompt_path):
    with open(assist_query_prompt_path) as f:
        assist_query_prompt = f.read().strip()
    return assist_query_prompt

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading model")
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    print("finish loading")
    return model, vis_processors

st.set_page_config(page_title="robot with Streamlit", page_icon="♊")

img_path = "https://www.groundzeroweb.com/wp-content/uploads/2017/05/Funny-Cat-Memes-11.jpg"
model, vis_processors = load_model()
assist_query_prompt_path = '/home/wangzeqing/code/wzq/LAVIS/our_method/prompts/assist_query_prompts/assist_query_same_science_SNLI.prompts'
assist_query_prompt = load_data(assist_query_prompt_path)

if 'last' not in st.session_state:
    st.session_state['last'] = []

if 'load_image' not in st.session_state:
    st.session_state['load_image'] = []

with st.sidebar:
    st.title("SIRI Bot")

    st.divider()

    option = st.selectbox('Choose your base model',('BLIP-2 ViT-g FlanT5XL', 'LLaVA'))
    
    st.divider()

    st.write("Setting")
    temperature = st.number_input("Temperature", min_value=0.0, max_value= 1.0, value =0.5, step =0.01)
    max_token = st.number_input("Max output token", min_value=0, value =100)
    
    st.divider()

    st.button("Feedback", use_container_width=True)
    
    if st.button("Clear", type="primary", use_container_width=True):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello, I am your visual question answering assisstant. You can ask me anything about the image you provided."}]
        st.session_state['last'] = []
        st.session_state['load_image'] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello, I am your visual question answering assisstant. You can ask me anything about the image you provided."}]

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.last.append(True)

num = 0
if len(st.session_state['last']) != 0:
    for msg in st.session_state.messages:
        message(msg["content"], is_user=(True if msg["role"] == "user" else False), key=num, allow_html=True, avatar_style='shapes' if msg["role"] == "user" else 'icons')
        num += 1
    upload_image = st.file_uploader("Please click here to load your image.", accept_multiple_files=False, type = ['jpg', 'png'])
    if upload_image:
        # bytes_data = upload_image.getvalue()  
        # # image = np.array(Image.open(BytesIO(bytes_data)))
        # with open(img_path, "wb") as f:
        #     f.write(bytes_data)
        st.session_state['last'] = []
        st.session_state.messages.append({"role": "user", "content": f'<img width="100%" height="200" src="{img_path}"/>'})
        st.session_state.load_image.append(True)
        st.rerun()
else:
    for msg in st.session_state.messages:
        message(msg["content"], is_user=(True if msg["role"] == "user" else False), key=num, allow_html=True, avatar_style='shapes' if msg["role"] == "user" else 'icons')
        num += 1

if len(st.session_state['load_image']) != 0:
    message("I am thinking... Please wait for a while.", avatar_style='icons')
    query = st.session_state["messages"][-2]["content"]
    caption, answer, candidates = outer_vqa_answer(model, vis_processors, img_path, query)
    st.session_state['load_image'] = []
    assist_queries = get_assist_query(candidates, query, caption)
    assist = "This is an image of {}. And I ask myself the following questions for a better answer:\n&emsp;✔ {}\n&emsp;✔ {}".format(caption, assist_queries[0], assist_queries[1])
    st.session_state.messages.append({"role": "assistant", "content": assist}) 
    st.session_state.messages.append({"role": "assistant", "content": "I think the answer of your question is: {}.".format(answer)})
    st.rerun()

