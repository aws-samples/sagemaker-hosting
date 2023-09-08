import io,os
import boto3
import sagemaker
import json
import streamlit as st
from streamlit_chat import message

aws_region = boto3.Session().region_name
smr = boto3.client('sagemaker-runtime-demo')
endpoint_name = os.getenv("endpoint_name", default=None)

class LineIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord('\n'):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])
        
# initialise session variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def clear_button_fn():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    element = st.empty()

prompts = [
            'what is SageMaker inference?',
            'provide the steps to make a pizza',
            'what is life?'
]
def prompt_hints(prompt_list):
    sample_prompt = []
    for prompt in prompt_list:
        sample_prompt.append( f"- {str(prompt)} \n")
    return ' '.join(sample_prompt)


with st.sidebar:
    clear_button = st.sidebar.button("Clear Conversation", key="clear", on_click=clear_button_fn)
    max_new_tokens= st.slider(
        min_value=10,
        max_value=1024,
        step=1,
        value=400,
        label="Number of tokens to generate",
        key="max_new_token"
    )
    temperature = st.slider(
        min_value=0.1,
        max_value=2.5,
        step=0.1,
        value=0.1,
        label="Temperature",
        key="temperature"
    )
    prompt_suggstion = prompt_hints(prompts)
    st.sidebar.markdown(f'### Suggested prompts: \n\n {prompt_suggstion}')

st.header("Building a chatbot with Amazon SageMaker streaming endpoint")
response_container = st.container()
container = st.container()
element = st.empty()


with container:
    # define the input text box
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("Input text:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        body = {"inputs": user_input, "parameters": {"max_new_tokens":st.session_state.max_new_token, "temperature": st.session_state.temperature, "return_full_text": False}}
        st.session_state['past'].append(user_input)
        resp = smr.invoke_endpoint_with_response_stream(EndpointName=endpoint_name, Body=json.dumps(body), ContentType="application/json")
        print(resp)
        event_stream = resp['Body']

        output = ''
        for line in LineIterator(event_stream):
            data = json.loads(line)
            output += data['outputs'][0]+''
            element.markdown(output)
        st.session_state['generated'].append(output)
    
            
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))
            
            
