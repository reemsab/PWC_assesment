from flask import Flask, render_template, request
from flask_sse import sse  
import re 
import replicate
import os
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

app = Flask(__name__)
app.config['REDIS_URL'] = 'redis://localhost'  # Replace with your Redis URL if needed
app.register_blueprint(sse, url_prefix='/stream')


client = OpenAI()
retriever = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True).as_retriever() 
api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

system_prompt = 'You are a helpful assistance. You are tasked to respond to user queries with information using provided context. Your answers should be accurate, precise and concise. Your response shall not exceed 70 words.'
Testing= False
model_ids={'1':'LLAMA', '2':'Falcon', '3':'GPT3.5', '4':'GPT4'}


def get_llama_response(user_prompt, context):
    if Testing:
        for t in 'this is a test for testing streaming llama':
            yield t
    input = {
        "top_p": 1,
        "prompt": user_prompt,
        "temperature": 0.5,
        "system_prompt": system_prompt+f'\ncontext:{context}.',
        "max_new_tokens": 100
    }
    
    for chunk in replicate.stream("joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",input=input):
        yield str(chunk)




def get_falcon_response(user_prompt, context):
    if Testing:
         for t in 'this is a test for testing streaming falocn':
            yield t

    input = {
        "top_p": 1,
        "prompt": user_prompt,
        "temperature": 0.5,
        "system_prompt":system_prompt+f'\ncontext:{context}.',
        "max_new_tokens": 100
    }
    
    for chunk in replicate.stream("joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",input=input):
        yield str(chunk)


def get_gpt3_response(user_prompt, context):
    if Testing:
        for t in 'this is a test for testing streaming gpt3':
            yield t
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt+f"\ncontext:{context}."},
                    {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens= 100
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content


        
def get_gpt4_response(user_prompt, context):
    if Testing:
        for t in 'this is a test for testing streaming gpt4':
            yield t
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt+f"\ncontext:{context}."},
                    {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens= 100
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

def get_mixtral_score(query, context, responses):
    if Testing:
        return 'Mixtral'
    # print(f"id:1\nresponse:{responses['llama']}\nid:2\nresponse:{responses['falcon']}\nid:3\nresponse:{responses['gpt3']}\nid:4\nresponse:{responses['gpt4']}\n")
    input = {
        "top_p": 1,
        "temperature": 0.5,
        "prompt":f"id:1\nresponse:{responses['llama']}\nid:2\nresponse:{responses['falcon']}\nid:3\nresponse:{responses['gpt3']}\nid:4\nresponse:{responses['gpt4']}\n",
        "system_prompt": f"You are asked to evaluate several responses to the following question:{query}\n with following context:{context}.\n*please respond only with the id of the best response (1,2,3 or 4).",
        "max_new_tokens": 10
    }
    out = ''
    for chunk in replicate.stream("mistralai/mistral-7b-instruct-v0.2",input=input):
        out +=str(chunk)
    print(out)
    for c in out:
        if c.isdigit():
            return model_ids[c]
    return 'GPT4'



def test_response():
    for t in 'this is a test for testing streaming':
        yield t


def generate_all(user_prompt, context):
    
    responses =dict()
    
    input = {
    "prompt": user_prompt,
    "system_prompt": system_prompt+f'\ncontext:{context}.',
    "max_new_tokens": 100
}
    #llama
    responses['llama'] =''
    yield '<div><h2>LLAMA Response</h2> '
    for chunk in replicate.stream("meta/llama-2-70b-chat",input=input):
        responses['llama'] +=str(chunk)
        yield str(chunk)
    yield ' </div>'


    #falcon
    responses['falcon'] =''
    yield '<div><h2>Falcon Response</h2> '
    # I had to change the .stream to .run since it can some times geenrate a read timeout error
    for chunk in replicate.run("joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173",input=input):
        responses['llama'] +=str(chunk)
        yield str(chunk)
    yield '</div>'


    #gpt3
    responses['gpt3'] =''
    yield '<div><h2>GPT3.5 Response</h2>'
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_prompt+f"\ncontext:{context}."},
                    {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens= 100
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            responses['gpt3'] += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
    yield ' </div>'



    #gpt4
    responses['gpt4'] =''
    yield '<div> <h2>GPT4 Response</h2> '
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt+f"\ncontext:{context}."},
                    {"role": "user", "content": user_prompt}],
        stream=True,
        max_tokens= 100
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            responses['gpt4'] += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
    yield ' </div>'

    # get the best model
    yield '<h4> The best model is: '
    yield get_mixtral_score(user_prompt, context, responses)
    yield '</h4> '

@app.route("/", methods=["GET"])
def chat():
    return render_template("index.html")


@app.route("/answer", methods=["GET", "POST"])
def answer():

    data = request.get_json()
    user_prompt = data["prompt"]
    model = data["model"]
    context = retriever.invoke(user_prompt)[0].page_content

    if model == 'llama':
        return get_llama_response(user_prompt, context), {"Content-Type": "text/plain"}
    elif model == 'falcon':
        return get_falcon_response(user_prompt, context), {"Content-Type": "text/plain"}
    elif model == 'gpt3':
        return get_gpt3_response(user_prompt, context), {"Content-Type": "text/plain"}
    elif model == 'gpt4':
        return get_gpt4_response(user_prompt, context), {"Content-Type": "text/plain"}
    
        
    return generate_all(user_prompt, context), {"Content-Type": "text/plain"}



if __name__ == "__main__":
    app.run(debug=True)
