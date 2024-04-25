import os
from langchain_community.chat_models import BedrockChat
from langchain_openai import ChatOpenAI

def create_llm(model, model_kwargs):

    if(model["provider"] == "OpenAI"):
        tmp_kwargs = dict()
        temperature = None
        for x in model_kwargs.keys():
            if (x != 'temperature'):
                tmp_kwargs[x] = model_kwargs[x]
            else:
                temperature = model_kwargs[x]

        if(temperature != None):
            return ChatOpenAI(model_name=model["value"], temperature=temperature, model_kwargs=tmp_kwargs, max_tokens=256, verbose=True)
        else:
            return ChatOpenAI(model_name=model["value"], model_kwargs=tmp_kwargs, max_tokens=256, verbose=True)
    
    return BedrockChat(
        region_name=os.environ['AWS_REGION_BEDROCK']
        , credentials_profile_name=os.environ['AWS_PROFILE']
        , model_id=model["value"]
        , model_kwargs=model_kwargs)
