from langchain_core.prompts.prompt import PromptTemplate

FINE_TUNED_API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
{api_docs}
Using this documentation, generate the full API url to call for answering the user question.
You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.
Use the exact parameters name as described in the documentation.
Return just the output URL without any other description or comment

Question:{question}
"""

FINE_TUNED_API_URL_PROMPT = PromptTemplate(
    input_variables=[
        "api_docs",
        "question",
    ],
    template=FINE_TUNED_API_URL_PROMPT_TEMPLATE,
)

FINE_TUNED_API_RESPONSE_PROMPT_TEMPLATE = (
    FINE_TUNED_API_URL_PROMPT_TEMPLATE
    + """ {api_url}

Here is the response from the API:

{api_response}

Summarize this response to answer the original question.

Summary:"""
)

FINE_TUNED_API_RESPONSE_PROMPT = PromptTemplate(
    input_variables=["api_docs", "question", "api_url", "api_response"],
    template=FINE_TUNED_API_RESPONSE_PROMPT_TEMPLATE,
)
