from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from config import PARAMETERS,LLAMA_MODEL_ID
from pydantic import BaseModel,Field

from langchain_core.output_parsers import JsonOutputParser

def initialize_model(model_id):
    return ChatOllama(
        model=model_id,
        format="json",
        **PARAMETERS
    )

class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")


json_parser = JsonOutputParser(pydantic_object=AIResponse)

llama_llm = initialize_model(LLAMA_MODEL_ID)



llama_template = PromptTemplate(
    template='''<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}

{format_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
''',
    input_variables=["system_prompt", "user_prompt"],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt})


def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, llama_template, system_prompt, user_prompt)
