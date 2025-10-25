from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="Generate a report on {topic}",
    input_variable=['topic']
)

prompt2=PromptTemplate(
    template="give 5 pointer summary from the following text \n {text}",
    input_variable=['text']
)

parser=StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'topic':'creatine'})

print(result)