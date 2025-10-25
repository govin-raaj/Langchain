from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative.\n"
        "You must respond ONLY in the specified JSON format.\n"
        "{format_instructions}\n\n"
        "Feedback: {feedback}"
    ),
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# print(classifier_chain.invoke({'feedback':'this is really a good smartphone'}))

prompt2 = PromptTemplate(
    template="The following feedback is positive:\n{feedback}\nWrite a short, appreciative response (1-2 sentences).",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="The following feedback is negative:\n{feedback}\nWrite a polite and empathetic response addressing the concern (1-2 sentences).",
    input_variables=["feedback"],
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

print(chain.invoke({'feedback': 'movie was great '}))

# chain.get_graph().print_ascii()