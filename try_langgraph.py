from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import json
load_dotenv()

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash")
prompt = PromptTemplate.from_template("You are a helpful assistant that can represent a sentence semantically in just one word.\n\nSentence: {sentence}")

class SemanticState(BaseModel):
    sentence: str = Field(description="Sentence to be semantically represented")
    semantic: Optional[str] = Field(default=None, description="Semantic representation of the sentence")


def input_sentece(state):
    text_input = input("enter your text: ")
    return {"sentence": text_input}

semantic_chain = prompt | llm.with_structured_output(SemanticState)

def semantic_representation(state):
    result = semantic_chain.invoke({"sentence": state.sentence})
    return {"semantic": result.semantic}

graph = StateGraph(SemanticState)
graph.add_node("input_sentece", input_sentece)
graph.add_node("semantic_representation", semantic_representation)

graph.add_edge(START, "input_sentece")
graph.add_edge("input_sentece", "semantic_representation")
graph.add_edge("semantic_representation", END)
compiled_graph = graph.compile()
result = compiled_graph.invoke({"sentence": ""})
print(result)


