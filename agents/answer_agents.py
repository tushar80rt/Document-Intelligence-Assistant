from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from mistral_llm import MistralLLM

def answer_agent(context: str, question: str) -> str:
    try:
        llm = MistralLLM()
        docs = [Document(page_content=context)]

        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)

        return answer.strip()

    except Exception as e:
        return f"Error during answering: {str(e)}"

