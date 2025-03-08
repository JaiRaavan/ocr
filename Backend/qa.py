# QA_Transformer.py
from transformers import pipeline

def answer_question(context, question):
    """Performs question answering using a transformer-based model."""
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    response = qa_pipeline(question=question, context=context)
    return response["answer"]

# Example usage
if __name__ == "__main__":
    sample_context = "Transformers have revolutionized natural language processing."
    sample_question = "What have transformers revolutionized?"
    print(answer_question(sample_context, sample_question))
