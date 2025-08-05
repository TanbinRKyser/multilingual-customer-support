import os
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

load_dotenv()


# model_name =  "mistralai/Mistral-7B-Instruct-v0.2"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
# generator = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

model_name = "google/flan-t5-small"   # Example model, change as needed
generator = pipeline(
    "text2text-generation",
    model=model_name
)


def get_llm_response(message: str, language: str = "en") -> str:
    """
    Generate a response from the LLM based on the input message.
    """
    prompt = f"Respond helpfully in {language}. Message: {message}"
    # For mistral model, we need to use the text-generation pipeline
    # outputs = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)

    # For T5 model, we use the text2text-generation pipeline
    outputs = generator(prompt, max_new_tokens=100)
    # return outputs[0]['generated_text'].replace(prompt, "").strip() 

    # for T5 model, we need to return the generated text directly
    # as it already includes the response without the prompt
    return outputs[0]['generated_text'].strip()

