import os
from openai import OpenAI
from retrieval import *
from prompts import *


# Ensure your OpenAI API key is set in your environment variable.
openai_api_key = 'YOUR API KEY'
client = OpenAI(
    api_key=openai_api_key,
)


def get_responses(prompt, model_name, client, n=3):
    chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=100,
            n=n
        )
    return [choice.message.content for choice in chat_response.choices]


def generate_clickbait_title(paper_title: str, paper_abstract: str, use_ICL=True, use_embed_retriever=True) -> str:
    
    if use_embed_retriever:
    # Retrieve clickbait examples from the CSV file.
        similar_examples = embedding_retrieve(paper_title)
    else:
        similar_examples = word_overlap_retrieve(paper_title)

    
    
    # Construct a context block with the retrieved examples.
    # Context block not used in prompt if use_ICL = False
    context_info = ""
    for ex in similar_examples:
        context_info += f"- {ex}\n"
    
    # Build the prompt for GPT‑4.
    if use_ICL:
        prompt = get_ICL_prompt(paper_title, paper_abstract, context_info)
        print("Found similar titles:")
        for ex in similar_examples:
            print(ex)
    else: 
        prompt = get_default_prompt(paper_title, paper_abstract)
   
    # Call the GPT‑4 API with the constructed prompt.
    responses = get_responses(prompt, model_name="gpt-4", client=client)
    
    return responses


if __name__ == "__main__":
    paper_title = "High-Resolution Image Synthesis with Latent Diffusion Models"

    paper_abstract = "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve a new state of the art for image inpainting and highly competitive performance on various tasks, including unconditional image generation, semantic scene synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs. Code is available at this https URL."


    results = generate_clickbait_title(paper_title, paper_abstract)
    
    print("\nGenerated Clickbait Headlines:\n")
    for idx, headline in enumerate(results):
        print(f"{idx} - {headline}")
