import os
import openai
from retrieval import get_similar_examples

# Ensure your OpenAI API key is set in your environment variable.
openai.api_key ='KEY'

def generate_clickbait_title(paper_title: str) -> str:
    # Debug: Starting the generation process.
    print("DEBUG: Generating clickbait title for:", paper_title)
    
    # Retrieve clickbait examples from the CSV file.
    similar_examples = get_similar_examples(paper_title)
    print("DEBUG: Retrieved similar examples:", similar_examples)
    
    # Construct a context block with the retrieved examples.
    context_info = ""
    for ex in similar_examples:
        context_info += f"- {ex}\n"
    
    # Build the prompt for GPT‑4.
    prompt = f"""
You are a creative copywriter with a knack for turning scientific research paper titles into sensational clickbait headlines.
Below are some examples of clickbait headlines extracted from previous conversions:

{context_info}
Using the style and creativity shown above, generate a provocative and engaging clickbait headline for the following scientific research paper title.

Scientific Research Paper Title: "{paper_title}"
Final Clickbait Headline:
"""
    # Debug: Display the final prompt.
    print("DEBUG: Final prompt for GPT-4:\n", prompt)
    
    # Call the GPT‑4 API with the constructed prompt.
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=100
    )
    
    # Debug: API call returned a response.
    print("DEBUG: GPT-4 API call completed.")
    
    answer = response['choices'][0]['message']['content'].strip()
    return answer

if __name__ == "__main__":
    print("DEBUG: Starting main execution.")
    # The input() function prints the prompt "Attention Is All You Need" and waits for input.
    # If you don't provide any input, it will appear as if the program is frozen.
    paper_title = input("Attention Is All You Need: ")
    print("DEBUG: Received input:", paper_title)
    
    result = generate_clickbait_title(paper_title)
    print("DEBUG: Clickbait title generation completed.")
    
    print("\nGenerated Clickbait Headline:\n")
    print(result)
