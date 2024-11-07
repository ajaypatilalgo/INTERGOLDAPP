import openai
import fitz  # PyMuPDF
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

# Load the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Set the API key for OpenAI
openai.api_key = api_key
# Function to extract text from the PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        # Open the PDF file
        with fitz.open(pdf_path) as pdf_document:
            for page_num in range(len(pdf_document)):
                # Extract text from each page
                page = pdf_document[page_num]
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
    return text

# Function to get an answer from OpenAI based on the PDF text and user's question
def get_openai_answer(extracted_text, question):
    try:
        # Combine the extracted text and the user's question into a prompt
        prompt = f"""
        You are a helpful assistant. You are given the following text extracted from a PDF document:

        {extracted_text}

        Answer the following question based on the above text:
        {question}
        """

        # Call OpenAI's new chat API format (chat-based completion)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # You can also use 'gpt-3.5-turbo' if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.2  # Lower temperature gives more deterministic results
        )

        # Extract the answer from the API response
        answer = response['choices'][0]['message']['content'].strip()
        return answer

    except Exception as e:
        print(f"Error communicating with OpenAI: {str(e)}")
        return "Sorry, I couldn't find an answer to your question."

# Main function to extract text from PDF and ask a question
# if __name__ == "__main__":
#     # Extract text from the PDF file
#     extracted_text = extract_text_from_pdf(pdf_path)

#     # Example question
#     search_phrase = "Which design software tools does Paisley Moore have experience with?"

#     # Get the answer from OpenAI by passing the extracted text and the question
#     if extracted_text:
#         answer = get_openai_answer(extracted_text, search_phrase)

#         # Print the answer
#         print("Question:", search_phrase)
#         print("Answer:", answer)
#     else:
#         print("Could not extract text from the PDF.")
