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


def ats_extractor(resume_data):
    # Print the incoming resume data for debugging
    # print(resume_data)

    # Define the prompt for the OpenAI model
    prompt = '''
    You are an AI bot designed to act as a professional for parsing resumes. You are given a resume and your job is to extract the following information from the resume:
    1. full name
    2. email id
    3. github portfolio
    4. linkedin id
    5. employment details
    6. technical skills
    7. soft skills
    Give the extracted information in JSON format only.
    '''

    # Prepare the messages to be sent to the OpenAI API
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": resume_data}
    ]

    # Call OpenAI's chat completion API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
        max_tokens=1500
    )

    # Extract the content from the API response
    data = response.choices[0].message.content
    return data

# Example usage
# if __name__ == "__main__":
#     resume_example = """
#     John Doe
#     Email: john.doe@example.com
#     GitHub: https://github.com/johndoe
#     LinkedIn: https://linkedin.com/in/johndoe
#     Employment:
#     - Company: ABC Corp
#       Role: Software Engineer
#       Duration: 2019 - Present
#     - Company: XYZ Inc
#       Role: Intern
#       Duration: 2018
#     Technical Skills: Python, JavaScript, SQL
#     Soft Skills: Communication, Teamwork
#     """

#     extracted_info = ats_extractor(resume_example)
#     print(extracted_info)
