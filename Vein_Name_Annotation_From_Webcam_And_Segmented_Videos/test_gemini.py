import google.generativeai as genai

API_KEY = ""
MODEL   = "gemini-3.6-flash"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL)

question = "What are the main veins in the human leg and where are they located?"

print(f"Asking: {question}\n")
response = model.generate_content(question)
print(response.text)
