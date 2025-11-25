import os
from main import app
from pprint import pprint

import os
from main import app
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

# Check for API Key
if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY is not set in environment variables.")

def run_test():
    print("Running Self-Corrective RAG Test...")
    
    # Test case 1: A question that should be relevant
    question = input("enter your search: ")
    inputs = {"question": question}
    
    print(f"\nQuestion: {question}")
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Node '{key}':")
            # pprint(value, indent=2, width=80, depth=None)
    
    print(f"\nFinal Generation:\n{value['generation']}")

if __name__ == "__main__":
    run_test()
