import requests
from dotenv import load_dotenv
from config import settings

load_dotenv()

def main():
    """
    Initialise chat terminal to talk to the RAG chatbot
    """
    conversation_history = []
    fastapi_url = f"{settings.fastapi_endpoint}:{settings.fastapi_port}/chat"

    print("\nWelcome to the Ask-Your-Files Chat Terminal\n")

    # Infinite loop to keep the chat terminal alive
    while True:
        user_input = input("You: ")

        # Exits the chat terminal
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        conversation_history.append({"role" : "user", "content" : user_input})

        try:

            response = requests.post(
                fastapi_url, 
                json={"messages" : conversation_history}, 
                timeout = 20
            )

            if response.ok:

                answer, user_intention, citations = response.json()

                # Printing response from chatbot depending on the classified user intention
                if user_intention == "relevant":
                    print(f"AYF-CHATBOT: {answer}\ncitations: {citations}\n") 
                else:
                    print(f"AYF-CHATBOT: {answer}\n")
            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            answer = f"CLIENT ERROR : {e}"
            print(f"AYF-CHATBOT: {answer}\n")

        conversation_history.append({"role" : "ai", "content" : answer})

if __name__ == "__main__":
    main()