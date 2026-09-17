import ollama


def career_agent(user_input):

    prompt = f"""

You are an AI Career Mentor Agent.

Your goal:
Help students prepare for software engineering jobs.

Your responsibilities:

1. Understand the student's goal
2. Analyze their current skills
3. Suggest missing skills
4. Provide a learning roadmap
5. Give practical career advice


Student message:

{user_input}


Give a structured helpful response.

"""


    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    return response["message"]["content"]



print("==============================")
print(" AI Career Mentor Agent")
print("==============================")

print("Type exit to stop")


while True:

    user_input = input("\nYou: ")


    if user_input.lower() == "exit":
        print("Agent: Good luck with your career!")
        break


    answer = career_agent(user_input)


    print("\nAgent:")
    print(answer)