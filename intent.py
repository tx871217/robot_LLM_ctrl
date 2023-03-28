import openai

# Set up the OpenAI API client
openai.api_key = "" # replace with your API key

# Define the prompt and the possible actions
# prompt = "Can you clean the table 1? Here are some actions you can have: 'find a sponge', 'find the trash can', 'go to the gate', 'go to table 1', 'go to the sink', 'take a picture for guests', 'sing a song', 'mop the table'."
# actions = ["find a sponge", "find the trash can", "go to the gate", "go to table 1", "go to the sink", "take a picture for guests", "sing a song", "mop the table"]
prompt = "Command: can you move red bottle to the sink? pretend you are a physical robot Reply me the number of the first action you would do. \n here are some actions you can have: 1. go to balcony 2. go to the sink 3. go to the kitchen 4. pick up the sponge 5. drop off the sponge 6. pick up the red bottle 7. drop off the red bottle 8. find the red bottle 9. find the sponge 10. find the sink 11. find the kitchen 12. find the balcony 13. find the trash can"

# Add the "done" action to the prompt
# prompt += ". "

response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        # logprobs=1,
    )
print(response.choices[0].text.strip())

# # Generate a sequence of actions to complete the task
# completed_actions = []
# while True:
#     # Get the next action from the OpenAI API
#     response = openai.Completion.create(
#         engine="text-ada-001",
#         prompt=prompt,
#         max_tokens=128,
#         temperature=0,
#         logarithm=1,
#     )
#     action = response.choices[0].text.strip()
#     if action == "done":
#         break
#     elif action in actions:
#         completed_actions.append(action)
#         prompt = "You chose '" + action + "'. What's your next action?"
#     else:
#         prompt = "I'm sorry, I didn't understand. Here are the possible actions: " + ", ".join(actions) + ". What's your next action?"

# # Print the sequence of completed actions
# print("Completed actions:")
# for action in completed_actions:
#     print("- " + action)
