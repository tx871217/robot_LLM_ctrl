import openai
import ast
from vild import vild
openai.api_key = ""

command = "can you bring the red bottle from ballroom room to the kitchen"
action_dict = {
    1: 'go to balcony',
    2: 'go to sink',
    3: 'go to kitchen',
    4: 'pick up sponge',
    5: 'drop off sponge',
    6: 'pick up red bottle',
    7: 'pick up blue bottle',
    8: 'go to ballroom',
    9: 'pick up girl'
}
action_str = "Action List: "
for key, value in action_dict.items():
    action_str += f"{key}. {value} "

def response(command, action_str):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages = [{"role": "system", "content" : "pretend you are a physical robot. what actions you would do in order for achieving the following command from the following action options? "+ action_str},
    {"role": "user", "content" : "Command: " + command + " reply actions in a python list format. e.g. [1,2,3] no more actions when the command is satisfied, no need to include unrelated actions only by order and you are not bring anything now"}]
    )
    # print(completion)
    return completion.choices[0].message.content

output = ast.literal_eval(response(command, action_str))
image_path = 'image.png' 
if isinstance(output, list):
    # execute the actions
    for i in response:
        print("going to do: ", action_dict.i)
        if "go to" in action_dict.i:
            location_name = action_dict.i.replace("go to ", "")
            # find the location
            # vild(image_path, location_name) # todo: return depth
            # go to the location
            pass
        elif "pick up" in action_dict.i:
            object_name = action_dict.i.replace("pick up ", "")
            # find the object
            # vild(image_path, object_name) # todo: return depth
            # pick up the object
            pass
        elif "drop off" in action_dict.i:
            # TODO
            # drop off the object (open the gripper)
            pass
else:
    print(output)