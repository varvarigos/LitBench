"""
Generate a prompt for generating the title of a paper based on its abstract.

Args:
    usr_input (str): A string containing the title and abstract of the paper in the format "Title: <title> Abstract: <abstract>".
    template (dict): A dictionary containing the template for the prompt with a key "prompt_input".

Returns:
    str: A formatted string with the instruction and abstract to be used as input for generating the title.
"""

def abs_2_title(usr_input, template):
    instruction = "Please generate the title of paper based on its abstract"

    abstract = usr_input.split("Abstract: ")[1]

    prompt_input = ""
    prompt_input = prompt_input + "Abstract: " + abstract + "\n"
    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)

    return res
