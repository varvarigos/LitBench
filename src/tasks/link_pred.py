"""
Determine if paper A will cite paper B.

Args:
    usr_input (str): The user-provided input containing the titles and abstracts of papers A and B.
    template (dict): A dictionary containing the template for generating the link prediction task.

Returns:
    str: The generated link prediction task based on the user input.
"""

def link_pred(usr_input, template):
    instruction = "Determine if paper A will cite paper B."

    title_A = usr_input.split("Title A: ")[1].split("\n\n")[0]
    abstract_A = usr_input.split("Abstract A: ")[1].split("\n\n")[0]
    title_B = usr_input.split("Title B: ")[1].split("\n\n")[0]
    abstract_B = usr_input.split("Abstract B: ")[1]

    prompt_input = ""
    prompt_input = prompt_input + "Title of Paper A: " + title_A + '\n'
    prompt_input = prompt_input + "Abstract of Paper A: " + abstract_A + '\n'
    prompt_input = prompt_input + "Title of Paper B: " + title_B + '\n'
    prompt_input = prompt_input + "Abstract of Paper B: " + abstract_B + '\n'

    prompt_input = prompt_input + " Give me a direct answer of yes or no."

    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    return res
