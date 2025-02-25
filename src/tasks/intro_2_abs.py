"""
Generate the abstract of a paper based on its introduction section.

Args:
    usr_prompt (str): The user-provided prompt containing the introduction section of the paper.
    template (dict): A dictionary containing the template for generating the abstract.
    context_window (int): The maximum length of the context window for the prompt input.

Returns:
    str: The generated abstract based on the introduction section.
"""


def intro_2_abs(usr_prompt, template, context_window):
    instruction = "Please generate the abstract of paper based on its introduction section."

    introduction = usr_prompt.split("Introduction: ")[1]

    prompt_input = ""
    prompt_input = prompt_input + "Introduction: " + introduction + "\n"

    # Reduce it to make it fit
    prompt_input = prompt_input[:int(context_window*2)]

    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)

    return res
