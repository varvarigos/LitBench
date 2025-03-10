"""
Determine if paper A will cite paper B.

Args:
    usr_input (str): The user-provided input containing the titles and abstracts of papers A and B.
    template (dict): A dictionary containing the template for generating the link prediction task.

Returns:
    str: The generated link prediction task based on the user input.
"""

def link_pred(usr_input, template, has_predefined_template=False):
    instruction = "Determine if paper A will cite paper B."

    if has_predefined_template:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": usr_input},
        ]
    else:
        res = template["prompt_input"].format(instruction=instruction, input=usr_input)

    return res
