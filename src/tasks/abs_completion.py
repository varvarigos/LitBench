"""
Generates a formatted prompt for completing the abstract of a paper.

Args:
    usr_input (str): The user input containing the title and part of the abstract.
                        Expected format:
                        "Title: <title>\nAbstract: <abstract>"
    template (dict): A dictionary containing the template for the prompt.
                        Expected format:
                        {"prompt_input": "<template_string>"}
                        The template string should contain placeholders for
                        'instruction' and 'input'.

Returns:
    str: A formatted string with the instruction and the input embedded in the template.
"""

def abs_completion(usr_input, template, has_predefined_template=False):
    instruction = "Please complete the abstract of a paper."

    if has_predefined_template:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": usr_input},
        ]
    else:
        res = template["prompt_input"].format(instruction=instruction, input=usr_input)

    return res
