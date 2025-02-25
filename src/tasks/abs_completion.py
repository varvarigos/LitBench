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

def abs_completion(usr_input, template):
    instruction = "Please complete the abstract of a paper."

    title = usr_input.split("Title: ")[1].split("\n\n")[0]
    abstract = usr_input.split("Abstract: ")[1]

    prompt_input = ""
    prompt_input = prompt_input + "Title: " + title + "\n"
    prompt_input = prompt_input + "Part of abstract: " + abstract
    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)

    return res
