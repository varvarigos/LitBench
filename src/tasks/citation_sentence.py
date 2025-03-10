"""
Generates a citation sentence based on the titles and abstracts of two papers.

Args:
    usr_input (str): A string containing the titles and abstracts of Paper A and Paper B.
                        The format should be:
                        "Title A: <title of paper A>\nAbstract A: <abstract of paper A>\nTitle B: <title of paper B>\nAbstract B: <abstract of paper B>"
    template (dict): A dictionary containing a template for the prompt input. The key "prompt_input" should map to a string with placeholders for the instruction and input.

Returns:
    str: A formatted string that combines the instruction and the prompt input with the provided titles and abstracts.
"""

def citation_sentence(usr_input, template, has_predefined_template=False):
    instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section. \n"

    if has_predefined_template:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": usr_input},
        ]
    else:
        res = template["prompt_input"].format(instruction=instruction, input=usr_input)

    return res
