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

def citation_sentence(usr_input, template):
    instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section. \n"

    title_A = usr_input.split("Title A: ")[1].split("\n\n")[0]
    abstract_A = usr_input.split("Abstract A: ")[1].split("\n\n")[0]
    title_B = usr_input.split("Title B: ")[1].split("\n\n")[0]
    abstract_B = usr_input.split("Abstract B: ")[1]

    prompt_input = ""
    prompt_input = prompt_input + "Title of Paper A: " + title_A + '\n'
    prompt_input = prompt_input + "Abstract of Paper A: " + abstract_A + '\n'
    prompt_input = prompt_input + "Title of Paper B: " + title_B + '\n'
    prompt_input = prompt_input + "Abstract of Paper B: " + abstract_B + '\n'

    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)
    return res
