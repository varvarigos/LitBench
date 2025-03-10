"""
Retrieves the most likely paper to be cited by Paper A from a list of candidate papers based on user input.
Args:
    usr_input (str): A string containing the title and abstract of Paper A followed by the titles and abstracts of candidate papers.
    template (dict): A dictionary containing a template for formatting the prompt input.
Returns:
    str: A string containing the prompt input for the user.
"""

def paper_retrieval(usr_input, template, has_predefined_template=False):
    instruction = "Please select the paper that is more likely to be cited by paper A from candidate papers."

    if has_predefined_template:
        res = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": usr_input},
        ]
    else:
        res = template["prompt_input"].format(instruction=instruction, input=usr_input)

    return res
