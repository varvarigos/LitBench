"""
Retrieves the most likely paper to be cited by Paper A from a list of candidate papers based on user input.
Args:
    usr_input (str): A string containing the title and abstract of Paper A followed by the titles and abstracts of candidate papers.
    template (dict): A dictionary containing a template for formatting the prompt input.
Returns:
    str: A string containing the prompt input for the user.
"""

def paper_retrieval(usr_input, template):
    instruction = "Please select the paper that is more likely to be cited by paper A from candidate papers."

    title_1 = usr_input.split("Title of the Paper A: ")[1].split("\n\n")[0]
    abstract_1 = usr_input.split("Abstract of Paper A: ")[1].split("\n\n")[0]
    
    candidate_papers = []
    for i in range(1, len(usr_input.split("Candidate "))-1):
        candidate_papers.append(usr_input.split("Candidate " + str(i) + ": ")[1].split("\n\n")[0])
    candidate_papers.append(usr_input.split("Candidate " + str(len(usr_input.split("Candidate "))-1) + ": ")[1])

    prompt_input = ""
    prompt_input = prompt_input + "Title of the Paper A: " + title_1 + "\n"
    prompt_input = prompt_input + "Abstract of the Paper A: " + abstract_1 + "\n"
    prompt_input = prompt_input + "candidate papers: " + "\n"
    for i in range(len(candidate_papers)):
        prompt_input = prompt_input + str(i) + '. ' + candidate_papers[i] + "\n"

    prompt_input = prompt_input + "Give me the title of your selected paper."

    res = template["prompt_input"].format(instruction=instruction, input=prompt_input)

    return res
