import sys
import regex
import yaml
import shutil
import bibtexparser
from charset_normalizer import from_path
from langdetect import detect
import os
import subprocess
import numpy as np
import networkx as nx
import re



def is_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def read_tex_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tex_content = file.read()
    return tex_content

def write_tex_file(file_path, s):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(s)

def get_core(s):
    start = '\\begin{document}'
    end = '\\end{document}'
    beginning_doc = s.find(start)
    end_doc = s.rfind(end)
    return s[beginning_doc+len(start):end_doc]


def retrieve_text(text, command, keep_text=False):
    """Removes '\\command{*}' from the string 'text'.

    Regex `base_pattern` used to match balanced parentheses taken from:
    https://stackoverflow.com/questions/546433/regular-expression-to-match-balanced-parentheses/35271017#35271017
    """
    base_pattern = (
        r'\\' + command + r"(?:\[(?:.*?)\])*\{((?:[^{}]+|\{(?1)\})*)\}(?:\[(?:.*?)\])*"
    )

    def extract_text_inside_curly_braces(text):
        """Extract text inside of {} from command string"""
        pattern = r"\{((?:[^{}]|(?R))*)\}"

        match = regex.search(pattern, text)

        if match:
            return match.group(1)
        else:
            return ""

    # Loops in case of nested commands that need to retain text, e.g. \red{hello \red{world}}.
    while True:
        all_substitutions = []
        has_match = False
        for match in regex.finditer(base_pattern, text):
            # In case there are only spaces or nothing up to the following newline,
            # adds a percent, not to alter the newlines.
            has_match = True

            if not keep_text:
                new_substring = ""
            else:
                temp_substring = text[match.span()[0] : match.span()[1]]
                return extract_text_inside_curly_braces(temp_substring)

            if match.span()[1] < len(text):
                next_newline = text[match.span()[1] :].find("\n")
                if next_newline != -1:
                    text_until_newline = text[
                        match.span()[1] : match.span()[1] + next_newline
                    ]
                    if (
                        not text_until_newline or text_until_newline.isspace()
                    ) and not keep_text:
                        new_substring = "%"
            all_substitutions.append((match.span()[0], match.span()[1], new_substring))

        for start, end, new_substring in reversed(all_substitutions):
            text = text[:start] + new_substring + text[end:]

        if not keep_text or not has_match:
            break


def reduce_linebreaks(s):
    return re.sub(r'(\n[ \t]*)+(\n[ \t]*)+', '\n\n', s)


def replace_percentage(s):
    return re.sub(r'% *\n', '\n', s)   


def reduce_spaces(s):
    return re.sub(' +', ' ', s)


def delete_urls(s):
    return re.sub(r'http\S+', '', s)


def remove_tilde(s):
    s1 = re.sub(r'[~ ]\.', '.', s)
    s2 = re.sub(r'[~ ],', ',', s1)
    return re.sub(r'{}', '', s2)


def remove_verbatim_words(s):
    with open("conf/specific_commands.yaml", "r") as stream:
        read_config = yaml.safe_load(stream)
    
    for command in read_config['verbatim_to_delete']:
        s = s.replace(command, '')

    for command in read_config['two_arguments']:
        pattern = r'\\' + command + r'{[^}]*}' + r'{[^}]*}'
        s = re.sub(pattern, '', s)

    for command in read_config['three_arguments']:
        pattern = r'\\' + command + r'{[^}]*}' + r'{[^}]*}' + r'{[^}]*}'
        s = re.sub(pattern, '', s)

    for command in read_config['two_arguments_elaborate']:
        s = remove_multargument(s, '\\' + command, 2)

    for command in read_config['three_arguments_elaborate']:
        s = remove_multargument(s, '\\' + command, 3)

    for command in read_config['replace_comments']:
        pattern = r'\\' + command
        s = re.sub(pattern, '%', s)
    
    s = re.sub(
      r'\\end{[\s]*abstract[\s]*}',
      '',
      s,
      flags=re.IGNORECASE
    )

    s = re.sub(
      r'\\begin{[\s]*abstract[\s]*}',
      'Abstract\n\n',
      s,
      flags=re.IGNORECASE
    )
    return s


def yes_or_no(s):
    return 1 if "Yes" == s[0:3] else 0 if "No" == s[0:2] else -1


def get_main(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    latex_paths = [f for f in file_paths if f.endswith('.tex')]
    number_tex = len(latex_paths)
    if number_tex == 0:
        return None
    if number_tex == 1:
        return latex_paths[0]
    adjacency = np.zeros((number_tex, number_tex))
    keys = [os.path.basename(path) for path in latex_paths]
    reg_ex = r'\\input{(.*?)}|\\include{(.*?)}|\\import{(.*?)}|\\subfile{(.*?)}|\\include[*]{(.*?)}|}'
    for i,file in enumerate(latex_paths):
        content = read_tex_file(file)
        find_pattern_input = re.findall(reg_ex, content)
        find_pattern_input = [tup for tup in find_pattern_input if not all(element == "" for element in tup)]
        number_matches = len(find_pattern_input)
        if number_matches == 0:
            continue
        else:
            content = replace_imports(file, content)
        reg_ex_clean = r'\\input{(.*?)}|\\include{(.*?)}'
        find_pattern_input = re.findall(reg_ex_clean, content)
        number_matches = len(find_pattern_input)  
        for j in range(number_matches):
            match = find_pattern_input[j]
            non_empty_match = [t for t in match if t]
            for non_empty in non_empty_match:
                base_match = os.path.basename(non_empty)
                if not base_match.endswith('.tex'):
                    base_match = base_match + '.tex'
                    if base_match not in keys:
                        continue
                ind = keys.index(base_match)
                adjacency[i][ind] = 1
    G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    connected_components = list(nx.weakly_connected_components(G))
    size_connected = [len(x) for x in connected_components]
    maximum_size = max(size_connected)
    biggest_connected = [x for x in connected_components if len(x) == maximum_size]
    if len(biggest_connected)>1:
        roots = [n for connected in biggest_connected for n in connected if not list(G.predecessors(n))]
        _check = []
        for r in roots:
            try: 
                _check.append(check_begin(latex_paths[r]))
            except Exception as e:
                _check.append(False)
        potentials_files = [latex_paths[x] for x, y in zip(roots, _check) if y == True]
        sizes_files = [os.path.getsize(x) for x in potentials_files]
        return potentials_files[sizes_files.index(max(sizes_files))]
        
    else:
        roots = [n for n in biggest_connected[0] if not list(G.predecessors(n))]
        return latex_paths[roots[0]]


def initial_clean(directory, config):
    config_cmd = ''
    if config == True:
        config_cmd = '--config conf/cleaning_config.yaml'
    temp_dir = directory[:directory.rfind('/')] + '_temp' + '/'
    shutil.copytree(directory, temp_dir)
    try:
        command_res = os.system('arxiv_latex_cleaner --keep_bib {} {}'.format(directory, config_cmd))
        if command_res != 0:
            raise Exception('Error cleaning')
        else:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        shutil.rmtree(directory)
        os.rename(temp_dir, directory)
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path) 
        latex_paths = [f for f in file_paths if f.endswith('.tex')]
        for p in latex_paths:
            results = from_path(p)
            with open(p, 'w', encoding='utf-8') as f:
                f.write(str(results.best()))
        os.system('arxiv_latex_cleaner --keep_bib {} {}'.format(directory, config_cmd))
    cleaned_directory = directory[:directory.rfind('/')] + '_arXiv'
    shutil.rmtree(directory)
    os.rename(cleaned_directory, directory)


def check_begin(directory):
    content = read_tex_file(directory)
    english = detect(content) == 'en'
    return True and english if re.findall(r'\\begin{document}', content) else False


def post_processing(extracted_dir, file):
    _dir = os.path.dirname(file) + '/'
    perl_expand(file)
    file = _dir + 'merged_latexpand.tex'
    try: 
        de_macro(file)
        file = _dir + 'merged_latexpand-clean.tex'
    except Exception as e:
        pass
    try:
        def_handle(file)
    except Exception as e:
        pass
    try:
        declare_operator(file) # has additional add-ons 
    except Exception as e:
        pass
    try: 
        de_macro(file)
        file = _dir + os.path.splitext(os.path.basename(file))[0] + '-clean' + '.tex'
    except Exception as e:
        pass
    initial_clean(_dir, config=True)
    initial_clean(_dir, config=False)
    tex_content = read_tex_file(file)
    final_tex = reduce_spaces(
        delete_urls(
            remove_tilde(
                reduce_linebreaks(
                    replace_percentage(
                        remove_verbatim_words(
                            tex_content
                        )
                    )
                )
            )
        )
    ).strip()
    shutil.rmtree(extracted_dir)
    os.makedirs(extracted_dir)
    write_tex_file(extracted_dir + 'final_cleaned.tex', final_tex)
    initial_clean(extracted_dir, config=False)    
    return extracted_dir + 'final_cleaned.tex'


def perl_expand(file):
    # Save the current working directory
    oldpwd = os.getcwd()
    target_dir = os.path.dirname(file) + '/'
    # Correctly construct the path
    target = os.path.join(target_dir, 'latexpand')
    src = './src/utils/latexpand'
    # Copy the `latexpand` script to the target directory
    shutil.copyfile(src, target)
    # Change to the target directory
    os.chdir(target_dir)

    # Run the perl command without shell=True and handle redirection within Python
    with open('merged_latexpand.tex', 'w') as output_file:
        subprocess.run(['perl', 'latexpand', os.path.basename(file)],
                       stdout=output_file, stderr=subprocess.DEVNULL)
    
    # Return to the original directory
    os.chdir(oldpwd)


def de_macro(file):
    # Save the current working directory\
    oldpwd = os.getcwd()
    target_dir = os.path.dirname(file) + '/'
    # Construct the target path
    target = os.path.join(target_dir, 'de-macro.py')
    src = '.src/utils/de-macro.py'

    # Copy the `de-macro.py` script to the target directory
    shutil.copyfile(src, target)
    # Change to the target directory
    os.chdir(target_dir)

    # Run the de-macro script without os.system and capture errors
    try:
        subprocess.run(['python3', 'de-macro.py', os.path.basename(file)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error de-macro: {e}") from e
    finally:
        # Always return to the original directory
        os.chdir(oldpwd)


def def_handle(file):
    h = os.system('python3 src/utils/def_handle.py {} --output {}'.format(file, file))
    if h != 0:
        raise Exception('Error def handle')


def declare_operator(file):
    s = read_tex_file(file)
    ## Operators
    pattern = r'\\DeclareMathOperator'
    s = re.sub(pattern, r'\\newcommand', s)
    pattern = {
    r'\\newcommand\*': r'\\newcommand',
    r'\\providecommand\*': r'\\newcommand',
    r'\\providecommand': r'\\newcommand',
    r'\\renewcommand\*': r'\\renewcommand',
    r'\\newenvironment\*': r'\\newenvironment',
    r'\\renewenvironment\*': r'\\renewenvironment'
    }
    s = re.sub(r'\\end +', r'\\end', s)
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    ## Title
    start = '\\begin{document}'
    beginning_doc = s.find(start)
    pattern = {
            r'\\icmltitlerunning\*': r'\\title',
            r'\\icmltitlerunning': r'\\title',
            r'\\inlinetitle\*': r'\\title',
            r'\\icmltitle\*': r'\\title',
            r'\\inlinetitle': r'\\title',
            r'\\icmltitle': r'\\title',
            r'\\titlerunning\*': r'\\title',
            r'\\titlerunning': r'\\title',
            r'\\toctitle': r'\\title',
            r'\\title\*': r'\\title',
            r'\\TITLE\*': r'\\title',
            r'\\TITLE': r'\\title',
            r'\\Title\*': r'\\title',
            r'\\Title': r'\\title',
        }
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    find_potential = s.find('\\title') 

    ## Remove \\
    title_content = retrieve_text(s, 'title', keep_text = True)
    if title_content != None:
        cleaned_title = re.sub(r'\\\\', ' ', title_content)
        cleaned_title = re.sub(r'\n',' ', cleaned_title)
        cleaned_title = re.sub(r'\~',' ', cleaned_title)
        s = s.replace(title_content, cleaned_title)
        if find_potential != -1 and find_potential < beginning_doc:
            s = s.replace('\\maketitle', cleaned_title)

    ##  Cite and ref commands 
    pattern = {
        r'\\citep\*': r'\\cite',
        r'\\citet\*': r'\\cite',
        r'\\citep': r'\\cite',
        r'\\citet': r'\\cite',
        r'\\cite\*': r'\\cite',
        r'\\citealt\*': r'\\cite',
        r'\\citealt': r'\\cite',
        r'\\citealtp\*': r'\\cite',
        r'\\citealp': r'\\cite',
        r'\\citeyear\*': r'\\cite',
        r'\\citeyear': r'\\cite',
        r'\\citeauthor\*': r'\\cite',
        r'\\citeauthor': r'\\cite',
        r'\\citenum\*': r'\\cite',
        r'\\citenum': r'\\cite',
        r'\\cref': r'\\ref',
        r'\\Cref': r'\\ref',
        r'\\factref': r'\\ref',
        r'\\appref': r'\\ref',
        r'\\thmref': r'\\ref',
        r'\\secref': r'\\ref',
        r'\\lemref': r'\\ref',
        r'\\corref': r'\\ref',
        r'\\eqref': r'\\ref',
        r'\\autoref': r'\\ref',
        r'begin{thm}': r'begin{theorem}',
        r'begin{lem}': r'begin{lemma}',
        r'begin{cor}': r'begin{corollary}',
        r'begin{exm}': r'begin{example}',
        r'begin{defi}': r'begin{definition}',
        r'begin{rem}': r'begin{remark}',
        r'begin{prop}': r'begin{proposition}',
        r'end{thm}': r'end{theorem}',
        r'end{lem}': r'end{lemma}',
        r'end{cor}': r'end{corollary}',
        r'end{exm}': r'end{example}',
        r'end{defi}': r'end{definition}',
        r'end{rem}': r'end{remark}',
        r'end{prop}': r'end{proposition}',
    }

    for key in pattern:
        s = re.sub(key, pattern[key], s)

    
    pattern = {
        r'subsubsection':  r'section',
        r'subsubsection ': r'section',
        r'subsubsection\*':  r'section',
        r'subsubsection\* ':  r'section',
        r'subsection': r'section',
        r'subsection ':  r'section',
        r'subsection\*': r'section',
        r'subsection\* ': r'section',
        r'section ':  r'section',
        r'section\*': r'section',
        r'section\* ': r'section',
        r'chapter':  r'section',
        r'chapter ': r'section',
        r'chapter\*':  r'section',
        r'chapter\* ':  r'section',
        r'mysubsubsection': r'section',
        r'mysubsection':  r'section',
        r'mysection':  r'section',
    }

    for key in pattern:
        s = re.sub(key, pattern[key], s)

    # In case any new commands for appendix/appendices 
    s = re.sub(r'newcommand{\\appendix}', '', s)
    s = re.sub(r'newcommand{\\appendices}', '', s)
    s = get_core(s)
    
    ## In case of double titles being defined 
    title_content = retrieve_text(s, 'title', keep_text = True)
    if title_content != None:
        cleaned_title = re.sub(r'\\\\', ' ', title_content)
        cleaned_title = re.sub(r'\n',' ', cleaned_title)
        cleaned_title = re.sub(r'\~',' ', cleaned_title)
        s = s.replace(title_content, cleaned_title)
    write_tex_file(file, s)
    

def replace_imports(file, s):
    regex_p1 = r'\\import{(.*?)}{(.*?)}'
    s = re.sub(regex_p1, r"\\input{\1\2}", s)
    regex_p2 = r'\\subfile{(.*?)}'
    s = re.sub(regex_p2, r"\\input{\1}", s)
    regex_p3 = r'\\include[*]{(.*?)}'
    s = re.sub(regex_p3, r"\\input{\1}", s)
    write_tex_file(file, s)
    return s


def remove_multargument(s, target, k):
    ind = s.find(target)
    while ind != -1:
        start_ind = ind + len(target)
        stack_open = 0
        stack_close = 0
        track_arg  = 0
        for i, char in enumerate(s[start_ind:]):
            if char == '{':
                stack_open += 1
            if char == '}':
                stack_close += 1
            if stack_open !=0 and stack_close !=0:
                if stack_open == stack_close:
                    track_arg += 1
                    stack_open = 0
                    stack_close = 0
            if track_arg == k:
                break
        s = s[:ind] + s[start_ind + i + 1:]
        ind = s.find(target)
    return s


def fix_citations(s):
    pattern = {
    r'\\citep\*': r'\\cite',
    r'\\citet\*': r'\\cite',
    r'\\citep': r'\\cite',
    r'\\citet': r'\\cite',
    r'\\cite\*': r'\\cite',
    r'\\citealt\*': r'\\cite',
    r'\\citealt': r'\\cite',
    r'\\citealtp\*': r'\\cite',
    r'\\citealp': r'\\cite',
    r'\\citeyear\*': r'\\cite',
    r'\\citeyear': r'\\cite',
    r'\\citeauthor\*': r'\\cite',
    r'\\citeauthor': r'\\cite',
    r'\\citenum\*': r'\\cite',
    r'\\citenum': r'\\cite'
    }
    for key in pattern:
        s = re.sub(key, pattern[key], s)
    return s

def find_bib(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path) 
    bib_paths = [f for f in file_paths if f.endswith('.bib')]
    return bib_paths

def create_bib_from_bbl(bibfile):
    with open(bibfile, 'r') as f:
        content = f.read()
    library_raw = bibtexparser.parse_string(content)
    library = {}
    for block in library_raw.blocks:
        if isinstance(
            block, 
            (bibtexparser.model.DuplicateBlockKeyBlock, bibtexparser.model.ParsingFailedBlock, bibtexparser.model.ImplicitComment)
        ):
            continue
        fields = {}
        for field in block.fields:
            fields[field.key] = field.value

        ## Get a good title one ##
        field_content = fields["note"]
        field_content = field_content.replace("\n", " ")
        field_content = re.sub(" +", " ", field_content)
        if field_content.find("``") != -1 and field_content.find("\'\'") != -1:
            title = (
                field_content[field_content.find("``") + 2 : field_content.find("\'\'")]
                    .replace("\\emph", "")
                    .replace("\\emp", "")
                    .replace("\\em", "")
                    .replace(",", "")
                    .replace("{", "")
                    .replace("}","")
                    .replace("``", "")
                    .replace("\'\'", "")
                    .strip(".")
                    .strip()
                    .strip(".")
                    .lower()
            )
            fields['title'] = title
        else:
            if field_content.count("\\newblock") == 2:
                field_content = field_content.replace("\\newblock", "``", 1)
                field_content = field_content.replace("\\newblock", "\'\'", 1)
                if field_content.find("``") != -1 and field_content.find("\'\'") != -1:
                    title = (
                        field_content[field_content.find("``") + 2 : field_content.find("\'\'")]
                        .replace("\\emph", "")
                        .replace("\\emp", "")
                        .replace("\\em", "")
                        .replace(",", "")
                        .replace("{", "")
                        .replace("}","")
                        .replace("``", "")
                        .replace("\'\'", "")
                        .strip(".")
                        .strip()
                        .strip(".")
                        .lower()
                    )
                    fields['title'] = title
        library[block.key] = fields
    return library


def create_bib(bibfile):
    with open(bibfile, 'r') as f:
        content = f.read()
    library_raw = bibtexparser.parse_string(content)

    library = {}
    for block in library_raw.blocks:
        if isinstance(
            block, 
            (bibtexparser.model.DuplicateBlockKeyBlock, bibtexparser.model.ParsingFailedBlock, bibtexparser.model.ImplicitComment)
        ):
            continue
        fields = {}
        for field in block.fields:
            fields[field.key] = field.value.replace('{', '').replace('}', '')
            if field.key == 'title':
                title = re.sub(r'[\n]+', ' ', field.value) # keep only one \n
                title = re.sub(r' +', ' ', title)
                fields[field.key] = (
                    title.replace("\\emph", "")
                    .replace("\\emp", "")
                    .replace("\\em", "")
                    .replace(",", "")
                    .replace("{", "")
                    .replace("}", "")
                    .strip(".")
                    .strip()
                    .strip(".")
                    .lower()
                )
        if 'title' not in fields:
            continue
        library[block.key] = fields
    return library


def find_bbl(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path) 
    bib_paths = [f for f in file_paths if f.endswith('.bbl')]
    return bib_paths


def textobib(file):
    oldpwd = os.getcwd() 
    target_dir = os.path.dirname(file) + '/' 
    target = target_dir + 'tex2bib'
    src = './tex2bib'
    shutil.copyfile(src, target)
    os.chdir(target_dir)
    output_file = os.path.splitext(os.path.basename(file))[0]  + '.bib'
    os.system('perl tex2bib -i {} -o {}'.format(os.path.basename(file), output_file))
    os.chdir(oldpwd)
    return target_dir + output_file


def get_library_bib(bib_files):
    library = []
    for bib_file in bib_files:
        library.append(create_bib(bib_file))
    final_library = {}
    for d in library:
        final_library.update(d)
    return final_library


def get_library_bbl(bbl_files):
    bib_files = []
    for bbl_file in bbl_files:
        bib_files.append(textobib(bbl_file))
    library = []
    for bib_file in bib_files:
        library.append(create_bib_from_bbl(bib_file))
    final_library = {}
    for d in library:
        final_library.update(d) 
    return final_library
