import nbformat
import os, sys
from utils.optim_utils import Parser, load_json
from easydict import EasyDict as edict

# this notebook has to be created at the top level.
# i.e all your images, videos, json files should be accessible by ./
# Check how I access the path to files and how I set the paths in markdown (does not have base_path coz notebook accesses them using ./)

def create_notebook(base_path, time_logger):
    # Create a new Jupyter Notebook
    notebook = nbformat.v4.new_notebook()

    # take gt data from config file:
    optimiser_config = os.path.join(base_path, "config.json")

    args = Parser(optimiser_config)
    # Training time:
    time = " Training time: "
    time += "{:.4f}".format(time_logger.train_time/60)
    time += "min \n"
    time += "Point Generation time: "
    time += "{:.4f}".format(time_logger.point_gen_pass)
    time += "s \n"
    time += "Point transfer time: "
    time += "{:.4f}".format(time_logger.point_transfer)
    time += "s \n"
    time += "Rendering time: "
    time += "{:.4f}".format(time_logger.render_pass)
    time += "s \n"
    time += "Loss compute: "
    time += "{:.6f}".format(time_logger.loss_pass)
    time += "s \n"
    time += "Backward pass: "
    time += "{:.4f}".format(time_logger.backward_pass)
    time += "s \n"
    time += "Simulated annealing: "
    time += "{:.4f}".format(time_logger.sa_pass)
    time += "s \n"
    markdown_cell = nbformat.v4.new_markdown_cell(time)
    notebook.cells.append(markdown_cell)
       
    # Init IFS Code
    # -------------
    init_ifs_code = "##### Init IFS Code\n\n"
    init_stats = os.path.join(base_path, "init", "init_weights_stats.json")
    args1 = edict(load_json(init_stats))

    init_matrix_list = args1.init_matrices
    init_vector_list = args1.init_vectors
    init_singular_list = args1.singular_values
    init_probs = args1.probs
    init_conditions = args1.condition_numbers
    init_opacities = args1.opacities
    
    init_ifs_code += create_matrix(init_matrix_list)
    init_ifs_code += "\n\n"
    init_ifs_code += create_vector(init_vector_list)
    init_ifs_code += "\n\n"

    init_ifs_code += create_list(init_singular_list, "SingularValue(s)")
    init_ifs_code += "\n\n"

    init_ifs_code += create_list(init_probs, "Init Probs")
    init_ifs_code += "\n\n"
    init_ifs_code += create_list(init_conditions, "ConditionNumbers")
    init_ifs_code += "\n\n"
    init_ifs_code += create_list(init_opacities, "Opacities")

    markdown_cell = nbformat.v4.new_markdown_cell(init_ifs_code)
    notebook.cells.append(markdown_cell)

    # Optimised IFS Code
    # -------------
    opt_ifs_code = "##### Optimised IFS Code\n\n"
    opt_stats = os.path.join(base_path, "output", "optimized_ifs_code.json")
    args2 = edict(load_json(opt_stats))

    opt_matrix_list = args2.opt_matrices
    opt_vector_list = args2.opt_vectors
    opt_singular_list = args2.singular_values
    opt_probs = args2.probs
    opt_dets_list = args2.determinants
    opt_conditions = args2.condition_numbers
    opt_opacities = args2.opacities
    opt_color = args2.optimized_color
    
    opt_ifs_code += create_matrix(opt_matrix_list)
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_vector(opt_vector_list)
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_singular_list, "SingularValue(s)")
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_probs, "Probs")
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_dets_list, "determinants")
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_conditions, "ConditionNumbers")
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_opacities, "Opacities")
    opt_ifs_code += "\n\n"
    opt_ifs_code += create_list(opt_color, "Foreground")

    markdown_cell = nbformat.v4.new_markdown_cell(opt_ifs_code)
    notebook.cells.append(markdown_cell)

    # -------------
    # Loss Curves
    # -------------
    img_path = os.path.join("output")
    markdown_content = "##### Loss Curves\n\n"
    loss = os.path.join(img_path, "Loss.png")
    loss_log = os.path.join(img_path, "Loss_log.png")

    markdown_content += f"![MSE]({loss}) &nbsp; &nbsp; "
    markdown_content += f"![MSE Log Scale]({loss_log})"
    
    markdown_cell = nbformat.v4.new_markdown_cell(markdown_content)
    notebook.cells.append(markdown_cell)
    # -------------

    # Optimization progress
    # -------------
    markdown_content = "##### Optimization progress\n\n"
    video_path = os.path.join("output", "sequence.mp4")
    markdown_content += f"<video controls src={video_path} width={1024}></video>"
    markdown_cell = nbformat.v4.new_markdown_cell(markdown_content)
    notebook.cells.append(markdown_cell)
    
    # -------------

    # Comparison Images
    # -------------
    img_path = os.path.join("output")
    markdown_content = "##### Results & Comparisons \n\n"

    gt = os.path.join(img_path,  f"gt.png")
    supersampled_result = os.path.join(img_path,  f"val", "supersampled.png")
    
    markdown_content += f"GT ----------- Result \n\n"
    markdown_content += f"![Super High-res result]({gt}) &nbsp; &nbsp; "
    markdown_content += f"![Super High-res result]({supersampled_result}) &nbsp; &nbsp; "    
    
    
    markdown_cell = nbformat.v4.new_markdown_cell(markdown_content)
    notebook.cells.append(markdown_cell)

    return notebook

def create_matrix(matrix_list):
    # Convert the matrix list to LaTeX format
    matrix_latex = ""
    markdown_content = ""
    for i in range(len(matrix_list)):
        matrix = matrix_list[i]
        matrix_latex += f"$ M_{i} = "
        matrix_latex += "\\begin{bmatrix}\n"
        
        for row in matrix:
            formatted_row = ["{:.2f}".format(element) for element in row]
            matrix_latex += " & ".join(map(str, formatted_row)) + " \\\\ \n"
        matrix_latex += "\\end{bmatrix}$"
        
        markdown_content += f"{matrix_latex}"
        markdown_content += "; "
        matrix_latex = ""

    return markdown_content

def create_list(input_list, name):
    formatted_list = ["{:.2f}".format(element) for element in input_list]
    formatted_string = ", ".join(formatted_list)
    
    markdown_content = ""
    markdown_content += f"$ {name} = [{formatted_string}] $ \n\n"
    return markdown_content


def create_vector(vector_list):
    # Convert the vector list to LaTeX format
    vector_latex = ""
    markdown_content = ""
    for i in range(len(vector_list)):
        vector = vector_list[i]
        vector_latex += f"$ V_{i} = "
        vector_latex += "\\begin{bmatrix}\n"
        
        # for row in vector:
        formatted_row = ["{:.2f}".format(element) for element in vector]
        vector_latex += " \\\\ ".join(map(str, formatted_row)) + " \\\\ \n"
        vector_latex += "\\end{bmatrix}$"
        
        markdown_content += f"{vector_latex}"
        markdown_content += "; "
        vector_latex = ""

    return markdown_content

def save_notebook(notebook, filename):
    # Save the notebook to a file
    with open(filename, "w") as f:
        nbformat.write(notebook, f)

def dump_notebook(base_path, time_logger):
    base_path = os.path.abspath(base_path)
    notebook = create_notebook(base_path, time_logger)
    notebook_filename = os.path.join(base_path, "summary_notebook.ipynb")
    save_notebook(notebook, notebook_filename)
    
    print("---- Created Summary notebook ----")

# if __name__ == "__main__":
#     dump_notebook("./log/script_test0/test_suite_1")
