import os, sys
import random

def getDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(getDir())

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils.optim_utils import load_json, safe_state, write_to_json
from cf.images.image_io import load_image, find_images
from cf.images.conversions import image_to_tensor
from cf.tools.train_tools import TrainingLog
from tqdm import tqdm

device = 'cuda'

class CNNModel(nn.Module):
    def __init__(self, output_funcs):
        super(CNNModel, self).__init__()
        self.num_funcs = output_funcs
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 128 * 128, 256)
        self.fc2 = nn.Linear(256, output_funcs * 6)
        
        # Create Identity matrix
        self.I = torch.eye(2, 3).unsqueeze(0).expand(self.num_funcs, -1, -1).to(device)
        self.epsilon = 0.01

        # Initialize weights and biases to zero
        self.apply(self._initialize_weights)

    def _initialize_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='sigmoid')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.size(1) == 3:
            x = x[:, 0, None, :, :]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        compressed_f = x.view(self.num_funcs, 2, 3) # has to be num_funcs * 6 values reshaped each to 2,3, where num_funcs should be the batch

        # deviations from identity
        result_matrix = self.I + self.epsilon * compressed_f
        
        # Sorting the output
        x_sorted = result_matrix
        
        return x_sorted
    

# Loss Function
def l2_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def convert_code_to_tensor(code):
    # code is an edict with ifs_m, ifs_t, ifs_p (not used)
    combined_tensors = []
    for num_f in range(len(code.ifs_m)):
        weight = torch.tensor(code.ifs_m[num_f], dtype = torch.float32) # 2x2
        bias = torch.tensor(code.ifs_t[num_f], dtype = torch.float32) # 2

        # Combine weight and bias into a 2x3 tensor
        combined_tensor = torch.cat((weight, bias.unsqueeze(-1)), dim=-1)  # Shape: 2x3
        combined_tensors.append(combined_tensor)
    
    # prepare it as a tensor where functions are stacked in the batch dimension and combine weight and bias to create a 2x3 tensor
    # Resulting shape: (batch_size, 2, 3)
    code_tensor = torch.stack(combined_tensors, dim=0)
    
    return code_tensor

def convert_tensor_to_code(code_tensor):
    # converts output codes by cnn to a dictionary for evaluation later on by model_inference
    
    code_tensor = code_tensor.detach().cpu()

    ifs_m = []
    ifs_t = []
    
    # Loop through each item in the batch dimension (each 2x3 tensor)
    for tensor in code_tensor:
        weight = tensor[:, :-1]  # Extract the first two columns (shape: 2x2)
        bias = tensor[:, -1]     # Extract the last column (shape: 2)
        
        ifs_m.append(weight.numpy().tolist())
        ifs_t.append(bias.numpy().tolist())
    
    # Create the dictionary with ifs_m and ifs_t
    code = {
        'ifs_m': ifs_m,
        'ifs_t': ifs_t,
        'ifs_p': []  # an empty list since it's not used
    }
    
    return code

def load_random_sample(train_data):
    image_folder = os.path.join(train_data, "all_images")
    code_folder = os.path.join(train_data, "all_codes")

    idx_range = len(os.listdir(image_folder))
    # Randomly sample an index
    idx = random.randint(0, idx_range - 1)  # Random integer between 0 and num_images - 1

    # Construct file names
    image_file = f"fdb_{idx}.png"
    code_file = f"fdb_{idx}.json"

    # Load the image
    image_path = os.path.join(image_folder, image_file)
    input_img = load_image(image_path)
    test_img_tensor = image_to_tensor(input_img).to(device)

    # Load the corresponding code
    code_path = os.path.join(code_folder, code_file)
    cnn_ifs_code = load_json(code_path)
    ground_truth = convert_code_to_tensor(cnn_ifs_code).to(device)
    sorted_gt = sort(ground_truth)

    return test_img_tensor, sorted_gt


# Not differentiiable sorting
def sort(tensors):
    def compute_eigenvalue_metric(tensors):
        # tensors should be of shape (N, 2, 3)
        # Extracting the (2,2) submatrix, shape (N, 2, 2)
        sub_matrices = tensors[:, :2, :2]
        _, singular_values, _ = torch.svd(sub_matrices)  # Shape (N, 2)

        # Use the maximum eigenvalue as the sorting metric
        eigenvalue_metric = torch.max(singular_values, dim=1).values
        _, sorting_indices = torch.sort(eigenvalue_metric, dim=0)

        return sorting_indices

    # Compute eigenvalue-based metric
    indices = compute_eigenvalue_metric(tensors)

    x_sorted = tensors[indices]
    
    return x_sorted

def train(log_path, train_data, num_functions):
    safe_state(0)

    # log files stuff
    print(os.path.abspath(log_path))
    imgs_log_path = os.path.join(log_path, "imgs")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(imgs_log_path, exist_ok=True)

    writer = TrainingLog(log_path, add_unique_str=False)

    model = CNNModel(output_funcs=num_functions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    training_iters = 10000

    # training loop:
    model.train()
    j = 0
    # ----------------------

    # run optimization
    for j in tqdm(range(j, training_iters)):
        optimizer.zero_grad()
        
        with torch.no_grad():
            img_tensor, sorted_gt = load_random_sample(train_data)

        outputs = model(img_tensor)
        loss = l2_loss(outputs, sorted_gt)   
        
        loss.backward()
        optimizer.step()        
        scheduler.step()
        
        tqdm.write(
                "Iteration: {}, Loss: {}".format(
                    j, loss.item()
                )
            ) 
        writer.add_scalar("Loss", loss.item(), j)

        if j % 1000 == 0:
            torch.save(model.state_dict(), f"{log_path}/iter_{j}.pth")

    # save
    torch.save(model.state_dict(), f"{log_path}/best_generator.pth")

def eval(log_path, dataset_path, num_functions):
    save_dir = os.path.join(f"{log_path}", "eval_cnn_codes")
    os.makedirs(save_dir, exist_ok=True)

    model = CNNModel(output_funcs=num_functions).to(device)

    model.load_state_dict(torch.load(f"{log_path}/best_generator.pth"))
    model.eval()
    img_path_list = find_images(dataset_path)

    for i in range(len(img_path_list)):
        image_path = img_path_list[i]
        filename = os.path.basename(image_path)
        foldername = os.path.splitext(filename)[0]
        
        input_img = load_image(image_path)
        test_img_tensor = image_to_tensor(input_img).to(device)

        # Forward pass through the model
        with torch.no_grad():
            outputs = model(test_img_tensor)

            # save output ifs codes as dictionaries so that I can use "naive=True in model_inference for evaluation later"
            code_dict = convert_tensor_to_code(outputs)
            write_to_json(code_dict, os.path.join(save_dir, f"{foldername}.json"))


