import torch
from model import RecursiveNN 
from data import name_to_tree, load_data
from utils import category_from_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

category_lines, all_categories = load_data()
n_categories = len(all_categories)

model = RecursiveNN(input_size=57, hidden_size=256, output_size=n_categories).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(name):
    with torch.no_grad():
        name_tree = name_to_tree(name, device) 
        output = model.classify(name_tree)      
        print("Model output:", output)
        guess, _ = category_from_output(output, all_categories)
        return guess


if __name__ == "__main__":
    while True:
        name = input("Enter a name (or press Enter to quit): ")
        if name == "":
            break
        print(f"Prediction: {predict(name)}")


