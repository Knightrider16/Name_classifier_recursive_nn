def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_index = top_i[0].item()  # Convert tensor to index
    print(f"Predicted index: {category_index}, Output score: {top_n}")
    
    if category_index >= len(all_categories):
        return "Unknown", category_index
    return all_categories[category_index], category_index
