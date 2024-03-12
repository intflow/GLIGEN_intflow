import json
from tqdm import tqdm

def generate_pseudo_caption(class_names):
    class_descriptions = {
        "person": "person",
        "car": "car",
        "bicycle": "bicycle",
        "dog": "dog",
        # Add more class mappings as needed
    }

    # Count occurrences of each class name
    class_counts = {}
    for class_name in class_names:
        descriptive_name = class_descriptions.get(class_name, class_name)
        if descriptive_name in class_counts:
            class_counts[descriptive_name] += 1
        else:
            class_counts[descriptive_name] = 1

    # Generate descriptive phrases, incorporating counts
    descriptions = []
    for class_name, count in class_counts.items():
        if count > 1:
            descriptions.append(f"{count} {class_name}s")  # Pluralize
        else:
            descriptions.append(f"a {class_name}")

    # Combine descriptions into a sentence
    if len(descriptions) == 1:
        caption = descriptions[0]
    else:
        caption = ", ".join(descriptions[:-1]) + " and " + descriptions[-1]

    return "There is " + caption + " in the image."

def load_json_file(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def add_pseudo_captions_to_annotations(json_data):
    # Assuming 'categories' is a list of category dictionaries with 'id' and 'name'
    category_map = {category['id']: category['name'] for category in json_data['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for annotation in json_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(category_map[category_id])
    
    # Generate captions
    captions = {}
    for image_id in tqdm(image_annotations.keys(), desc="Generating captions"):
        class_names = image_annotations[image_id]
        captions[image_id] = generate_pseudo_caption(class_names)
    return captions

def save_json_with_captions(json_data, captions, output_filepath):
    for image in tqdm(json_data['images'], desc="Updating image data"):
        image_id = image['id']
        if image_id in captions:
            image['caption'] = captions[image_id]
    with open(output_filepath, 'w') as file:
        json.dump(json_data, file, indent=4)

# Example usage
json_filepath = '/data/efc20k/json/annotation.json'
output_json_filepath = '/data/efc20k/json/annotation_caption.json'
json_data = load_json_file(json_filepath)
captions = add_pseudo_captions_to_annotations(json_data)
save_json_with_captions(json_data, captions, output_json_filepath)

# Print captions for the first few images (for demonstration)
for image_id in list(captions.keys())[:5]:
    print(f"Image ID {image_id}: {captions[image_id]}")