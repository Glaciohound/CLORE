import argparse
from tqdm import tqdm
import clip
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument("--test_classes_file", type=str)
parser.add_argument("--info_type", choices=["names", "definitions"])
parser.add_argument("--class_names_file", type=str)
parser.add_argument("--definitions_file", type=str)
parser.add_argument("--batch_size", default=64)
parser.add_argument("--remove_name", action="store_true")
parser.add_argument("--remove_first_sentence", action="store_true")
args = parser.parse_args()

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
print("loaded model")
dataset = ImageFolder(args.data_dir, transform=preprocess)
print("loaded image folder")
with open(args.test_classes_file, "r") as f:
    test_classes = [_line.strip().split(".")[1].replace("_", " ")
                    for _line in f.readlines()]
with open(args.class_names_file, "r") as f:
    class_names = [_line.strip().split(".")[1].replace("_", " ")
                   for _line in f.readlines()]
is_test_class = torch.BoolTensor([
    _name in test_classes for _name in class_names
]).to(device)
if args.info_type == "names":
    tokenized = clip.tokenize(class_names).to(device)
else:
    with open(args.definitions_file, "r") as f:
        definition_lines = f.readlines()
    definitions = []
    definition_indices = []
    for i in range(200):
        assert definition_lines.pop(0).strip() == str(i+1)
        group = []
        while True and len(definition_lines) > 0:
            sentence = definition_lines.pop(0).strip()
            if sentence != "":
                if args.remove_name:
                    _name = class_names[i]
                    for _part in _name.split(" ")[1:]:
                        sentence = sentence.replace(_part, "bird")
                        sentence = sentence.replace(
                            _part[0].upper()+_part[1:], "bird")
                        sentence = sentence.replace(
                            _part[0].lower()+_part[1:], "bird")
                group.append(sentence)
            else:
                break
        definitions.extend(group)
        definition_indices.extend([int(i)] * len(group))
    tokenized = clip.tokenize(definitions).to(device)
    definition_indices = torch.LongTensor(definition_indices).to(device)

# Calculate features
with torch.no_grad():
    dataloader = DataLoader(
        dataset, args.batch_size, shuffle=False)
    all_image_features = []
    all_labels = []
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        image_features = model.encode_image(images)
        all_image_features.append(image_features)
        all_labels.append(labels)
    all_image_features = torch.cat(all_image_features)
    all_labels = torch.cat(all_labels).to(device)
    text_features = model.encode_text(tokenized)

all_image_features /= all_image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
if args.info_type == "names":
    similarity = (10.0 * all_image_features @ text_features.T).softmax(dim=-1)
else:
    similarity = 10.0 * all_image_features @ text_features.T
    if not args.remove_first_sentence:
        similarity = torch.stack([
            similarity[:, definition_indices == class_i].max(1)[0]
            for class_i in range(200)
        ], 1)
    else:
        similarity_list = []
        for class_i in range(200):
            this_sim = similarity[:, definition_indices == class_i]
            if this_sim.shape[1] > 1:
                this_sim = this_sim[:, 1:]
            similarity_list.append(this_sim.max(1)[0])
        similarity = torch.stack(similarity_list, 1)
    similarity = similarity.softmax(dim=-1)
predictions = similarity.argmax(1)
correctness = predictions == all_labels
class_samples = all_labels[None] == torch.arange(200)[:, None].to(device)
class_acc = (
    (class_samples * correctness).sum(1) /
    class_samples.sum(1).clamp_min(1)
)
unseen_avg_acc = class_acc[is_test_class].mean()
seen_avg_acc = class_acc[is_test_class.logical_not()].mean()
print("seen_acc:", seen_avg_acc, "unseen_acc:", unseen_avg_acc, "acc_H:",
      2*seen_avg_acc*unseen_avg_acc/(seen_avg_acc+unseen_avg_acc))
