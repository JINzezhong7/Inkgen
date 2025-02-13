import os
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import jiwer
from tqdm import tqdm

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")


image_folder = "../repo_results/Iam_first100/pngs"
image_template = "no_priming_{}.png"  


predictions_file = "../repo_results/Iam_first100/predictions.txt"  
output_file = "../repo_results/Iam_first100/output_results_cer.txt"  


cer_list = []
results = []


with open(predictions_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()
    for line in tqdm(lines):
        if "Ground_truth Text:" in line:
            parts = line.split(";")
            image_index = parts[0].replace("Image ", "").strip()  
            ground_truth = parts[1].replace("Ground_truth Text: ", "").strip() 

            image_name = image_template.format(image_index)
            image_path = os.path.join(image_folder, image_name)

            if not os.path.exists(image_path):
                print(f"Image {image_path} does not exist, skipping.")
                continue

            
            image = Image.open(image_path).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values)
            predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            cer = jiwer.cer(ground_truth, predicted_text)
            cer_list.append(cer)

            
            results.append(
                f"Image: {image_name}; Ground Truth: {ground_truth}; Predicted Text: {predicted_text}\CER: {cer:.4f}"
            )

average_cer = sum(cer_list) / len(cer_list) if cer_list else 0

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(f"Average CER: {average_cer:.4f}\n\n")
    outfile.write("Details for each prediction:\n")
    outfile.write("\n".join(results))

