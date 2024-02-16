from PIL import Image, ImageDraw, ImageFont

# Run inference no one image example
def run_inference(image, model, processor, output_image=True):
    """_summary_

    Args:
        image (JIL): Document image to perform token
        model (_type_): Huggingface model for token classification
        processor (_type_): Huggingface processor
        output_image (bool, optional): Whether or not to display image with bbox and labels.

    Returns:
        _type_: Labels for tokens on image
    """    
    # create model input
    encoding = processor(image, truncation=True ,return_tensors="pt")
    del encoding["pixel_values"]
    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels)
    else:
        return labels


# draw bboxes and labels around tokens on the image
def draw_boxes(image, boxes, predictions):
    
    label2color = {
    "Caption":"red",
    "Footnote":"blue",
    "Formula":"green",
    "List-item":"yellow",
    "Page-footer":"black",
    "Page-header":"red",
    "Picture":"blue",
    "Section-header":"green",
    "Table": "yellow",
    "Text": "black",
    "Title": "blue"
    }
    
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image

# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]