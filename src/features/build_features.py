from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, concatenate_datasets

def build_features(dataset, processor):
    
    # We define a functino to apply on the dataset to process
    def prepare_examples(examples):
        images = examples[image_column_name]
        words = examples[text_column_name]
        boxes = examples[boxes_column_name]
        word_labels = examples[label_column_name]
        
        encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length")
        return encoding
    
    # We get the column names of the features used
    features = dataset.features
    column_names = dataset.column_names
    image_column_name = "image"
    text_column_name = "texts"
    boxes_column_name = "bboxes_block"
    label_column_name = "categories"
    label_list = features[label_column_name].feature.names
    
    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    # Process evaluation dataset
    feat_dataset = dataset.select([i for i in range(len(dataset))]).map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features)
    
    for i in range(len(dataset)//150 + 1):
        feat_dataset_part = dataset.select([i for i in range(i*150, min((i+1)*150, len(dataset))) ]).map(
        prepare_examples,
        batched=True,
        remove_columns=column_names,
        features=features)
    if i == 0:  feat_dataset = feat_dataset_part
    else:       feat_dataset = concatenate_datasets([feat_dataset_part,feat_dataset])
    
    return feat_dataset, label_list
    