import pandas as pd
from datasets import Dataset
from AtakapaTokenizer import AtakapaTokenizer

class AtakapaDatasetBuilder:
# ==========================================
# HOW TO USE THIS WITH YOUR EXCEL FILES
# ==========================================
# builder = AtakapaDatasetBuilder()

# 1. Load your files
# texts_df = pd.read_excel('excel_file_name.xlsx')
# examples_df = pd.read_excel('excel_file_name.xlsx')

# 2. Convert to Hugging Face Datasets
# hf_texts = builder.build_from_dataframe(texts_df, atakapa_col='atakapa_column_name', gloss_col='english_column_name')
# hf_examples = builder.build_from_dataframe(examples_df, atakapa_col='atakapa_column_name', gloss_col='english_column_name')

# 3. Combine them into one massive training dataset
# from datasets import concatenate_datasets
# final_training_dataset = concatenate_datasets([hf_texts, hf_examples])

# 4. Pass 'final_training_dataset' directly into the SFTTrainer we wrote earlier!
    def __init__(self):
        # Load your custom tokenizer to ensure perfect train/inference symmetry
        self.tokenizer = AtakapaTokenizer()
        
    def segment_for_training(self, raw_text):
        """Passes training text through your exact tokenizer logic."""
        words = str(raw_text).split()
        segmented_words = []
        for w in words:
            # Using your rollback tokenizer's segmenter
            morphemes = self.tokenizer._segment_word(w)
            segmented_words.append(" ".join(morphemes))
        return " ".join(segmented_words)

    def format_chatml_row(self, segmented_atakapa, english_gloss):
        """Wraps the pair in the exact prompt template the SLM will see in production."""
        # Using the standard ChatML format optimized for Qwen / modern SLMs
        template = (
            f"<|im_start|>user\n"
            f"Gloss this segmented Atakapa: {segmented_atakapa}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{english_gloss}<|im_end|>"
        )
        return template

    def build_from_dataframe(self, df, atakapa_col, gloss_col, is_pre_segmented=False):
        """Converts an Excel dataframe into a Hugging Face Dataset."""
        formatted_texts = []
        
        for _, row in df.iterrows():
            atakapa_text = str(row[atakapa_col]).strip()
            gloss_text = str(row[gloss_col]).strip()
            
            # Skip empty rows
            if not atakapa_text or not gloss_text or atakapa_text == 'nan':
                continue
                
            # If your Excel file already has hyphens/segmentation from the linguist, 
            # we can skip the tokenizer. Otherwise, force it through your tokenizer.
            if not is_pre_segmented:
                atakapa_text = self.segment_for_training(atakapa_text)
                
            chatml_string = self.format_chatml_row(atakapa_text, gloss_text)
            formatted_texts.append(chatml_string)
            
        # Convert to a Hugging Face Dataset object
        return Dataset.from_dict({"text": formatted_texts})

