import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

class AtakapaSLMTrainer:
    def __init__(self, model_id="Qwen/Qwen2.5-0.5B", morph_rules_file='atakapa_morphology_UTF8.xlsx'):
        self.model_id = model_id
        self.morph_rules = pd.read_excel(morph_rules_file)
        
    def _extract_glosses(self):
        """Extracts unique English glosses from the morphology file."""
        # Assuming your excel file has a column named 'gloss' or 'meaning'
        # Update the column name 'gloss' to match your actual file
        raw_glosses = self.morph_rules['gloss_tag'].astype(str).dropna().unique().tolist()
        
        # Format them as special tokens (e.g., "1SG" -> "[1SG]") 
        # Skip if they are already formatted this way in your sheet
        gloss_tokens = [f"[{g.strip()}]" for g in raw_glosses if g.strip()]
        return gloss_tokens

    def setup_model_and_tokenizer(self, custom_morphemes):
        print(f"Loading {self.model_id}...")
        
        # 1. Load Tokenizer & Extract Glosses
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        gloss_tokens = self._extract_glosses()
        
        # Combine morphemes and glosses into the custom vocabulary
        custom_vocab = custom_morphemes + gloss_tokens
        
        # Add to tokenizer
        num_added = self.tokenizer.add_tokens(custom_vocab)
        print(f"Added {num_added} new tokens (Morphemes + Glosses).")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
        
        # 3. Resize Embeddings to fit the new vocabulary
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # ---------------------------------------------------------
        # 4. APPLY LoRA (PEFT)
        # ---------------------------------------------------------
        peft_config = LoraConfig(
            r=16,               # Rank of the adapter (16 or 32 is great for vocab expansion)
            lora_alpha=32,      # Scaling factor
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Target standard attention layers (Target names vary slightly by model, 
            # these match Llama/Qwen architectures)
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # CRITICAL: Tells PEFT to unfreeze and train the embedding layers!
            modules_to_save=["embed_tokens", "lm_head"] 
        )
        
        # Wrap the model in PEFT
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer

    def train(self, formatted_dataset: Dataset):
        """Initializes the SFTTrainer and starts training."""
        
        # Define Training Hyperparameters
        training_args = SFTConfig(
            output_dir="./atakapa-glosser-model",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,      # Simulate larger batch sizes on small GPUs
            learning_rate=2e-4,                 # 2e-4 is standard for LoRA, but new embeddings might need slightly higher
            logging_steps=10,
            max_steps=500,                      # Adjust based on dataset size
            save_strategy="epoch",
            num_train_epochs=10,
            optim="paged_adamw_8bit",           # Saves VRAM
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            max_length=256, 
            dataset_text_field="text"
        )
        
        # Initialize TRL's Supervised Fine-Tuning Trainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted_dataset,
            peft_config=None, # Already applied via get_peft_model above
            processing_class=self.tokenizer,
            args=training_args,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save the final adapter and embedding weights
        trainer.model.save_pretrained("./atakapa-glosser-final")
        self.tokenizer.save_pretrained("./atakapa-glosser-final")
        print("Training complete and model saved.")