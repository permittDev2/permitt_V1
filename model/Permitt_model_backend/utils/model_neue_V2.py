import os
import random
import glob
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Diffusers und Transformers Importe
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel
)
from diffusers.optimization import get_scheduler
# LoRA spezifische Importe - PEFT ist der empfohlene Weg
try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
    print("PEFT-Bibliothek gefunden und wird verwendet.")
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT-Bibliothek NICHT gefunden. Fallback auf manuelle LoRA-Prozessoren (kann fehleranfällig sein).")
    # Fallback, falls PEFT nicht da ist (ältere diffusers oder manuelle Installation)
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor, LoRAAttnAddedKVProcessor

from transformers import CLIPTextModel, CLIPTokenizer


# Standard-Transformation für die Bilder
default_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Stable Diffusion arbeitet oft mit 512x512
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class FloorPlanDataset(Dataset):
    """Dataset für Grundrisse mit zugehörigen Metadaten."""
    def __init__(self, dataframe, image_paths, tokenizer, transform=None, text_template="A floor plan with {} bedrooms, {} bathrooms{}", max_tokenizer_length=77):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_template = text_template
        self.max_tokenizer_length = max_tokenizer_length
        self.image_paths_map = {os.path.basename(path): path for path in image_paths}
        print(f"Dataset init mit {len(self.dataframe)} Einträgen, {len(self.image_paths_map)} Bildern.")

        # DEBUG: Zeige einige Dateinamen zur Überprüfung
        if 'Filename' in self.dataframe.columns:
            print("Beispiele für 'Filename' in CSV:", self.dataframe['Filename'].head().tolist())
        print("Beispiele für gefundene Bilddateinamen (Keys in map):", list(self.image_paths_map.keys())[:5])

        self.dataframe = self.dataframe[self.dataframe['Filename'].isin(self.image_paths_map.keys())].reset_index(drop=True)
        print(f"Nach Filterung (nur existierende Bilder): {len(self.dataframe)} Einträge.")
        if len(self.dataframe) == 0:
            print("WARNUNG: Das Dataset ist nach der Filterung leer! Überprüfen Sie die Dateinamenübereinstimmung zwischen CSV und Bilddateien.")


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if idx >= len(self.dataframe):
            raise IndexError("Index außerhalb des Bereichs nach Filterung.")
        row = self.dataframe.iloc[idx]
        img_filename = row.get('Filename')
        img_path = self.image_paths_map[img_filename] # Sollte jetzt existieren, da gefiltert

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Fehler beim Laden {img_path}: {e}")
            image_size = (512,512)
            image = torch.zeros((3, *image_size)) if self.transform else Image.new('RGB', image_size)

        bedrooms = int(row.get('Bedrooms', 0) if pd.notna(row.get('Bedrooms')) else 0)
        bathrooms = int(row.get('Bathrooms_Toilets', 0) if pd.notna(row.get('Bathrooms_Toilets')) else 0)
        kitchens = int(row.get('Kitchens', 0) if pd.notna(row.get('Kitchens')) else 0)
        living_rooms = int(row.get('Living_Rooms', 0) if pd.notna(row.get('Living_Rooms')) else 0)

        details = []
        if kitchens > 0: details.append("kitchen")
        if living_rooms > 0: details.append("living room")
        # Fügen Sie hier weitere Details hinzu, z.B. "garage", "balcony" etc.
        has_balcony = bool(row.get('Has_Balcony_Terrace', False) if pd.notna(row.get('Has_Balcony_Terrace')) else False)
        if has_balcony: details.append("balcony")
        garages = int(row.get('Garages', 0) if pd.notna(row.get('Garages')) else 0)
        if garages > 0: details.append(f"{garages} garage" + ("s" if garages > 1 else ""))


        details_str = ""
        if details:
            details_str = ", " + ", ".join(details)

        condition_text = self.text_template.format(bedrooms, bathrooms, details_str)

        input_ids = self.tokenizer(
            condition_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokenizer_length,
            return_tensors="pt",
        ).input_ids

        return {
            'image': image,
            'input_ids': input_ids.squeeze(),
            'condition_text': condition_text
        }


def find_metadata_file():
    possible_paths = ['./imgs/output_data.csv'] # Bitte sicherstellen, dass diese CSV die KORREKTEN Dateinamen enthält
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Metadaten-Datei gefunden: {path}")
            return path
    print("WARNUNG: Keine Metadaten-Datei unter den Standardpfaden gefunden.")
    return None

def find_image_directory():
    possible_dirs = ['./imgs/preprocessed_Data_V1/preprocessed_Data_V1'] # Pfad zu Ihren Bildern
    for directory in possible_dirs:
        if os.path.exists(directory):
            print(f"Bildverzeichnis gefunden: {directory}")
            return directory
    print("WARNUNG: Kein Bildverzeichnis unter den Standardpfaden gefunden.")
    return None

def create_dataloaders(tokenizer, metadata_path=None, image_folder=None, batch_size=4, transform=default_transform, val_split=0.1, min_data_for_split=10):
    if metadata_path is None: metadata_path = find_metadata_file()
    if image_folder is None: image_folder = find_image_directory()
    if not metadata_path or not image_folder:
        raise FileNotFoundError("Metadaten oder Bildverzeichnis nicht gefunden.")

    df_metadata = pd.read_csv(metadata_path)
    print(f"Metadaten CSV geladen mit {len(df_metadata)} Zeilen.")
    image_paths = glob.glob(os.path.join(image_folder, "**", "*.png"), recursive=True) + \
                  glob.glob(os.path.join(image_folder, "**", "*.jpg"), recursive=True)
    if not image_paths: raise FileNotFoundError(f"Keine Bilder (.png, .jpg) in {image_folder} gefunden.")
    print(f"Gefunden: {len(image_paths)} Bilder.")

    dataset = FloorPlanDataset(df_metadata, image_paths, tokenizer, transform=transform)
    if len(dataset) == 0:
        print("FATAL: Dataset ist nach Filterung leer. Training nicht möglich. Bitte Dateinamen prüfen!")


    if len(dataset) < min_data_for_split:
        train_dataset, val_dataset = dataset, None
        print(f"WARNUNG: Nur {len(dataset)} samples, kein Validierungssplit.")
    else:
        val_size = int(val_split * len(dataset))
        train_size = len(dataset) - val_size
        if train_size <=0 or val_size <=0:
             train_dataset, val_dataset = dataset, None
             print(f"WARNUNG: Ungültiger Split ({train_size}/{val_size}), kein Validierungssplit.")
        else:
             train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1,
        shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True if torch.cuda.is_available() else False
    )
    val_dataloader = None
    if val_dataset and len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset, batch_size=min(batch_size, len(val_dataset)), shuffle=False,
            num_workers=2, pin_memory=True, persistent_workers=True if torch.cuda.is_available() else False
        )
    return train_dataloader, val_dataloader, dataset


def show_batch(dataloader, num_images=4):
    if not dataloader or len(dataloader) == 0:
        print("DataLoader ist leer, kann keinen Batch anzeigen.")
        return
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("DataLoader ist erschöpft, kann keinen Batch anzeigen.")
        return

    images = batch['image'][:num_images]
    texts = batch['condition_text'][:num_images]
    actual_num_images = images.shape[0]
    if actual_num_images == 0:
        print("Batch enthält keine Bilder.")
        return

    fig, axes = plt.subplots(1, actual_num_images, figsize=(3 * actual_num_images, 6))
    if actual_num_images == 1: axes = [axes]

    for i in range(actual_num_images):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        axes[i].imshow(img)
        axes[i].set_title(texts[i], fontsize=7)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


class FloorPlanDiffusionLoRA:
    def __init__(self, device, pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5", lora_rank=4):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.lora_rank = lora_rank
        self.image_size = 512

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

        self.vae_scale_factor = self.vae.config.scaling_factor

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False) # Grundsätzlich alles frieren

        print(f"Versuche LoRA-Adapter mit PEFT (Rank: {self.lora_rank})...")
        if PEFT_AVAILABLE:
            # Bereite das Modell für das Training mit geringerer Präzision vor, falls gewünscht (optional, aber gut für Speicher)
            # if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            #     self.unet = prepare_model_for_kbit_training(self.unet, use_gradient_checkpointing=True) # use_gradient_checkpointing ist gut für Speicher
            # else:
            #     print("BF16 nicht unterstützt oder kein CUDA, kein kbit Training.")
            # Gradient Checkpointing kann auch ohne kbit Training nützlich sein
            self.unet.enable_gradient_checkpointing()


            # Zielmodule: Dies sind typische Namen für lineare Layer in Attention-Blöcken von SD-UNets.
            # Es kann sein, dass Sie diese anpassen müssen, wenn Sie ein anderes Basis-UNet verwenden.
            # Untersuchen Sie `self.unet.named_modules()` für die genauen Namen.
            target_modules = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"]
            # Manchmal auch spezifischer: z.B. 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q'
            # Eine generischere (aber potenziell weniger präzise) Methode wäre, nach Modultypen zu filtern.

            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank, # Oft gleich r oder 2*r
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none", # "none", "all", oder "lora_only"
                # task_type="CAUSAL_LM" # Für Textmodelle, für Diffusionsmodelle oft nicht nötig oder anders
            )
            # Prüfen ob unet schon ein PEFT Modell ist, oder `add_adapter` Methode hat.
            if hasattr(self.unet, 'add_adapter') and not isinstance(self.unet, type(get_peft_model(self.unet, lora_config))): # Einfaches UNet
                print("Verwende unet.add_adapter()")
                self.unet.add_adapter(lora_config, adapter_name="floorplan_adapter")
            else: # Entweder schon PEFT oder add_adapter nicht da, dann mit get_peft_model umwickeln
                print("Verwende get_peft_model() aus der PEFT-Bibliothek.")
                self.unet = get_peft_model(self.unet, lora_config)

            # PEFT sollte die requires_grad Flags korrekt setzen.
            # self.unet.print_trainable_parameters() # Nützlich zum Debuggen
        else: # Fallback, wenn PEFT nicht verfügbar ist
            print("PEFT nicht verfügbar. Fallback auf manuelle LoRAAttnProcessor Zuweisung.")
            unet_lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
                # Bestimme hidden_size basierend auf dem Block
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                else: continue # Nicht relevanter Prozessor

                if cross_attention_dim is not None:
                    # Aktualisierte Implementierung für neuere diffusers-Versionen
                    unet_lora_attn_procs[name] = LoRAAttnProcessor()
                else:
                    # Aktualisierte Implementierung für neuere diffusers-Versionen
                    unet_lora_attn_procs[name] = LoRAAttnProcessor()

                # Manuelles Erstellen der LoRA-Gewichte mit dem richtigen Rang
                if isinstance(self.unet.attn_processors[name], LoRAAttnProcessor) or isinstance(self.unet.attn_processors[name], LoRAAttnAddedKVProcessor):
                    attn_processor = self.unet.attn_processors[name]
                    hidden_size = processor.processor.hidden_size
                    cross_attention_dim = processor.processor.cross_attention_dim if hasattr(processor.processor, "cross_attention_dim") else None

                    # Manuelles Erstellen der LoRA-Gewichte
                    if hasattr(attn_processor, "to_q"):
                        attn_processor.to_q_lora = torch.nn.Linear(hidden_size, self.lora_rank, bias=False)
                        attn_processor.to_q_lora_weight = torch.nn.Parameter(torch.zeros(hidden_size, self.lora_rank))
                        attn_processor.to_k_lora = torch.nn.Linear(hidden_size, self.lora_rank, bias=False)
                        attn_processor.to_k_lora_weight = torch.nn.Parameter(torch.zeros(hidden_size, self.lora_rank))
                        attn_processor.to_v_lora = torch.nn.Linear(hidden_size, self.lora_rank, bias=False)
                        attn_processor.to_v_lora_weight = torch.nn.Parameter(torch.zeros(hidden_size, self.lora_rank))
                        attn_processor.to_out_lora = torch.nn.Linear(hidden_size, self.lora_rank, bias=False)
                        attn_processor.to_out_lora_weight = torch.nn.Parameter(torch.zeros(hidden_size, self.lora_rank))

                        # Initialisierung
                        torch.nn.init.kaiming_uniform_(attn_processor.to_q_lora_weight)
                        torch.nn.init.kaiming_uniform_(attn_processor.to_k_lora_weight)
                        torch.nn.init.kaiming_uniform_(attn_processor.to_v_lora_weight)
                        torch.nn.init.zeros_(attn_processor.to_out_lora_weight)

            self.unet.set_attn_processor(unet_lora_attn_procs)
            # Manuell trainierbare Parameter setzen, wenn set_attn_processor verwendet wird
            for name, module in self.unet.named_modules():
                if isinstance(module, (LoRAAttnProcessor, LoRAAttnAddedKVProcessor)):
                    for param_name, param_val in module.named_parameters():
                        param_val.requires_grad = True # Mache alle Parameter im LoRA-Prozessor trainierbar

        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

        self.text_encoder.to(device)
        self.vae.to(device)
        self.unet.to(device)

        os.makedirs("./models_lora_V1", exist_ok=True)
        os.makedirs("./results_lora_V1", exist_ok=True)

        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.unet.parameters())
        print(f"LoRA Rank: {self.lora_rank}")
        print(f"Anzahl trainierbarer Parameter im UNet (LoRA): {trainable_params} (von {all_params} insgesamt, {(100 * trainable_params / all_params):.2f}%)")
        if trainable_params == 0:
            print("WARNUNG: Keine trainierbaren LoRA-Parameter gefunden! Überprüfen Sie die LoRA-Konfiguration, Zielmodule oder den Fallback-Pfad.")


    def get_text_embeddings(self, text_prompts):
        if not isinstance(text_prompts, list): text_prompts = [text_prompts]
        text_input_ids = self.tokenizer(
            text_prompts, padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids.to(self.device)
        text_embeddings = self.text_encoder(text_input_ids)[0]
        return text_embeddings

    def train(self, train_dataloader, val_dataloader=None, num_epochs=100, lr=1e-4,
              save_checkpoint_every_n_steps=200, generate_samples_every_n_steps=100,
              gradient_accumulation_steps=1, weight_decay=1e-2, use_cfg_during_training_prob=0.0):

        lora_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        if not lora_params:
            print("FEHLER: Keine trainierbaren LoRA-Parameter für den Optimizer gefunden. Training kann nicht starten.")
            return

        optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
        num_training_steps_for_scheduler = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps

        lr_scheduler = get_scheduler(
            "cosine", optimizer=optimizer,
            num_warmup_steps=int(0.05 * num_training_steps_for_scheduler), # 5% Warmup
            num_training_steps=num_training_steps_for_scheduler,
        )

        global_step = 0
        for epoch in range(num_epochs):
            self.unet.train()
            self.text_encoder.eval()
            self.vae.eval()
            epoch_loss = 0.0
            progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoche {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)

                with torch.no_grad():
                    encoder_hidden_states = self.text_encoder(input_ids)[0]

                if use_cfg_during_training_prob > 0 and random.random() < use_cfg_during_training_prob:
                    uncond_input_ids = self.tokenizer(
                        [""] * input_ids.shape[0], padding="max_length", truncation=True,
                        max_length=self.tokenizer.model_max_length, return_tensors="pt"
                    ).input_ids.to(self.device)
                    with torch.no_grad():
                        uncond_embeddings = self.text_encoder(uncond_input_ids)[0]
                    # Wähle zufällig für einige Samples im Batch die unkonditionierten Embeddings
                    # Hier wird ein kleiner Prozentsatz (z.B. 10%) des Batches unkonditioniert trainiert, wenn use_cfg_during_training_prob=0.1
                    # Für dieses Beispiel: Wenn der Wurf < prob, dann alle im Batch unkond.
                    # Besser: eine Maske erzeugen, um nur einige Samples unkonditioniert zu machen.
                    # Vereinfachung hier: wenn der Wurf zutrifft, wird der *gesamte* Batch-Teil unkonditioniert.
                    # Realistischer:
                    # if random.random() < 0.5 : # 50% der Zeit, wenn CFG-Training aktiv ist
                    #    encoder_hidden_states = uncond_embeddings
                    encoder_hidden_states = uncond_embeddings # Einfache Version für dieses Beispiel


                with torch.no_grad():
                    model_input = self.vae.encode(clean_images).latent_dist.sample()
                latents = model_input * self.vae_scale_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                loss = loss / gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item() * gradient_accumulation_steps
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_steps, lr=lr_scheduler.get_last_lr()[0])
                global_step += 1

                if global_step % save_checkpoint_every_n_steps == 0:
                    self.save_lora_checkpoint(f"./models_lora_V1/floor_plan_lora_step_{global_step}.pt")
                if global_step % generate_samples_every_n_steps == 0:
                    print(f"\nGeneriere Samples bei Schritt {global_step}...")
                    sample_prompts = [
                        "A floor plan with 1 bedroom, 1 bathroom",
                        "A floor plan with 3 bedrooms, 2 bathrooms, kitchen, living room, balcony",
                    ]
                    self.generate_samples(text_prompts=sample_prompts, save_path=f"./results_lora_V1/samples_step_{global_step}.png")
                    self.unet.train()

            progress_bar.close()
            avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
            print(f"Epoche {epoch + 1} abgeschlossen. Avg Loss: {avg_epoch_loss:.4f}")

            if val_dataloader and len(val_dataloader) > 0:
                self.unet.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_val in tqdm(val_dataloader, desc="Validierung"):
                        clean_images_val = batch_val["image"].to(self.device)
                        input_ids_val = batch_val["input_ids"].to(self.device)
                        encoder_hidden_states_val = self.text_encoder(input_ids_val)[0]
                        model_input_val = self.vae.encode(clean_images_val).latent_dist.sample()
                        latents_val = model_input_val * self.vae_scale_factor
                        noise_val = torch.randn_like(latents_val)
                        timesteps_val = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents_val.shape[0],), device=self.device).long()
                        noisy_latents_val = self.noise_scheduler.add_noise(latents_val, noise_val, timesteps_val)
                        model_pred_val = self.unet(noisy_latents_val, timesteps_val, encoder_hidden_states=encoder_hidden_states_val).sample
                        loss_val = F.mse_loss(model_pred_val, noise_val)
                        val_loss += loss_val.item()
                avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
                print(f"Validierungsverlust nach Epoche {epoch + 1}: {avg_val_loss:.4f}")
            self.save_lora_checkpoint(f"./models_lora_V1/floor_plan_lora_epoch_{epoch + 1}.pt")
        print("LoRA Training abgeschlossen.")
        self.save_lora_checkpoint(f"./models_lora_V1/floor_plan_lora_final.pt")

    def save_lora_checkpoint(self, path):
        lora_state_dict = {}
        # Wenn PEFT verwendet wird, gibt es eine spezielle Methode zum Speichern
        if PEFT_AVAILABLE and hasattr(self.unet, 'save_pretrained'): # Check ob es ein PEFT Model ist
            try:
                # PEFT Modelle speichern das gesamte Modell, wir wollen aber nur die LoRA Gewichte
                # Besser ist, die state_dict der LoRA Layer manuell zu extrahieren
                # oder `get_peft_model_state_dict` zu verwenden
                from peft import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(self.unet)
                torch.save(lora_state_dict, path)
                print(f"PEFT LoRA Checkpoint (nur Adapter) gespeichert: {path}")
                return
            except Exception as e:
                print(f"Fehler beim Speichern mit get_peft_model_state_dict: {e}. Fallback auf manuelle Extraktion.")
        # Fallback oder wenn nicht PEFT get_peft_model verwendet wurde
        for name, param in self.unet.named_parameters():
            if param.requires_grad:
                lora_state_dict[name] = param.data.cpu().clone()
        if lora_state_dict:
            torch.save(lora_state_dict, path)
            print(f"LoRA Checkpoint (manuell extrahiert) gespeichert: {path}")
        else:
            print("WARNUNG: Keine trainierbaren Parameter zum Speichern im LoRA Checkpoint gefunden.")


    def load_lora_checkpoint(self, path, strict_loading=False):
        print(f"Lade LoRA Gewichte von: {path}")
        lora_state_dict = torch.load(path, map_location=self.device)

        if PEFT_AVAILABLE and hasattr(self.unet, 'load_adapter'): # Für PEFT Modelle
            try:
                # Wenn der Checkpoint nur den Adapter state_dict enthält (wie von get_peft_model_state_dict)
                # und das Basismodell schon da ist.
                # Manchmal muss man das Basismodell neu laden und dann den Adapter.
                # Oder, wenn das UNet schon ein PEFT-Modell ist:
                self.unet.load_state_dict(lora_state_dict, strict=strict_loading)
                print("LoRA Gewichte in PEFT-Modell geladen.")
                self.unet.to(self.device)
                return
            except Exception as e:
                print(f"Fehler beim direkten Laden in PEFT-Modell: {e}. Versuche Fallback.")

        # Fallback oder wenn nicht PEFT `get_peft_model` verwendet wurde, oder wenn `load_adapter` nicht direkt passt
        incompatible_keys = self.unet.load_state_dict(lora_state_dict, strict=strict_loading)
        if hasattr(incompatible_keys, 'missing_keys') and incompatible_keys.missing_keys:
            print(f"Warnung: Fehlende Schlüssel beim Laden des LoRA Checkpoints: {incompatible_keys.missing_keys}")
        if hasattr(incompatible_keys, 'unexpected_keys') and incompatible_keys.unexpected_keys:
            print(f"Warnung: Unerwartete Schlüssel beim Laden des LoRA Checkpoints: {incompatible_keys.unexpected_keys}")
        print(f"LoRA Gewichte (manuell) geladen von: {path}")
        self.unet.to(self.device)


    @torch.no_grad()
    def generate_samples(self, text_prompts, num_samples_per_prompt=1, num_inference_steps=50, guidance_scale=7.5, save_path=None, seed=None):
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()

        if isinstance(text_prompts, str): text_prompts = [text_prompts]
        total_num_samples = len(text_prompts) * num_samples_per_prompt

        prompt_embeds = self.get_text_embeddings(text_prompts)
        uncond_prompts = [""] * len(text_prompts)
        uncond_embeds = self.get_text_embeddings(uncond_prompts)

        if num_samples_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_samples_per_prompt, dim=0)
            uncond_embeds = uncond_embeds.repeat_interleave(num_samples_per_prompt, dim=0)

        text_embeddings_for_cfg = torch.cat([uncond_embeds, prompt_embeds])

        # Latent shape
        latent_channels = self.vae.config.latent_channels
        # VAE downsampling factor ist 2^(anzahl_down_blocks - 1)
        # SD VAE hat typischerweise einen Faktor von 8 (2^3), was zu 512/8 = 64 Latents führt.
        # len(self.vae.config.block_out_channels) ist die Anzahl der Stufen im Encoder/Decoder.
        # Die tatsächliche Anzahl der Downsampling-Operationen ist oft len(block_out_channels) - 1
        downscale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        height = self.image_size // downscale_factor
        width = self.image_size // downscale_factor

        # Generator für reproduzierbare Ergebnisse
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed() # Zufälliger Seed

        latents = torch.randn(
            (total_num_samples, latent_channels, height, width),
            generator=generator, device=self.device, dtype=prompt_embeds.dtype # dtype anpassen
        )
        latents = latents * self.noise_scheduler.init_noise_sigma

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.noise_scheduler.timesteps

        for t in tqdm(timesteps_tensor, desc="Sampling"):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(sample=latent_model_input, timestep=t, encoder_hidden_states=text_embeddings_for_cfg).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.noise_scheduler.step(noise_pred_cfg, t, latents).prev_sample

        # Cast zu float32 vor dem Dekodieren, falls VAE das erwartet und Latents z.B. float16 sind
        if self.vae.dtype == torch.float32 and latents.dtype != torch.float32:
            latents = latents.to(torch.float32)

        images = self.vae.decode(latents / self.vae_scale_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images_np = images.cpu().permute(0, 2, 3, 1).numpy()

        num_generated_images = images_np.shape[0]
        cols = min(num_generated_images, 4) # Max 4 Bilder pro Zeile
        rows = (num_generated_images -1) // cols + 1
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.5 * rows), squeeze=False)
        axes_flat = axes.flatten()

        for i, img_np in enumerate(images_np):
            prompt_idx = i // num_samples_per_prompt
            title_text = text_prompts[prompt_idx]
            if num_samples_per_prompt > 1: title_text += f" (S {i % num_samples_per_prompt + 1})"
            axes_flat[i].imshow(img_np)
            axes_flat[i].set_title(title_text, fontsize=8)
            axes_flat[i].axis('off')
        for j in range(i + 1, len(axes_flat)): # Verbleibende Achsen ausblenden
            axes_flat[j].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Beispielbilder gespeichert: {save_path}")
        # plt.show()
        return images_np


if __name__ == "__main__":
    print("Starte Floor Plan Diffusion LoRA Fine-Tuning Skript...")

    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    LORA_RANK = 8  # WICHTIG: Dies MUSS der gleiche Rank sein, mit dem ein geladenes Modell trainiert wurde!
    BATCH_SIZE = 2
    NUM_EPOCHS = 10  # Anzahl der Epochen für dieses Training
    LEARNING_RATE = 5e-5  # Kleinere Lernrate für das Weitertrainieren ist oft gut
    WEIGHT_DECAY = 1e-2
    IMAGE_SIZE_TRANSFORM = (512,512)
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_CFG_IN_TRAINING_PROB = 0.1

    METADATA_FILE_PATH = None  # Wird automatisch gesucht, falls None
    IMAGE_FOLDER_PATH = None   # Wird automatisch gesucht, falls None

    # Zielverzeichnisse für Modelle und Ergebnisse
    MODELS_OUTPUT_DIR = "./models_lora_V1"  # Hier werden neue Modelle gespeichert
    RESULTS_OUTPUT_DIR = "./results_lora_V1"
    # Verzeichnis, in dem nach vorhandenen LoRA-Modellen zum Fortsetzen gesucht wird
    LORA_CHECKPOINT_SEARCH_DIR = "models_lora_V1"

    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Verwende Gerät: {device}")

        print("\nInitialisiere LoRA Diffusionsmodell und Tokenizer...")
        lora_diffusion_model = FloorPlanDiffusionLoRA(
            device=device,
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            lora_rank=LORA_RANK
        )
        tokenizer_for_dataset = lora_diffusion_model.tokenizer

        print("Erstelle DataLoaders...")
        current_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_TRANSFORM),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dl, val_dl, full_dataset = create_dataloaders(
            tokenizer=tokenizer_for_dataset,
            metadata_path=METADATA_FILE_PATH, image_folder=IMAGE_FOLDER_PATH,
            batch_size=BATCH_SIZE, transform=current_transform, val_split=0.1
        )
        print(f"Dataset erfolgreich erstellt mit {len(full_dataset)} Einträgen.")
        if train_dl: print(f"Trainingsdaten: {len(train_dl.dataset)} ({len(train_dl)} Batches)")
        if val_dl: print(f"Validierungsdaten: {len(val_dl.dataset)} ({len(val_dl)} Batches)")

        if train_dl and len(train_dl.dataset) > 0:
            print("\nZeige einen Beispiel-Batch aus dem Trainings-DataLoader:")
            show_batch(train_dl, num_images=min(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 4))
        else:
            print("Keine Daten im Trainings-DataLoader. Training nicht möglich.")
            if len(full_dataset) == 0: exit()

        # <<< --- AUTOMATISCHES FINDEN UND LADEN DES NEUESTEN CHECKPOINTS --- >>>
        checkpoint_to_load = None

        if os.path.exists(LORA_CHECKPOINT_SEARCH_DIR):
            pt_files = glob.glob(os.path.join(LORA_CHECKPOINT_SEARCH_DIR, "*.pt"))
            if not pt_files:
                pt_files = glob.glob(os.path.join(LORA_CHECKPOINT_SEARCH_DIR, "*.safetensors"))

            if pt_files:
                try:
                    latest_checkpoint = max(pt_files, key=os.path.getmtime)
                    checkpoint_to_load = latest_checkpoint
                    print(f"Neuester Checkpoint gefunden (nach Änderungsdatum): {checkpoint_to_load}")
                except Exception as e:
                    print(f"Fehler beim Ermitteln des neuesten Checkpoints per Datum: {e}. Versuche Fallbacks.")
                    final_checkpoints = [f for f in pt_files if "final" in os.path.basename(f).lower()]
                    if final_checkpoints:
                        checkpoint_to_load = max(final_checkpoints, key=os.path.getmtime)
                        print(f"Fallback: Neuester 'final' Checkpoint gefunden: {checkpoint_to_load}")
                    else:
                        pt_files.sort()
                        if pt_files:
                            checkpoint_to_load = pt_files[-1]
                            print(f"Fallback: Alphabetisch letzter Checkpoint gefunden: {checkpoint_to_load}")
            else:
                print(f"Keine .pt oder .safetensors Dateien im Verzeichnis {LORA_CHECKPOINT_SEARCH_DIR} gefunden.")
        else:
            print(f"LoRA Modellverzeichnis {LORA_CHECKPOINT_SEARCH_DIR} zum Laden nicht gefunden.")

        # Lade den Checkpoint, WENN einer gefunden wurde
        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            print(f"Lade LoRA Checkpoint: {checkpoint_to_load}")
            lora_diffusion_model.load_lora_checkpoint(checkpoint_to_load, strict_loading=False)
        else:
            if checkpoint_to_load:
                print(f"WARNUNG: Referenzierter Checkpoint {checkpoint_to_load} existiert nicht. Starte neues Training.")
            elif os.path.exists(LORA_CHECKPOINT_SEARCH_DIR) and not pt_files:
                print(f"Keine .pt/.safetensors Checkpoints im Verzeichnis {LORA_CHECKPOINT_SEARCH_DIR} gefunden. Starte neues Training.")
            elif not os.path.exists(LORA_CHECKPOINT_SEARCH_DIR):
                print(f"LoRA Modellverzeichnis {LORA_CHECKPOINT_SEARCH_DIR} zum Laden nicht gefunden. Starte neues Training.")
            else:
                print("Kein passender LoRA Checkpoint zum Laden gefunden oder angegeben. Starte neues Training von Grund auf.")
        # <<< --- ENDE DES AUTOMATISCHEN FINDENS --- >>>

        print("\nStarte LoRA Fine-Tuning...")
        lora_diffusion_model.train(
            train_dataloader=train_dl, val_dataloader=val_dl,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            save_checkpoint_every_n_steps=200,
            generate_samples_every_n_steps=100,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            use_cfg_during_training_prob=USE_CFG_IN_TRAINING_PROB
        )

        print("\nGeneriere finale Beispielbilder nach dem LoRA Training...")
        final_prompts = [
            "A floor plan with 1 bedroom, 1 bathroom",
            "A floor plan with 2 bedrooms, 1 bathroom, kitchen",
            "A floor plan with 3 bedrooms, 2 bathrooms, kitchen, living room",
            "A floor plan with 4 bedrooms, 2 bathrooms, kitchen, living room, garage, balcony"
        ]
        lora_diffusion_model.generate_samples(
            text_prompts=final_prompts, num_samples_per_prompt=2,
            num_inference_steps=50, guidance_scale=7.5,
            save_path=os.path.join(RESULTS_OUTPUT_DIR, "final_generated_lora_samples.png"),
            seed=42
        )
        print("\nSkript erfolgreich abgeschlossen.")

    except FileNotFoundError as e:
        print(f"FEHLER: Datei oder Verzeichnis nicht gefunden: {e}")
    except ValueError as e:
        print(f"FEHLER: Wertfehler aufgetreten: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
    print("Starte Floor Plan Diffusion LoRA Fine-Tuning Skript...")

    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    LORA_RANK = 8  # WICHTIG: Dies MUSS der gleiche Rank sein, mit dem ein geladenes Modell trainiert wurde!
    BATCH_SIZE = 2
    NUM_EPOCHS = 10 # Anzahl der Epochen für dieses Training (ggf. anpassen für Weitertrainieren)
    LEARNING_RATE = 5e-5 # Kleinere Lernrate für das Weitertrainieren ist oft gut
    WEIGHT_DECAY = 1e-2
    IMAGE_SIZE_TRANSFORM = (512,512)
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_CFG_IN_TRAINING_PROB = 0.1

    METADATA_FILE_PATH = None # Wird automatisch gesucht, falls None
    IMAGE_FOLDER_PATH = None  # Wird automatisch gesucht, falls None

    # Zielverzeichnisse für Modelle und Ergebnisse
    MODELS_OUTPUT_DIR = "./models_lora_V1" # Hier werden neue Modelle gespeichert
    RESULTS_OUTPUT_DIR = "./results_lora_V1"
    # Verzeichnis, in dem nach vorhandenen LoRA-Modellen zum Fortsetzen gesucht wird
    LORA_CHECKPOINT_SEARCH_DIR = "models_lora_V1"


    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Verwende Gerät: {device}")

        print("\nInitialisiere LoRA Diffusionsmodell und Tokenizer...")
        lora_diffusion_model = FloorPlanDiffusionLoRA(
            device=device,
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            lora_rank=LORA_RANK # Dieser Rank wird verwendet, um die LoRA-Layer zu initialisieren
        )
        tokenizer_for_dataset = lora_diffusion_model.tokenizer

        print("Erstelle DataLoaders...")
        current_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_TRANSFORM),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dl, val_dl, full_dataset = create_dataloaders(
            tokenizer=tokenizer_for_dataset,
            metadata_path=METADATA_FILE_PATH, image_folder=IMAGE_FOLDER_PATH,
            batch_size=BATCH_SIZE, transform=current_transform, val_split=0.1
        )
        print(f"Dataset erfolgreich erstellt mit {len(full_dataset)} Einträgen.")
        if train_dl: print(f"Trainingsdaten: {len(train_dl.dataset)} ({len(train_dl)} Batches)")
        if val_dl: print(f"Validierungsdaten: {len(val_dl.dataset)} ({len(val_dl)} Batches)")

        if train_dl and len(train_dl.dataset) > 0:
            print("\nZeige einen Beispiel-Batch aus dem Trainings-DataLoader:")
            show_batch(train_dl, num_images=min(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 4))
        else:
            print("Keine Daten im Trainings-DataLoader. Training nicht möglich.")
            if len(full_dataset) == 0 : exit()


        # <<< --- AUTOMATISCHES FINDEN UND LADEN DES NEUESTEN CHECKPOINTS --- >>>
        checkpoint_to_load = None

        if os.path.exists(LORA_CHECKPOINT_SEARCH_DIR):
            pt_files = glob.glob(os.path.join(LORA_CHECKPOINT_SEARCH_DIR, "*.pt"))
            if not pt_files:
                 pt_files = glob.glob(os.path.join(LORA_CHECKPOINT_SEARCH_DIR, "*.safetensors")) # Auch nach safetensors suchen

            if pt_files:
                try:
                    # Sortiere nach Änderungsdatum (neueste zuerst)
                    latest_checkpoint = max(pt_files, key=os.path.getmtime)
                    checkpoint_to_load = latest_checkpoint
                    print(f"Neuester Checkpoint gefunden (nach Änderungsdatum): {checkpoint_to_load}")
                except Exception as e:
                    print(f"Fehler beim Ermitteln des neuesten Checkpoints per Datum: {e}. Versuche Fallbacks.")
                    # Fallback 1: Suche nach "final" im Namen
                    final_checkpoints = [f for f in pt_files if "final" in os.path.basename(f).lower()]
                    if final_checkpoints:
                        # Wenn mehrere "final" da sind, nimm den neuesten davon
                        checkpoint_to_load = max(final_checkpoints, key=os.path.getmtime)
                        print(f"Fallback: Neuester 'final' Checkpoint gefunden: {checkpoint_to_load}")
                    else:
                        # Fallback 2: Nimm den alphabetisch letzten (funktioniert bei step/epoch im Namen)
                        pt_files.sort()
                        if pt_files:
                            checkpoint_to_load = pt_files[-1]
                            print(f"Fallback: Alphabetisch letzter Checkpoint gefunden: {checkpoint_to_load}")
            else:
                print(f"Keine .pt oder .safetensors Dateien im Verzeichnis {LORA_CHECKPOINT_SEARCH_DIR} gefunden.")
        else:
            print(f"LoRA Modellverzeichnis {LORA_CHECKPOINT_SEARCH_DIR} zum Laden nicht gefunden.")

        # Lade den Checkpoint, WENN einer gefunden wurde
        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            print(f"Lade LoRA Checkpoint: {checkpoint_to_load}")
            lora_diffusion_model.load_lora_checkpoint(checkpoint_to_load, strict_loading=False)
        else:
            if checkpoint_to_load: # checkpoint_to_load war gesetzt, aber Datei existiert nicht (sollte nicht passieren)
                print(f"WARNUNG: Referenzierter Checkpoint {checkpoint_to_load} existiert nicht. Starte neues Training.")
            elif os.path.exists(LORA_CHECKPOINT_SEARCH_DIR) and not pt_files:
                 print(f"Keine .pt/.safetensors Checkpoints im Verzeichnis {LORA_CHECKPOINT_SEARCH_DIR} gefunden. Starte neues Training.")
            elif not os.path.exists(LORA_CHECKPOINT_SEARCH_DIR):
                 print(f"LoRA Modellverzeichnis {LORA_CHECKPOINT_SEARCH_DIR} zum Laden nicht gefunden. Starte neues Training.")
            else: # Allgemeiner Fall
                print("Kein passender LoRA Checkpoint zum Laden gefunden oder angegeben. Starte neues Training von Grund auf.")
        # <<< --- ENDE DES AUTOMATISCHEN FINDENS --- >>>


        print("\nStarte LoRA Fine-Tuning...")
        lora_diffusion_model.train(
            train_dataloader=train_dl, val_dataloader=val_dl,
            num_epochs=NUM_EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            save_checkpoint_every_n_steps=200, # Speichert in MODELS_OUTPUT_DIR
            generate_samples_every_n_steps=100, # Speichert in RESULTS_OUTPUT_DIR
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            use_cfg_during_training_prob=USE_CFG_IN_TRAINING_PROB
        )

        print("\nGeneriere finale Beispielbilder nach dem LoRA Training...")
        final_prompts = [
            "A floor plan with 1 bedroom, 1 bathroom",
            "A floor plan with 2 bedrooms, 1 bathroom, kitchen",
            "A floor plan with 3 bedrooms, 2 bathrooms, kitchen, living room",
            "A floor plan with 4 bedrooms, 2 bathrooms, kitchen, living room, garage, balcony"
        ]
        lora_diffusion_model.generate_samples(
            text_prompts=final_prompts, num_samples_per_prompt=2,
            num_inference_steps=50, guidance_scale=7.5,
            save_path=os.path.join(RESULTS_OUTPUT_DIR, "final_generated_lora_samples.png"),
            seed=42
        )
        print("\nSkript erfolgreich abgeschlossen.")

    except FileNotFoundError as e:
        print(f"FEHLER: Datei oder Verzeichnis nicht gefunden: {e}")
    except ValueError as e:
        print(f"FEHLER: Wertfehler aufgetreten: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
    print("Starte Floor Plan Diffusion LoRA Fine-Tuning Skript...")

    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    LORA_RANK = 8  # WICHTIG: Dies MUSS der gleiche Rank sein, mit dem das zu ladende Modell trainiert wurde!
    BATCH_SIZE = 2
    NUM_EPOCHS = 10 # Du kannst hier weniger Epochen für das Weitertrainieren einstellen, z.B. 10 weitere
    LEARNING_RATE = 5e-5 # Eventuell eine kleinere Lernrate für das Weitertrainieren
    WEIGHT_DECAY = 1e-2
    IMAGE_SIZE_TRANSFORM = (512,512)
    GRADIENT_ACCUMULATION_STEPS = 2
    USE_CFG_IN_TRAINING_PROB = 0.1


    METADATA_FILE_PATH = None # Stelle sicher, dass dies auf deine CSV-Datei zeigt oder None bleibt, damit sie gesucht wird
    IMAGE_FOLDER_PATH = None  # Stelle sicher, dass dies auf deinen Bildordner zeigt oder None bleibt, damit er gesucht wird

    os.makedirs("./models_lora_V1", exist_ok=True)
    os.makedirs("./results_lora_V1", exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Verwende Gerät: {device}")

        print("\nInitialisiere LoRA Diffusionsmodell und Tokenizer...")
        lora_diffusion_model = FloorPlanDiffusionLoRA(
            device=device,
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            lora_rank=LORA_RANK # Dieser Rank wird verwendet, um die LoRA-Layer zu initialisieren
        )
        tokenizer_for_dataset = lora_diffusion_model.tokenizer

        print("Erstelle DataLoaders...")
        current_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_TRANSFORM),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dl, val_dl, full_dataset = create_dataloaders(
            tokenizer=tokenizer_for_dataset,
            metadata_path=METADATA_FILE_PATH, image_folder=IMAGE_FOLDER_PATH,
            batch_size=BATCH_SIZE, transform=current_transform, val_split=0.1
        )
        print(f"Dataset erfolgreich erstellt mit {len(full_dataset)} Einträgen.")
        if train_dl: print(f"Trainingsdaten: {len(train_dl.dataset)} ({len(train_dl)} Batches)")
        if val_dl: print(f"Validierungsdaten: {len(val_dl.dataset)} ({len(val_dl)} Batches)")

        if train_dl and len(train_dl.dataset) > 0:
            print("\nZeige einen Beispiel-Batch aus dem Trainings-DataLoader:")
            show_batch(train_dl, num_images=min(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 4))
        else:
            print("Keine Daten im Trainings-DataLoader. Training nicht möglich.")
            if len(full_dataset) == 0 : exit()


        # <<< --- HIER IST DIE WICHTIGE ÄNDERUNG --- >>>
        # Optional: LoRA Checkpoint laden
        # Passe den Pfad zu deinem spezifischen Checkpoint an.
        # Stelle sicher, dass Backslashes in Pfaden unter Windows entweder verdoppelt (\\) oder
        # als Raw-String (r"...") oder mit Slashes (/) geschrieben werden.
        checkpoint_to_load = "models_V1/floor_plan_diffusion_step_10000_1.pt"

        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            print(f"Lade LoRA Checkpoint: {checkpoint_to_load}")

            lora_diffusion_model.load_lora_checkpoint(checkpoint_to_load, strict_loading=False) # strict_loading=False ist oft robuster
        else:
            if checkpoint_to_load:
                print(f"WARNUNG: Checkpoint {checkpoint_to_load} nicht gefunden. Starte neues Training.")
            else:
                print("Kein LoRA Checkpoint zum Laden angegeben, starte neues Training.")
        # <<< --- ENDE DER WICHTIGEN ÄNDERUNG --- >>>


        print("\nStarte LoRA Fine-Tuning...")
        lora_diffusion_model.train(
            train_dataloader=train_dl, val_dataloader=val_dl,
            num_epochs=NUM_EPOCHS,       # z.B. nur noch 10 weitere Epochen
            lr=LEARNING_RATE,            # z.B. eine kleinere Lernrate wie 5e-5 oder 1e-5
            weight_decay=WEIGHT_DECAY,
            save_checkpoint_every_n_steps=200,
            generate_samples_every_n_steps=100,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            use_cfg_during_training_prob=USE_CFG_IN_TRAINING_PROB
        )



    except FileNotFoundError as e:
        print(f"FEHLER: Datei oder Verzeichnis nicht gefunden: {e}")
    except ValueError as e:
        print(f"FEHLER: Wertfehler aufgetreten: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()
    print("Starte Floor Plan Diffusion LoRA Fine-Tuning Skript...")

    PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
    LORA_RANK = 8
    BATCH_SIZE = 2 # Kleiner für 512x512 + TextEncoder
    NUM_EPOCHS = 20 # LoRA kann schnell lernen, aber mehr Epochen können helfen
    LEARNING_RATE = 1e-4 # AdamW default ist 1e-3, für LoRA oft 1e-4 bis 5e-4
    WEIGHT_DECAY = 1e-2
    IMAGE_SIZE_TRANSFORM = (512,512)
    GRADIENT_ACCUMULATION_STEPS = 2 # Effektiv Batch Size 4
    USE_CFG_IN_TRAINING_PROB = 0.1 # Wahrscheinlichkeit für unkonditioniertes Training eines Teils des Batches


    METADATA_FILE_PATH = None
    IMAGE_FOLDER_PATH = None

    os.makedirs("./models_lora_V1", exist_ok=True)
    os.makedirs("./results_lora_V1", exist_ok=True)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Verwende Gerät: {device}")


        print("\nInitialisiere LoRA Diffusionsmodell und Tokenizer...")
        lora_diffusion_model = FloorPlanDiffusionLoRA(
            device=device,
            pretrained_model_name_or_path=PRETRAINED_MODEL,
            lora_rank=LORA_RANK
        )
        tokenizer_for_dataset = lora_diffusion_model.tokenizer

        print("Erstelle DataLoaders...")
        current_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE_TRANSFORM),
            transforms.RandomHorizontalFlip(p=0.5), # Datenaugmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Datenaugmentation
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dl, val_dl, full_dataset = create_dataloaders(
            tokenizer=tokenizer_for_dataset,
            metadata_path=METADATA_FILE_PATH, image_folder=IMAGE_FOLDER_PATH,
            batch_size=BATCH_SIZE, transform=current_transform, val_split=0.1
        )
        print(f"Dataset erfolgreich erstellt mit {len(full_dataset)} Einträgen.")
        if train_dl: print(f"Trainingsdaten: {len(train_dl.dataset)} ({len(train_dl)} Batches)")
        if val_dl: print(f"Validierungsdaten: {len(val_dl.dataset)} ({len(val_dl)} Batches)")

        if train_dl and len(train_dl.dataset) > 0:
            print("\nZeige einen Beispiel-Batch aus dem Trainings-DataLoader:")
            show_batch(train_dl, num_images=min(BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS, 4))
        else:
            print("Keine Daten im Trainings-DataLoader. Training nicht möglich.")
            if len(full_dataset) == 0 : exit()

        # Optional: LoRA Checkpoint laden
        checkpoint_to_load = r"C:\Users\jawoosh\PycharmProjects\PermittAI_model\models_V1\floor_plan_diffusion_step_10000_1.pt"
        if checkpoint_to_load and os.path.exists(checkpoint_to_load):
            print(f"Lade LoRA Checkpoint: {checkpoint_to_load}")
            lora_diffusion_model.load_lora_checkpoint(checkpoint_to_load)
        else:
            if checkpoint_to_load: print(f"Checkpoint {checkpoint_to_load} nicht gefunden.")
            print("Kein LoRA Checkpoint geladen, starte neues Training.")


        print("\nStarte LoRA Fine-Tuning...")
        lora_diffusion_model.train(
            train_dataloader=train_dl, val_dataloader=val_dl,
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            save_checkpoint_every_n_steps=200, generate_samples_every_n_steps=100,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            use_cfg_during_training_prob=USE_CFG_IN_TRAINING_PROB
        )

        print("\nGeneriere finale Beispielbilder nach dem LoRA Training...")
        final_prompts = [
            "A floor plan with 1 bedroom, 1 bathroom",
            "A floor plan with 2 bedrooms, 1 bathroom, kitchen",
            "A floor plan with 3 bedrooms, 2 bathrooms, kitchen, living room",
            "A floor plan with 4 bedrooms, 2 bathrooms, kitchen, living room, garage, balcony"
        ]
        lora_diffusion_model.generate_samples(
            text_prompts=final_prompts, num_samples_per_prompt=2,
            num_inference_steps=50, guidance_scale=7.5,
            save_path="./results_lora_V1/final_generated_lora_samples.png",
            seed=42 # Für reproduzierbare finale Bilder
        )
        print("\nSkript erfolgreich abgeschlossen.")

    except FileNotFoundError as e:
        print(f"FEHLER: Datei oder Verzeichnis nicht gefunden: {e}")
    except ValueError as e:
        print(f"FEHLER: Wertfehler aufgetreten: {e}")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
        import traceback
        traceback.print_exc()

