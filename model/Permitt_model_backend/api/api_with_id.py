import os
import sys
import time
import uuid
import json
import torch
import traceback
import contextlib
import io
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Warnungen unterdrücken
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Projektpfade definieren
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Projekt-Imports
# ACHTUNG: Angenommen, create_prompt_from_json gibt jetzt ein Tuple (prompt, preferences_id) zurück
from utils.prompt_generator import create_prompt_from_json
from utils.model_neue_V2 import FloorPlanDiffusionLoRA

# Globale Konstanten
MODEL = None
DEVICE = None
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "floor_plan_lora_final.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "api", "output", "generated_images")

# Sicherstellen, dass der Ausgabeordner existiert
os.makedirs(OUTPUT_DIR, exist_ok=True)


###################
# Utility Funktionen
###################

@contextlib.contextmanager
def suppress_stdout():
    """Temporär die Standardausgabe unterdrücken."""
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout


def clean_gpu_memory():
    """GPU-Speicher freigeben, wenn verfügbar."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


###################
# Pydantic-Modelle
###################

class RoomExtra(BaseModel):
    """Zusätzliche Raumeigenschaften."""
    name: str


class Room(BaseModel):
    """Raum-Definition für einen Grundriss."""
    type: str
    number: int
    extras: List[str] = []


class Floor(BaseModel):
    """Stockwerk-Definition mit Räumen."""
    name: str
    rooms: List[Room]


class Location(BaseModel):
    """Standort-Informationen."""
    street: Optional[str] = None
    postalCode: Optional[str] = None
    city: Optional[str] = None


class FloorPlanRequest(BaseModel):
    """Anfrage-Modell für die Grundriss-Generierung."""
    location: Optional[Location] = None
    measurementUnit: str = Field(default="metric", description="Maßeinheit (metric oder imperial)")
    numberOfFloors: int = Field(default=1, description="Anzahl der Stockwerke")
    area: Optional[float] = None
    floors: List[Floor]
    additionalNotes: Optional[str] = ""


class FloorPlanResponse(BaseModel):
    """Antwort-Modell für die Grundriss-Generierung."""
    imageUrl: str
    prompt: str
    preferencesId: str  # Feld für die Preferences ID hinzugefügt
    generationTimeSeconds: float


###################
# Modell-Funktionen
###################

def load_model():
    """
    Lädt das Modell, wenn es noch nicht geladen ist.

    Returns:
        Das geladene Modell

    Raises:
        HTTPException: Bei Problemen mit dem Modell-Loading
    """
    global MODEL, DEVICE

    if MODEL is not None:
        return MODEL

    try:
        # GPU Verfügbarkeit überprüfen
        if torch.cuda.is_available():
            DEVICE = torch.device("cuda")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            clean_gpu_memory()
        else:
            DEVICE = torch.device("cpu")
            print("Verwende CPU (Hinweis: GPU-Nutzung wird empfohlen)")

        print(f"Lade Modell auf Gerät: {DEVICE}")
        start_time = time.time()

        # Modell-Instanziierung mit unterdrückter Ausgabe
        with suppress_stdout():
            MODEL = FloorPlanDiffusionLoRA(
                device=DEVICE,
                pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
                lora_rank=8
            )
            MODEL.load_lora_checkpoint(MODEL_PATH)

        load_time = time.time() - start_time
        print(f"Modell in {load_time:.2f} Sekunden geladen")

        return MODEL

    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Fehler beim Laden des Modells: {str(e)}"
        )


def generate_image(
        prompt: str,
        preferences_id: str,  # Preferences ID hinzugefügt
        image_width: int = 1280,
        image_height: int = 1280
) -> Tuple[str, float]:
    """
    Generiert ein Bild basierend auf dem Prompt und speichert es unter der Preferences ID.

    Args:
        prompt: Text-Prompt für die Bildgenerierung
        preferences_id: Die ID, die für den Dateinamen verwendet wird
        image_width: Breite des zu generierenden Bildes
        image_height: Höhe des zu generierenden Bildes

    Returns:
        Tuple aus Dateipfad und Generierungszeit

    Raises:
        HTTPException: Bei Problemen mit der Bildgenerierung
    """
    try:
        # Modell laden
        model = load_model()

        # Sicherstellen, dass die Bildgrößen Integers sind
        image_width = int(image_width)
        image_height = int(image_height)

        # Dateinamen generieren unter Verwendung der Preferences ID
        filename = f"{preferences_id}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        start_time = time.time()

        # Bild generieren
        images_np = model.generate_samples(
            text_prompts=[prompt],
            num_samples_per_prompt=1,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42
        )

        # Konvertiere zu PIL Image und speichere
        img = Image.fromarray((images_np[0] * 255).astype(np.uint8))

        # Größe anpassen
        if img.size != (image_width, image_height):
            img = img.resize((image_width, image_height), Image.LANCZOS)

        # Bild speichern mit hoher Qualität
        img.save(filepath, quality=95)

        generation_time = time.time() - start_time
        print(f"Bild wurde in {generation_time:.2f} Sekunden generiert: {filepath}")

        # Speicher freigeben
        clean_gpu_memory()

        return filepath, generation_time

    except Exception as e:
        print(f"Fehler bei der Bildgenerierung: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Fehler bei der Bildgenerierung: {str(e)}"
        )


###################
# FastAPI App
###################

app = FastAPI(
    title="FloorPlan API",
    description="API zum Generieren von Grundriss-Bildern basierend auf Raumbeschreibungen",
    version="1.0.0"
)

# CORS-Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion sollte dies eingeschränkt werden
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


###################
# API-Endpunkte
###################

@app.post("/generate-floorplan_V1", response_model=FloorPlanResponse)
async def generate_floorplan(request: Request, background_tasks: BackgroundTasks):
    """
    Generiert ein Grundriss-Bild basierend auf der JSON-Beschreibung.
    Erwartet eine flache JSON-Struktur mit designPreferencesId vom Frontend.

    Args:
        request: HTTP-Request mit JSON-Daten
        background_tasks: FastAPI BackgroundTasks für Aufräumarbeiten

    Returns:
        FloorPlanResponse mit Bild-URL, Prompt, Preferences ID und Generierungszeit
    """
    try:
        # JSON-Daten aus dem Request-Body extrahieren
        json_data = await request.json()

        # Direkt die flache Struktur verwenden (keine Preferences-Verschachtelung mehr nötig)
        print("Verarbeite JSON-Struktur vom Frontend")

        # Rufe create_prompt_from_json auf, um Prompt und Preferences ID zu erhalten
        prompt, preferences_id = create_prompt_from_json(json_data)

        print(f"Generierter Prompt: {prompt}")
        print(f"Design Preferences ID: {preferences_id}")

        # Bild generieren und Preferences ID übergeben
        filepath, generation_time = generate_image(prompt, preferences_id)

        # Aufräumen im Hintergrund
        background_tasks.add_task(clean_gpu_memory)

        # Response mit der URL und Preferences ID zurückgeben
        return FloorPlanResponse(
            imageUrl=f"/images/{preferences_id}",
            prompt=prompt,
            preferencesId=preferences_id,
            generationTimeSeconds=generation_time
        )

    except HTTPException:
        # HTTPException weitergeben
        raise
    except Exception as e:
        print(f"Fehler: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/{preferences_id}")
async def get_image(preferences_id: str):
    """
    Liefert das angeforderte Bild basierend auf der Preferences ID.

    Args:
        preferences_id: Die ID des Bildes, das abgerufen werden soll.

    Returns:
        FileResponse mit dem angeforderten Bild
    """
    # Entferne mögliche Dateierweiterung aus der preferences_id
    preferences_id = preferences_id.split('.')[0]

    # Erzeuge den Dateinamen mit .png-Erweiterung
    filename = f"{preferences_id}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")

    return FileResponse(filepath, media_type="image/png")


###################
# Hauptausführung
###################

if __name__ == "__main__":
    import uvicorn

    # Modell beim Start laden, um die erste Anfrage zu beschleunigen
    try:
        load_model()
    except Exception as e:
        print(f"Warnung: Modell konnte nicht vorab geladen werden: {e}")

    # Server starten
    # reload=True sollte in Produktion nicht verwendet werden
    uvicorn.run("api_with_id:app", host="0.0.0.0", port=8000, reload=True)