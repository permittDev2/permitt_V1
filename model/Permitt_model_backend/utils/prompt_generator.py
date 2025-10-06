from typing import Dict, Optional, Any
import json


def create_prompt_from_json(json_data: Dict[str, Any]) -> tuple:
    """
    Konvertiert JSON-Daten in einen Prompt für das Grundriss-Generierungsmodell.
    Verwendet nur die Raumtypen und ihre Anzahl (keine Extras).

    Args:
        json_data: JSON-Daten im Format des Frontend-Requests

    Returns:
        Ein Tupel aus (prompt_string, preferences_id)
    """
    # Extrahiere die Preferences und Stockwerke
    preferences = json_data.get("Preferences", json_data)
    
    # ID aus designPreferencesId extrahieren (neues Frontend-Format)
    # Fallback auf alte Formate (id, _id) für Rückwärtskompatibilität
    preferences_id = preferences.get("designPreferencesId",
                                     preferences.get("id",
                                     preferences.get("_id", "")))

    # Wenn preferences_id leer ist, generiere eine neue UUID
    if not preferences_id:
        import uuid
        preferences_id = str(uuid.uuid4())
    
    # Extrahiere die Stockwerke und Räume
    floors = preferences.get("Floors", preferences.get("floors", []))

    # Räume nach Typen zusammenfassen
    room_counts = count_rooms_by_type(floors)

    # Prompt-Teile erstellen
    prompt_parts = create_prompt_parts(room_counts, preferences)

    # Zusammenführen mit Kommas getrennt
    prompt_text = ", ".join(prompt_parts)

    # Präfix hinzufügen
    return f"floorplan with {prompt_text}", preferences_id


def count_rooms_by_type(floors: list) -> Dict[str, int]:
    """
    Zählt die Anzahl der Räume nach Typ aus allen Stockwerken.

    Args:
        floors: Liste der Stockwerke mit Raumdaten

    Returns:
        Dictionary mit Raumtypen als Schlüssel und Anzahl als Wert
    """
    room_counts = {}

    for floor in floors:
        for room in floor.get("Rooms", floor.get("rooms", [])):
            room_type = room.get("Type", room.get("type", "")).lower()
            room_count = room.get("Number", room.get("number", 0))

            # Leere Raumtypen überspringen
            if not room_type:
                continue

            # Anzahl der Räume aktualisieren
            if room_type in room_counts:
                room_counts[room_type] += room_count
            else:
                room_counts[room_type] = room_count

    return room_counts


def create_prompt_parts(room_counts: Dict[str, int], json_data: Dict[str, Any]) -> list:
    """
    Erstellt die einzelnen Teile des Prompts basierend auf Raumzahlen und zusätzlichen Daten.

    Args:
        room_counts: Dictionary mit Raumtypen und deren Anzahl
        json_data: Vollständige JSON-Daten für zusätzliche Informationen

    Returns:
        Liste mit Prompt-Teilstrings
    """
    prompt_parts = []

    # Spezifische Räume in gewünschter Reihenfolge
    priority_rooms = {
        "bed room": "bedroom",
        "bedroom": "bedroom",
        "bathroom": "bathroom",
        "kitchen": "kitchen",
        "living room": "living room",
        "balcony": "balcony"
    }

    # Zuerst die priorisierten Raumtypen verarbeiten
    for room_type, display_name in priority_rooms.items():
        if room_type in room_counts and room_counts[room_type] > 0:
            count = room_counts[room_type]
            suffix = "s" if count > 1 else ""
            prompt_parts.append(f"{count} {display_name}{suffix}")

    # Dann alle anderen Raumtypen
    for room_type, count in room_counts.items():
        # Überspringen der bereits verarbeiteten Raumtypen
        if room_type in priority_rooms:
            continue

        # Leerzeichen entfernen für einheitliche Formatierung
        normalized_type = room_type.replace(" ", "")
        suffix = "s" if count > 1 else ""
        prompt_parts.append(f"{count} {normalized_type}{suffix}")

    # Anzahl der Stockwerke hinzufügen (wenn mehr als 1)
    number_of_floors = json_data.get("NumberOfFloors", json_data.get("numberOfFloors", 1))
    if number_of_floors > 1:
        prompt_parts.append(f"{number_of_floors} floors")

    # Fläche hinzufügen (wenn vorhanden)
    area = json_data.get("Area", json_data.get("area", json_data.get("AreaPerFloor")))
    if area:
        unit = "sqm" if json_data.get("MeasurementUnit", json_data.get("measurementUnit", "metric")).lower() == "metric" else "sqft"
        prompt_parts.append(f"{area} {unit}")

    return prompt_parts


def parse_json_request(json_str: str) -> Dict[str, Any]:
    """
    Parst einen JSON-String und gibt ein Dictionary zurück.

    Args:
        json_str: JSON-String

    Returns:
        Ein Dictionary mit den JSON-Daten

    Raises:
        ValueError: Wenn das JSON-Format ungültig ist
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ungültiges JSON-Format: {str(e)}")


# Testfunktion (auskommentiert für Produktionscode)

# def test_prompt_generator():
#     # Beispiel-JSON für Tests
#     test_json = {
#         "UserId": "68cc5a2f1e26cc819c3bcaec",
#         "Preferences": {
#             "_id": "49cc5814-ac7c-45c7-821f-3f7e2fbfa10b",
#             "Location": {
#                 "Street": "Hainstrasse 1234",
#                 "PostalCode": 4109,
#                 "City": "Leipzig"
#             },
#             "MeasurementUnit": "metric",
#             "NumberOfFloors": 1,
#             "AreaPerFloor": 0,
#             "Floors": [
#                 {
#                     "Name": "First Floor",
#                     "Rooms": [
#                         {"Type": "", "Number": 0, "Extras": [""]},
#                         {"Type": "Kitchen", "Number": 1, "Extras": ["Balcony"]},
#                         {"Type": "Living room", "Number": 1, "Extras": []},
#                         {"Type": "Laundry", "Number": 1, "Extras": ["Storage"]},
#                         {"Type": "Garage", "Number": 1, "Extras": []}
#                     ]
#                 }
#             ],
#             "AdditionalNotes": ""
#         },
#         "Status": "Pending",
#         "CreatedAt": "2025-09-18T19:22:06.605Z",
#         "PdfUrl": ""
#     }
#
#     prompt, preferences_id = create_prompt_from_json(test_json)
#     print(f"Generierter Prompt: {prompt}")
#     print(f"Preferences ID: {preferences_id}")
#     return prompt, preferences_id
#
#
# if __name__ == "__main__":
#     test_prompt_generator()
