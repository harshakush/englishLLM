import json
import requests
import os
from datetime import datetime
from openai import OpenAI


def load_and_validate_json(json_file):
    """
    Load the JSON file and validate that it contains a valid 'grammar_concepts_reference' section.
    Returns: A dictionary containing all grammar concepts, or raises an error if invalid.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data.get("grammar_concepts_reference"), dict):
            raise ValueError("JSON must contain a valid 'grammar_concepts_reference' dictionary.")
        return data["grammar_concepts_reference"]
    except FileNotFoundError:
        print(f"⚠️ Error: File '{json_file}' not found.")
        exit(1)
    except Exception as e:
        print(f"⚠️ JSON load failed with error: {e}")
        exit(1)


def generate_essay_prompt(concept_entry):
    """
    Generate a prompt for the model based on the concept, description, and examples.
    """
    # Ensure that necessary fields are present
    if "concept" not in concept_entry or "description" not in concept_entry or "examples" not in concept_entry:
        raise ValueError("Concept must contain 'concept', 'description', and 'examples' fields.")

    description = concept_entry["description"]
    examples = "\n- " + "\n- ".join(concept_entry["examples"])

    # Build prompt
    return f"""
{description}

Examples:
{examples}
"""


def save_essay_to_file(concept_id, concept_name, essay_content, data_folder="data"):
    """
    Save the generated essay to a file with a unique date+time filename.
    Returns: True if successful, False otherwise.
    """
    try:
        # Ensure the data folder exists
        os.makedirs(data_folder, exist_ok=True)

        # Generate timestamp string
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename with concept_id and timestamp
        filename = f"concept_{concept_id}_{timestamp}.txt"
        path = os.path.join(data_folder, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"## {concept_name}\n\n{essay_content}")
        print(f"💾 Saved essay for concept {concept_id} to: '{path}'")
        return True
    except Exception as e:
        print(f"❌ Error saving file for concept {concept_id}: {e}")
        return False


def process_concepts(grammar_concepts, model="qwen3_32b", data_folder="data"):
    """
    Process each grammar concept and generate an essay.
    """
    if not isinstance(grammar_concepts, dict):
        raise ValueError("Input must be a dictionary of concepts.")

    print(f"🔄 Processing {len(grammar_concepts)} grammar concepts...")
    for key in grammar_concepts:
        try:
            # Convert the key to integer (assuming it's a numeric string)
            concept_id = int(key)
            concept_entry = grammar_concepts[key]

            prompt = generate_essay_prompt(concept_entry) + "Write 5 eassays 2000 word each on this topic and use aggriculture, defence sector, car manufacturing, travel and toruism ,aerospace, IT service sector  as a reference for each essay"

            print(f"\n📝 Processing Concept {concept_id}: '{concept_entry['concept']}'")
            print(prompt)
            essay_content = call_lm_studio_api(prompt, model)  # You need to define this function
            save_essay_to_file(concept_id, concept_entry["concept"], essay_content, data_folder)
        except Exception as e:
            print(f"⚠️ Error processing concept {key}: {e}")


def call_ollama_api(prompt, model):
    """
    Calls Ollama's API with the given prompt and model.
    This function was originally named for LM Studio but now uses Ollama.

    Assumes that Ollama is running on http://localhost:11434
    """

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # Change to True for real-time output if needed
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()
        return result["response"]

    except Exception as e:
        print(f"❌ Error calling Ollama API: {e}")
        return ""


def call_lm_studio_api(prompt, model):
    # Initialize the LM Studio client
    client = OpenAI(base_url="http://localhost:23232/v1", api_key="lm-studio")

    # Make the API call with the passed prompt
    completion = client.chat.completions.create(
        model=model,  # LM Studio usually serves one model, so this can be any string
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    # Return the generated response text
    return completion.choices[0].message.content


if __name__ == "__main__":
    json_file = "grammar_concepts_database.json"
    grammar_concepts = load_and_validate_json(json_file)
    process_concepts(grammar_concepts, model="qwen3:14b", data_folder="data")