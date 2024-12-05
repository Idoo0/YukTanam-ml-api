import requests
import json

def getPlantName(image_path):

    base_url = "https://my-api.plantnet.org/v2/identify/all"
    api_key = "2b10mNhQN9AMKIJpFy30Duuu"

    image_file_path = image_path

    headers = {
        "Content-Type": "multipart/form-data",
    }
    params = {
        "api-key": api_key,
        "include-related-images": "false",
        "no-reject": "false",
        "nb-results": 10,
        "lang": "en",
        "type": "-",
    }

    files = {
        "images": open(image_file_path, "rb"),
        "organs": (None, "leaf"),
    }

    response = requests.post(base_url, params=params, files=files)

    if response.status_code == 200:
        print("Request berhasil:")
        response = response.json()
        top3 = response["results"][:3]
        plant_name = []
        for l in top3:
            plant_name.append(
                {"score": l["score"], "commonNames": l["species"]["commonNames"]})
        data_str = json.dumps(plant_name, ensure_ascii=False)
        return data_str
    else:
        print(f"Request gagal dengan kode status {
              response.status_code}: {response.text}")
        return None
