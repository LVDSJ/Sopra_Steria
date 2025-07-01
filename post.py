import requests
import base64

# Load and encode image
with open("images/circle_game/a1f0a821cded1a9e33f605091f0766f8.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    payload = {
        "image": f"data:image/jpeg;base64,{img_base64}"
    }

# Send POST request
response = requests.post("http://localhost:5000/predict", json=payload)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)