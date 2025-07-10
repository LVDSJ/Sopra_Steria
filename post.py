import requests
import base64

# Encode the image as base64
with open("images/peace_sign/will-smith-celebrity-skydivers.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

payload = {
    "image": f"data:image/jpeg;base64,{img_base64}"
}

response = requests.post("http://127.0.0.1:5000/predict", json=payload)

try:
    print("Response status:", response.status_code)
    print("Prediction:", response.json())
except Exception as e:
    print("Failed to decode response:", e)
    print("Raw response text:", response.text)



