import os
import sys
import re
import base64
import numpy as np
import cv2
import requests as http_requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract versión: {version}")
except Exception as e:
    print(f"⚠️ Tesseract no disponible: {e}")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_with_openai_vision(image_bytes, monto_esperado):
    if not OPENAI_API_KEY:
        print("⚠️ No hay API key de OpenAI")
        return None
    
    try:
        img_base64 = image_to_base64(image_bytes)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analiza este comprobante de pago peruano (Yape, Plin, BCP, etc).
Extrae SOLO el monto principal pagado en soles.
El monto esperado es S/ {monto_esperado}.

Responde SOLO con JSON sin texto adicional:
{{"monto": 260.00, "descripcion": "Plin envio exitoso"}}

Si no identificas el monto:
{{"monto": null, "descripcion": "No se pudo leer"}}"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 100
        }
        
        response = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content'].strip()
            content = content.replace('```json', '').replace('```', '').strip()
            print(f"🤖 OpenAI Vision: {content}")
            import json
            return json.loads(content)
        else:
            print(f"⚠️ Error OpenAI: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error OpenAI Vision: {e}")
        return None

def extract_with_tesseract(image_bytes):
    try:
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        textos = []
        for lang, config in [('spa', '--psm 3'), ('spa', '--psm 6'), ('eng', '--psm 3')]:
            try:
                textos.append(pytesseract.image_to_string(binary, lang=lang, config=config))
            except:
                pass
        
        texto = '\n'.join(textos)
        print(f"📝 Tesseract:\n{texto[:300]}")
        
        amounts = []
        for pattern in [r'S/\s*(\d{1,5}(?:[.,]\d{1,2})?)', r'monto\s*[:.]?\s*S?/?\s*(\d{1,5}(?:[.,]\d{1,2})?)', r'total\s*[:.]?\s*S?/?\s*(\d{1,5}(?:[.,]\d{1,2})?)']:
            for match in re.findall(pattern, texto, re.IGNORECASE):
                try:
                    a = float(str(match).replace(',', '.').replace(' ', ''))
                    if 10 <= a <= 9999 and not (2020 <= a <= 2030):
                        amounts.append(round(a, 2))
                except:
                    pass
        
        return list(set(amounts))
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return []

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "tesseract": str(pytesseract.get_tesseract_version()),
        "openai_vision": "enabled" if OPENAI_API_KEY else "disabled"
    })

@app.route('/validar-pago', methods=['POST'])
def validar_pago():
    try:
        monto_esperado = request.form.get('monto_esperado')
        if not monto_esperado:
            return jsonify({"error": "Falta monto_esperado", "valido": False}), 400
        
        monto_esperado = float(monto_esperado)
        print(f"\n💰 Monto esperado: S/ {monto_esperado}")
        
        if 'imagen' not in request.files:
            return jsonify({"error": "No imagen", "valido": False}), 400
        
        file = request.files['imagen']
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Archivo inválido", "valido": False}), 400
        
        image_bytes = file.read()
        es_valido = False
        monto_pagado = None
        mensaje = ""

        # MÉTODO 1: OpenAI Vision
        if OPENAI_API_KEY:
            resultado = extract_with_openai_vision(image_bytes, monto_esperado)
            if resultado and resultado.get('monto') is not None:
                monto_pagado = float(resultado['monto'])
                diferencia = abs(monto_pagado - monto_esperado)
                es_valido = diferencia <= 1.0
                mensaje = f"✅ Válido: S/ {monto_pagado:.2f}" if es_valido else f"❌ Monto incorrecto: S/ {monto_pagado:.2f} (esperado S/ {monto_esperado:.2f})"
                return jsonify({"valido": es_valido, "monto_esperado": monto_esperado, "monto_encontrado": monto_pagado, "mensaje": mensaje, "metodo": "OpenAI Vision"})

        # MÉTODO 2: Tesseract fallback
        montos = extract_with_tesseract(image_bytes)
        if montos:
            monto_pagado = min(montos, key=lambda x: abs(x - monto_esperado))
            diferencia = abs(monto_pagado - monto_esperado)
            es_valido = diferencia <= 0.50
            mensaje = f"✅ Válido: S/ {monto_pagado:.2f}" if es_valido else f"❌ Incorrecto: S/ {monto_pagado:.2f} (esperado S/ {monto_esperado:.2f})"
        else:
            mensaje = "❌ No se pudo identificar el monto"
        
        return jsonify({"valido": es_valido, "monto_esperado": monto_esperado, "monto_encontrado": monto_pagado, "montos_detectados": montos, "mensaje": mensaje, "metodo": "Tesseract"})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "valido": False}), 500

if __name__ == '__main__':
    print(f"🚀 Servidor iniciado - OpenAI Vision: {'ACTIVO' if OPENAI_API_KEY else 'INACTIVO'}")
    app.run(debug=False, host='0.0.0.0', port=5000)