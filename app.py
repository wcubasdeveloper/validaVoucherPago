import os
import sys
import re
import subprocess
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# En Linux Tesseract se instala en /usr/bin/tesseract
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# Verificar que funciona
try:
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract versión: {version}")
except Exception as e:
    print(f"❌ Error al inicializar Tesseract: {e}")
    sys.exit(1)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_amount_from_text(text):
    """Extrae montos en soles del texto - optimizado para vouchers peruanos"""
    text_original = text
    text = text.replace(',', '.')
    
    print(f"\n📄 Texto completo para análisis:\n{text}\n")
    
    amounts_priority = []  # Montos con contexto claro (S/, monto, total)
    amounts_general = []   # Montos generales
    
    # PRIORIDAD 1: Patrones con contexto explícito de soles
    priority_patterns = [
        r'S/\s*(\d{1,5}(?:\.\d{1,2})?)',           # S/ 260.00
        r'S/\.?\s*(\d{1,5}(?:\.\d{1,2})?)',         # S/260
        r'monto\s*[:.]?\s*S?/?\s*(\d{1,5}(?:\.\d{1,2})?)',
        r'total\s*[:.]?\s*S?/?\s*(\d{1,5}(?:\.\d{1,2})?)',
        r'pago\s*[:.]?\s*S?/?\s*(\d{1,5}(?:\.\d{1,2})?)',
        r'enviado\s*[:.]?\s*S?/?\s*(\d{1,5}(?:\.\d{1,2})?)',
        r'importe\s*[:.]?\s*S?/?\s*(\d{1,5}(?:\.\d{1,2})?)',
        r'(\d{1,5}(?:\.\d{1,2})?)\s*soles',
    ]
    
    for pattern in priority_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount_str = re.sub(r'[^\d.]', '', str(match))
                if amount_str and amount_str.count('.') <= 1:
                    amount = float(amount_str)
                    if 1 <= amount <= 10000:
                        amounts_priority.append(amount)
                        print(f"   ⭐ Monto prioritario: S/ {amount} (patrón: {pattern})")
            except ValueError:
                continue
    
    # PRIORIDAD 2: Números grandes aislados (probablemente el monto principal)
    # Buscar números >= 100 que estén solos o cerca de puntos decimales
    general_patterns = [
        r'\b(\d{3,5}(?:\.\d{1,2})?)\b',  # 100-99999
        r'\b(\d{2,5}),(\d{2})\b',         # formato 260,00
    ]
    
    for pattern in general_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                if isinstance(match, tuple):
                    amount_str = match[0] + '.' + match[1]
                else:
                    amount_str = str(match)
                amount_str = re.sub(r'[^\d.]', '', amount_str)
                if amount_str and amount_str.count('.') <= 1:
                    amount = float(amount_str)
                    # Excluir años (2020-2030) y números de operación largos
                    if 10 <= amount <= 9999 and not (2020 <= amount <= 2030):
                        amounts_general.append(amount)
                        print(f"   📊 Monto general: S/ {amount}")
            except ValueError:
                continue
    
    # Combinar: priorizar los montos con contexto
    all_amounts = amounts_priority if amounts_priority else amounts_general
    
    if all_amounts:
        all_amounts = list(set([round(a, 2) for a in all_amounts]))
    
    return all_amounts

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "tesseract": {
            "path": pytesseract.pytesseract.tesseract_cmd,
            "version": str(pytesseract.get_tesseract_version()),
        }
    })

@app.route('/validar-pago', methods=['POST'])
def validar_pago():
    try:
        monto_esperado = request.form.get('monto_esperado')
        if not monto_esperado:
            return jsonify({"error": "Falta el parámetro 'monto_esperado'", "valido": False}), 400
        
        monto_esperado = float(monto_esperado)
        print(f"\n💰 Monto esperado: S/ {monto_esperado}")
        
        if 'imagen' not in request.files:
            return jsonify({"error": "No se recibió imagen", "valido": False}), 400
        
        file = request.files['imagen']
        if not file or file.filename == '':
            return jsonify({"error": "Archivo vacío", "valido": False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Tipo de archivo no permitido", "valido": False}), 400
        
        print(f"📸 Procesando imagen: {file.filename}")
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "No se pudo decodificar la imagen", "valido": False}), 400
        
        # Preprocesamiento
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        textos = []
        
        try:
            texto1 = pytesseract.image_to_string(binary, lang='spa', config='--psm 3')
            textos.append(texto1)
            print("✅ OCR con español (psm 3) completado")
        except Exception as e:
            print(f"⚠️ Error con configuración 1: {e}")
        
        try:
            texto2 = pytesseract.image_to_string(binary, lang='eng', config='--psm 6')
            textos.append(texto2)
            print("✅ OCR con inglés (psm 6) completado")
        except Exception as e:
            print(f"⚠️ Error con configuración 2: {e}")
        
        try:
            texto3 = pytesseract.image_to_string(binary, config='--psm 3')
            textos.append(texto3)
            print("✅ OCR sin idioma específico completado")
        except Exception as e:
            print(f"⚠️ Error con configuración 3: {e}")
        
        texto_completo = '\n'.join(textos)
        print(f"\n📝 Texto extraído:\n{texto_completo[:500]}")
        
        montos_encontrados = extract_amount_from_text(texto_completo)
        print(f"\n💰 Montos encontrados: {montos_encontrados}")
        
        es_valido = False
        monto_pagado = None
        mensaje = ""
        
        if montos_encontrados:
            monto_pagado = min(montos_encontrados, key=lambda x: abs(x - monto_esperado))
            diferencia = abs(monto_pagado - monto_esperado)
            
            if diferencia <= 0.50:
                es_valido = True
                mensaje = f"✅ Pago válido: S/ {monto_pagado:.2f} (esperado: S/ {monto_esperado:.2f})"
            else:
                mensaje = f"❌ Monto incorrecto: S/ {monto_pagado:.2f} (esperado: S/ {monto_esperado:.2f})"
        else:
            mensaje = "❌ No se pudo identificar ningún monto en la imagen"
        
        print(f"\n📊 Resultado: {mensaje}")
        
        return jsonify({
            "valido": es_valido,
            "monto_esperado": monto_esperado,
            "monto_encontrado": monto_pagado,
            "montos_detectados": montos_encontrados,
            "mensaje": mensaje
        })
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "valido": False}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SERVIDOR DE VALIDACIÓN DE PAGOS - LINUX")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   GET  /health")
    print("   POST /validar-pago")
    print("\n📍 Servidor corriendo en: http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)