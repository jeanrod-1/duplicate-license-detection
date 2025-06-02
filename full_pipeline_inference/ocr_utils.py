import pytesseract
import cv2
import string
import re

def preprocess_plate_for_ocr(image, scale_factor=4):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def clean_plate_text(text):

    # Correcciones OCR comunes
    # replacements = {
    #     'O': '0', 'Q': '0',
    #     'I': '1', 'L': '1',
    #     'Z': '2',
    #     'S': '5',
    #     'B': '8'
    # }

    # Convertir a mayúsculas y eliminar todos los espacios en blanco (incluyendo tabs, saltos de línea)
    text = re.sub(r'\s+', '', text.upper())

    # Eliminar caracteres que no sean letras o números
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Aplicar reemplazos
    # text = ''.join(replacements.get(c, c) for c in text)

    # Buscar coincidencias exactas: 3 letras + 3 números
    match = re.search(r'[A-Z]{3}[0-9]{3}', text)
    if match:
        return match.group(0)

    # Si no hay match, intentar "forzar" estructura AAA111
    letters = ''.join([c for c in text if c in string.ascii_uppercase])
    digits = ''.join([c for c in text if c in string.digits])

    # Si hay al menos 3 letras y 3 números, los usamos
    if len(letters) >= 3 and len(digits) >= 3:
        return letters[:3] + digits[:3]

    # Si no se puede reconstruir una placa válida
    return ""
