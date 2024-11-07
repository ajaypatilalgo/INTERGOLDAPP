import easyocr as ocr
from PIL import Image
import numpy as np
import re
import Levenshtein

# Load the OCR model once at startup
def load_model():
    reader = ocr.Reader(['en'])
    return reader

reader = load_model()

def string_similarity(s1, s2):
    distance = Levenshtein.distance(s1, s2)
    similarity = 1 - (distance / max(len(s1), len(s2)))
    return similarity * 100

def extract_data_from_image(image):
    # Read text from image using OCR
    result = reader.readtext(np.array(image))
    result_text = [text[1] for text in result]
    
    PH, PHID = [], []
    ADD, AID = set(), []
    EMAIL, EID, PIN, PID, WEB, WID = '', '', '', '', '', ''

    # Extract details using regex and heuristics
    for i, string in enumerate(result_text):
        # Email extraction
        if re.search(r'@', string.lower()):
            EMAIL = string.lower()
            EID = i

        # Pincode extraction
        match = re.search(r'\d{6,7}', string.lower())
        if match:
            PIN = match.group()
            PID = i

        # Phone number extraction
        match = re.search(r'(?:ph|phone|phno)?\s*(?:[+-]?\d\s*[\(\)]*){7,}', string)
        if match and len(re.findall(r'\d', string)) > 7:
            PH.append(string)
            PHID.append(i)

        # Address extraction
        keywords = ['road', 'floor', ' st ', 'st,', 'street', ' dt ', 'district',
                    'near', 'beside', 'opposite', ' at ', ' in ', 'center', 'main road',
                    'state','country', 'post','zip','city','zone','mandal','town','rural',
                    'circle','next to','across from','area','building','towers','village',
                    ' ST ',' VA ',' VA,',' EAST ',' WEST ',' NORTH ',' SOUTH ']
        digit_pattern = r'\d{6,7}'
        if any(keyword in string.lower() for keyword in keywords) or re.search(digit_pattern, string):
            ADD.add(string)
            AID.append(i)

        # State extraction
        states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
                  'Haryana','Hyderabad', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh',
                  'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
                  'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal',
                  "United States", "China", "Japan", "Germany", "United Kingdom", "France", "India",
                  "Canada", "Italy", "South Korea", "Russia", "Australia", "Brazil", "Spain", "Mexico", 'USA','UK']

        for x in states:
            similarity = string_similarity(x.lower(), string.lower())
            if similarity > 50:
                ADD.add(string)
                AID.append(i)

        # Website URL
        if re.match(r"(?!.*@)(www|.*com$)", string):
            WEB = string.lower()
            WID = i

    # Combine details for display
    IDS = [EID, PID, WID]
    IDS.extend(AID)
    IDS.extend(PHID)

    # fin = []
    # for i, string in enumerate(result_text):
    #     if i not in IDS:
    #         if len(string) >= 4 and ',' not in string and '.' not in string and 'www.' not in string:
    #             if not re.match("^[0-9]{0,3}$", string) and not re.match("^[^a-zA-Z0-9]+$", string):
    #                 numbers = re.findall('\d+', string)
    #                 if len(numbers) == 0 or all(len(num) < 3 for num in numbers):
    #                     fin.append(string)
    fin = []
    card_holder = ''
    company_name = ''

    # Common name prefixes
    name_prefixes = ['Mr.', 'Ms.', 'Mrs.', 'Dr.', 'Prof.']
    name_pattern = r'^(?:' + '|'.join(name_prefixes) + r')?\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$'  # Enhanced name regex

    for i, string in enumerate(result_text):
        if i not in IDS:
            if len(string) >= 4 and ',' not in string and '.' not in string and 'www.' not in string:
                if not re.match("^[0-9]{0,3}$", string) and not re.match("^[^a-zA-Z0-9]+$", string):
                    numbers = re.findall('\d+', string)
                    if len(numbers) == 0 or all(len(num) < 3 for num in numbers):
                        # Check if the string is likely to be a cardholder name using improved regex
                        if re.match(name_pattern, string):
                            card_holder = string.strip()  # Set cardholder name, strip whitespace

                        # Check for company name with enhanced regex
                        elif re.search(r'^(Company|Corp|Limited|Ltd|Inc)\s*[: -]?\s*([A-Za-z0-9\s&.-]+)', string):
                            company_name = string.split(':')[-1].strip()  # Extract company name

                        fin.append(string)  # Append to final list

    # Return extracted data
    return {
        'website': WEB,
        'email': EMAIL,
        'pincode': PIN,
        'phone_numbers': PH,
        'address': list(ADD),
        'card_holder': fin[0],
        'company_name': company_name,
        'other_details': fin  # Include other extracted strings if needed
    }
