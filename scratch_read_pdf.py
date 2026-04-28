import PyPDF2

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
        return text

if __name__ == "__main__":
    text = read_pdf('Projet_WAMS_2025.pdf')
    with open('Projet_WAMS_2025.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    print("Extraction successful.")
