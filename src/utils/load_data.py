from PyPDF2 import PdfReader
from io import BytesIO
import chainlit as cl
import os

# process the uplaoded document and create embeddings for the text
def pdf_to_text(file):
    """
    Reads the content of a PDF file and extracts text from all pages.

    Parameters:
    - file: PDF file object (assumes it has a 'content' attribute)

    Returns:
    - pdf_text (str): Extracted text from the PDF file as a single string
    """
    # Convert the content of the PDF file to a BytesIO stream
    text_stream = BytesIO(file.content)

    # Create a PdfReader object from the stream to extract text 
    pdf = PdfReader(text_stream)  
    
    pdf_text = ""
    # Iterate through each page in the PDF and extract text
    for page in pdf.pages:
        pdf_text += page.extract_text()  # Concatenate the text from each page
    
    return pdf_text  # Return the extracted text as a single string
