
import os
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Add CORS middleware
origins = [
    "http://localhost:3000",  # Allow your React frontend
    "http://localhost:5173",  # Allow your Vite React frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(pdf_file):
    """Extracts text from a given PDF file."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")

def get_info_from_llm(text: str):
    """Sends extracted text to OpenAI API and returns structured data."""
    prompt = f"""
    You are an expert data extractor. From the following invoice text, extract the specified fields and return the data in a clean JSON format.
    If a field is not present, its value should be "N/A".

    The JSON keys should be exactly as specified in the "Fields to Extract" list.

    **Fields to Extract:**
    - "Buyer's Order No."
    - "Quantity"
    - "Rate"
    - "Basic amount without tax"
    - "IGST"
    - "Total Amount"
    - "InvoiceNo"
    - "Ack Date"
    - "GSTIN Number"
    - "TML GSTIN"
    - "IRN"

    **Extraction Guidelines:**
    - "Quantity", "Rate", "Basic amount without tax", and "Total Amount" are usually found within the table describing the goods.
    - For "Quantity" and "Rate", extract only the integer value. For example, if the quantity is "5 Nos", the value should be 5.
    - For "InvoiceNo", look for a label like "Invoice No.". The value can be alphanumeric with slashes, like "SW/25-26/2513".
    - For "Ack Date", look for a label like "Dated" or "Ack Date".
    - For "IGST", find the value for IGST tax. It might be under a description of taxes.
    - For "GSTIN Number", this is the GSTIN for the "Sunrise Wheels".
    - For "TML GSTIN", this is the GSTIN for the "Buyer" or "Bill to" party.

    **Invoice Text:**
    ---
    {text}
    ---
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to extract information from documents and return it as a single JSON object. The keys in the JSON should be exactly as requested in the prompt."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {e}")

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms the DataFrame to the final required format."""
    if "Buyer's Order No." in df.columns:
        df.rename(columns={"Buyer's Order No.": "PoNumber"}, inplace=True)
    if "InvoiceNo" in df.columns:
        df.rename(columns={"InvoiceNo": "Vendor Challan No"}, inplace=True)
    if "Ack Date" in df.columns:
        df.rename(columns={"Ack Date": "Challan Date"}, inplace=True)
    if "Basic amount without tax" in df.columns:
        df.rename(columns={"Basic amount without tax": "Basic value"}, inplace=True)
    if "IGST" in df.columns:
        df.rename(columns={"IGST": "IGST VALUE"}, inplace=True)
    if "Total Amount" in df.columns:
        df.rename(columns={"Total Amount": "INVOICE VALUE"}, inplace=True)
    if "Rate" in df.columns:
        df.rename(columns={"Rate": "Gross Rate"}, inplace=True)

    final_df = pd.DataFrame()
    final_df['PO NUMBER'] = df.get('PoNumber')
    final_df['PO Item No'] = ''
    final_df['Quantity'] = df.get('Quantity')
    final_df['VENDOR CHALLAN NO'] = df.get('Vendor Challan No')
    final_df['Challan Date'] = df.get('Challan Date')
    final_df['Gross Rate'] = df.get('Gross Rate')
    final_df['Net P O Rate'] = df.get('Gross Rate')
    final_df['Basic Value'] = df.get('Basic value')
    final_df['Taxable Value'] = df.get('Basic value')
    final_df['SGST VALUE'] = ''
    final_df['CGST VALUE'] = ''
    final_df['IGST VALUE'] = df.get('IGST VALUE')
    final_df['SGST RATE'] = ''
    final_df['CGST RATE'] = ''
    final_df['IGST RATE'] = ''
    final_df['Packing Amount'] = ''
    final_df['Freight Amount'] = ''
    final_df['Others Amount'] = ''
    final_df['INVOICE VALUE'] = df.get('INVOICE VALUE')
    final_df['Currency'] = 'INR'
    final_df['E Way Bill'] = ''
    final_df['57F4 NUMBER'] = ''
    final_df['57F4 NO DATE'] = ''
    final_df['GSTIN Number'] = df.get('GSTIN Number')
    final_df['Vehicle Number'] = ''
    final_df['PART REV Level'] = ''
    final_df['COP Certificate'] = ''
    final_df['Certificate Date'] = ''
    final_df['TML GSTIN'] = df.get('TML GSTIN')
    final_df['Digital Invoice File Name'] = ''
    final_df['IRN'] = df.get('IRN')
    final_df['TCS Value'] = ''
    final_df['Field4'] = ''
    final_df['Field5'] = ''

    return final_df

@app.post("/api/extract")
async def extract_data_from_pdfs(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="You can only upload a maximum of 10 files at a time.")

    all_data = []
    for file in files:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF.")
        
        text = extract_text_from_pdf(file.file)
        
        extracted_data_str = get_info_from_llm(text)
        
        try:
            if '```json' in extracted_data_str:
                extracted_data_str = extracted_data_str.split('```json\n')[1].split('```')[0]
            extracted_data = json.loads(extracted_data_str)
            all_data.append(extracted_data)
        except (json.JSONDecodeError, IndexError) as e:
            # Maybe log the error and the problematic string
            print(f"Error parsing JSON for {file.filename}: {e}")
            print(f"LLM Response: {extracted_data_str}")
            # Decide if you want to skip the file or return an error
            continue # simple skip for now

    if not all_data:
        raise HTTPException(status_code=400, detail="No data could be extracted from the provided files.")

    df = pd.DataFrame(all_data)
    final_df = transform_data(df)

    return final_df.to_dict(orient='records')

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Data Extractor API"}
