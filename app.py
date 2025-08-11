
import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(pdf_file):
    """Extracts text from a given PDF file."""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_info_from_llm(text):
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
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to extract information from documents and return it as a single JSON object. The keys in the JSON should be exactly as requested in the prompt."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("📄 PDF Data Extractor")
    
    st.write("Upload up to 10 PDF files to extract key information.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload up to 10 PDF files at a time."
    )
    
    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("You can only upload a maximum of 10 files at a time.")
        else:
            all_data = []
            for uploaded_file in uploaded_files:
                st.write(f"Processing `{uploaded_file.name}`...")
                
                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)
                
                # Get structured data from LLM
                extracted_data_str = get_info_from_llm(text)
                
                # Convert string response to dictionary (handle potential errors)
                try:
                    # The response from the LLM might be a string representation of a JSON.
                    # We might need to clean it up before parsing.
                    import json
                    # A common issue is the LLM returning markdown with the JSON.
                    if '```json' in extracted_data_str:
                        extracted_data_str = extracted_data_str.split('```json\n')[1].split('```')[0]
                    
                    extracted_data = json.loads(extracted_data_str)
                    all_data.append(extracted_data)
                except (json.JSONDecodeError, IndexError) as e:
                    st.error(f"Error parsing data for `{uploaded_file.name}`: {e}")
                    st.text_area("LLM Response", extracted_data_str, height=200)

            if all_data:
                # Convert the list of dictionaries to a pandas DataFrame
                df = pd.DataFrame(all_data)
                
                # Rename the 'Buyer's Order No.' column to 'PoNumber'
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
                final_df['PO NUMBER'] = df['PoNumber']
                final_df['PO Item No'] = ''
                final_df['Quantity'] = df['Quantity']
                final_df['VENDOR CHALLAN NO'] = df['Vendor Challan No']
                final_df['Challan Date'] = df['Challan Date']
                final_df['Gross Rate'] = df['Gross Rate']
                final_df['Net P O Rate'] = df['Gross Rate']
                final_df['Basic Value'] = df['Basic value']
                final_df['Taxable Value'] = df['Basic value']
                final_df['SGST VALUE'] = ''
                final_df['CGST VALUE'] = ''
                final_df['IGST VALUE'] = df['IGST VALUE']
                final_df['SGST RATE'] = ''
                final_df['CGST RATE'] = ''
                final_df['IGST RATE'] = ''
                final_df['Packing Amount'] = ''
                final_df['Freight Amount'] = ''
                final_df['Others Amount'] = '' 
                final_df['INVOICE VALUE'] = df['INVOICE VALUE']
                final_df['Currency'] = 'INR'
                final_df['E Way Bill'] = ''
                final_df['57F4 NUMBER'] = ''
                final_df['57F4 NO DATE'] = ''
                final_df['GSTIN Number'] = df['GSTIN Number']
                final_df['Vehicle Number'] = ''
                final_df['PART REV Level'] = ''
                final_df['COP Certificate'] = ''
                final_df['Certificate Date'] = ''
                final_df['TML GSTIN'] = df['TML GSTIN']
                final_df['Digital Invoice File Name'] = ''
                final_df['IRN'] = df['IRN']
                final_df['TCS Value'] = ''
                final_df['Field4'] = ''
                final_df['Field5'] = ''
                # Display the data in a table
                st.success("Successfully extracted data from all files!")
                st.dataframe(final_df)

                # Add a download button for the CSV
                csv = final_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='extracted_data.csv',
                    mime='text/csv',
                )

if __name__ == "__main__":
    main()