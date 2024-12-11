import os
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

def llamaparse_pdf_files(input_dir: str, output_dir: str): 
    """
    Parse pdf files into LlamaParse from input_dir then save it to output_dir

    Args:
        input_dir (str): input directory that stores .pdf files
        output_dir (str): target directory for storing .md files
    """
    load_dotenv()
    
    
    file_name = os.path.basename(input_dir).replace('.pdf', '')
    
    # set up parser
    doc_parsed = LlamaParse(result_type="markdown", target_pages="2,3,4").load_data(input_dir)
    
    data = ""

    for doc in doc_parsed: 
        data += doc.text
        
    with open(f'{output_dir}{file_name}.md', 'w', encoding='utf-8') as f:
        f.write(data + '\n')
            
    return "Parsing documents using LlamaParse completed."


if __name__ == '__main__':

    print('Start parsing...')
    
    import glob
    
    for file in list(glob.glob('documents/pdf/*.pdf')):
        llamaparse_pdf_files(input_dir=file, output_dir='./documents/markdown/')