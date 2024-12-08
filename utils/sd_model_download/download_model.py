# from https://github.com/Engineer-of-Stuff/stable-diffusion-paperspace
import re
import requests
import gdown
import json
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlparse

load_dotenv()

model_storage_dir = os.environ['MODEL_DIR']
hf_token = os.environ.get('HF_TOKEN', None)
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
civitai_token = 'bf5a73346bccc8ab11cd99e1386a0e1b'

def is_url(url_str):
    return re.search(r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}', url_str)

def check_disk_space(required_bytes, path="/tmp"):
    """Check if there's enough disk space available"""
    st = os.statvfs(path)
    free_bytes = st.f_frsize * st.f_bavail
    return free_bytes > required_bytes

def dl_web_file(web_dl_file, filename=None, token=None):
    web_dl_file = is_url(web_dl_file)[0]  # clean the URL string
    
    # Get file size before downloading
    response = requests.head(web_dl_file, headers={'Authorization': f'Bearer {token}'} if token else {})
    file_size = int(response.headers.get('content-length', 0))
    
    # Check if we have enough space (file size + 1GB buffer)
    if not check_disk_space(file_size + (1024 * 1024 * 1024)):
        print(f"Error: Not enough disk space to download {filename}")
        print(f"Required: {file_size / (1024*1024*1024):.2f}GB")
        return False
        
    filename_cmd = f'--out="{filename}"' if filename else ''
    token_cmd = f"--header='Authorization: Bearer {token}'" if token else ''
    
    command = f'''aria2c {token_cmd} --file-allocation=none -c -x 16 -s 16 --summary-interval=0 --console-log-level=warn --continue --user-agent "{user_agent}" {filename_cmd} "{web_dl_file}" '''
    
    # Try the download up to 3 times
    for attempt in range(3):
        try:
            result = os.system(command)
            if result == 0:
                return True
            print(f"Download attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            print(f"Error during download: {str(e)}")
            if attempt < 2:
                print("Retrying...")
            continue
    
    print(f"Failed to download {filename} after 3 attempts")
    return False

def get_model_name_from_hf_url(url):
    # Extract model name from URL like https://huggingface.co/bartowski/magnum-v4-22b-GGUF/blob/main/magnum-v4-22b-Q6_K.gguf
    parts = urlparse(url).path.split('/')
    if len(parts) >= 3:
        return parts[2]  # Get the model name part (e.g., 'magnum-v4-22b-GGUF')
    return None

def prepare_llm_folder(model_name):
    """Create a specific folder for each LLM model"""
    folder_path = Path(f"{model_storage_dir}/llm_checkpoints/{model_name}")
    folder_path.mkdir(parents=True, exist_ok=True)
    os.chdir(folder_path)
    return folder_path

def get_hf_repo_files(repo_id, headers):
    """Get list of files from a Hugging Face repository"""
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return [item['path'] for item in response.json()]
    return []

def download_hf_model(model_uri, headers, model_type=None):
    # Extract repo_id from URL
    parts = urlparse(model_uri).path.split('/')
    repo_id = '/'.join(parts[1:3])
    model_name = repo_id.split('/')[-1]
    
    # If this is not an LLM model, don't create a specific subfolder
    if model_type != 'llm':
        folder_path = Path(model_storage_dir) / model_type
    else:
        folder_path = prepare_llm_folder(model_name)
    
    os.chdir(folder_path)
    
    # Check if this is a specific file URL (contains /blob/main/)
    if '/blob/' in model_uri:
        filename = os.path.basename(model_uri)
        file_url = model_uri.replace('/blob/', '/resolve/')
        print(f"Downloading single file: {filename}...")
        if not dl_web_file(file_url, filename=filename, token=hf_token):
            print(f"Failed to download {filename}, skipping...")
        return
        
    # If not a specific file, download all repository files
    files = get_hf_repo_files(repo_id, headers)
    for file in files:
        if file in ['.gitattributes']:  # Only skip git files
            continue
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
        print(f"Downloading {file}...")
        if not dl_web_file(file_url, filename=file, token=hf_token):
            print(f"Failed to download {file}, skipping...")

def downlaod_model(model_uri, model_type=None):
    model_uri = model_uri.strip()
    headers={'User-Agent': user_agent}
    
    # Handle Hugging Face downloads
    if 'https://huggingface.co/' in model_uri:
        if hf_token:
            headers['Authorization'] = f'Bearer {hf_token}'
        response = requests.head(model_uri, allow_redirects=True, headers=headers)
        if response.status_code == 401:
            print('Huggingface token is invalid or not provided, please check your HF_TOKEN environment variable.')
        else:
            download_hf_model(model_uri, headers, model_type)
        return
    
    magnet_match = re.search(r'magnet:\?xt=urn:btih:[\-_A-Za-z0-9&=%.]*', model_uri)
    civitai_match = re.search(r'^https?:\/\/(?:www\.|(?!www))civitai\.com\/(models\/\d+|api\/download\/models\/\d+)', model_uri)
    web_match = is_url(model_uri)

    if magnet_match:
        bash_var = magnet_match[0]
        command = f'''aria2c --seed-time=0 --max-overall-upload-limit=1K --bt-max-peers=120 --summary-interval=0 --console-log-level=warn --file-allocation=none "{bash_var}"'''
        os.system(command)
    elif 'https://drive.google.com' in model_uri:
        gdrive_file_id, _ = gdown.parse_url.parse_url(model_uri)
        gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}&confirm=t")
    elif civitai_match:
        if '/api/download/' in model_uri:
            model_id = model_uri.split('/')[-1].split('?')[0]  # Extract model ID
            download_url = f"https://civitai.com/api/download/models/{model_id}"
            if civitai_token:
                download_url += f"?token={civitai_token}"
            dl_web_file(download_url)
        else:
            if not is_url(civitai_match[0]):
                print('URL does not match known civitai.com pattern.')
            else:
                soup = BeautifulSoup(requests.get(model_uri, headers=headers).text, features="html.parser")
                data = json.loads(soup.find('script', {'id': '__NEXT_DATA__'}).text)
                model_data = data["props"]["pageProps"]["trpcState"]["json"]["queries"][0]["state"]["data"]
                latest_model = model_data['modelVersions'][0]
                
                latest_model_url = f"https://civitai.com/api/download/models/{latest_model['id']}"
                if civitai_token:
                    latest_model_url += f"?token={civitai_token}"
                
                print('Downloading model:', model_data['name'])
                dl_web_file(latest_model_url)
    elif web_match:
        # Always do the web match last
        with requests.get(web_match[0], allow_redirects=True, stream=True, headers=headers) as r:
            # Uing GET since some servers respond differently to HEAD.
            # Using `with` so we can close the connection and not download the entire file.
            response = r
            r.close()
        if response.headers.get('content-type') or response.headers.get('content-disposition'):
            if 'octet-stream' in response.headers.get('content-type', '') or 'attachment' in response.headers.get('content-disposition', ''):
                dl_web_file(model_uri)
                # clean exit here
            else:
                print('Required HTTP headers are incorrect. One of these needs to be correct:', end='\n\n')
                print('Content-Type:', response.headers['content-type'].split(";")[0] if response.headers.get('content-type') else 'None')
                print('Must be "application/octet-stream"', end='\n\n')
                print('Content-Disposition:', response.headers['content-disposition'] if response.headers.get('content-disposition') else 'None')
                print('Must start with "attachment;"')
                # clean exit here
        else:
            print('Required HTTP headers are missing. You need at lease one of these:', end='\n\n')
            print('Content-Type:', response.headers['content-type'].split(";")[0] if response.headers.get('content-type') else 'None')
            print('Must be "application/octet-stream"', end='\n\n')
            print('Content-Disposition:', response.headers['content-disposition'] if response.headers.get('content-disposition') else 'None')
            print('Must start with "attachment;"')
    else:
        print('Could not parse your URI.')
        # clean exit here

def prepare_folder(name):
    os.makedirs(f"{model_storage_dir}/{name}",exist_ok=True)
    os.chdir(f"{model_storage_dir}/{name}")  

prepare_folder("sd")
model_list = os.environ.get('MODEL_LIST', "").split(',')
for uri in model_list:
    if uri != '':
        # If it's a HuggingFace URL, treat it as an LLM model
        if 'https://huggingface.co/' in uri:
            downlaod_model(uri, 'llm')
        else:
            downlaod_model(uri, 'sd')

prepare_folder("lora")
lora_list = os.environ.get('LORA_LIST', "").split(',')
for uri in lora_list:
    if uri != '':
        downlaod_model(uri, 'lora')

prepare_folder("controlnet")
controlnet_list = os.environ.get('CONTROLNET_LIST', "").split(',')
for uri in controlnet_list:
    if uri != '':
        downlaod_model(uri, 'controlnet')

prepare_folder("vae")
vae_list = os.environ.get('VAE_LIST', "").split(',')
for uri in vae_list:
    if uri != '':
        downlaod_model(uri, 'vae')
        
prepare_folder("embedding") 
embedding_list = os.environ.get('EMBEDDING_LIST', "").split(',')
for uri in embedding_list:
    if uri != '':
        downlaod_model(uri, 'embedding')
        
prepare_folder("upscaler") 
upscaler_list = os.environ.get('UPSCALER_LIST', "").split(',')
for uri in upscaler_list:
    if uri != '':
        downlaod_model(uri, 'upscaler')

# Add new LLM model list handling
prepare_folder("LLM_checkpoints")
llm_list = os.environ.get('LLM_LIST', "").split(',')
for uri in llm_list:
    if uri != '':
        downlaod_model(uri, 'llm')
