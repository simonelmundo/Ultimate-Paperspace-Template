# from https://github.com/Engineer-of-Stuff/stable-diffusion-paperspace
import re
import requests
import gdown
import json
from bs4 import BeautifulSoup
import os
import shutil
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

load_dotenv()

model_storage_dir = os.environ['MODEL_DIR']
hf_token = os.environ.get('HF_TOKEN', '')
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'

# Add new environment variable for Civitai token
civitai_token = 'bf5a73346bccc8ab11cd99e1386a0e1b'
civitai_token = 'be4ab0abfc4f8ff45247122f0ccd0196'

# Hugging Face CDN signed URLs + multi-range aria2c often hit 403s; keep HF conservative.
HF_ARIA2_CONNECTIONS = 4
DEFAULT_ARIA2_CONNECTIONS = 16

def is_url(url_str):
    return re.search(r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}', url_str)

def parse_hf_url(url):
    """Parse huggingface.co resolve/blob URL into (repo_id, revision, file_path)."""
    url = url.replace('/blob/', '/resolve/')
    match = re.match(
        r'https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+?)(?:\?|#|$)',
        url,
    )
    if not match:
        return None
    repo_id, revision, file_path = match.groups()
    return repo_id, revision, unquote(file_path)

def dl_web_file(web_dl_file, filename=None, token=None, connections=DEFAULT_ARIA2_CONNECTIONS):
    web_dl_file = is_url(web_dl_file)[0] # clean the URL string
    filename_cmd = f'--out="{filename}"' if filename else ''
    token_cmd = f'--header="Authorization: Bearer {token}"' if token else ''
    # Split across connections for speed. Use fewer connections for HF (see dl_huggingface).
    command = (
        f'aria2c --check-certificate=false {token_cmd} --file-allocation=none -c '
        f'-x {connections} -s {connections} --max-connection-per-server={connections} '
        f'--retry-wait=2 --max-tries=10 '
        f'--summary-interval=0 --console-log-level=warn --continue '
        f'--enable-http-keep-alive=false --user-agent "{user_agent}" '
        f'{filename_cmd} "{web_dl_file}" '
    )
    os.system(command)

def _ensure_hf_transfer():
    """Enable Rust hf_transfer when installed (installed by main.sh / sd_comfy)."""
    try:
        import hf_transfer  # noqa: F401
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        return True
    except ImportError:
        os.environ.pop('HF_HUB_ENABLE_HF_TRANSFER', None)
        return False

def _place_hf_file(downloaded_path, dest_filename, from_cache=False):
    """Place downloaded HF file at cwd/basename (flat model layout)."""
    dest = os.path.abspath(os.path.join(os.getcwd(), dest_filename))
    downloaded_path = os.path.abspath(downloaded_path)
    if downloaded_path == dest:
        return dest
    os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
    if os.path.exists(dest) or os.path.islink(dest):
        os.remove(dest)
    if from_cache:
        # Keep HF cache intact; prefer hardlink/symlink to avoid doubling large files.
        try:
            os.link(downloaded_path, dest)
        except OSError:
            try:
                os.symlink(downloaded_path, dest)
            except OSError:
                shutil.copy2(downloaded_path, dest)
        return dest
    shutil.move(downloaded_path, dest)
    parent = os.path.dirname(downloaded_path)
    cwd = os.path.abspath(os.getcwd())
    while parent and parent.startswith(cwd) and parent != cwd:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)
    return dest

def dl_via_hf_hub(repo_id, file_path, revision, dest_filename, token=None):
    """Download via huggingface_hub (+ hf_transfer when available). Returns True on success."""
    try:
        import inspect
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('huggingface_hub not installed; falling back to tuned aria2c for Hugging Face.')
        return False

    transfer_ok = _ensure_hf_transfer()
    print(
        f'Downloading from Hugging Face via huggingface_hub'
        f'{" + hf_transfer" if transfer_ok else ""}: {repo_id}/{file_path}'
    )
    try:
        params = inspect.signature(hf_hub_download).parameters
        kwargs = {
            'repo_id': repo_id,
            'filename': file_path,
            'revision': revision,
        }
        if token:
            if 'token' in params:
                kwargs['token'] = token
            elif 'use_auth_token' in params:
                kwargs['use_auth_token'] = token

        # Older hub builds (common on Paperspace base images) lack local_dir.
        if 'local_dir' in params:
            kwargs['local_dir'] = os.getcwd()
            downloaded = hf_hub_download(**kwargs)
            _place_hf_file(downloaded, dest_filename, from_cache=False)
        else:
            downloaded = hf_hub_download(**kwargs)
            _place_hf_file(downloaded, dest_filename, from_cache=True)
        return True
    except Exception as exc:
        print(f'huggingface_hub download failed ({exc}); falling back to tuned aria2c.')
        return False

def dl_huggingface(model_uri, token=None):
    """Hybrid HF download: huggingface_hub/hf_transfer first, tuned aria2c fallback."""
    resolve_uri = model_uri.replace('/blob/', '/resolve/')
    filename = os.path.basename(urlparse(resolve_uri).path)
    parsed = parse_hf_url(resolve_uri)

    if parsed:
        repo_id, revision, file_path = parsed
        if dl_via_hf_hub(repo_id, file_path, revision, filename, token=token):
            return

    print(f'Using tuned aria2c for Hugging Face (-x {HF_ARIA2_CONNECTIONS}): {filename}')
    dl_web_file(resolve_uri, filename, token=token, connections=HF_ARIA2_CONNECTIONS)

def downlaod_model(model_uri):
    model_uri = model_uri.strip()
    headers={'User-Agent': user_agent}
    magnet_match = re.search(r'magnet:\?xt=urn:btih:[\-_A-Za-z0-9&=%.]*', model_uri)
    civitai_match = re.search(r'^https?:\/\/(?:www\.|(?!www))civitai\.com\/(models\/\d+|api\/download\/models\/\d+)', model_uri)
    web_match = is_url(model_uri)

    if magnet_match:
        bash_var = magnet_match[0]
        command = f'''aria2c --seed-time=0 --max-overall-upload-limit=1K --bt-max-peers=120 --summary-interval=0 --console-log-level=warn --file-allocation=none "{bash_var}"'''
        os.system(command)
        # clean exit here
    elif 'https://huggingface.co/' in model_uri:
        if hf_token:
            headers['Authorization'] = f'Bearer {hf_token}'
        response = requests.head(model_uri, allow_redirects=True, headers=headers)
        if response.status_code == 401:
            print('Huggingface token is invalid or not provided, please check your HF_TOKEN environment variable.')
        else:
            dl_huggingface(model_uri, token=hf_token)
            # clean exit here
    elif 'https://drive.google.com' in model_uri:
        gdrive_file_id, _ = gdown.parse_url.parse_url(model_uri)
        gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}&confirm=t")
        # clean exit here
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

# Download order: VAE -> ControlNet -> Upscaler -> LoRA -> SD -> Embedding
# VAE first (needed for image encoding/decoding)
prepare_folder("vae")
vae_list = os.environ.get('VAE_LIST', "").split(',')
for uri in vae_list:
    if uri != '':
        downlaod_model(uri)

# ControlNet second (needed for control features)
prepare_folder("controlnet")
controlnet_list = os.environ.get('CONTROLNET_LIST', "").split(',')
for uri in controlnet_list:
    if uri != '':
        downlaod_model(uri)

# Upscaler third (needed for image upscaling)
prepare_folder("upscaler") 
upscaler_list = os.environ.get('UPSCALER_LIST', "").split(',')
for uri in upscaler_list:
    if uri != '':
        downlaod_model(uri)

# LoRA fourth (needed for model fine-tuning)
prepare_folder("lora")
lora_list = os.environ.get('LORA_LIST', "").split(',')
for uri in lora_list:
    if uri != '':
        downlaod_model(uri)

# SD models fifth (main stable diffusion models)
prepare_folder("sd")
model_list = os.environ.get('MODEL_LIST', "").split(',')
for uri in model_list:
    if uri != '':
        downlaod_model(uri)

# Embedding last (text embeddings)
prepare_folder("embedding") 
embedding_list = os.environ.get('EMBEDDING_LIST', "").split(',')
for uri in embedding_list:
    if uri != '':
        downlaod_model(uri)
