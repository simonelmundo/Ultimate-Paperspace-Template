# from https://github.com/Engineer-of-Stuff/stable-diffusion-paperspace
import re
import requests
import gdown
import json
from bs4 import BeautifulSoup
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Parallelize only files smaller than this; large files stay one-at-a-time.
PARALLEL_MAX_BYTES = 3 * 1024 ** 3  # 3 GiB
MAX_PARALLEL_DOWNLOADS = int(os.environ.get('MAX_PARALLEL_DOWNLOADS', '3'))

_print_lock = threading.Lock()

def log(msg):
    with _print_lock:
        print(msg, flush=True)

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

def probe_remote_size(uri):
    """Return remote Content-Length in bytes, or None if unknown."""
    headers = {'User-Agent': user_agent}
    probe_uri = uri.strip()
    try:
        if 'huggingface.co' in probe_uri:
            probe_uri = probe_uri.replace('/blob/', '/resolve/')
            if hf_token:
                headers['Authorization'] = f'Bearer {hf_token}'
        elif 'civitai.com' in probe_uri and civitai_token and 'token=' not in probe_uri:
            sep = '&' if '?' in probe_uri else '?'
            if '/api/download/' in probe_uri:
                probe_uri = f'{probe_uri}{sep}token={civitai_token}'

        response = requests.head(probe_uri, allow_redirects=True, headers=headers, timeout=30)
        length = response.headers.get('Content-Length') or response.headers.get('content-length')
        if length and length.isdigit():
            return int(length)

        # Some CDNs omit Content-Length on HEAD; try a 1-byte range GET.
        range_headers = dict(headers)
        range_headers['Range'] = 'bytes=0-0'
        with requests.get(probe_uri, allow_redirects=True, headers=range_headers, timeout=30, stream=True) as response:
            content_range = response.headers.get('Content-Range') or response.headers.get('content-range')
            if content_range and '/' in content_range:
                total = content_range.rsplit('/', 1)[-1]
                if total.isdigit():
                    return int(total)
            length = response.headers.get('Content-Length') or response.headers.get('content-length')
            if length and length.isdigit():
                return int(length)
    except Exception as exc:
        log(f'Could not probe size for {uri}: {exc}')
    return None

def format_size(num_bytes):
    if num_bytes is None:
        return 'unknown'
    gib = num_bytes / (1024 ** 3)
    if gib >= 1:
        return f'{gib:.2f} GiB'
    mib = num_bytes / (1024 ** 2)
    return f'{mib:.1f} MiB'

def dl_web_file(web_dl_file, filename=None, token=None, connections=DEFAULT_ARIA2_CONNECTIONS, dest_dir=None):
    web_dl_file = is_url(web_dl_file)[0] # clean the URL string
    filename_cmd = f'--out="{filename}"' if filename else ''
    token_cmd = f'--header="Authorization: Bearer {token}"' if token else ''
    dir_cmd = f'--dir="{dest_dir}"' if dest_dir else ''
    # Split across connections for speed. Use fewer connections for HF (see dl_huggingface).
    command = (
        f'aria2c --check-certificate=false {token_cmd} --file-allocation=none -c '
        f'-x {connections} -s {connections} --max-connection-per-server={connections} '
        f'--retry-wait=2 --max-tries=10 '
        f'--summary-interval=0 --console-log-level=warn --continue '
        f'--enable-http-keep-alive=false --user-agent "{user_agent}" '
        f'{dir_cmd} {filename_cmd} "{web_dl_file}" '
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

def _place_hf_file(downloaded_path, dest_filename, dest_dir, from_cache=False):
    """Place downloaded HF file at dest_dir/basename (flat model layout)."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.abspath(os.path.join(dest_dir, dest_filename))
    downloaded_path = os.path.abspath(downloaded_path)
    if downloaded_path == dest:
        return dest
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
    dest_dir_abs = os.path.abspath(dest_dir)
    while parent and parent.startswith(dest_dir_abs) and parent != dest_dir_abs:
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)
    return dest

def dl_via_hf_hub(repo_id, file_path, revision, dest_filename, token=None, dest_dir=None):
    """Download via huggingface_hub (+ hf_transfer when available). Returns True on success."""
    dest_dir = dest_dir or os.getcwd()
    try:
        import inspect
        from huggingface_hub import hf_hub_download
    except ImportError:
        log('huggingface_hub not installed; falling back to tuned aria2c for Hugging Face.')
        return False

    transfer_ok = _ensure_hf_transfer()
    log(
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
            kwargs['local_dir'] = dest_dir
            downloaded = hf_hub_download(**kwargs)
            _place_hf_file(downloaded, dest_filename, dest_dir, from_cache=False)
        else:
            downloaded = hf_hub_download(**kwargs)
            _place_hf_file(downloaded, dest_filename, dest_dir, from_cache=True)
        return True
    except Exception as exc:
        log(f'huggingface_hub download failed ({exc}); falling back to tuned aria2c.')
        return False

def dl_huggingface(model_uri, token=None, dest_dir=None):
    """Hybrid HF download: huggingface_hub/hf_transfer first, tuned aria2c fallback."""
    dest_dir = dest_dir or os.getcwd()
    resolve_uri = model_uri.replace('/blob/', '/resolve/')
    filename = os.path.basename(urlparse(resolve_uri).path)
    parsed = parse_hf_url(resolve_uri)

    if parsed:
        repo_id, revision, file_path = parsed
        if dl_via_hf_hub(repo_id, file_path, revision, filename, token=token, dest_dir=dest_dir):
            return

    log(f'Using tuned aria2c for Hugging Face (-x {HF_ARIA2_CONNECTIONS}): {filename}')
    dl_web_file(
        resolve_uri,
        filename,
        token=token,
        connections=HF_ARIA2_CONNECTIONS,
        dest_dir=dest_dir,
    )

def downlaod_model(model_uri, dest_dir=None):
    model_uri = model_uri.strip()
    dest_dir = dest_dir or os.getcwd()
    os.makedirs(dest_dir, exist_ok=True)
    headers={'User-Agent': user_agent}
    magnet_match = re.search(r'magnet:\?xt=urn:btih:[\-_A-Za-z0-9&=%.]*', model_uri)
    civitai_match = re.search(r'^https?:\/\/(?:www\.|(?!www))civitai\.com\/(models\/\d+|api\/download\/models\/\d+)', model_uri)
    web_match = is_url(model_uri)

    if magnet_match:
        bash_var = magnet_match[0]
        command = (
            f'aria2c --seed-time=0 --max-overall-upload-limit=1K --bt-max-peers=120 '
            f'--summary-interval=0 --console-log-level=warn --file-allocation=none '
            f'--dir="{dest_dir}" "{bash_var}"'
        )
        os.system(command)
    elif 'https://huggingface.co/' in model_uri:
        if hf_token:
            headers['Authorization'] = f'Bearer {hf_token}'
        response = requests.head(model_uri, allow_redirects=True, headers=headers)
        if response.status_code == 401:
            log('Huggingface token is invalid or not provided, please check your HF_TOKEN environment variable.')
        else:
            dl_huggingface(model_uri, token=hf_token, dest_dir=dest_dir)
    elif 'https://drive.google.com' in model_uri:
        gdrive_file_id, _ = gdown.parse_url.parse_url(model_uri)
        # Trailing sep tells gdown to keep the remote filename inside dest_dir.
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_file_id}&confirm=t",
            output=dest_dir if dest_dir.endswith(os.sep) else dest_dir + os.sep,
        )
    elif civitai_match:
        if '/api/download/' in model_uri:
            model_id = model_uri.split('/')[-1].split('?')[0]  # Extract model ID
            download_url = f"https://civitai.com/api/download/models/{model_id}"
            if civitai_token:
                download_url += f"?token={civitai_token}"
            dl_web_file(download_url, dest_dir=dest_dir)
        else:
            if not is_url(civitai_match[0]):
                log('URL does not match known civitai.com pattern.')
            else:
                soup = BeautifulSoup(requests.get(model_uri, headers=headers).text, features="html.parser")
                data = json.loads(soup.find('script', {'id': '__NEXT_DATA__'}).text)
                model_data = data["props"]["pageProps"]["trpcState"]["json"]["queries"][0]["state"]["data"]
                latest_model = model_data['modelVersions'][0]

                latest_model_url = f"https://civitai.com/api/download/models/{latest_model['id']}"
                if civitai_token:
                    latest_model_url += f"?token={civitai_token}"

                log(f'Downloading model: {model_data["name"]}')
                dl_web_file(latest_model_url, dest_dir=dest_dir)
    elif web_match:
        # Always do the web match last
        with requests.get(web_match[0], allow_redirects=True, stream=True, headers=headers) as r:
            # Uing GET since some servers respond differently to HEAD.
            # Using `with` so we can close the connection and not download the entire file.
            response = r
            r.close()
        if response.headers.get('content-type') or response.headers.get('content-disposition'):
            if 'octet-stream' in response.headers.get('content-type', '') or 'attachment' in response.headers.get('content-disposition', ''):
                dl_web_file(model_uri, dest_dir=dest_dir)
            else:
                log('Required HTTP headers are incorrect. One of these needs to be correct:\n')
                log('Content-Type: ' + (response.headers['content-type'].split(";")[0] if response.headers.get('content-type') else 'None'))
                log('Must be "application/octet-stream"\n')
                log('Content-Disposition: ' + (response.headers['content-disposition'] if response.headers.get('content-disposition') else 'None'))
                log('Must start with "attachment;"')
        else:
            log('Required HTTP headers are missing. You need at lease one of these:\n')
            log('Content-Type: ' + (response.headers['content-type'].split(";")[0] if response.headers.get('content-type') else 'None'))
            log('Must be "application/octet-stream"\n')
            log('Content-Disposition: ' + (response.headers['content-disposition'] if response.headers.get('content-disposition') else 'None'))
            log('Must start with "attachment;"')
    else:
        log('Could not parse your URI.')

def prepare_folder(name):
    path = f"{model_storage_dir}/{name}"
    os.makedirs(path, exist_ok=True)
    return path

def collect_jobs():
    """Build (uri, dest_dir) list in category order."""
    categories = [
        ("vae", "VAE_LIST"),
        ("controlnet", "CONTROLNET_LIST"),
        ("upscaler", "UPSCALER_LIST"),
        ("lora", "LORA_LIST"),
        ("sd", "MODEL_LIST"),
        ("embedding", "EMBEDDING_LIST"),
    ]
    jobs = []
    for folder, env_key in categories:
        dest_dir = prepare_folder(folder)
        for uri in os.environ.get(env_key, "").split(','):
            uri = uri.strip()
            if uri:
                jobs.append((uri, dest_dir))
    return jobs

def run_downloads():
    jobs = collect_jobs()
    if not jobs:
        log('No models to download.')
        return

    small_jobs = []
    large_jobs = []
    log(f'Probing sizes for {len(jobs)} downloads (parallel if < 3 GiB)...')
    for uri, dest_dir in jobs:
        size = probe_remote_size(uri)
        label = os.path.basename(urlparse(uri.replace('/blob/', '/resolve/')).path) or uri
        if size is not None and size < PARALLEL_MAX_BYTES:
            small_jobs.append((uri, dest_dir, size))
            log(f'  [parallel] {label} ({format_size(size)})')
        else:
            large_jobs.append((uri, dest_dir, size))
            reason = 'unknown size' if size is None else format_size(size)
            log(f'  [sequential] {label} ({reason})')

    workers = max(1, min(MAX_PARALLEL_DOWNLOADS, len(small_jobs) or 1))
    log(f'Starting {len(small_jobs)} parallel downloads (workers={workers}), then {len(large_jobs)} sequential.')

    if small_jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(downlaod_model, uri, dest_dir): (uri, size)
                for uri, dest_dir, size in small_jobs
            }
            for future in as_completed(futures):
                uri, size = futures[future]
                label = os.path.basename(urlparse(uri.replace('/blob/', '/resolve/')).path) or uri
                try:
                    future.result()
                    log(f'Finished parallel download: {label} ({format_size(size)})')
                except Exception as exc:
                    log(f'Parallel download failed for {label}: {exc}')

    for uri, dest_dir, size in large_jobs:
        label = os.path.basename(urlparse(uri.replace('/blob/', '/resolve/')).path) or uri
        log(f'Starting sequential download: {label} ({format_size(size)})')
        try:
            downlaod_model(uri, dest_dir)
            log(f'Finished sequential download: {label}')
        except Exception as exc:
            log(f'Sequential download failed for {label}: {exc}')

run_downloads()
