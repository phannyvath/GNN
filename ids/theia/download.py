"""Verified FLASH notebook archive IDs; bounded streaming, no archive extraction."""
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import tarfile
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from .common import save_json, sha256

ARCHIVES = {
    '1r': ('10cecNtR3VsHfV0N-gNEeoVeB89kCnse5', 'ta1-theia-e3-official-1r.json.tar.gz'),
    '6r': ('1Kadc6CUTb4opVSDE4x6RFFnEy0P1cRp0', 'ta1-theia-e3-official-6r.json.tar.gz'),
}
SOURCE = 'https://github.com/DART-Laboratory/Flash-IDS/blob/main/Theia.ipynb'


class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.action = None; self.fields = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('type') == 'hidden':
            self.fields[attrs['name']] = attrs.get('value', '')


def open_archive(file_id):
    url = 'https://drive.google.com/uc?' + urllib.parse.urlencode({'export': 'download', 'id': file_id})
    response = urllib.request.urlopen(url, timeout=60)
    if 'text/html' in response.headers.get('Content-Type', ''):
        page = response.read(1024 * 1024).decode('utf-8'); response.close()
        form = Form(); form.feed(page)
        if not form.action or urllib.parse.urlparse(form.action).netloc != 'drive.usercontent.google.com':
            raise RuntimeError('Drive returned login/quota/access HTML. Open the official file link manually; see docs/THEIA.md.')
        response = urllib.request.urlopen(form.action + '?' + urllib.parse.urlencode(form.fields), timeout=60)
    if 'text/html' in response.headers.get('Content-Type', ''):
        body = response.read(1024 * 1024).decode('utf-8', errors='replace'); response.close()
        title = re.search(r'<title>(.*?)</title>', body, re.I | re.S)
        reason = title.group(1) if title else 'HTML access/quota response'
        raise RuntimeError(f'Drive did not return an archive: {reason}. File ID: {file_id}')
    return response


class BudgetReader:
    def __init__(self, stream, maximum):
        self.stream = stream; self.maximum = maximum; self.count = 0; self.digest = hashlib.sha256()
    def read(self, size=-1):
        size = min(size if size >= 0 else 1024 * 1024, self.maximum - self.count + 1)
        data = self.stream.read(size); self.count += len(data); self.digest.update(data)
        if self.count > self.maximum:
            raise RuntimeError('Download byte budget exceeded; increase --max-download-mib explicitly.')
        return data


def download(output, archive='1r', sample_records=0, max_download_mib=2048):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    file_id, name = ARCHIVES[archive]
    destination = output / (name.replace('.tar.gz', '.sample.jsonl') if sample_records else name)
    if destination.exists():
        raise FileExistsError(f'Refusing to replace {destination}')
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists(): raise FileExistsError(f'Remove or inspect interrupted download: {partial}')
    record_count = 0; member_name = None
    try:
        response = open_archive(file_id)
    except Exception as error:
        with open(output/'download_attempts.jsonl','a') as log:
            log.write(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),
                'archive':name,'file_id':file_id,'status':'failed','error':str(error)})+'\n')
        raise
    with response:
        reader = BudgetReader(response, max_download_mib * 1024 ** 2)
        with open(partial, 'wb') as out:
            if sample_records:
                with tarfile.open(fileobj=reader, mode='r|gz') as tar:
                    for member in tar:
                        if not member.isfile() or not re.search(r'\.json(?:\.\d+)?$', member.name): continue
                        member_name = member.name
                        with tar.extractfile(member) as source:
                            while record_count < sample_records:
                                line = source.readline(16 * 1024 ** 2 + 1)
                                if not line: break
                                if len(line) > 16 * 1024 ** 2: raise ValueError('Oversized CDM record')
                                if not line.strip(): continue
                                json.loads(line)  # do not preserve a partial JSON line
                                out.write(line.rstrip(b'\n') + b'\n'); record_count += 1
                        break  # prefix of first JSON member, deliberately not a representative sample
                if not record_count: raise ValueError('No JSON records found in archive')
            else:
                for block in iter(lambda: reader.read(1024 * 1024), b''): out.write(block)
        if not sample_records:
            with open(partial, 'rb') as check:
                if check.read(2) != b'\x1f\x8b': raise ValueError('Expected gzip archive')
    partial.rename(destination)
    save_json(str(destination) + '.source.json', {
        'source': SOURCE, 'file_url': f'https://drive.google.com/file/d/{file_id}/view',
        'archive': name, 'sample': bool(sample_records), 'member': member_name,
        'records': record_count if sample_records else None, 'downloaded_bytes': reader.count,
        'downloaded_prefix_sha256': reader.digest.hexdigest(), 'file_sha256': sha256(destination),
        'warning': 'Chronological prefix for engineering checks only; not an attack benchmark.' if sample_records else None,
    })
    print(destination)
    return destination
