"""Check all public-export mappings without modifying files."""
from pathlib import Path
import json
from export_provenance import current_digest, numerical_payload
ROOT=Path(__file__).resolve().parent

def verify():
    record=json.loads((ROOT/'EXPORT_PROVENANCE.json').read_text())
    numerical=0
    for name, entry in record['files'].items():
        path=ROOT/name
        if current_digest(path)!=entry['public_sha256']:
            raise ValueError('Public-export checksum mismatch: '+name)
        if 'numeric_payload_sha256' in entry:
            if numerical_payload(json.loads(path.read_text()))!=entry['numeric_payload_sha256']:
                raise ValueError('Numerical-payload mismatch: '+name)
            numerical+=1
    return {'status':'passed','mapped_files':len(record['files']),
            'numeric_payloads':numerical,'scope':'Public bytes and preserved numeric payloads; no measurements or simulations.'}
if __name__=='__main__': print(json.dumps(verify(),indent=2))
