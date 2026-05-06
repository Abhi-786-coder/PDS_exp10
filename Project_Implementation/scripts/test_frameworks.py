import requests
import json

resp = requests.post(
    "http://localhost:8000/analyze",
    json={"smiles": "Nc1ccccc1"},
    timeout=180
)
d = resp.json()

print("=" * 50)
print("STBI Assessment:", d.get('stbi', {}).get('assessment'))
print("STBI Score:     ", d.get('stbi', {}).get('stbi'))
print("STBI Message:   ", d.get('stbi', {}).get('message', '')[:80])
print()
print("CONSTELLATION:  ", d.get('constellation', {}).get('constellation_name'))
print("PROXIMITY:      ", d.get('constellation', {}).get('proximity_label'))
print("MECHANISM:      ", d.get('constellation', {}).get('mechanism_hint'))
print()
ep = d.get('escape_path', {})
summ = ep.get('summary', {})
print("PROGNOSIS:      ", summ.get('overall_prognosis'))
print("EASY/HARD/TRAP: ", f"{summ.get('easy_count',0)}E / {summ.get('hard_count',0)}H / {summ.get('trapped_count',0)}T")
print("MESSAGE:        ", summ.get('prognosis_message', '')[:120])
print("=" * 50)
print("ALL KEYS IN RESPONSE:", list(d.keys()))
