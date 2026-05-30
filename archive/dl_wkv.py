"""Download RWKV-v8 CUDA kernels (latest)."""
import urllib.request, json, os

req = urllib.request.Request(
    "https://api.github.com/repos/BlinkDL/RWKV-LM/git/trees/main?recursive=1",
    headers={"Accept": "application/vnd.github.v3+json"},
)
resp = urllib.request.urlopen(req, timeout=15)
data = json.loads(resp.read())
os.makedirs("kernels", exist_ok=True)

for item in data["tree"]:
    p = item["path"]
    if p.startswith("RWKV-v8/cuda/") and p.endswith((".cu", ".cpp", ".h")):
        fname = p.split("/")[-1]
        url = f"https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/{p}"
        urllib.request.urlretrieve(url, f"kernels/{fname}")
        size = os.path.getsize(f"kernels/{fname}")
        print(f"  {fname} ({size} bytes)")
print("Done")
