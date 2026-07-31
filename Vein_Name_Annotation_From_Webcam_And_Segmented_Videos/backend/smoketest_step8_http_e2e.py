"""
Build step 10: drive the real Flask HTTP API end to end (not calling pipeline.py
directly) — upload -> poll status -> download both result videos. Requires app.py
already running (py -3.12 app.py) on port 7862.
Run: py -3.12 smoketest_step8_http_e2e.py
"""
import os
import time
import requests

BASE = "http://127.0.0.1:7862"
US = os.path.join(os.path.dirname(__file__), "outputs", "ed_ref", "real_us_20s.mp4")
WC = os.path.join(os.path.dirname(__file__), "outputs", "ed_ref", "real_wc_20s.mp4")


def main():
    r = requests.get(BASE + "/")
    print(f"[smoketest] GET / -> {r.status_code}")
    assert r.status_code == 200

    with open(US, "rb") as us_f, open(WC, "rb") as wc_f:
        files = {
            "ultrasound_video": ("ultrasound.mp4", us_f, "video/mp4"),
            "webcam_video": ("webcam.mp4", wc_f, "video/mp4"),
        }
        r = requests.post(BASE + "/api/jobs", files=files)
    print(f"[smoketest] POST /api/jobs -> {r.status_code} {r.json()}")
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    t0 = time.time()
    while True:
        r = requests.get(f"{BASE}/api/jobs/{job_id}/status")
        data = r.json()
        print(f"  status={data['status']} stage={data['stage']} pct={data['progress_pct']}")
        if data["status"] in ("done", "error"):
            break
        if time.time() - t0 > 600:
            raise SystemExit("timed out waiting for job")
        time.sleep(3)

    assert data["status"] == "done", f"job failed: {data.get('error')}"
    print(f"[smoketest] job done in {time.time()-t0:.1f}s")

    for which in ("intermediate", "final"):
        r = requests.get(f"{BASE}/api/jobs/{job_id}/result/{which}")
        print(f"[smoketest] GET .../result/{which} -> {r.status_code}, "
              f"content-type={r.headers.get('content-type')}, bytes={len(r.content)}")
        assert r.status_code == 200
        assert r.headers.get("content-type") == "video/mp4"
        assert len(r.content) > 1000
        out_path = os.path.join(os.path.dirname(__file__), "outputs", f"smoketest_step8_{which}.mp4")
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"    saved to {out_path}")

    print("[smoketest] OK — full HTTP round trip succeeded")


if __name__ == "__main__":
    main()
