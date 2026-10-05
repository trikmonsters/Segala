import asyncio
import json
import os
import random
import re

import botright
import cv2
import numpy as np

TARGET_URL = "https://www.dola.com/chat/38417995536110865"
PROMPT = (
    "A funny video of a kitten being scolded by its mother cat. "
    "The mother cat looks stern and scolds the kitten, while the kitten "
    "looks guilty and cute with big eyes. Natural lighting, stable exposure, "
    "realistic style, 15 seconds duration, 9:16 vertical aspect ratio, "
    "suitable for TikTok/Reels."
)
MAX_CAPTCHA_TRIES = 4
VIDEO_TIMEOUT_S = 600
POLL_S = 15


# --------------------------------------------------------------------------
# Util
# --------------------------------------------------------------------------
def load_cookies():
    raw = os.environ.get("DOLA_COOKIES", "").strip()
    if not raw:
        return []
    out = []
    for c in json.loads(raw):
        if "name" not in c or "value" not in c:
            continue
        ck = {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain", ".dola.com"),
            "path": c.get("path", "/"),
        }
        ss = str(c.get("sameSite", "")).capitalize()
        if ss in ("Strict", "Lax", "None"):
            ck["sameSite"] = ss
        exp = c.get("expires", c.get("expirationDate"))
        if isinstance(exp, (int, float)) and exp > 0:
            ck["expires"] = exp
        if c.get("secure") is not None:
            ck["secure"] = bool(c["secure"])
        if c.get("httpOnly") is not None:
            ck["httpOnly"] = bool(c["httpOnly"])
        out.append(ck)
    return out


async def captcha_visible(page):
    for ctx in [page] + list(page.frames):
        try:
            if await ctx.locator("text=Verify to continue").count() > 0:
                return True
        except Exception:
            pass
    return False


async def fetch_bytes(page, src):
    """Ambil bytes gambar (mendukung data: URI dan URL biasa)."""
    if src.startswith("data:"):
        import base64
        return base64.b64decode(src.split(",", 1)[1])
    resp = await page.context.request.get(src)
    return await resp.body()


# --------------------------------------------------------------------------
# Solver 1: Botright (hanya untuk hCaptcha / reCAPTCHA / GeeTest)
# NOTE: page.solve_captcha() TIDAK ADA di Botright -> itu penyebab error lama.
# --------------------------------------------------------------------------
async def try_botright(page):
    for name in ("solve_hcaptcha", "solve_recaptcha", "solve_geetest"):
        fn = getattr(page, name, None)
        if not fn:
            continue
        try:
            await asyncio.wait_for(fn(), timeout=60)
            print(f"[✓] Botright {name} selesai")
            return True
        except Exception as e:
            print(f"[-] Botright {name}: {type(e).__name__}: {str(e)[:100]}")
    return False


# --------------------------------------------------------------------------
# Solver 2: Slider puzzle (gaya TikTok/ByteDance) via OpenCV template matching
# --------------------------------------------------------------------------
async def solve_slider_once(page):
    # Cari context (page atau iframe) yang memuat captcha
    ctx = None
    for c in [page] + list(page.frames):
        try:
            if await c.locator("[class*='captcha' i] img").count() >= 2:
                ctx = c
                break
        except Exception:
            pass
    if ctx is None:
        print("[!] Elemen gambar captcha tidak ditemukan (cek selector)")
        return False

    imgs = ctx.locator("[class*='captcha' i] img")
    items = []
    for i in range(await imgs.count()):
        el = imgs.nth(i)
        box = await el.bounding_box()
        src = await el.get_attribute("src")
        if box and src and box["width"] > 5:
            items.append((box["width"] * box["height"], el, box, src))
    if len(items) < 2:
        print("[!] Gambar captcha < 2")
        return False
    items.sort(key=lambda t: t[0], reverse=True)
    _, _, bg_box, bg_src = items[0]      # terbesar = background
    _, _, pc_box, pc_src = items[-1]     # terkecil = potongan puzzle

    bg = cv2.imdecode(np.frombuffer(await fetch_bytes(page, bg_src), np.uint8), cv2.IMREAD_COLOR)
    pc = cv2.imdecode(np.frombuffer(await fetch_bytes(page, pc_src), np.uint8), cv2.IMREAD_UNCHANGED)
    if bg is None or pc is None:
        print("[!] Gagal decode gambar")
        return False

    # Potong area transparan pada piece
    if pc.ndim == 3 and pc.shape[2] == 4:
        ys, xs = np.where(pc[:, :, 3] > 20)
        if len(xs):
            pc = pc[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        pc_gray = cv2.cvtColor(pc[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        pc_gray = cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY) if pc.ndim == 3 else pc

    bg_edge = cv2.Canny(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), 100, 200)
    pc_edge = cv2.Canny(pc_gray, 100, 200)
    res = cv2.matchTemplate(bg_edge, pc_edge, cv2.TM_CCOEFF_NORMED)
    _, score, _, max_loc = cv2.minMaxLoc(res)

    scale = bg_box["width"] / bg.shape[1]
    target_left = max_loc[0] * scale                      # posisi gap di layar
    start_left = pc_box["x"] - bg_box["x"]                # posisi awal piece
    distance = target_left - start_left
    print(f"[+] match score={score:.2f} jarak drag={distance:.1f}px")

    # Handle slider
    handle = ctx.locator(
        "[class*='captcha' i] [class*='slider' i], "
        "[class*='captcha' i] [class*='drag' i], "
        "[class*='captcha' i] [class*='knob' i]"
    ).last
    hb = await handle.bounding_box()
    if not hb:
        print("[!] Handle slider tidak ditemukan")
        return False

    x0 = hb["x"] + hb["width"] / 2
    y0 = hb["y"] + hb["height"] / 2
    await page.mouse.move(x0, y0)
    await page.wait_for_timeout(random.randint(200, 500))
    await page.mouse.down()

    # Gerakan manusiawi: ease-out + jitter + overshoot kecil
    steps = random.randint(25, 40)
    for i in range(1, steps + 1):
        t = i / steps
        ease = 1 - (1 - t) ** 3
        x = x0 + distance * ease + random.uniform(-0.6, 0.6)
        y = y0 + random.uniform(-1.5, 1.5)
        await page.mouse.move(x, y)
        await page.wait_for_timeout(random.randint(8, 28))
    await page.mouse.move(x0 + distance + random.uniform(2, 4), y0)
    await page.wait_for_timeout(random.randint(60, 140))
    await page.mouse.move(x0 + distance, y0)
    await page.wait_for_timeout(random.randint(100, 250))
    await page.mouse.up()
    await page.wait_for_timeout(2500)
    return not await captcha_visible(page)


async def handle_captcha(page):
    for attempt in range(1, MAX_CAPTCHA_TRIES + 1):
        if not await captcha_visible(page):
            print("[✓] Tidak ada captcha")
            return True
        print(f"[+] Captcha terdeteksi, percobaan {attempt}/{MAX_CAPTCHA_TRIES}")
        await page.screenshot(path=f"captcha_try{attempt}.png")

        if await try_botright(page) and not await captcha_visible(page):
            return True
        try:
            if await solve_slider_once(page):
                print("[✓] Slider puzzle lolos")
                return True
        except Exception as e:
            print(f"[!] Slider error: {type(e).__name__}: {e}")

        # Refresh captcha bila ada tombol refresh
        for ctx in [page] + list(page.frames):
            try:
                r = ctx.locator("[class*='refresh' i]").first
                if await r.count():
                    await r.click(timeout=2000)
                    break
            except Exception:
                pass
        await page.wait_for_timeout(2000)
    return not await captcha_visible(page)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
async def main():
    client = await botright.Botright(headless=True)
    browser = await client.new_browser()
    page = await browser.new_page()
    status = {"captcha_solved": "NO", "video_found": "NO", "error": "NO"}

    try:
        cookies = load_cookies()
        if cookies:
            await page.context.add_cookies(cookies)
            print(f"[+] {len(cookies)} cookie disuntikkan")

        print(f"[+] Membuka {TARGET_URL}")
        resp = await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
        print(f"[+] HTTP {resp.status if resp else 'no response'}")
        await page.wait_for_timeout(8000)

        status["captcha_solved"] = "YES" if await handle_captcha(page) else "NO"
        await page.wait_for_timeout(3000)

        # Cari input
        input_el = None
        for sel in ("textarea", "div[contenteditable='true']"):
            if await page.locator(sel).count() > 0:
                input_el = page.locator(sel).first
                print(f"[+] Input: {sel}")
                break
        if input_el is None:
            print("[!] Input tidak ditemukan")
            await page.screenshot(path="dola_debug.png", full_page=True)
            status["error"] = "YES"
            return

        await input_el.click()
        await input_el.fill(PROMPT)
        await page.wait_for_timeout(1000)
        await page.keyboard.press("Enter")
        print("[+] Prompt terkirim")

        # Captcha bisa muncul lagi setelah submit
        await page.wait_for_timeout(5000)
        await handle_captcha(page)

        # Polling sampai ada <video> (bukan tidur buta 5 menit)
        elapsed = 0
        video_src = None
        while elapsed < VIDEO_TIMEOUT_S:
            vids = page.locator("video")
            for i in range(await vids.count()):
                src = await vids.nth(i).get_attribute("src") or ""
                if not src:
                    s = vids.nth(i).locator("source").first
                    if await s.count():
                        src = await s.get_attribute("src") or ""
                if src.startswith("http"):
                    video_src = src
                    break
            if video_src:
                break
            await page.wait_for_timeout(POLL_S * 1000)
            elapsed += POLL_S
            print(f"[.] menunggu video... {elapsed}s")

        await page.screenshot(path="dola_video_result.png", full_page=True)

        if video_src:
            status["video_found"] = "YES"
            try:
                data = await (await page.context.request.get(video_src)).body()
                with open("result.mp4", "wb") as f:
                    f.write(data)
                print(f"[+] Video disimpan ({len(data)//1024} KB)")
            except Exception as e:
                print(f"[!] Gagal unduh video: {e}")

        content = (await page.content()).lower()
        if any(k in content for k in ("failed", "gagal", "try again", "coba lagi")):
            status["error"] = "YES"
    finally:
        with open("result_status.txt", "w") as f:
            f.write(f"target_url={TARGET_URL}\n")
            for k, v in status.items():
                f.write(f"{k}={v}\n")
        await browser.close()
        await client.close()


asyncio.run(main())
