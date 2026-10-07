import asyncio
import json
import os
import random
import sys
import traceback

import cv2
import ddddocr
import numpy as np
from patchright.async_api import async_playwright

TARGET_URL = "https://www.dola.com/chat/38417995536110865"
PROMPT = (
    "A funny video of a kitten being scolded by its mother cat. "
    "The mother cat looks stern and scolds the kitten, while the kitten "
    "looks guilty and cute with big eyes. Natural lighting, stable exposure, "
    "realistic style, 15 seconds duration, 9:16 vertical aspect ratio, "
    "suitable for TikTok/Reels."
)
MAX_CAPTCHA_TRIES = 5
VIDEO_TIMEOUT_S = 600
POLL_S = 15

slide_det = ddddocr.DdddOcr(det=False, ocr=False, show_ad=False)


def log(*a):
    print(*a, flush=True)


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
    if src.startswith("data:"):
        import base64
        return base64.b64decode(src.split(",", 1)[1])
    resp = await page.context.request.get(src)
    return await resp.body()


def gap_x_ddddocr(piece_bytes, bg_bytes):
    """Return x kiri celah pada gambar background (piksel asli)."""
    res = slide_det.slide_match(piece_bytes, bg_bytes, simple_target=True)
    return res["target"][0]


def gap_x_opencv(piece_bytes, bg_bytes):
    bg = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_COLOR)
    pc = cv2.imdecode(np.frombuffer(piece_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if pc.ndim == 3 and pc.shape[2] == 4:
        ys, xs = np.where(pc[:, :, 3] > 20)
        if len(xs):
            pc = pc[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        pc = cv2.cvtColor(pc[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif pc.ndim == 3:
        pc = cv2.cvtColor(pc, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(
        cv2.Canny(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), 100, 200),
        cv2.Canny(pc, 100, 200),
        cv2.TM_CCOEFF_NORMED,
    )
    return cv2.minMaxLoc(res)[3][0]


async def solve_slider_once(page):
    ctx = None
    for c in [page] + list(page.frames):
        try:
            if await c.locator("[class*='captcha' i] img").count() >= 2:
                ctx = c
                break
        except Exception:
            pass
    if ctx is None:
        log("[!] Gambar captcha tidak ditemukan (cek selector)")
        return False

    imgs = ctx.locator("[class*='captcha' i] img")
    items = []
    for i in range(await imgs.count()):
        el = imgs.nth(i)
        box = await el.bounding_box()
        src = await el.get_attribute("src")
        if box and src and box["width"] > 5:
            items.append((box["width"] * box["height"], box, src))
    if len(items) < 2:
        log("[!] Gambar captcha < 2")
        return False
    items.sort(key=lambda t: t[0], reverse=True)
    _, bg_box, bg_src = items[0]
    _, pc_box, pc_src = items[-1]

    bg_bytes = await fetch_bytes(page, bg_src)
    pc_bytes = await fetch_bytes(page, pc_src)
    bg_w = cv2.imdecode(np.frombuffer(bg_bytes, np.uint8), cv2.IMREAD_COLOR).shape[1]

    try:
        gx = gap_x_ddddocr(pc_bytes, bg_bytes)
        how = "ddddocr"
    except Exception as e:
        log(f"[-] ddddocr gagal ({e}), pakai OpenCV")
        gx = gap_x_opencv(pc_bytes, bg_bytes)
        how = "opencv"

    scale = bg_box["width"] / bg_w
    distance = gx * scale - (pc_box["x"] - bg_box["x"])
    log(f"[+] {how}: gap_x={gx}px asli, jarak drag={distance:.1f}px layar")

    handle = ctx.locator(
        "[class*='captcha' i] [class*='slider' i], "
        "[class*='captcha' i] [class*='drag' i], "
        "[class*='captcha' i] [class*='knob' i]"
    ).last
    hb = await handle.bounding_box()
    if not hb:
        log("[!] Handle slider tidak ditemukan")
        return False

    x0, y0 = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    await page.mouse.move(x0 - 40, y0 + 25, steps=10)
    await page.mouse.move(x0, y0, steps=8)
    await page.wait_for_timeout(random.randint(250, 600))
    await page.mouse.down()
    steps = random.randint(28, 45)
    for i in range(1, steps + 1):
        t = i / steps
        ease = 1 - (1 - t) ** 3
        await page.mouse.move(
            x0 + distance * ease + random.uniform(-0.6, 0.6),
            y0 + random.uniform(-1.5, 1.5),
        )
        await page.wait_for_timeout(random.randint(8, 30))
    await page.mouse.move(x0 + distance + random.uniform(2, 4), y0)
    await page.wait_for_timeout(random.randint(60, 140))
    await page.mouse.move(x0 + distance, y0)
    await page.wait_for_timeout(random.randint(120, 260))
    await page.mouse.up()
    await page.wait_for_timeout(2500)
    return not await captcha_visible(page)


async def handle_captcha(page):
    for attempt in range(1, MAX_CAPTCHA_TRIES + 1):
        if not await captcha_visible(page):
            log("[✓] Tidak ada captcha")
            return True
        log(f"[+] Captcha terdeteksi, percobaan {attempt}/{MAX_CAPTCHA_TRIES}")
        await page.screenshot(path=f"captcha_try{attempt}.png")
        try:
            if await solve_slider_once(page):
                log("[✓] Slider puzzle lolos")
                return True
        except Exception:
            log("[!] Slider error:\n" + traceback.format_exc())
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


async def main():
    status = {"captcha_solved": "NO", "video_found": "NO", "error": "NO"}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # jalan di bawah xvfb
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            viewport={"width": 1366, "height": 768}, locale="en-US"
        )
        page = await context.new_page()
        try:
            cookies = load_cookies()
            if cookies:
                await context.add_cookies(cookies)
                log(f"[+] {len(cookies)} cookie disuntikkan")

            log(f"[+] Membuka {TARGET_URL}")
            resp = await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
            log(f"[+] HTTP {resp.status if resp else 'no response'}")
            await page.wait_for_timeout(8000)

            status["captcha_solved"] = "YES" if await handle_captcha(page) else "NO"
            await page.wait_for_timeout(3000)

            input_el = None
            for sel in ("textarea", "div[contenteditable='true']"):
                if await page.locator(sel).count() > 0:
                    input_el = page.locator(sel).first
                    log(f"[+] Input: {sel}")
                    break
            if input_el is None:
                log("[!] Input tidak ditemukan")
                await page.screenshot(path="dola_debug.png", full_page=True)
                status["error"] = "YES"
                return

            await input_el.click()
            await input_el.fill(PROMPT)
            await page.wait_for_timeout(1000)
            await page.keyboard.press("Enter")
            log("[+] Prompt terkirim")
            await page.wait_for_timeout(5000)
            await handle_captcha(page)

            elapsed, video_src = 0, None
            while elapsed < VIDEO_TIMEOUT_S and not video_src:
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
                if not video_src:
                    await page.wait_for_timeout(POLL_S * 1000)
                    elapsed += POLL_S
                    log(f"[.] menunggu video... {elapsed}s")

            await page.screenshot(path="dola_video_result.png", full_page=True)
            if video_src:
                status["video_found"] = "YES"
                data = await (await context.request.get(video_src)).body()
                with open("result.mp4", "wb") as f:
                    f.write(data)
                log(f"[+] Video disimpan ({len(data)//1024} KB)")
        except Exception:
            log("[!] Fatal:\n" + traceback.format_exc())
            status["error"] = "YES"
            try:
                await page.screenshot(path="dola_debug.png", full_page=True)
            except Exception:
                pass
        finally:
            with open("result_status.txt", "w") as f:
                f.write(f"target_url={TARGET_URL}\n")
                for k, v in status.items():
                    f.write(f"{k}={v}\n")
            await browser.close()


asyncio.run(main())
