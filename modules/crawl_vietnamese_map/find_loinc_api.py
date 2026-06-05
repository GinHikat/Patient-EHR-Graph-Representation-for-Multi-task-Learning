from playwright.sync_api import sync_playwright
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def run():
    print("Launching Playwright to intercept LOINC Table network calls...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        def handle_response(response):
            url = response.url
            # Filter out assets
            if any(ext in url.lower() for ext in [".js", ".css", ".png", ".jpg", ".woff", ".svg", ".ico", ".otf"]):
                return
            try:
                content_type = response.headers.get("content-type", "")
                status = response.status
                print(f"Intercepted: {url} | Status: {status} | Content-Type: {content_type}")
            except Exception as e:
                pass

        page.on("response", handle_response)
        
        print("Navigating to https://icd.kcb.vn/loinc-table ...")
        page.goto("https://icd.kcb.vn/loinc-table", timeout=60000)
        
        # Wait 10 seconds for the SPA to load and retrieve data
        time.sleep(10)
        browser.close()
        print("Done.")

if __name__ == "__main__":
    run()
