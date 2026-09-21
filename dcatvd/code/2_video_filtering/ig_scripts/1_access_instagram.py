from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir="playwright_profile",
        headless=False
    )

    page = browser.new_page()
    page.goto("https://www.instagram.com/")

    print("Log into Instagram, then press Enter")
    input()

    browser.close()