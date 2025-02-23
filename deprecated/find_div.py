from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

MAIN_WIKI_URL = "https://hoi4.paradoxwikis.com/Modding"

# Setup Selenium WebDriver
options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("start-maximized")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# Start WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Load the page
driver.get(MAIN_WIKI_URL)

# Wait for the content section
try:
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "mw-content-text")))
    print("✅ Page fully loaded!")
except:
    print("⚠ ERROR: Page did not load in time.")
    driver.quit()
    exit()

# Scroll down to trigger JavaScript loading
for _ in range(3):
    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
    time.sleep(3)

# Extract the page source
soup = BeautifulSoup(driver.page_source, "html.parser")
driver.quit()

# ✅ Print all divs inside #mw-content-text
modding_sections = soup.select("#mw-content-text > div.mw-parser-output > div")

print(f"✅ Found {len(modding_sections)} div elements inside 'mw-content-text'.")

# Print the first few divs to inspect their content
for i, div in enumerate(modding_sections[:5]):  # Print first 5 divs
    print(f"\n🔹 Div {i+1}:\n{div.prettify()[:1000]}...\n")  # Show first 1000 characters
