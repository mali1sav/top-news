import streamlit as st
import aiohttp
import asyncio
import json
import time
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# ──────────────────────────────────────────────  CONFIG  ───────────────────────────────────────────── #

st.set_page_config(
    page_title="Top Stories Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")

DEFAULT_CRYPTO_KEYWORDS = [
    "XRP",
    "Bitcoin", "BTC",
    "Ethereum", "ETH",
    "Dogecoin",
    "Solana",
    "BNB",
    "SUI",
    "Pi Network",
    "Shiba Inu",
]

LOCATIONS = {"US": "us", "UK": "uk", "Germany": "de", "Netherlands": "nl"}

RESULTS_FILE = "results.json"
CONFIG_FILE = "config.json"

UNWANTED_PATTERNS = [
    "reddit.com", "youtube.com", "x.com", "twitter.com",
    "wikipedia.org", "airbnb.",
    "coinmarketcap.com/currencies", "coingecko.com",
    "/price/", "/prices/", "coinbase.com/price",
    "mexc.com/price", "coincheckup.com", "binance.com",
    "bnb.bg", "calendar", "events-calendar"
]

# ────────────────────────────────────────────  PERSISTENCE  ─────────────────────────────────────────── #

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except:
        return {"last_run": None}

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def load_persisted_results():
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        cutoff = datetime.now() - timedelta(days=2)
        return [
            r for r in data
            if datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
        ]
    except:
        return []

def save_persisted_results(results: list):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ───────────────────────────────────────────────  SERP CALL  ──────────────────────────────────────────── #

async def perform_search_async(query: str, location: str) -> list:
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "gl": location,
        "hl": "en",
        "tbs": "qdr:d",
        "sort": "date",
        "num": 10,
    }
    url = "https://serpapi.com/search"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    st.error(f"SERPAPI error: {resp.status}")
                    return []
                results_json = await resp.json()
                out = []
                for item in results_json.get("organic_results", []):
                    link = item.get("link", "")
                    title = item.get("title", "")
                    pos = item.get("position")
                    if any(pat in link.lower() for pat in UNWANTED_PATTERNS):
                        continue
                    try:
                        if int(pos) not in (1, 2):
                            continue
                    except:
                        continue
                    out.append({
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Country": None,
                        "Crypto": None,
                        "Keyword": query,
                        "Position": pos,
                        "Title": title,
                        "URL": link,
                    })
                return out
    except Exception as e:
        st.error(f"Error performing search: {e}")
        return []

# ───────────────────────────────────────────────  LOGIC  ────────────────────────────────────────────── #

def deduplicate_results(new, old):
    old_urls = {item["URL"] for item in old}
    return [r for r in new if r["URL"] not in old_urls]

async def run_monitoring_job(cryptos, locs):
    all_res, pb = [], st.progress(0)
    total = len(cryptos) * len(locs)
    done = 0

    for c in cryptos:
        for loc_name, loc_code in locs.items():
            res = await perform_search_async(c, loc_code)
            for r in res:
                r["Country"] = loc_name
                r["Crypto"] = c
            all_res.extend(res)
            done += 1
            pb.progress(done / total)

    unique = list({r["URL"]: r for r in all_res}.values())
    persisted = load_persisted_results()
    new_res = deduplicate_results(unique, persisted)

    if new_res:
        st.success(f"✔ {len(new_res)} new content page(s) found")
    else:
        st.info("No new results worth translating")

    save_persisted_results(persisted + new_res)
    return new_res, persisted + new_res

# ─────────────────────────────────────────────  UI / MAIN  ──────────────────────────────────────────── #

def init_state():
    cfg = load_config()
    st.session_state.setdefault("last_run", cfg.get("last_run"))
    st.session_state.setdefault("selected_cryptos", DEFAULT_CRYPTO_KEYWORDS.copy())
    st.session_state.setdefault("new_coin_input", "")
    st.session_state.setdefault("selected_locations", list(LOCATIONS.keys()))

def fmt_since(ts):
    if not ts:
        return "Never"
    try:
        last = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        diff = datetime.now() - last
        h, m = divmod(int(diff.total_seconds() / 60), 60)
        return f"{h} h {m} m ago" if h else f"{m} m ago"
    except:
        return "Never"

def main():
    st.title("Top Stories Monitor (content pages only)")
    init_state()

    col1, col2, col3 = st.columns([2, 2, 1])

    # ────────────── Column 1: Keywords & Add New ────────────── #
    with col1:
        with st.container(border=True):
            st.subheader("Cryptocurrencies")
            combined_options = list(dict.fromkeys(
                DEFAULT_CRYPTO_KEYWORDS + st.session_state.selected_cryptos
            ))
            selected_cryptos = st.multiselect(
                "Select or Add Cryptocurrencies",
                options=combined_options,
                default=st.session_state.selected_cryptos,
                help="Pick from defaults or any you’ve added",
                label_visibility="collapsed"
            )
            st.session_state.selected_cryptos = selected_cryptos # Update session state

        st.markdown("&nbsp;") # Adds a bit of vertical space

        with st.container(border=True):
            st.subheader("Add a Keyword")
            new_coin = st.text_input(
                "Enter new coin or keyword", 
                st.session_state.new_coin_input,
                label_visibility="collapsed"
            )
            
            add_button_col, quick_run_button_col = st.columns(2)
            with add_button_col:
                add_btn = st.button("Add", key="add_btn", use_container_width=True)
            with quick_run_button_col:
                quick_btn = st.button("Run just this keyword", key="quick_btn", use_container_width=True, type="primary")

            if add_btn:
                coin = new_coin.strip()
                if coin and coin not in st.session_state.selected_cryptos:
                    st.session_state.selected_cryptos.append(coin)
                    st.session_state.new_coin_input = ""  # Clear input
                    st.rerun()  # Rerun only after a successful addition
                elif not coin:
                    st.toast("Keyword cannot be empty")
                else: # This implies coin is not empty but is already in the list
                    st.toast(f"'{coin}' is already in the list")

    # ────────────── Column 2: Locations ───────────── #
    with col2:
        with st.container(border=True):
            st.subheader("Regions")
            selected_locations = st.multiselect(
                "Select Regions",
                options=list(LOCATIONS.keys()),
                default=st.session_state.selected_locations,
                label_visibility="collapsed"
            )
            st.session_state.selected_locations = selected_locations # Update session state

    # ────────────── Column 3: Run & Info ─────────────────── #
    quick_run_crypto = None
    if quick_btn: # This logic needs to be outside the column if it depends on button press
        coin_to_quick_run = new_coin.strip() # Use the value from text_input directly
        if coin_to_quick_run:
            quick_run_crypto = coin_to_quick_run
            if coin_to_quick_run not in st.session_state.selected_cryptos:
                st.session_state.selected_cryptos.append(coin_to_quick_run)
            st.session_state.new_coin_input = "" # Clear input after successful setup for quick run
            # st.rerun() # Rerun will happen after job completion
        else:
            st.toast("Keyword cannot be empty for quick run")
            quick_btn = False # Reset flag if keyword is empty

    with col3:
        with st.container(border=True):
            st.subheader("Monitor Actions")
            
            disable_run_all = False
            next_run_available_in = ""
            if st.session_state.last_run:
                last_run_dt = datetime.strptime(st.session_state.last_run, "%Y-%m-%d %H:%M:%S")
                time_since_last_run = datetime.now() - last_run_dt
                if time_since_last_run < timedelta(hours=2):
                    disable_run_all = True
                    time_until_next_run = timedelta(hours=2) - time_since_last_run
                    hours, remainder = divmod(time_until_next_run.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    next_run_available_in = f"Next full scan available in approx. {int(hours)}h {int(minutes)}m."
            
            run_all_btn = st.button(
                "Run Monitor (all keywords)", 
                type="primary", 
                use_container_width=True, 
                disabled=disable_run_all
            )
            st.caption("To save SERPAPI credits, full scans are limited to once every 2 hours.")
            info_message = f"Last full scan: {fmt_since(st.session_state.last_run)}."
            if disable_run_all and next_run_available_in:
                info_message += f" {next_run_available_in}"
            st.info(info_message)

    # ────────────────────────────────────────────  RESULTS  ───────────────────────────────────────────── #

    results_placeholder = st.empty()
    persisted_results = load_persisted_results()
    if persisted_results:
        results_df = pd.DataFrame(persisted_results).sort_values(by="Timestamp", ascending=False)
        results_placeholder.dataframe(results_df, use_container_width=True, height=min(500, len(results_df) * 35 + 38))
    else:
        results_placeholder.info("No results yet. Run the monitor to fetch news.")

    if run_all_btn or quick_run_crypto:
        cfg = load_config()
        now = datetime.now()

        # The button disabling logic should prevent this, but as a safeguard:
        if run_all_btn and st.session_state.last_run:
            last_run_dt = datetime.strptime(st.session_state.last_run, "%Y-%m-%d %H:%M:%S")
            if datetime.now() - last_run_dt < timedelta(hours=2):
                st.warning("Full run initiated too soon. Please wait for the cooldown. This should have been disabled.")
                st.stop()
        
        # API Key Check
        if not SERP_API_KEY:
            st.error("SERP_API_KEY missing → check .env")
            st.stop()

        cryptos_to_run = [quick_run_crypto] if quick_run_crypto else st.session_state.selected_cryptos
        locs_to_run = {k: LOCATIONS[k] for k in st.session_state.selected_locations}

        if not cryptos_to_run:
            st.error("No keywords selected.")
            st.stop()
        if not locs_to_run:
            st.error("No locations selected.")
            st.stop()
        
        current_action = f"'{quick_run_crypto}' in selected regions" if quick_run_crypto else "all selected keywords/regions"
        with st.spinner(f"Running monitor for {current_action}..."):
            new_results, all_results = asyncio.run(
                run_monitoring_job(cryptos_to_run, locs_to_run)
            )
        
        if all_results:
            all_results_df = pd.DataFrame(all_results).sort_values(by="Timestamp", ascending=False)
            results_placeholder.dataframe(all_results_df, use_container_width=True, height=min(500, len(all_results_df) * 35 + 38))
        else:
            results_placeholder.info("No results found for the current selection.")

        # Update last_run time only for full runs
        if run_all_btn:
            st.session_state.last_run = now.strftime("%Y-%m-%d %H:%M:%S")
            save_config({"last_run": st.session_state.last_run})
        
        if quick_run_crypto:
            st.toast(f"Quick run for '{quick_run_crypto}' complete!")
        
        st.rerun() # Rerun to update displays and clear button states

if __name__ == "__main__":
    main()
