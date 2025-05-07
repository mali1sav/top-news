import streamlit as st
import aiohttp
import asyncio
import json
import time
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

# ──────────────────────────────────────────────  CONFIG  ───────────────────────────────────────────── #

st.set_page_config(page_title="Top Stories Monitor",
                   layout="wide",
                   initial_sidebar_state="collapsed")

load_dotenv()
SERP_API_KEY = os.getenv('SERP_API_KEY')

DEFAULT_CRYPTO_KEYWORDS = [
    "XRP",
    "Bitcoin", "BTC",
    "Ethereum", "ETH",
    "Dogecoin",
    "Solana",
    "BNB",
    "SUI",
    "Pi Network",
    "Shiba Inu"
]

LOCATIONS = {"US": "us", "UK": "uk", "Germany": "de", "Netherlands": "nl"}

RESULTS_FILE = "results.json"
CONFIG_FILE   = "config.json"

# 100 % unwanted sources / listing-pages / price-feeds
UNWANTED_PATTERNS = [
    # social / junk
    "reddit.com", "youtube.com", "x.com", "twitter.com",
    # encyclopaedia
    "wikipedia.org",
    # travel
    "airbnb.",
    # price / chart pages & exchange listings
    "coinmarketcap.com/currencies",
    "coingecko.com",
    "/price/", "/prices/",
    "coinbase.com/price",
    "mexc.com/price",
    "coincheckup.com",
    "binance.com",
    "bnb.bg",
    # random calendars / events
    "calendar", "events-calendar"
]

# ────────────────────────────────────────────  PERSISTENCE  ─────────────────────────────────────────── #

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_run": None}

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def load_persisted_results():
    try:
        with open(RESULTS_FILE, "r") as f:
            data = json.load(f)
        cutoff = datetime.now() - timedelta(days=1)
        return [
            r for r in data
            if datetime.strptime(r["Timestamp"], "%Y-%m-%d %H:%M:%S") >= cutoff
        ]
    except Exception:
        return []

def save_persisted_results(results: list):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

# ─────────────────────────────────────────────  SERP CALL  ──────────────────────────────────────────── #

async def perform_search_async(query: str, location: str) -> list:
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "gl": location,
        "hl": "en",
        "tbs": "qdr:d",     # past 24 h
        "sort": "date",
        "num": 10
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
                    link  = item.get("link", "")
                    title = item.get("title", "")
                    pos   = item.get("position")

                    # HARD filter: chuck anything that obviously isn't an article
                    link_low = link.lower()
                    if any(pat in link_low for pat in UNWANTED_PATTERNS):
                        continue

                    # keep only SERP pos 1-2
                    try:
                        if int(pos) not in (1, 2):
                            continue
                    except Exception:
                        continue

                    out.append({
                        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Country":   None,
                        "Crypto":    None,
                        "Keyword":   query,
                        "Position":  pos,
                        "Title":     title,
                        "URL":       link
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
    done  = 0

    for c in cryptos:
        for loc_name, loc_code in locs.items():
            res = await perform_search_async(c, loc_code)
            for r in res:
                r["Country"] = loc_name
                r["Crypto"]  = c
            all_res.extend(res)

            done += 1
            pb.progress(done / total)

    unique = list({r["URL"]: r for r in all_res}.values())
    persisted = load_persisted_results()
    new_res   = deduplicate_results(unique, persisted)

    if new_res:
        st.success(f"✔ {len(new_res)} new content page(s) found")
    else:
        st.info("No new results worth translating")

    save_persisted_results(persisted + new_res)
    return new_res, persisted + new_res

# ─────────────────────────────────────────────  UI / MAIN  ──────────────────────────────────────────── #

def init_state():
    cfg = load_config()
    st.session_state.setdefault("last_run",           cfg.get("last_run"))
    st.session_state.setdefault("selected_cryptos",   DEFAULT_CRYPTO_KEYWORDS)
    st.session_state.setdefault("selected_locations", list(LOCATIONS.keys()))

def fmt_since(ts):
    if not ts:
        return "Never"
    try:
        last = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") if isinstance(ts, str) else ts
        diff = datetime.now() - last
        h, m = divmod(int(diff.total_seconds() / 60), 60)
        return f"{h} h {m} m ago" if h else f"{m} m ago"
    except Exception:
        return "Never"

def main():
    st.title("Top Stories Monitor (content pages only)")
    init_state()

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.session_state.selected_cryptos = st.multiselect(
            "Cryptocurrencies",
            DEFAULT_CRYPTO_KEYWORDS,
            default=st.session_state.selected_cryptos
        )
    with col2:
        st.session_state.selected_locations = st.multiselect(
            "Regions",
            list(LOCATIONS.keys()),
            default=st.session_state.selected_locations
        )
    with col3:
        st.write("")
        run_btn = st.button("Run Monitor", type="primary", use_container_width=True)

    all_results = load_persisted_results()
    tab_all, tab_new = st.tabs(["All Results", "New Results"])

    def show(df_list, empty_msg):
        if df_list:
            df = pd.DataFrame(df_list)[[
                "Timestamp", "Crypto", "Country", "Position",
                "Title", "URL", "Keyword"
            ]]
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "URL":  st.column_config.LinkColumn("URL",   width="medium"),
                    "Title":st.column_config.TextColumn("Title", width="large"),
                    "Timestamp": st.column_config.DatetimeColumn("Time", format="D MMM, HH:mm")
                }
            )
        else:
            st.info(empty_msg)

    with tab_all: show(all_results, "No stored articles yet")
    with tab_new: st.info("Run the monitor to populate")

    last_run_msg = f"⏱ Last run: {fmt_since(st.session_state.last_run)}"
    if st.session_state.last_run:
        last = datetime.strptime(st.session_state.last_run, "%Y-%m-%d %H:%M:%S")
        if (datetime.now() - last).total_seconds() < 7200:
            st.warning(last_run_msg + " – chill for 2 h to save credits")
        else:
            st.info(last_run_msg)
    else:
        st.info(last_run_msg)

    if run_btn:
        if not SERP_API_KEY:
            st.error("SERP_API_KEY missing → check .env")
            return
        if not st.session_state.selected_cryptos:
            st.warning("Pick at least one coin")
            return
        if not st.session_state.selected_locations:
            st.warning("Pick at least one region")
            return

        with st.spinner("Scraping fresh top-stories…"):
            new, all_now = asyncio.run(
                run_monitoring_job(
                    st.session_state.selected_cryptos,
                    {k: LOCATIONS[k] for k in st.session_state.selected_locations}
                )
            )

        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_run = ts_now
        save_config({"last_run": ts_now})

        with tab_all: show(all_now, "No stored articles yet")
        with tab_new: show(new, "No fresh content this run")

if __name__ == "__main__":
    main()
