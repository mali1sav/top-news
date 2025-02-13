import streamlit as st
import aiohttp
import asyncio
import json
import time
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Crypto News Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()
SERP_API_KEY = os.getenv('SERP_API_KEY')

# Default crypto keywords
DEFAULT_CRYPTO_KEYWORDS = ["XRP", "Bitcoin", "Ethereum", "Dogecoin", "Solana", "BNB"]

# Define locations
LOCATIONS = {"US": "us", "UK": "uk", "Germany": "de", "Netherlands": "nl"}

# Persistence files
RESULTS_FILE = "results.json"
CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from file."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"last_run": None}

def save_config(config):
    """Save configuration to file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def load_persisted_results():
    """Load persisted results from local file (if exists)."""
    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_persisted_results(results):
    """Save results to local file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

async def perform_search_async(query: str, location: str) -> list:
    """Query SERPAPI for a given query and location."""
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "engine": "google",
        "gl": location,
        "hl": "en",
        "tbs": "qdr:d",  # news from the past day
        "sort": "date",
        "num": 10
    }
    url = "https://serpapi.com/search"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    output = []
                    if "organic_results" in results:
                        for item in results["organic_results"]:
                            link = item.get("link", "")
                            position = item.get("position", None)
                            title = item.get("title", "")
                            # Filter out unwanted domains
                            if any(bad in link for bad in ["reddit.com", "youtube.com", "wikipedia.org", "airbnb.co.uk", "airbnb.com", "yahoo.com", "x.com", "twitter.com"]):
                                continue
                            # Only include results with SERP positions 1 or 2
                            try:
                                pos_int = int(position)
                                if pos_int not in (1, 2):
                                    continue
                            except Exception:
                                continue
                            output.append({
                                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "Country": None,
                                "Crypto": None,
                                "Keyword": query,
                                "Position": position,
                                "Title": title,
                                "URL": link
                            })
                    return output
                else:
                    st.error(f"SERPAPI error: {resp.status}")
                    return []
    except Exception as e:
        st.error(f"Error performing search: {e}")
        return []

def deduplicate_results(new_results, persisted):
    """Remove duplicates based on URL."""
    persisted_urls = {item["URL"] for item in persisted}
    return [res for res in new_results if res["URL"] not in persisted_urls]

async def run_monitoring_job(selected_cryptos, selected_locations):
    """Run the monitoring job with selected cryptocurrencies and locations."""
    all_results = []
    progress_bar = st.progress(0)
    total_combinations = len(selected_cryptos) * len(selected_locations)
    current_progress = 0

    for crypto in selected_cryptos:
        for loc_name, loc_code in selected_locations.items():
            results = await perform_search_async(crypto, loc_code)
            # Annotate results with country and crypto info
            for res in results:
                res["Country"] = loc_name
                res["Crypto"] = crypto
            all_results.extend(results)
            
            # Update progress
            current_progress += 1
            progress_bar.progress(current_progress / total_combinations)

    # Deduplicate by URL
    unique_results = {res["URL"]: res for res in all_results}.values()
    unique_results = list(unique_results)

    # Load previously persisted results
    persisted = load_persisted_results()

    # Find new results not already persisted
    new_results = deduplicate_results(unique_results, persisted)

    if new_results:
        st.success(f"Found {len(new_results)} new result(s)")
    else:
        st.info("No new results found")

    # Save updated results
    updated_results = persisted + new_results
    save_persisted_results(updated_results)

    return new_results, updated_results

def initialize_session_state():
    """Initialize session state variables."""
    # Load saved config
    config = load_config()
    
    if 'last_run' not in st.session_state:
        st.session_state.last_run = config.get('last_run')
    if 'selected_cryptos' not in st.session_state:
        st.session_state.selected_cryptos = DEFAULT_CRYPTO_KEYWORDS
    if 'selected_locations' not in st.session_state:
        st.session_state.selected_locations = list(LOCATIONS.keys())

def get_time_since_last_run():
    """Get a formatted string of time since last run."""
    if not st.session_state.last_run:
        return "Never"
    
    try:
        # Convert string to datetime if needed
        last_run = st.session_state.last_run
        if isinstance(last_run, str):
            last_run = datetime.strptime(last_run, "%Y-%m-%d %H:%M:%S")
        
        time_diff = datetime.now() - last_run
        hours = int(time_diff.total_seconds() / 3600)
        minutes = int((time_diff.total_seconds() % 3600) / 60)
        
        if hours > 0:
            return f"{hours} hours {minutes} minutes ago"
        return f"{minutes} minutes ago"
    except Exception:
        return "Never"

def main():    
    # Initialize session state
    initialize_session_state()
    
    # Create columns for the controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Multi-select for cryptocurrencies
        st.session_state.selected_cryptos = st.multiselect(
            "Select Cryptocurrencies",
            DEFAULT_CRYPTO_KEYWORDS,
            default=st.session_state.selected_cryptos
        )
        
    with col2:
        # Multi-select for locations
        st.session_state.selected_locations = st.multiselect(
            "Select Locations",
            list(LOCATIONS.keys()),
            default=st.session_state.selected_locations
        )
    
    with col3:
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        run_button = st.button("Run Monitor", type="primary", use_container_width=True)
    
    # Load and display previous results
    all_results = load_persisted_results()
    
    # Display results in tabs (All Results first)
    tab1, tab2 = st.tabs(["All Results", "New Results"])
    
    def display_results(results, empty_message):
        if results:
            # Create DataFrame and order columns
            df = pd.DataFrame(results)
            column_order = [
                "Timestamp", "Crypto", "Country", "Position",
                "Title", "URL", "Keyword"
            ]
            df = df[column_order]
            
            # Configure and display the dataframe
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn(
                        "URL",
                        width="medium",
                        help="Click to open article"
                    ),
                    "Title": st.column_config.TextColumn(
                        "Title",
                        width="large",
                        help="Article title"
                    ),
                    "Timestamp": st.column_config.DatetimeColumn(
                        "Time",
                        format="D MMM, HH:mm",
                        width="small"
                    ),
                    "Position": st.column_config.NumberColumn(
                        "Pos",
                        width="small"
                    ),
                    "Crypto": st.column_config.TextColumn(
                        "Crypto",
                        width="small"
                    ),
                    "Country": st.column_config.TextColumn(
                        "Region",
                        width="small"
                    ),
                    "Keyword": st.column_config.TextColumn(
                        "Search Term",
                        width="small"
                    )
                }
            )
        else:
            st.info(empty_message)
    
    # Display all results first
    with tab1:
        display_results(all_results, "No results found")
    
    with tab2:
        st.info("Run the monitor to see new results")
    
    # Display last run time and warning
    last_run_time = get_time_since_last_run()
    
    if st.session_state.last_run:
        try:
            last_run = st.session_state.last_run
            if isinstance(last_run, str):
                last_run = datetime.strptime(last_run, "%Y-%m-%d %H:%M:%S")
            time_diff = datetime.now() - last_run
            hours_since_last_run = time_diff.total_seconds() / 3600
            
            if hours_since_last_run < 2:
                st.warning(
                    f"⏱️ Last run: {last_run_time}\n\n"
                    "⚠️ Running too frequently may use up SERP credits. We recommend waiting at least 2 hours between runs."
                )
            else:
                st.info(f"⏱️ Last run: {last_run_time}")
        except Exception:
            st.info(f"⏱️ Last run: {last_run_time}")
    else:
        st.info("⏱️ Last run: Never")
    
    # Convert selected locations to the format needed
    selected_locations_dict = {k: LOCATIONS[k] for k in st.session_state.selected_locations}
    
    if run_button:
        if not SERP_API_KEY:
            st.error("SERP API key not found. Please check your .env file.")
            return
            
        if not st.session_state.selected_cryptos:
            st.warning("Please select at least one cryptocurrency.")
            return
            
        if not st.session_state.selected_locations:
            st.warning("Please select at least one location.")
            return
            
        with st.spinner("Running monitoring job..."):
            new_results, all_results = asyncio.run(
                run_monitoring_job(st.session_state.selected_cryptos, selected_locations_dict)
            )
            
            # Update and save last run time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_run = current_time
            save_config({"last_run": current_time})
            
            # Update the tabs with new results
            with tab1:
                display_results(all_results, "No results found")
                    
            with tab2:
                display_results(new_results, "No new results in this run")

if __name__ == "__main__":
    main()
