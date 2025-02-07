import streamlit as st
import asyncio
import aiohttp
import os
import nest_asyncio
import pandas as pd
import json
import ast
import re
from dotenv import load_dotenv
from rapidfuzz import fuzz

# Configure the page for full-screen (wide) layout
st.set_page_config(layout="wide", page_title="Top Stories Monitor")

# Allow nested event loops (useful in Streamlit)
nest_asyncio.apply()

# ------------------------------
# Global Debug Flag: Set to True to show warnings; False to hide them.
# ------------------------------
DEBUG = False

def log_warning(message):
    if DEBUG:
        st.warning(message)
    else:
        print("Warning:", message)

# ------------------------------
# Environment Setup and Config
# ------------------------------
load_dotenv()
SERPAPI_API_KEY = os.getenv('SERP_API_KEY')
OPENROUTER_URL = os.getenv('OPENROUTER_URL', "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEFAULT_MODEL = "qwen/qwen-turbo"  # Adjust as needed

# ------------------------------
# OpenRouter LLM Call Function with Concurrency Control
# ------------------------------
openrouter_semaphore = asyncio.Semaphore(5)

async def call_openrouter_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "OpenDeepResearcher, by Mali",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    async with openrouter_semaphore:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        try:
                            await asyncio.sleep(0.3)
                            return result['choices'][0]['message']['content']
                        except (KeyError, IndexError) as e:
                            log_warning("Unexpected OpenRouter response structure:" + str(result))
                            return ""
                    else:
                        text = await resp.text()
                        log_warning(f"OpenRouter API error: {resp.status} - {text}")
                        return ""
        except Exception as e:
            log_warning("Error calling OpenRouter: " + str(e))
            return ""

# ------------------------------
# Asynchronous Helper Functions
# ------------------------------
def get_predefined_variations(base_query: str) -> list:
    coin = base_query.strip().lower()
    mapping = {
        "bitcoin": ["Bitcoin", "BTC", "Bitcoin Price News", "Bitcoin News"],
        "xrp": ["XRP", "XRP news", "Ripple"],
        "ethereum": ["Ethereum", "ETH", "Ethereum news"],
        "dogecoin": ["Dogecoin", "DOGE", "Dogecoin news", "Dogecoin price prediction"]
    }
    return mapping.get(coin, None)

async def generate_alternate_queries_async(base_query: str) -> list:
    predefined = get_predefined_variations(base_query)
    if predefined:
        st.write(f"Using predefined variations for {base_query}: {predefined}")
        return predefined
    else:
        prompt = (
            f"Rewrite the query '{base_query}' into at least three distinct search queries that capture different angles. "
            "Return your answer as a JSON array of strings with no additional text. For example: "
            f"[\"{base_query} news\", \"{base_query} price\", \"{base_query} analysis\"]"
        )
        result = await call_openrouter_llm(prompt)
        st.write("Raw output for alternate queries:", result)
        try:
            queries = json.loads(result.strip())
            if isinstance(queries, list) and len(queries) >= 3:
                return queries
            else:
                log_warning("OpenRouter did not return at least three queries. Using the base query as fallback.")
                return [base_query]
        except Exception as e:
            log_warning(f"Error parsing alternate queries: {e}. Using the base query.")
            return [base_query]

async def perform_search_async(query: str, location: str) -> list:
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "gl": location,
        "hl": "en",
        "tbs": "qdr:d",
        "sort": "date",
        "num": 10
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search", params=params) as resp:
                if resp.status == 200:
                    results = await resp.json()
                    output = []
                    if "organic_results" in results:
                        for item in results["organic_results"]:
                            link = item.get("link")
                            position = item.get("position", "N/A")
                            title = item.get("title", "")
                            output.append({"link": link, "position": position, "title": title})
                        return output[:5]
                    else:
                        return []
                else:
                    st.error(f"SERPAPI error: {resp.status}")
                    return []
    except Exception as e:
        st.error(f"Search error for location {location}: {e}")
        return []

async def translate_title(title: str) -> str:
    prompt = f"Translate the following title into English: \"{title}\". Only return the translated title."
    result = await call_openrouter_llm(prompt)
    return result.strip()

# ------------------------------
# Grouping Functionality using Fuzzy Matching
# ------------------------------
def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'[^a-z0-9\s]', '', title)
    return title.strip()

def group_articles(rows, similarity_threshold=55):
    groups = []
    assigned = set()
    for i, row in enumerate(rows):
        if i in assigned:
            continue
        current_group = [i]
        assigned.add(i)
        title_i = normalize_title(row["Title (English)"])
        for j in range(i+1, len(rows)):
            if j in assigned:
                continue
            title_j = normalize_title(rows[j]["Title (English)"])
            score = fuzz.token_set_ratio(title_i, title_j)
            if score >= similarity_threshold:
                current_group.append(j)
                assigned.add(j)
        groups.append(current_group)
    return groups

def select_top_article(rows, group):
    best_index = group[0]
    best_position = float('inf')
    for i in group:
        try:
            pos = float(rows[i]["Position"])
        except ValueError:
            pos = float('inf')
        if pos < best_position:
            best_position = pos
            best_index = i
    return best_index

# ------------------------------
# New Function: Select Overall Top Articles
# ------------------------------
async def select_top_articles(articles: list, top_n: int = 7) -> list:
    """
    Given a list of article dictionaries (each representing a top article from a group),
    prepare a summary and ask OpenRouter to select the top_n overall articles.
    If the LLM returns an empty or unparseable result, fall back to sorting by SERP position.
    Returns a list (with at most top_n objects) of selected articles with an added key 'Reason'.
    """
    if not articles:
        return []
    
    summary_lines = []
    for article in articles:
        line = (
            f"Country: {article.get('Country')}, "
            f"Keyword: {article.get('Keyword')}, "
            f"Position: {article.get('Position')}, "
            f"Title: {article.get('Title (English)')}, "
            f"Star Story: {article.get('Star Story?')}"
        )
        summary_lines.append(line)
    summary_text = "\n".join(summary_lines)
    
    prompt = (
        f"Below is a list of top articles from different groups. Each line describes an article with details: "
        f"Country, Keyword, SERP position (lower is better), Title, and whether it is a Star Story (appears multiple times).\n\n"
        f"{summary_text}\n\n"
        f"Based on this information, please select the top {top_n} most important news articles for editorial assignment. "
        "Choose one primary article per major event if applicable, and provide a brief reason why that article was selected. "
        "Return your answer as a JSON array of objects, where each object has the keys: 'Country', 'Keyword', 'Position', "
        "'Title (English)', 'URL', and 'Reason'. Only return the JSON array."
    )
    result = await call_openrouter_llm(prompt)
    try:
        selected = json.loads(result.strip())
        if isinstance(selected, list) and len(selected) > 0:
            return selected[:top_n]
        else:
            raise ValueError("Empty selection")
    except Exception as e:
        log_warning(f"Error parsing top article selection: {e}")
        # Fallback: sort the articles by SERP position and return top_n articles.
        try:
            sorted_articles = sorted(articles, key=lambda x: float(x.get("Position", float('inf'))))
        except Exception as e:
            sorted_articles = articles
        fallback = sorted_articles[:top_n]
        for article in fallback:
            article["Reason"] = "Fallback selection based on SERP position"
        return fallback

# ------------------------------
# Main Asynchronous Routine for a Single Keyword
# ------------------------------
async def run_monitoring(base_query: str):
    locations = {"US": "us", "UK": "uk", "Germany": "de", "Netherlands": "nl"}
    alternate_queries = await generate_alternate_queries_async(base_query)
    st.write(f"Alternate queries generated for {base_query}: {alternate_queries}")
    
    aggregated_results = []
    for loc_name, loc_code in locations.items():
        for query in alternate_queries:
            search_results = await perform_search_async(query, loc_code)
            for res in search_results:
                row = {
                    "Country": loc_name,
                    "Keyword": query,
                    "Position": res.get("position"),
                    "URL": res.get("link"),
                    "SERP Title": res.get("title")
                }
                aggregated_results.append(row)
    
    async def enrich_row(row):
        translated_title = await translate_title(row["SERP Title"])
        row["Title (English)"] = translated_title if translated_title else row["SERP Title"]
        return row

    enriched_rows = await asyncio.gather(*(enrich_row(row) for row in aggregated_results))
    
    # Determine Star Story status based on duplicate URLs
    url_counts = {}
    for row in enriched_rows:
        url = row["URL"]
        url_counts[url] = url_counts.get(url, 0) + 1
    for row in enriched_rows:
        row["Star Story?"] = "Yes" if url_counts.get(row["URL"], 0) > 1 else "No"
    
    # Group similar articles using fuzzy matching on "Title (English)"
    groups = group_articles(enriched_rows, similarity_threshold=55)
    st.write("Article groups (by row indices):", groups)
    
    # Assign group labels
    for group_id, group in enumerate(groups, start=1):
        for idx in group:
            enriched_rows[idx]["Group"] = f"Group {group_id}"
    
    # Create assignment summary: select top article (lowest SERP position) per group
    assignment_summary = []
    for group in groups:
        top_idx = select_top_article(enriched_rows, group)
        assignment_summary.append(enriched_rows[top_idx])
    
    # Use the LLM to select overall top articles from the assignment summary
    top_articles = await select_top_articles(assignment_summary, top_n=7)
    
    # Prepare final rows for display
    final_rows = []
    for row in enriched_rows:
        final_rows.append({
            "Country": row["Country"],
            "Keyword": row["Keyword"],
            "Position": row["Position"],
            "Title (English)": row["Title (English)"],
            "URL": row["URL"],
            "Star Story?": row["Star Story?"],
            "Group": row.get("Group", ""),
            "Owner": ""  # leave blank
        })
    
    return final_rows, assignment_summary, top_articles

# ------------------------------
# Main Routine for Multiple Keywords
# ------------------------------
async def run_monitoring_multiple(keywords: list):
    all_rows = []
    assignment_all = []
    top_all = []
    for crypto in keywords:
        rows, assignment_summary, top_articles = await run_monitoring(crypto.strip())
        for row in rows:
            row["Crypto"] = crypto.strip()
        all_rows.extend(rows)
        for row in assignment_summary:
            row["Crypto"] = crypto.strip()
        assignment_all.extend(assignment_summary)
        for row in top_articles:
            row["Crypto"] = crypto.strip()
        top_all.extend(top_articles)
    return all_rows, assignment_all, top_all

# ------------------------------
# Streamlit Interface
# ------------------------------
def main():
    st.title("Top Stories Monitor")
    st.write(
        "This full-screen app monitors top Google news stories for one or more cryptocurrencies from the US, UK, Germany, and Netherlands (articles published today). "
        "For each result, it translates the article title into English and displays the following columns:\n\n"
        "Country | Keyword | Position | Title (English) | URL | Star Story? | Group | Owner\n\n"
        "Articles appearing under more than one query variant are marked as a Star Story. "
        "Similar articles are grouped together. An assignment summary is provided below to help you pick the top article from each group, "
        "and an overall top-7 selection is also provided."
    )
    
    crypto_input = st.text_input("Enter crypto keywords (comma separated)", value="XRP")
    keywords = [k.strip() for k in crypto_input.split(",") if k.strip()]
    
    if st.button("Run Monitoring"):
        with st.spinner("Fetching and processing data..."):
            table_data, assignment_summary, top_articles = asyncio.run(run_monitoring_multiple(keywords))
        st.success("Monitoring complete!")
        df = pd.DataFrame(table_data)
        st.dataframe(df)
        
        st.subheader("Assignment Summary (Top Article per Group)")
        df_assign = pd.DataFrame(assignment_summary)
        st.dataframe(df_assign)
        
        st.subheader("Overall Top 7 Selection")
        df_top = pd.DataFrame(top_articles)
        st.dataframe(df_top)

if __name__ == "__main__":
    main()
