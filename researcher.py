import os
import re
import streamlit as st
from dotenv import load_dotenv
import tenacity
import time
import requests
import aiohttp
import json
from serpapi import GoogleSearch  # Use the official import
from datetime import datetime
import asyncio
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Global error log list
error_log = []

def log_error(message):
    """
    Log errors to the global error_log list.
    Rate-limit errors (429 and ResourceExhausted) are only printed to console.
    """
    timestamp = datetime.now().isoformat()
    formatted_message = f"[{timestamp}] {message}"
    if "429" in message or "ResourceExhausted" in message:
        print(f"Rate limit error (hidden): {formatted_message}")
    else:
        error_log.append(formatted_message)
        print(formatted_message)

def log_progress(message):
    """
    Log progress messages to both console and UI.
    """
    timestamp = datetime.now().isoformat()
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    if 'progress_log' not in st.session_state:
        st.session_state.progress_log = []
    st.session_state.progress_log.append(formatted_message)

# Initialize clients (store only the SERP API key)
def init_clients():
    api_key = os.getenv('SERP_API_KEY')
    gemini_key = os.getenv('GOOGLE_GEMINI_API_KEY')
    if not api_key:
        st.error("Missing SERP_API_KEY in .env")
    if not gemini_key:
        st.error("Missing GOOGLE_GEMINI_API_KEY in .env")
    return {'serp': api_key, 'gemini': gemini_key}

# Initialize Gemini client
def init_gemini_client():
    """Initialize Google Gemini client."""
    try:
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_GEMINI_API_KEY')
        if not api_key:
            st.error("Gemini API key not found. Please set GEMINI_API_KEY in your environment variables.")
            return None
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return {
            'model': model,
            'name': 'gemini-2.0-flash'
        }
    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Failed to initialize Gemini client: {str(e)}")
        return None

def make_gemini_request(client, prompt):
    """Make a request to Gemini API."""
    try:
        response = client['model'].generate_content(prompt)
        if response and response.text:
            return response.text
        return None
    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Gemini request failed: {str(e)}")
        return None

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
    retry=tenacity.retry_if_exception_type(Exception)
)
def search_web(query, num_results=10):
    """
    Perform a SerpAPI search using the official GoogleSearch client.
    """
    params = {
        "q": query,
        "api_key": st.session_state.clients['serp'],
        "engine": "google",
        "num": num_results,
        "hl": "en",        # Host language English
        "gl": "us",        # Geolocation: United States
        "lr": "lang_en"    # Restrict results to English
    }
    print(f"[search_web] Searching for: {query}")
    return GoogleSearch(params).get_dict()

# Use Google Gemini's API to analyze content
@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=30, max=180),
    retry=tenacity.retry_if_exception_type(Exception)
)
def analyze_with_llm(query, context, word_range, keywords=None):
    """
    Use Google Gemini to generate a Markdown report of key findings.
    """
    keyword_instruction = ""
    if keywords:
        keyword_list = ", ".join(keywords)
        keyword_instruction = f"\nAdditionally, naturally integrate the following keywords: {keyword_list}."
    
    prompt = f"""Analyze the following research context and provide ONLY the key findings in a well-formatted Markdown report. Prioritize sources that provide concrete data, statistics, or numerical insights, and favor newer articles when multiple sources are available. For each source, extract concrete quotes, numerical data, or specific insights that add value to the reader. Use plain text for all content except for hyperlinks, which should be formatted as [SourceName](url). Do not use any Markdown formatting for bold, italics, or other styles; ensure that any characters that might trigger unintended Markdown formatting (such as asterisks, underscores, or backticks) are escaped if they are not part of a hyperlink.

Research Context:
{context}

Requirements:
- Write your key findings in Thai language.
- The report should be between {word_range} words.
- Each section must include 3-4 paragraphs with detailed insights. Each paragraph should incorporate concrete quotes, numerical data, or specific insights.
- Mention each source by embedding its name in the narrative using [SourceDomainName](url) in a natural, contextual manner.
- Keep technical terms and entity names in English; the rest should be in Thai.
- For monetary numbers, remove '$' and append " ดอลลาร์" (with a single space before and after the number).
- Do not include any explanations or disclaimers.
- Ensure that the Markdown is correctly escaped to prevent unintended formatting in Streamlit.
- Naturally integrate the keyword "{keywords[0] if keywords else ''}" into the report when discussing investment insights or market trends.
- Highlight any available figures or trends from the valid sources to make the report more informative.
- Conclude with a brief summary that highlights key data points, risks, and recommendations, while reiterating the significance of the keyword.

Return ONLY the Markdown formatted report.
"""
    try:
        gemini_client = init_gemini_client()
        if gemini_client:
            response = make_gemini_request(gemini_client, prompt)
            if response:
                return response
        return "Analysis failed: No response from API"
    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Analysis failed: {str(e)}")
        return f"Analysis failed: {str(e)}"

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
    retry=tenacity.retry_if_exception_type(Exception)
)
async def generate_search_queries(session, original_query):
    """
    Generate multiple search queries using Google Gemini based on the English query.
    """
    prompt = f"""As an expert researcher, generate 3-5 precise search queries based on:
{original_query}

Return ONLY a Python list of strings. Example: ['query 1', 'query 2']"""
    try:
        gemini_client = init_gemini_client()
        if gemini_client:
            response = make_gemini_request(gemini_client, prompt)
            if response:
                if response.startswith('[') and response.endswith(']'):
                    queries = [q.strip(" '") for q in response[1:-1].split(',')]
                    return [q for q in queries if q][:5]
                return [original_query]
        return [original_query]
    except asyncio.TimeoutError:
        log_error(f"[{datetime.now().isoformat()}] Query generation timed out")
        return [original_query]
    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Query generation failed: {str(e)}")
        return [original_query]

async def extract_relevant_context_async(session, user_query, search_query, page_text):
    """
    Given the original query, the search query used, and the page content,
    have the LLM extract all information relevant for answering the user's query.
    Return only the relevant context as plain text without commentary.
    """
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    message = (
        f"User Query: {user_query}\nSearch Query: {search_query}\n\n"
        f"Webpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"
    )
    try:
        gemini_client = init_gemini_client()
        if gemini_client:
            response = make_gemini_request(gemini_client, message)
            if response:
                return response
        return ""
    except asyncio.TimeoutError:
        log_error(f"[{datetime.now().isoformat()}] Context extraction timed out")
        return ""
    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Context extraction failed: {str(e)}")
        return ""

def fix_unwanted_newlines(text):
    """
    Remove newlines that occur within words (e.g. 'S\nO\nL\nX' becomes 'SOLX').
    This regex replaces a newline that is preceded and followed by an alphanumeric character.
    """
    return re.sub(r'(?<=\w)\n(?=\w)', '', text)

async def main_async():
    try:
        st.session_state.clients = init_clients()
        if not st.session_state.clients['serp']:
            return

        with st.sidebar:
            st.header("Research Parameters")
            query = st.text_area(
                "Research Query",
                "reasons why meme coins performance can't be ignored",
                height=150
            )
            num_results = st.slider("Search Results (per sub-query)", 5, 10, 5)
            max_queries = st.slider("Max Generated Sub-Queries", 1, 5, 3)
            report_length = st.selectbox(
                "Report Length",
                ["Long (850-1500 words)", "Medium (450-800 words)", "Short (200-400 words)"],
                index=1
            )
            keywords_input = st.text_area("Keywords (one per line)", "เหรียญคริปโตที่น่าลงทุน", height=80, help="Enter up to 3 keywords, one per line")
        
        keywords = [line.strip() for line in keywords_input.splitlines() if line.strip()] if keywords_input else []
        if len(keywords) > 3:
            keywords = keywords[:3]
        
        st.title("Deep Researcher Pro")
        
        if st.button("Start Analysis"):
            global error_log
            error_log = []
            
            english_query = query  # using the Thai text directly
            
            # Clear previous progress log
            st.session_state.progress_log = []
            
            # STEP 1: Generate Search Queries
            with st.spinner("Generating search queries..."):
                log_progress("[main] Starting to generate search queries...")
                async with aiohttp.ClientSession() as session:
                    try:
                        queries = await generate_search_queries(session, english_query)
                    except Exception as e:
                        log_error(f"[{datetime.now().isoformat()}] Query generation failed: {str(e)}")
                        st.error("Query generation failed. Please check the logs for details.")
                        return
                log_progress(f"[main] Generated queries: {queries}")
            
            # STEP 2: Perform the searches
            with st.spinner(f"Searching across {len(queries)} sub-queries..."):
                log_progress("[main] Searching across sub-queries...")
                all_organic = []
                for q in queries:
                    try:
                        search_results = search_web(q, num_results)
                        all_organic.extend(search_results.get('organic_results', []))
                    except Exception as e:
                        log_error(f"Search failed for '{q}': {str(e)}")
                seen_links = set()
                organic_results = []
                for result in all_organic:
                    if result.get('link') and result['link'] not in seen_links:
                        seen_links.add(result['link'])
                        organic_results.append(result)
                log_progress(f"[main] Found {len(organic_results)} unique organic results.")
            
            # STEP 3: Fetch webpage content
            with st.spinner("Fetching webpage content..."):
                log_progress("[main] Fetching webpage content for each result link...")
                urls = [r.get('link') for r in organic_results if r.get('link')]
                
                async def fetch_content():
                    async with aiohttp.ClientSession() as session:
                        tasks = []
                        for url in urls:
                            async def fetch(url):
                                try:
                                    # We'll wait up to 30 seconds for each response
                                    async with session.get(
                                        f"https://r.jina.ai/{url}",
                                        headers={"Authorization": f"Bearer {os.getenv('JINA_API_KEY')}"},
                                        timeout=30
                                    ) as response:
                                        return await asyncio.wait_for(response.text(), timeout=30)
                                except asyncio.TimeoutError:
                                    log_error(f"Timeout fetching {url}")
                                    return ""
                                except Exception as e:
                                    log_error(f"Error fetching {url}: {str(e)}")
                                    return ""
                            tasks.append(fetch(url))
                        return await asyncio.gather(*tasks)
                
                try:
                    webpage_texts = await fetch_content()
                except Exception as e:
                    log_error(f"[{datetime.now().isoformat()}] Webpage content fetching failed: {str(e)}")
                    st.error("Webpage content fetching failed. Please check the logs for details.")
                    return
                
                # Attach the fetched text to each result
                for i, result in enumerate(organic_results):
                    if i < len(webpage_texts) and webpage_texts[i]:
                        result['full_text'] = webpage_texts[i]
                    else:
                        result['full_text'] = result.get('snippet', '')
            
            # STEP 4: Evaluate relevance
            with st.spinner("Processing search results..."):
                log_progress("[main] Evaluating relevance of each search result...")
                valid_results = []
                relevance_scores = []
                
                for idx, result in enumerate(organic_results):
                    if not result.get('link'):
                        continue
                    content = f"Title: {result.get('title', '')}\nContent: {result.get('full_text', '')}"
                    prompt = f"""Evaluate relevance to query: {english_query}
Content: {content}

Output ONLY a numerical score between 0-1 with:
- 1 = Perfect match
- 0.7-0.9 = Strong relevance
- 0.4-0.6 = Partial relevance
- 0.1-0.3 = Weak relevance
- 0 = Irrelevant
"""
                    try:
                        gemini_client = init_gemini_client()
                        if gemini_client:
                            response = make_gemini_request(gemini_client, prompt)
                            if response:
                                try:
                                    score = float(response.strip())
                                except ValueError:
                                    score = 0.0
                            else:
                                score = 0.0
                            
                            if 0 <= score <= 1:
                                valid_results.append(result)
                                relevance_scores.append(score)
                            else:
                                log_error(f"Invalid score for result {idx+1}: {score}")
                    except Exception as e:
                        log_error(f"Scoring failed for result {idx+1}: {str(e)}")
                
                log_progress(f"[main] Valid results after scoring: {len(valid_results)}")
            
            # STEP 5: Generate final insights
            with st.spinner("Generating insights..."):
                log_progress("[main] Generating final insights from LLM...")
                context = "\n".join([
                    f"{r.get('title', '')}: {r.get('full_text', r.get('snippet', ''))[:500]} [Source]({r.get('link', '')})" 
                    for r in valid_results
                ])
                
                if report_length == "Short (200-400 words)":
                    word_range = "200-400"
                elif report_length == "Medium (450-800 words)":
                    word_range = "450-800"
                elif report_length == "Long (850-1500 words)":
                    word_range = "850-1500"
                else:
                    word_range = "450-800"
                
                markdown_report = analyze_with_llm(english_query, context, word_range, keywords=keywords)
                markdown_report = fix_unwanted_newlines(markdown_report)
            
            # Display progress log
            with st.expander("Progress Log", expanded=True):
                for log in st.session_state.progress_log:
                    st.write(log)

            # Display results
            st.subheader("Deep Report - Markdown Version")
            st.markdown(markdown_report, unsafe_allow_html=True)
            
            
            # Show any logged errors
            if error_log:
                with st.expander("Error Log (click to expand)", expanded=False):
                    for err in error_log:
                        st.write(err)

    except Exception as e:
        log_error(f"[{datetime.now().isoformat()}] Application error: {str(e)}")
        st.error("An unexpected error occurred. Please check the logs for details.")

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_async())
    finally:
        loop.close()

if __name__ == "__main__":
    main()
