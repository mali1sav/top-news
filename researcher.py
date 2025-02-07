import os
import streamlit as st
from dotenv import load_dotenv
import tenacity
import time
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import requests
from serpapi import GoogleSearch  # Use the official import
from transformers import GPT2Tokenizer
from datetime import datetime

# Load environment variables
load_dotenv()

# Global error log list
error_log = []

def log_error(message):
    """
    Log errors to the global error_log list.
    Rate-limit errors (429) are only printed to console.
    """
    if "429" in message:
        print(f"Rate limit error (hidden): {message}")
    else:
        error_log.append(message)
        print(message)

# Initialize clients
def init_clients():
    return {
        'gemini': init_gemini_client(),
        'jina': init_jina_client(),
        'serp': os.getenv('SERP_API_KEY')
    }

def init_gemini_client():
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.error("Missing GEMINI_API_KEY in .env")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash-exp')
    except Exception as e:
        log_error(f"Gemini init failed: {str(e)}")
        return None

def init_jina_client():
    if not os.getenv('JINA_API_KEY'):
        st.error("Missing JINA_API_KEY in .env")
        return None
    return True

def translate_to_english(query):
    """
    Translate a Thai query to English using the Gemini LLM.
    """
    prompt = f"Translate the following Thai text to English without any explanations:\n\n{query}"
    try:
        response = st.session_state.clients['gemini'].generate_content(prompt)
        translation = response.text.strip()
        return translation if translation else query
    except Exception as e:
        log_error(f"Translation failed: {str(e)}")
        return query

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
    retry=tenacity.retry_if_exception_type((Exception,))
)
def search_web(query, num_results=10):
    # Added parameters to target global English sources
    params = {
        "q": query,
        "api_key": st.session_state.clients['serp'],
        "engine": "google",
        "num": num_results,
        "hl": "en",        # Host language English
        "gl": "us",        # Geolocation: United States
        "lr": "lang_en"    # Restrict results to English
    }
    return GoogleSearch(params).get_dict()

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
    retry=tenacity.retry_if_exception_type((Exception,))
)
def analyze_with_gemini_dual(query, context, word_range, keywords=None):
    """
    Ask the LLM to produce a detailed analysis of the research context in two versions:
      - Version 1: Markdown format.
      - Version 2: Clean HTML for WordPress Gutenberg.
    The prompt instructs the LLM to extract specific quotes, numerical data, and detailed insights,
    and to clearly attribute each detail with a clickable hyperlink.
    The report should be between the specified word range (e.g., "1500+" for very long reports).
    """
    keyword_instruction = ""
    if keywords:
        keyword_list = ", ".join(keywords)
        keyword_instruction = f"\nAdditionally, integrate the following keywords naturally: {keyword_list}."
    
    prompt = f"""Analyze this research context and provide ONLY key findings in two versions.
In your analysis, for each source include concrete details such as specific quotes, numerical data, examples, or relevant insights that add value.
Clearly indicate the source for each detail using a clickable hyperlink in the following format:
(อ้างอิง: [SourceName](https://sourceurl.com)).
Keep technical terms and entity names in English.
For monetary numbers, remove '$' and add "ดอลลาร์" after the number with a single space before and after.

Requirements for both versions:
- Write your key findings in Thai language in 3-5 sections.
- The report should be between {word_range} words.
- Each section must include 2-4 paragraphs with detailed insights from each source.
- No explanations or disclaimers.
- Follow the example format strictly:
  • Detailed finding with quote or data [SourceName](https://url.com){keyword_instruction}

Produce two versions of the report:
Version 1 (Markdown):
Provide the report in Markdown format using appropriate markdown syntax.

Version 2 (WordPress Gutenberg):
Provide the same report in clean HTML structure for WordPress Gutenberg. 
Use proper <p>, <ul>, <li>, <table> tags as needed, and do not use any markdown syntax.

Separate the two versions with the delimiter: ---GUTENBERG---

Research Context:
{context}

User Query: {query}
"""
    response = st.session_state.clients['gemini'].generate_content(prompt).text
    if "---GUTENBERG---" in response:
        markdown_version, gutenberg_version = response.split("---GUTENBERG---", 1)
    else:
        markdown_version = response
        gutenberg_version = response
    return markdown_version.strip(), gutenberg_version.strip()

def get_freshness_score(result):
    """
    Calculate a freshness score for the result based on its published date.
    Expects result.get('date') in a known format.
    Attempts multiple formats and returns 0.5 if parsing fails.
    """
    date_str = result.get('date')
    if date_str:
        for fmt in ("%Y-%m-%d", "%b %d, %Y"):
            try:
                published_date = datetime.strptime(date_str, fmt)
                days_old = (datetime.now() - published_date).days
                if days_old <= 7:
                    return 1.0
                elif days_old >= 365:
                    return 0.0
                else:
                    return 1 - (days_old - 7) / (365 - 7)
            except ValueError:
                continue
        return 0.5
    else:
        return 0.5

@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
    retry=tenacity.retry_if_exception_type((Exception,))
)
async def generate_search_queries(session, original_query):
    """Generate multiple search queries using Gemini based on the English query."""
    prompt = f"""As an expert researcher, generate 3-5 precise search queries based on:
{original_query}

Return ONLY a Python list of strings. Example: ['query 1', 'query 2']"""
    try:
        response = await st.session_state.clients['gemini'].generate_content_async(prompt)
        if response.text:
            cleaned_response = response.text.replace('"', "'").strip()
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                queries = [q.strip(" '\"") for q in cleaned_response[1:-1].split(',')]
                return [q for q in queries if q][:5]
            return [original_query]
    except Exception as e:
        log_error(f"Query generation failed: {str(e)}")
        return [original_query]

async def extract_relevant_context_async(session, user_query, search_query, page_text):
    """
    Given the original query, the search query used, and the page content,
    have the LLM extract all information relevant for answering the query.
    """
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    message = (f"User Query: {user_query}\nSearch Query: {search_query}\n\n"
               f"Webpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}")
    try:
        response = await st.session_state.clients['gemini'].generate_content_async(message)
        if response.text:
            return response.text.strip()
        return ""
    except Exception as e:
        log_error(f"Extracting relevant context failed: {str(e)}")
        return ""

def main():
    st.set_page_config(page_title="Deep Researcher Pro", layout="wide")
    
    if 'clients' not in st.session_state:
        st.session_state.clients = init_clients()
    
    with st.sidebar:
        st.header("Research Parameters")
        query = st.text_area("Research Query (ภาษาไทย)", "Bitcoin ยังน่าลงทุนอยู่หรือไม่?", height=150)
        num_results = st.slider("Search Results (per sub-query)", 5, 50, 10)
        max_queries = st.slider("Max Generated Queries", 1, 5, 3)
        report_length = st.selectbox(
            "Report Length",
            ["Short (200-300 words)", "Medium (400-600 words)", "Long (800-1000 words)"],
            index=1  # default to "Medium (400-600 words)"
        )
        # For now, only a single option for report format is used (LLM chooses the best format)
        report_format = st.selectbox("Report Format", ["Default (LLM chooses)"])
        keywords_input = st.text_area("Keywords (one per line)", "", height=80, help="Enter up to 3 keywords, one per line")
    
    keywords = [line.strip() for line in keywords_input.splitlines() if line.strip()] if keywords_input else []
    if len(keywords) > 3:
        keywords = keywords[:3]
    
    st.title("Deep Researcher Pro")
    
    if st.button("Start Analysis"):
        global error_log
        error_log = []
        
        with st.spinner("Translating query to English..."):
            english_query = translate_to_english(query)
            st.write(f"**Translated Query:** {english_query}")
        
        with st.spinner("Generating search queries..."):
            import aiohttp, asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def generate_queries():
                async with aiohttp.ClientSession() as session:
                    return await generate_search_queries(session, english_query)
            all_queries = loop.run_until_complete(generate_queries())[:max_queries]
        
        with st.spinner(f"Searching across {len(all_queries)} queries..."):
            all_organic = []
            for q in all_queries:
                try:
                    search_results = search_web(q, num_results)
                    all_organic.extend(search_results.get('organic_results', []))
                except Exception as e:
                    log_error(f"Search failed for '{q}': {str(e)}")
            seen_links = set()
            organic_results = []
            for result in all_organic:
                if result.get('link') not in seen_links:
                    seen_links.add(result['link'])
                    organic_results.append(result)
        
        with st.spinner("Processing information..."):
            import aiohttp, asyncio
            urls = [r.get('link') for r in organic_results if r.get('link')]
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            async def fetch_content():
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for url in urls:
                        async def fetch(url):
                            try:
                                async with session.get(f"https://r.jina.ai/{url}", headers={
                                    "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}"
                                }) as response:
                                    return await response.text()
                            except Exception as e:
                                log_error(f"Error fetching {url}: {str(e)}")
                                return ""
                        tasks.append(fetch(url))
                    return await asyncio.gather(*tasks)
            webpage_texts = loop.run_until_complete(fetch_content())
            
            for i, result in enumerate(organic_results):
                result['full_text'] = webpage_texts[i] if i < len(webpage_texts) and webpage_texts[i] else result.get('snippet', '')
        
        with st.spinner("Evaluating relevance and freshness..."):
            valid_results = []
            relevance_scores = []
            freshness_scores = []
            for idx, result in enumerate(organic_results):
                if not result.get('link'):
                    continue
                content = f"Title: {result.get('title', '')}\nSnippet: {result.get('snippet', '')}"
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
                    response = st.session_state.clients['gemini'].generate_content(prompt)
                    score = float(response.text.strip())
                    if 0 <= score <= 1:
                        valid_results.append(result)
                        relevance_scores.append(score)
                        freshness_scores.append(get_freshness_score(result))
                    else:
                        log_error(f"Invalid score for result {idx+1}: {score}")
                except Exception as e:
                    log_error(f"Scoring failed for result {idx+1}: {str(e)}")
        
        with st.spinner("Generating insights..."):
            context = "\n".join([
                f"{r.get('title', '')}: {r.get('snippet', '')} [Source]({r.get('link', '')})" 
                for r in valid_results
            ])
            if report_length == "Short (200-300 words)":
                word_range = "200-300"
            elif report_length == "Medium (400-600 words)":
                word_range = "400-600"
            elif report_length == "Long (800-1000 words)":
                word_range = "800-1000"
            else:
                word_range = "400-600"
            markdown_report, gutenberg_report = analyze_with_gemini_dual(english_query, context, word_range, keywords=keywords)
        
        st.subheader("Deep Report - Markdown Version")
        st.markdown(markdown_report)
        st.subheader("Deep Report - WordPress Gutenberg Version")
        # Render the Gutenberg version as raw HTML using st.components.v1.html
        st.components.v1.html(gutenberg_report, height=800)
        
        if relevance_scores and valid_results:
            df = pd.DataFrame({
                'Title': [r.get('title', '') for r in valid_results],
                'Position': [r.get('position', 0) for r in valid_results],
                'Relevance': relevance_scores,
                'Freshness': freshness_scores
            })
            fig = px.scatter(df, x='Position', y='Relevance', size='Freshness', color='Title',
                             title="LLM Relevance & Freshness Analysis",
                             labels={'Relevance': 'LLM Relevance Score', 'Freshness': 'Freshness Score'})
            st.plotly_chart(fig)
        else:
            st.warning("No valid data available for visualization")
        
        if error_log:
            with st.expander("Error Log (click to expand)", expanded=False):
                for err in error_log:
                    st.write(err)

if __name__ == "__main__":
    main()
