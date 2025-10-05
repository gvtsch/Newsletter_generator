import streamlit as st
from helper_fncs import *


def main():

    st.set_page_config(page_title="AI Newsletter Generator", layout="centered")
    st.header("📰 AI Newsletter Generator")
    query = st.text_input("Enter a topic for the newsletter:", value="LangChain Deep Agents")
    if st.button("Generate Newsletter"):
        with st.spinner("Generating newsletter..."):
            query = query
            resp = search_serp(query)
            urls = pick_best_articles_urls(response_json=resp, query=query)
            data = get_article_content(urls=urls)
            summaries = summarize_content(db_faiss=data, query=query, k=3)
            newsletter = generate_newsletter(summaries=summaries, query=query)

            st.success("Newsletter generated!")

            with st.expander("Search resulsts"):
                st.info(resp)

            with st.expander("Top Article URLs"):
                for url in urls:
                    st.info(url)
             
            with st.expander("Data"):
                data_raw = " ".join(d.page_content for d in data.similarity_search(query,k=4))
                st.info(data_raw)

            with st.expander("Article Summaries"):
                st.info(summaries)

            with st.expander("See Full Newsletter"):
                st.markdown(newsletter)

if __name__ == "__main__":
    main()