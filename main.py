import streamlit as st
import requests
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud
import nltk
nltk.download('punkt')

# === 스팀 리뷰 가져오는 함수 ===
def get_steam_reviews(appid, num_reviews=20):
    url = f"https://store.steampowered.com/appreviews/{appid}"
    params = {
        'json': 1,
        'num_per_page': num_reviews,
        'filter': 'recent',  # 최신순
        'language': 'all'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error("Failed to connect to Steam server.")
        return []

    data = response.json()
    reviews = data.get('reviews', [])

    result = []
    for review in reviews:
        content = review.get('review')
        voted_up = review.get('voted_up')
        if content:
            result.append((content, voted_up))

    return result

# === Sentiment analysis pipeline ===
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# === Main ===
st.title("Steam Review Sentiment Analysis")

# AppID 입력 받기 (웹 입력 창)
appid = st.text_input("Enter Steam Game AppID (e.g., 730 for CS:GO, 570 for Dota 2):", "730")

if appid:
    reviews = get_steam_reviews(appid, 100)

    if reviews:
        pos_count = 0
        neg_count = 0
        neutral_count = 0
        positive_texts = []
        negative_texts = []

        tokenizer = sentiment_pipeline.tokenizer

        for idx, (text, voted) in enumerate(reviews, 1):
            # 정확히 BERT tokenizer로 텍스트 자르기 (512 토큰 제한)
            encoded_input = tokenizer.encode(text, truncation=True, max_length=512, return_tensors='pt')
            truncated_text = tokenizer.decode(encoded_input[0], skip_special_tokens=True)

            result = sentiment_pipeline(truncated_text)[0]
            label = result['label']
            stars = int(label[0])

            sentiment = ""
            if stars >= 4:
                sentiment = "Positive"
                pos_count += 1
                positive_texts.append(truncated_text)
            elif stars <= 2:
                sentiment = "Negative"
                neg_count += 1
                negative_texts.append(truncated_text)
            else:
                sentiment = "Neutral"
                neutral_count += 1

            voted_state = "Recommended" if voted else "Not Recommended"

            st.write(f"{idx}. [{voted_state}] ({sentiment}) {truncated_text}")

        # === Summary ===
        st.write("\n--- Summary ---")
        st.write(f"Positive reviews: {pos_count}")
        st.write(f"Negative reviews: {neg_count}")
        st.write(f"Neutral reviews: {neutral_count}")

        # 파이 차트 추가
        fig, ax = plt.subplots()
        ax.pie([pos_count, neg_count, neutral_count], labels=['Positive', 'Negative', 'Neutral'], autopct='%1.1f%%', startangle=90, colors=['#8bc34a', '#f44336', '#ffeb3b'])
        ax.axis('equal')
        st.pyplot(fig)

        # === WordCloud for Positive Reviews ===
        positive_combined = " ".join(positive_texts)
        positive_wordcloud = WordCloud(width=800, height=800, background_color='white').generate(positive_combined)
        st.subheader("Positive Review WordCloud")
        st.image(positive_wordcloud.to_array(), use_container_width=True)

        # === WordCloud for Negative Reviews ===
        negative_combined = " ".join(negative_texts)
        negative_wordcloud = WordCloud(width=800, height=800, background_color='white').generate(negative_combined)
        st.subheader("Negative Review WordCloud")
        st.image(negative_wordcloud.to_array(), use_container_width=True)

    else:
        st.error("No reviews found.")
