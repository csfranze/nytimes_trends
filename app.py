import streamlit as st
import pandas as pd
from bertopic import BERTopic

model = BERTopic.load('ST_Data/model_dir')
df = pd.read_csv('ST_Data/reduced_df.csv')
docs = pd.read_csv('ST_Data/docs.csv')
docs_list = docs['cleaned_docs'].tolist()
docs_list = [str(doc) if not isinstance(doc, str) else doc for doc in docs_list]
timestamps = df.pub_date.to_list()

with open("ST_Data/short_ollama_labels.txt", "r") as file:
    ollama_labels = file.read().splitlines()

topic_dict = dict(zip(model.get_topic_info().Topic, ollama_labels))
model.set_topic_labels(topic_dict)
topics_over_time_df = pd.read_csv("ST_Data/topics_over_time.csv")

# Streamlit app
st.image("ST_Data/poweredby_nytimes_200a.png", width=200)
st.title("Emergent Topics for the Year")
st.subheader("Trends of Selected Topics")
st.sidebar.header("Select Topics")
selected_topics = st.sidebar.multiselect(
    "Choose topics to visualize",
    options=list(topic_dict.keys()),
    format_func=lambda x: topic_dict[x]
)

if selected_topics:
    st.write(model.visualize_topics_over_time(topics_over_time_df, topics=selected_topics, custom_labels=True, title=''))
else:
    st.write(model.visualize_topics_over_time(topics_over_time_df, topics=[8, 18], custom_labels=True, title=''))

st.subheader("Recent Articles on Selected Topics")
# Display the topics and their corresponding entries
for topic in selected_topics:
    st.write(f"Topic: {topic_dict[topic]}")
    filtered_entries = model.get_document_info(docs_list)[model.get_document_info(docs_list)['Topic'] == topic].tail(5).index.tolist()
    for entry in filtered_entries:
        st.markdown('['+str(df['headline'].iloc[entry])+']'+'('+str(df['web_url'].iloc[entry])+')')
